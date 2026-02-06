"""
Unit tests for 02-full_python_pipeline.py DLT pipeline.

This module tests the Python-based DLT CDC pipeline functionality including:
- Dynamic pipeline creation for multiple tables
- Data quality expectations (@dlt.expect_or_drop)
- APPLY CHANGES logic with SCD Type 1 and SCD Type 2
- Multi-table processing patterns

Tests use pytest fixtures and follow the dbx_test framework patterns.
These tests validate the transformation logic independently of DLT runtime.

Note: DLT decorators (@dlt.table, @dlt.view) cannot be tested directly,
so we test the underlying transformation logic using equivalent PySpark code.
"""

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, row_number, when, lit, current_timestamp,
    max as spark_max, expr, struct
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType, DoubleType
)
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
from unittest.mock import MagicMock, patch


# ============================================================================
# Helper Functions - Simulating Python DLT Pipeline Logic
# ============================================================================

def create_raw_cdc_reader(table_name: str, catalog: str, schema: str) -> str:
    """
    Generate the path for raw CDC data.
    Simulates: spark.readStream.format("cloudFiles").load(path)
    
    Args:
        table_name: Name of the source table
        catalog: Catalog name
        schema: Schema name
        
    Returns:
        Path string for the raw data location
    """
    return f"/Volumes/{catalog}/{schema}/raw_data/{table_name}"


def apply_dlt_expectations(df: DataFrame,
                           no_rescued_data: bool = True,
                           valid_id: bool = True,
                           valid_operation: bool = True) -> DataFrame:
    """
    Apply DLT data quality expectations.
    Simulates:
        @dlt.expect_or_drop("no_rescued_data", "_rescued_data IS NULL")
        @dlt.expect_or_drop("valid_id", "id IS NOT NULL")
        @dlt.expect_or_drop("valid_operation", "operation IN ('APPEND', 'DELETE', 'UPDATE')")
    
    Args:
        df: Input DataFrame
        no_rescued_data: Apply _rescued_data IS NULL check
        valid_id: Apply id IS NOT NULL check
        valid_operation: Apply operation IN (...) check
        
    Returns:
        DataFrame with invalid records filtered out
    """
    valid_ops = ['APPEND', 'DELETE', 'UPDATE']
    
    result = df
    
    if no_rescued_data and "_rescued_data" in df.columns:
        result = result.filter(col("_rescued_data").isNull())
    
    if valid_id:
        result = result.filter(col("id").isNotNull())
    
    if valid_operation:
        result = result.filter(col("operation").isin(valid_ops))
    
    return result


def apply_changes_scd1(source_df: DataFrame,
                       target_df: DataFrame,
                       keys: List[str] = ["id"],
                       sequence_by: str = "operation_date",
                       delete_expr: str = "operation = 'DELETE'",
                       except_columns: List[str] = None) -> DataFrame:
    """
    Simulate dlt.apply_changes() with default SCD Type 1 behavior.
    
    Args:
        source_df: Source DataFrame with CDC operations
        target_df: Target DataFrame (current state)
        keys: Key columns for matching
        sequence_by: Column for ordering (latest wins)
        delete_expr: Expression to identify DELETE operations
        except_columns: Columns to exclude from result
        
    Returns:
        Merged DataFrame
    """
    if except_columns is None:
        except_columns = ["operation", "operation_date", "_rescued_data"]
    
    # Deduplicate source by sequence
    window_spec = Window.partitionBy(*keys).orderBy(col(sequence_by).desc())
    source_deduped = (source_df
                      .withColumn("_rank", row_number().over(window_spec))
                      .filter(col("_rank") == 1)
                      .drop("_rank"))
    
    # Handle deletes
    deletes = source_deduped.filter(expr(delete_expr))
    delete_ids = set(row["id"] for row in deletes.select(keys).collect())
    
    # Handle upserts (non-deletes)
    upserts = source_deduped.filter(~expr(delete_expr))
    upsert_ids = set(row["id"] for row in upserts.select(keys).collect())
    
    # Filter target
    all_affected = delete_ids | upsert_ids
    target_filtered = target_df.filter(~col("id").isin(list(all_affected)))
    
    # Prepare upserts
    existing_except = [c for c in except_columns if c in upserts.columns]
    upserts_prepared = upserts.drop(*existing_except)
    
    # Align schemas
    result_cols = target_filtered.columns
    upserts_final = upserts_prepared.select(
        *[col(c) if c in upserts_prepared.columns else lit(None).alias(c)
          for c in result_cols]
    )
    
    return target_filtered.union(upserts_final)


def apply_changes_scd2(source_df: DataFrame,
                       target_df: DataFrame,
                       keys: List[str] = ["id"],
                       sequence_by: str = "operation_date",
                       delete_expr: str = "operation = 'DELETE'",
                       except_columns: List[str] = None) -> DataFrame:
    """
    Simulate dlt.apply_changes() with stored_as_scd_type="2".
    
    Maintains full history with __START_AT and __END_AT columns.
    
    Args:
        source_df: Source DataFrame with CDC operations
        target_df: Target SCD2 DataFrame
        keys: Key columns for matching
        sequence_by: Column for ordering
        delete_expr: Expression to identify DELETE operations
        except_columns: Columns to exclude from result
        
    Returns:
        SCD2 DataFrame with history
    """
    if except_columns is None:
        except_columns = ["operation", "operation_date", "_rescued_data"]
    
    # Deduplicate source
    window_spec = Window.partitionBy(*keys).orderBy(col(sequence_by).desc())
    source_deduped = (source_df
                      .withColumn("_rank", row_number().over(window_spec))
                      .filter(col("_rank") == 1)
                      .drop("_rank"))
    
    result_records = []
    source_records = source_deduped.collect()
    target_records = {}
    
    if target_df.count() > 0:
        for row in target_df.filter(col("__END_AT").isNull()).collect():
            target_records[row["id"]] = row
    
    # Get all historical records (already closed)
    if target_df.count() > 0:
        historical = target_df.filter(col("__END_AT").isNotNull()).collect()
        result_records.extend([row.asDict() for row in historical])
    
    for src_row in source_records:
        record_id = src_row["id"]
        change_ts = src_row[sequence_by]
        is_delete = eval(delete_expr.replace("operation", f"'{src_row['operation']}'"))
        
        # Close existing current record if present
        if record_id in target_records:
            closed = dict(target_records[record_id].asDict())
            closed["__END_AT"] = change_ts
            result_records.append(closed)
        
        # Add new record if not a delete
        if not is_delete:
            new_record = {
                "id": record_id,
                "name": src_row.get("name"),
                "address": src_row.get("address"),
                "email": src_row.get("email"),
                "__START_AT": change_ts,
                "__END_AT": None
            }
            result_records.append(new_record)
    
    # Add unchanged current records
    processed_ids = set(row["id"] for row in source_records)
    for rid, row in target_records.items():
        if rid not in processed_ids:
            result_records.append(row.asDict())
    
    if result_records:
        return target_df.sparkSession.createDataFrame(result_records, target_df.schema)
    return target_df


def get_table_names_from_folders(folder_listing: List[Any]) -> List[str]:
    """
    Extract table names from folder listing.
    Simulates: for folder in dbutils.fs.ls(path): table_name = folder.name[:-1]
    
    Args:
        folder_listing: List of folder info objects with .name attribute
        
    Returns:
        List of table names (folder names without trailing slash)
    """
    return [folder.name[:-1] for folder in folder_listing]


def create_dynamic_pipeline_config(table_names: List[str], 
                                   catalog: str, 
                                   schema: str) -> Dict[str, Dict]:
    """
    Create configuration for dynamic multi-table pipeline.
    
    Args:
        table_names: List of table names to process
        catalog: Catalog name
        schema: Schema name
        
    Returns:
        Dictionary with table configurations
    """
    return {
        table_name: {
            "raw_table": f"{table_name}_cdc",
            "clean_view": f"{table_name}_cdc_clean",
            "target_table": table_name,
            "source_path": create_raw_cdc_reader(table_name, catalog, schema)
        }
        for table_name in table_names
    }


# ============================================================================
# Test Class: Dynamic Pipeline Creation
# ============================================================================

class TestDynamicPipelineCreation:
    """
    Tests for dynamic pipeline creation pattern.
    The Python pipeline loops over folders to create pipelines dynamically.
    """
    
    def test_table_name_extraction_from_folders(self, mock_folder_listing):
        """
        Test that table names are correctly extracted from folder listing.
        
        Why: The pipeline discovers tables by listing /Volumes/.../raw_data/
        """
        # Act
        table_names = get_table_names_from_folders(mock_folder_listing)
        
        # Assert
        assert len(table_names) == 3
        assert "customers" in table_names
        assert "orders" in table_names
        assert "products" in table_names
    
    def test_trailing_slash_removed_from_folder_names(self, mock_folder_listing):
        """
        Test that trailing slashes are properly removed.
        
        Why: Folder names from dbutils.fs.ls() include trailing slash.
        """
        # Act
        table_names = get_table_names_from_folders(mock_folder_listing)
        
        # Assert: no trailing slashes
        for name in table_names:
            assert not name.endswith("/")
    
    def test_pipeline_config_generation(self, test_catalog, test_schema):
        """
        Test that pipeline configuration is correctly generated.
        
        Why: Each table needs proper naming for raw, clean, and target tables.
        """
        # Arrange
        table_names = ["customers", "orders"]
        
        # Act
        config = create_dynamic_pipeline_config(table_names, test_catalog, test_schema)
        
        # Assert
        assert "customers" in config
        assert config["customers"]["raw_table"] == "customers_cdc"
        assert config["customers"]["clean_view"] == "customers_cdc_clean"
        assert config["customers"]["target_table"] == "customers"
        assert f"/Volumes/{test_catalog}/{test_schema}/raw_data/customers" in config["customers"]["source_path"]
    
    @pytest.mark.parametrize("num_tables", [1, 5, 10, 20])
    def test_pipeline_scales_with_table_count(self, test_catalog, test_schema, num_tables):
        """
        Test that pipeline creation scales with number of tables.
        
        Why: Real deployments may have many tables to process.
        """
        # Arrange
        table_names = [f"table_{i}" for i in range(num_tables)]
        
        # Act
        config = create_dynamic_pipeline_config(table_names, test_catalog, test_schema)
        
        # Assert
        assert len(config) == num_tables
        for name in table_names:
            assert name in config


# ============================================================================
# Test Class: DLT Expectations
# ============================================================================

class TestDLTExpectations:
    """
    Tests for @dlt.expect_or_drop expectations in the Python pipeline.
    """
    
    def test_no_rescued_data_expectation(self, spark, generic_cdc_schema, base_timestamp):
        """
        Test @dlt.expect_or_drop("no_rescued_data", "_rescued_data IS NULL").
        
        Why: Records with rescued data indicate schema mismatch.
        """
        # Arrange
        data = [
            (1, "valid", "APPEND", base_timestamp, None),
            (2, "invalid", "APPEND", base_timestamp, '{"bad": "schema"}'),
        ]
        df = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act
        result = apply_dlt_expectations(df, no_rescued_data=True, 
                                        valid_id=False, valid_operation=False)
        
        # Assert
        assert result.count() == 1
        assert result.collect()[0]["id"] == 1
    
    def test_valid_id_expectation(self, spark, generic_cdc_schema, base_timestamp):
        """
        Test @dlt.expect_or_drop("valid_id", "id IS NOT NULL").
        
        Why: ID is required for deduplication and merge operations.
        """
        # Arrange
        data = [
            (1, "valid", "APPEND", base_timestamp, None),
            (None, "no_id", "APPEND", base_timestamp, None),
        ]
        df = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act
        result = apply_dlt_expectations(df, no_rescued_data=False,
                                        valid_id=True, valid_operation=False)
        
        # Assert
        assert result.count() == 1
        assert result.collect()[0]["id"] == 1
    
    def test_valid_operation_expectation(self, spark, generic_cdc_schema, base_timestamp):
        """
        Test @dlt.expect_or_drop("valid_operation", "operation IN (...)").
        
        Why: Only APPEND, UPDATE, DELETE are valid CDC operations.
        """
        # Arrange
        data = [
            (1, "append_ok", "APPEND", base_timestamp, None),
            (2, "update_ok", "UPDATE", base_timestamp, None),
            (3, "delete_ok", "DELETE", base_timestamp, None),
            (4, "invalid", "MERGE", base_timestamp, None),
            (5, "also_invalid", "INSERT", base_timestamp, None),
        ]
        df = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act
        result = apply_dlt_expectations(df, no_rescued_data=False,
                                        valid_id=False, valid_operation=True)
        
        # Assert
        assert result.count() == 3
        valid_ids = set(row["id"] for row in result.collect())
        assert valid_ids == {1, 2, 3}
    
    def test_all_expectations_combined(self, spark, generic_cdc_schema, base_timestamp):
        """
        Test all expectations applied together.
        
        Why: In production, all expectations are active simultaneously.
        """
        # Arrange
        data = [
            # Valid record
            (1, "valid", "APPEND", base_timestamp, None),
            # NULL id - fails valid_id
            (None, "no_id", "APPEND", base_timestamp, None),
            # Invalid operation - fails valid_operation
            (2, "bad_op", "INVALID", base_timestamp, None),
            # Has rescued data - fails no_rescued_data
            (3, "bad_schema", "APPEND", base_timestamp, '{"x": 1}'),
            # Another valid record
            (4, "also_valid", "UPDATE", base_timestamp, None),
        ]
        df = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act
        result = apply_dlt_expectations(df)
        
        # Assert
        assert result.count() == 2
        valid_ids = set(row["id"] for row in result.collect())
        assert valid_ids == {1, 4}


# ============================================================================
# Test Class: Apply Changes SCD Type 1
# ============================================================================

class TestApplyChangesSCD1:
    """
    Tests for dlt.apply_changes() with default SCD Type 1 (upsert) behavior.
    """
    
    def test_append_inserts_new_records(self, spark, customers_silver_schema,
                                        customers_cdc_schema, base_timestamp):
        """
        Test that APPEND operations insert new records.
        
        Why: New customer records should be added to the target table.
        """
        # Arrange
        source_data = [
            (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "456 Oak Ave", "bob@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert result.count() == 2
    
    def test_update_replaces_existing_records(self, spark, customers_silver_schema,
                                              customers_cdc_schema, base_timestamp):
        """
        Test that UPDATE operations replace existing records (SCD1).
        
        Why: SCD1 maintains only current state, no history.
        """
        # Arrange
        target_data = [(1, "Alice Old", "Old Address", "old@email.com")]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        source_data = [
            (1, "Alice New", "New Address", "new@email.com", "UPDATE", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["name"] == "Alice New"
    
    def test_delete_removes_records(self, spark, customers_silver_schema,
                                    customers_cdc_schema, base_timestamp):
        """
        Test that DELETE operations remove records.
        
        Why: Pipeline uses apply_as_deletes=expr("operation = 'DELETE'")
        """
        # Arrange
        target_data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
        ]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        source_data = [
            (1, None, None, None, "DELETE", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        assert result.filter(col("id") == 1).count() == 0
    
    def test_sequence_by_deduplication(self, spark, customers_silver_schema,
                                       customers_cdc_schema, base_timestamp):
        """
        Test that sequence_by=col("operation_date") correctly deduplicates.
        
        Why: Out-of-order CDC events must be resolved by timestamp.
        """
        # Arrange
        source_data = [
            (1, "Version 1", "Addr1", "v1@email.com", "APPEND", base_timestamp, None),
            (1, "Version 3 - LATEST", "Addr3", "v3@email.com", "UPDATE",
             base_timestamp + timedelta(hours=2), None),
            (1, "Version 2", "Addr2", "v2@email.com", "UPDATE",
             base_timestamp + timedelta(hours=1), None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["name"] == "Version 3 - LATEST"
    
    def test_except_columns_excluded(self, spark, customers_silver_schema,
                                     customers_cdc_schema, base_timestamp):
        """
        Test except_column_list=["operation", "operation_date", "_rescued_data"].
        
        Why: Target table should not contain CDC metadata columns.
        """
        # Arrange
        source_data = [
            (1, "Alice", "Addr", "alice@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert "operation" not in result.columns
        assert "operation_date" not in result.columns
        assert "_rescued_data" not in result.columns


# ============================================================================
# Test Class: Apply Changes SCD Type 2
# ============================================================================

class TestApplyChangesSCD2:
    """
    Tests for dlt.apply_changes() with stored_as_scd_type="2".
    """
    
    def test_initial_insert_creates_current_version(self, spark, scd2_schema,
                                                    customers_cdc_schema, base_timestamp):
        """
        Test that initial INSERT creates a current version record.
        
        Why: SCD2 tracks validity with __START_AT and __END_AT columns.
        """
        # Arrange
        source_data = [
            (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], scd2_schema)
        
        # Act
        result = apply_changes_scd2(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["__START_AT"] == base_timestamp
        assert row["__END_AT"] is None  # Current version
    
    def test_update_creates_history_chain(self, spark, scd2_schema,
                                          customers_cdc_schema, base_timestamp):
        """
        Test that UPDATE creates a new version and closes the previous.
        
        Why: SCD2 must maintain complete change history.
        """
        # Arrange: existing current version
        target_data = [
            (1, "Alice", "Old Address", "alice@email.com", base_timestamp, None),
        ]
        target_df = spark.createDataFrame(target_data, scd2_schema)
        
        # Update
        update_ts = base_timestamp + timedelta(days=30)
        source_data = [
            (1, "Alice", "New Address", "alice@email.com", "UPDATE", update_ts, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result = apply_changes_scd2(source_df, target_df)
        
        # Assert: 2 versions
        assert result.count() == 2
        
        # Old version closed
        old = result.filter(col("address") == "Old Address").collect()[0]
        assert old["__END_AT"] == update_ts
        
        # New version current
        new = result.filter(col("address") == "New Address").collect()[0]
        assert new["__START_AT"] == update_ts
        assert new["__END_AT"] is None
    
    def test_delete_closes_current_version(self, spark, scd2_schema,
                                           customers_cdc_schema, base_timestamp):
        """
        Test that DELETE closes the current version (soft delete).
        
        Why: SCD2 preserves history even for deleted records.
        """
        # Arrange
        target_data = [
            (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, None),
        ]
        target_df = spark.createDataFrame(target_data, scd2_schema)
        
        delete_ts = base_timestamp + timedelta(days=30)
        source_data = [
            (1, None, None, None, "DELETE", delete_ts, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result = apply_changes_scd2(source_df, target_df)
        
        # Assert: record closed, not removed
        assert result.count() == 1
        row = result.collect()[0]
        assert row["__END_AT"] == delete_ts
    
    def test_multiple_updates_build_complete_history(self, spark, scd2_schema,
                                                     customers_cdc_schema, base_timestamp):
        """
        Test building complete history through multiple updates.
        
        Why: Real customers have many changes over time.
        """
        # Process updates sequentially
        target_df = spark.createDataFrame([], scd2_schema)
        
        updates = [
            (base_timestamp, "APPEND", "Alice V1", "Addr1", "v1@email.com"),
            (base_timestamp + timedelta(days=30), "UPDATE", "Alice V2", "Addr2", "v2@email.com"),
            (base_timestamp + timedelta(days=60), "UPDATE", "Alice V3", "Addr3", "v3@email.com"),
            (base_timestamp + timedelta(days=90), "UPDATE", "Alice V4", "Addr4", "v4@email.com"),
        ]
        
        for ts, op, name, addr, email in updates:
            source_data = [(1, name, addr, email, op, ts, None)]
            source_df = spark.createDataFrame(source_data, customers_cdc_schema)
            source_df = apply_dlt_expectations(source_df)
            target_df = apply_changes_scd2(source_df, target_df)
        
        # Assert: 4 versions
        assert target_df.count() == 4
        
        # Only 1 current version
        current = target_df.filter(col("__END_AT").isNull())
        assert current.count() == 1
        assert current.collect()[0]["name"] == "Alice V4"
        
        # 3 historical versions
        historical = target_df.filter(col("__END_AT").isNotNull())
        assert historical.count() == 3


# ============================================================================
# Test Class: Multi-Table Processing
# ============================================================================

class TestMultiTableProcessing:
    """
    Tests for processing multiple tables in the dynamic pipeline.
    """
    
    def test_independent_table_processing(self, spark, generic_cdc_schema,
                                          base_timestamp, multi_table_names):
        """
        Test that each table is processed independently.
        
        Why: Changes to one table should not affect others.
        """
        # Arrange: create data for each table
        table_dataframes = {}
        for i, table_name in enumerate(multi_table_names):
            data = [
                (j + 1, f"{table_name}_data_{j}", "APPEND",
                 base_timestamp + timedelta(hours=i), None)
                for j in range(3)
            ]
            table_dataframes[table_name] = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act: process each table
        results = {}
        for table_name, df in table_dataframes.items():
            clean_df = apply_dlt_expectations(df)
            results[table_name] = clean_df.count()
        
        # Assert: each table has correct count
        for table_name in multi_table_names:
            assert results[table_name] == 3
    
    def test_table_isolation(self, spark, generic_cdc_schema, base_timestamp):
        """
        Test that tables are isolated (IDs don't conflict).
        
        Why: ID 1 in customers is different from ID 1 in orders.
        """
        # Arrange
        customers_data = [(1, "Customer 1", "APPEND", base_timestamp, None)]
        orders_data = [(1, "Order 1", "APPEND", base_timestamp, None)]
        
        customers_df = spark.createDataFrame(customers_data, generic_cdc_schema)
        orders_df = spark.createDataFrame(orders_data, generic_cdc_schema)
        
        # Act
        customers_clean = apply_dlt_expectations(customers_df)
        orders_clean = apply_dlt_expectations(orders_df)
        
        # Assert: both have ID 1, but they're different records
        assert customers_clean.collect()[0]["data"] == "Customer 1"
        assert orders_clean.collect()[0]["data"] == "Order 1"
    
    def test_pipeline_creation_for_discovered_tables(self, mock_folder_listing,
                                                     test_catalog, test_schema):
        """
        Test that pipeline configs are created for all discovered tables.
        
        Why: Pipeline dynamically creates processing for each folder.
        """
        # Arrange
        table_names = get_table_names_from_folders(mock_folder_listing)
        
        # Act
        configs = create_dynamic_pipeline_config(table_names, test_catalog, test_schema)
        
        # Assert: config for each table
        assert len(configs) == len(mock_folder_listing)
        for table_name in table_names:
            assert table_name in configs
            assert configs[table_name]["raw_table"].endswith("_cdc")
            assert configs[table_name]["clean_view"].endswith("_cdc_clean")


# ============================================================================
# Test Class: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""
    
    def test_empty_source_produces_no_changes(self, spark, customers_cdc_schema,
                                              customers_silver_schema):
        """
        Test that empty source DataFrame doesn't modify target.
        
        Why: Empty batches should be handled gracefully.
        """
        # Arrange
        target_data = [(1, "Alice", "Addr", "alice@email.com")]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        source_df = spark.createDataFrame([], customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert
        assert result.count() == 1
    
    def test_all_records_filtered_by_expectations(self, spark, generic_cdc_schema,
                                                  base_timestamp):
        """
        Test handling when all records fail expectations.
        
        Why: Bad data batches shouldn't corrupt the pipeline.
        """
        # Arrange: all invalid records
        data = [
            (None, "no_id", "APPEND", base_timestamp, None),
            (1, "bad_op", "INVALID", base_timestamp, None),
            (2, "bad_schema", "APPEND", base_timestamp, '{"x": 1}'),
        ]
        df = spark.createDataFrame(data, generic_cdc_schema)
        
        # Act
        result = apply_dlt_expectations(df)
        
        # Assert
        assert result.count() == 0
    
    def test_null_values_in_optional_columns(self, spark, customers_cdc_schema,
                                             customers_silver_schema, base_timestamp):
        """
        Test handling of NULL values in non-key columns.
        
        Why: Address and email may be NULL for some customers.
        """
        # Arrange
        source_data = [
            (1, "Alice", None, "alice@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "123 Main St", None, "APPEND", base_timestamp, None),
            (3, None, "456 Oak Ave", "charlie@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert: all records inserted
        assert result.count() == 3
    
    @pytest.mark.parametrize("batch_size", [100, 500, 1000])
    def test_large_batch_processing(self, spark, generic_cdc_schema, base_timestamp, batch_size):
        """
        Test processing of large CDC batches.
        
        Why: Verify performance with realistic data volumes.
        """
        # Arrange
        data = [
            (i, f"data_{i}", "APPEND", base_timestamp + timedelta(seconds=i), None)
            for i in range(batch_size)
        ]
        source_df = spark.createDataFrame(data, generic_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        # Act
        result_count = source_df.count()
        
        # Assert
        assert result_count == batch_size
    
    def test_same_timestamp_handling(self, spark, customers_cdc_schema,
                                     customers_silver_schema, base_timestamp):
        """
        Test handling of records with identical timestamps.
        
        Why: Verify deterministic behavior for edge case.
        """
        # Arrange: same timestamp for all versions
        source_data = [
            (1, "Version A", "Addr A", "a@email.com", "APPEND", base_timestamp, None),
            (1, "Version B", "Addr B", "b@email.com", "UPDATE", base_timestamp, None),
            (1, "Version C", "Addr C", "c@email.com", "UPDATE", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_dlt_expectations(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_changes_scd1(source_df, target_df)
        
        # Assert: exactly 1 record (deterministic selection)
        assert result.count() == 1


# ============================================================================
# Test Class: Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests for the Python pipeline."""
    
    def test_complete_customer_lifecycle(self, spark, customers_cdc_schema,
                                         customers_silver_schema, base_timestamp):
        """
        Test complete customer lifecycle: create → update → delete.
        
        Why: Validates full pipeline behavior for typical use case.
        """
        target = spark.createDataFrame([], customers_silver_schema)
        
        # Create customers
        create_batch = spark.createDataFrame([
            (1, "Alice", "Addr1", "alice@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "Addr2", "bob@email.com", "APPEND", base_timestamp, None),
            (3, "Charlie", "Addr3", "charlie@email.com", "APPEND", base_timestamp, None),
        ], customers_cdc_schema)
        create_batch = apply_dlt_expectations(create_batch)
        target = apply_changes_scd1(create_batch, target)
        
        assert target.count() == 3
        
        # Update Alice
        update_batch = spark.createDataFrame([
            (1, "Alice Updated", "New Addr", "alice.new@email.com", "UPDATE",
             base_timestamp + timedelta(hours=1), None),
        ], customers_cdc_schema)
        update_batch = apply_dlt_expectations(update_batch)
        target = apply_changes_scd1(update_batch, target)
        
        alice = target.filter(col("id") == 1).collect()[0]
        assert alice["name"] == "Alice Updated"
        
        # Delete Bob
        delete_batch = spark.createDataFrame([
            (2, None, None, None, "DELETE", base_timestamp + timedelta(hours=2), None),
        ], customers_cdc_schema)
        delete_batch = apply_dlt_expectations(delete_batch)
        target = apply_changes_scd1(delete_batch, target)
        
        assert target.count() == 2
        assert target.filter(col("id") == 2).count() == 0
    
    def test_scd2_complete_history(self, spark, customers_cdc_schema,
                                   scd2_schema, base_timestamp):
        """
        Test complete SCD2 history building.
        
        Why: Validates SCD2 for SCD2_customers table.
        """
        target = spark.createDataFrame([], scd2_schema)
        
        # Initial insert
        batch1 = spark.createDataFrame([
            (1, "Alice", "Addr V1", "alice@email.com", "APPEND", base_timestamp, None),
        ], customers_cdc_schema)
        batch1 = apply_dlt_expectations(batch1)
        target = apply_changes_scd2(batch1, target)
        
        # Update 1
        batch2 = spark.createDataFrame([
            (1, "Alice", "Addr V2", "alice@email.com", "UPDATE",
             base_timestamp + timedelta(days=30), None),
        ], customers_cdc_schema)
        batch2 = apply_dlt_expectations(batch2)
        target = apply_changes_scd2(batch2, target)
        
        # Update 2
        batch3 = spark.createDataFrame([
            (1, "Alice Smith", "Addr V3", "alice.smith@email.com", "UPDATE",
             base_timestamp + timedelta(days=60), None),
        ], customers_cdc_schema)
        batch3 = apply_dlt_expectations(batch3)
        target = apply_changes_scd2(batch3, target)
        
        # Assert: 3 versions in history
        assert target.count() == 3
        
        # 1 current version
        current = target.filter(col("__END_AT").isNull())
        assert current.count() == 1
        assert current.collect()[0]["name"] == "Alice Smith"
        
        # 2 historical versions
        historical = target.filter(col("__END_AT").isNotNull())
        assert historical.count() == 2

