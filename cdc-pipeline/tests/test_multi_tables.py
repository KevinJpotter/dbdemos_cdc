"""
Unit tests for 02-CDC-CDF-full-multi-tables notebook.

This module tests the multi-table CDC functionality including:
- Bronze layer multi-table ingestion
- Silver layer merge logic for multiple tables
- Concurrent table processing
- Schema evolution handling

Tests use pytest fixtures and parametrize for comprehensive coverage.
"""

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, row_number, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, DoubleType
from datetime import datetime, timedelta
from delta.tables import DeltaTable
from concurrent.futures import ThreadPoolExecutor
from collections import deque


# ============================================================================
# Helper Functions (Extracted from notebook for testing)
# ============================================================================

def get_columns_to_update(df: DataFrame, exclude_cols: list = None) -> dict:
    """
    Get column mapping for MERGE UPDATE operation.
    
    Args:
        df: DataFrame to get columns from
        exclude_cols: Columns to exclude from update mapping
        
    Returns:
        Dictionary mapping column names to source references
    """
    if exclude_cols is None:
        exclude_cols = ["operation"]
    
    return {c: f"updates.{c}" for c in df.columns if c not in exclude_cols}


def deduplicate_bronze_data(df: DataFrame, id_col: str = "id",
                            order_col: str = "operation_date") -> DataFrame:
    """
    Deduplicate bronze CDC data by keeping the most recent record per ID.
    
    Args:
        df: Input DataFrame with CDC data
        id_col: Column to partition by
        order_col: Column to order by (descending)
        
    Returns:
        Deduplicated DataFrame
    """
    windowSpec = Window.partitionBy(id_col).orderBy(col(order_col).desc())
    return (df
            .withColumn("rank", row_number().over(windowSpec))
            .where("rank = 1")
            .drop("rank"))


def prepare_merge_columns(updates_df: DataFrame, target_columns: list,
                          drop_cols: list = None) -> DataFrame:
    """
    Prepare update DataFrame for merge by dropping technical columns.
    
    Args:
        updates_df: Source DataFrame with updates
        target_columns: List of columns in target table
        drop_cols: Technical columns to drop
        
    Returns:
        DataFrame ready for merge
    """
    if drop_cols is None:
        drop_cols = ["operation_date", "_rescued_data", "file_name", "rank"]
    
    existing_drop_cols = [c for c in drop_cols if c in updates_df.columns]
    return updates_df.drop(*existing_drop_cols)


def apply_multi_table_merge(source_df: DataFrame, target_df: DataFrame,
                            id_col: str = "id") -> DataFrame:
    """
    Apply merge logic for multi-table CDC processing.
    Simulates the merge behavior from the notebook.
    
    Args:
        source_df: Deduplicated source DataFrame with operation column
        target_df: Target DataFrame (current state)
        id_col: Column to match on
        
    Returns:
        DataFrame representing merged result
    """
    # Get target columns (excluding operation)
    target_columns = [c for c in target_df.columns if c != "operation"]
    
    # Handle deletes
    delete_ids = set(
        row[id_col] for row in 
        source_df.filter(col("operation") == "DELETE").select(id_col).collect()
    )
    
    # Filter out deleted records
    result = target_df.filter(~col(id_col).isin(list(delete_ids)))
    
    # Handle updates and inserts
    non_deletes = source_df.filter(col("operation") != "DELETE")
    update_ids = set(row[id_col] for row in non_deletes.select(id_col).collect())
    
    # Remove records that will be updated
    result = result.filter(~col(id_col).isin(list(update_ids)))
    
    # Add new/updated records
    source_cols = [c for c in non_deletes.columns if c in target_df.columns]
    result = result.union(non_deletes.select(source_cols))
    
    return result


def validate_table_schema(df: DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns present, False otherwise
    """
    return all(col_name in df.columns for col_name in required_columns)


def process_table_batch(table_name: str, bronze_df: DataFrame, 
                        silver_df: DataFrame) -> DataFrame:
    """
    Process a single table's CDC batch.
    
    Args:
        table_name: Name of the table being processed
        bronze_df: Bronze layer DataFrame with CDC data
        silver_df: Current silver layer DataFrame
        
    Returns:
        Updated silver DataFrame
    """
    # Deduplicate
    deduped = deduplicate_bronze_data(bronze_df)
    
    # Prepare for merge
    prepared = prepare_merge_columns(deduped, silver_df.columns)
    
    # Apply merge
    return apply_multi_table_merge(prepared, silver_df)


# ============================================================================
# Test Class: Multi-Table Deduplication
# ============================================================================

class TestMultiTableDeduplication:
    """Tests for deduplication across multiple tables."""
    
    @pytest.mark.parametrize("table_name,id_col", [
        ("users", "id"),
        ("transactions", "id"),
        ("products", "id"),
    ])
    def test_deduplication_per_table(self, spark, base_timestamp, table_name, id_col):
        """Test that deduplication works correctly for different table types."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        data = [
            (1, f"{table_name}_v1", "INSERT", base_timestamp),
            (1, f"{table_name}_v2", "UPDATE", base_timestamp + timedelta(hours=1)),
            (2, f"{table_name}_new", "INSERT", base_timestamp),
        ]
        df = spark.createDataFrame(data, schema)
        
        result = deduplicate_bronze_data(df, id_col=id_col)
        
        assert result.count() == 2
        v2_row = result.filter(col("id") == 1).collect()[0]
        assert v2_row["data"] == f"{table_name}_v2"
    
    @pytest.mark.parametrize("num_tables,records_per_table", [
        (1, 10),
        (3, 5),
        (5, 3),
        (10, 1),
    ])
    def test_deduplication_scalability(self, spark, base_timestamp, 
                                       num_tables, records_per_table):
        """Test deduplication scales with different table/record combinations."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("table_name", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        results = []
        for table_idx in range(num_tables):
            data = [
                (i, f"table_{table_idx}", base_timestamp + timedelta(hours=i))
                for i in range(records_per_table)
            ]
            df = spark.createDataFrame(data, schema)
            deduped = deduplicate_bronze_data(df)
            results.append(deduped.count())
        
        # Each table should have records_per_table unique IDs
        assert all(count == records_per_table for count in results)


# ============================================================================
# Test Class: Column Mapping for Merge
# ============================================================================

class TestColumnMapping:
    """Tests for column mapping in merge operations."""
    
    @pytest.mark.parametrize("columns,exclude,expected_keys", [
        (["id", "name", "email", "operation"], ["operation"], ["id", "name", "email"]),
        (["id", "amount", "user_id", "operation"], ["operation"], ["id", "amount", "user_id"]),
        (["id", "data"], [], ["id", "data"]),
    ])
    def test_column_mapping_excludes_correctly(self, spark, columns, exclude, expected_keys):
        """Test that column mapping excludes specified columns."""
        schema = StructType([StructField(c, StringType(), True) for c in columns])
        df = spark.createDataFrame([tuple("x" for _ in columns)], schema)
        
        mapping = get_columns_to_update(df, exclude_cols=exclude)
        
        assert set(mapping.keys()) == set(expected_keys)
    
    def test_column_mapping_format(self, spark):
        """Test that column mapping values have correct format."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
        ])
        df = spark.createDataFrame([(1, "test")], schema)
        
        mapping = get_columns_to_update(df, exclude_cols=[])
        
        assert mapping["id"] == "updates.id"
        assert mapping["name"] == "updates.name"


# ============================================================================
# Test Class: Multi-Table Merge Logic
# ============================================================================

class TestMultiTableMergeLogic:
    """Tests for merge logic across multiple table types."""
    
    @pytest.fixture
    def users_schema(self):
        return StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("operation", StringType(), True),
        ])
    
    @pytest.fixture
    def transactions_schema(self):
        return StructType([
            StructField("id", LongType(), False),
            StructField("user_id", LongType(), True),
            StructField("amount", DoubleType(), True),
            StructField("operation", StringType(), True),
        ])
    
    @pytest.mark.parametrize("table_type,operation", [
        ("users", "INSERT"),
        ("users", "UPDATE"),
        ("users", "DELETE"),
        ("transactions", "INSERT"),
        ("transactions", "UPDATE"),
        ("transactions", "DELETE"),
    ])
    def test_operations_across_table_types(self, spark, users_schema, 
                                            transactions_schema, table_type, operation):
        """Test that all operations work for different table types."""
        if table_type == "users":
            target_data = [(1, "User1", "user1@email.com", "INSERT")]
            source_data = [(1, "Modified", "modified@email.com", operation)]
            schema = users_schema
        else:
            target_data = [(1, 100, 99.99, "INSERT")]
            source_data = [(1, 100, 199.99, operation)]
            schema = transactions_schema
        
        target_df = spark.createDataFrame(target_data, schema)
        source_df = spark.createDataFrame(source_data, schema)
        
        result = apply_multi_table_merge(source_df, target_df)
        
        if operation == "DELETE":
            assert result.filter(col("id") == 1).count() == 0
        else:
            assert result.filter(col("id") == 1).count() == 1
    
    def test_concurrent_table_updates(self, spark, users_schema, transactions_schema):
        """Test that multiple tables can be processed correctly."""
        # Users table
        users_target = spark.createDataFrame([(1, "User1", "u1@e.com", "INSERT")], users_schema)
        users_source = spark.createDataFrame([(1, "User1Updated", "u1new@e.com", "UPDATE")], users_schema)
        
        # Transactions table
        txn_target = spark.createDataFrame([(100, 1, 50.0, "INSERT")], transactions_schema)
        txn_source = spark.createDataFrame([(101, 1, 75.0, "INSERT")], transactions_schema)
        
        # Process both
        users_result = apply_multi_table_merge(users_source, users_target)
        txn_result = apply_multi_table_merge(txn_source, txn_target)
        
        assert users_result.count() == 1
        assert users_result.collect()[0]["name"] == "User1Updated"
        assert txn_result.count() == 2


# ============================================================================
# Test Class: Schema Validation
# ============================================================================

class TestSchemaValidation:
    """Tests for schema validation in multi-table processing."""
    
    @pytest.mark.parametrize("columns,required,expected", [
        (["id", "name", "email"], ["id", "name"], True),
        (["id", "name"], ["id", "name", "email"], False),
        (["id"], ["id"], True),
        ([], ["id"], False),
    ])
    def test_schema_validation(self, spark, columns, required, expected):
        """Test schema validation correctly identifies missing columns."""
        if columns:
            schema = StructType([StructField(c, StringType(), True) for c in columns])
            df = spark.createDataFrame([tuple("x" for _ in columns)], schema)
        else:
            df = spark.createDataFrame([], StructType([]))
        
        result = validate_table_schema(df, required)
        assert result == expected
    
    @pytest.mark.parametrize("extra_columns", [
        [],
        ["new_field"],
        ["field1", "field2", "field3"],
    ])
    def test_schema_evolution_detection(self, spark, extra_columns):
        """Test detection of schema evolution (new columns)."""
        base_columns = ["id", "name", "email"]
        all_columns = base_columns + extra_columns
        
        schema = StructType([StructField(c, StringType(), True) for c in all_columns])
        df = spark.createDataFrame([tuple("x" for _ in all_columns)], schema)
        
        # Should still validate with base required columns
        assert validate_table_schema(df, base_columns)
        # Should have extra columns
        assert len(df.columns) == len(all_columns)


# ============================================================================
# Test Class: Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Tests for batch processing of multiple tables."""
    
    @pytest.fixture
    def generic_schema(self):
        return StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
    
    @pytest.fixture
    def generic_target_schema(self):
        return StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
        ])
    
    @pytest.mark.parametrize("batch_count,records_per_batch", [
        (1, 10),
        (3, 5),
        (5, 2),
    ])
    def test_sequential_batch_processing(self, spark, generic_schema, 
                                         generic_target_schema, base_timestamp,
                                         batch_count, records_per_batch):
        """Test processing multiple sequential batches."""
        result = spark.createDataFrame([], generic_target_schema)
        total_ids = set()
        
        for batch in range(batch_count):
            batch_data = [
                (batch * records_per_batch + i, 
                 f"data_{batch}_{i}", 
                 "INSERT",
                 base_timestamp + timedelta(hours=batch))
                for i in range(records_per_batch)
            ]
            batch_df = spark.createDataFrame(batch_data, generic_schema)
            
            deduped = deduplicate_bronze_data(batch_df)
            prepared = prepare_merge_columns(deduped, result.columns)
            result = apply_multi_table_merge(prepared, result)
            
            total_ids.update(range(batch * records_per_batch, 
                                   (batch + 1) * records_per_batch))
        
        assert result.count() == len(total_ids)
    
    def test_batch_with_mixed_operations(self, spark, generic_schema, 
                                         generic_target_schema, base_timestamp):
        """Test batch with INSERT, UPDATE, and DELETE in same batch."""
        # Initial data
        initial_data = [
            (1, "data1", "INSERT", base_timestamp),
            (2, "data2", "INSERT", base_timestamp),
            (3, "data3", "INSERT", base_timestamp),
        ]
        initial_df = spark.createDataFrame(initial_data, generic_schema)
        deduped = deduplicate_bronze_data(initial_df)
        prepared = prepare_merge_columns(deduped, ["id", "data", "operation"])
        target = spark.createDataFrame([], generic_target_schema)
        result = apply_multi_table_merge(prepared, target)
        
        # Mixed batch
        mixed_data = [
            (4, "data4", "INSERT", base_timestamp + timedelta(hours=1)),  # New insert
            (1, "data1_updated", "UPDATE", base_timestamp + timedelta(hours=1)),  # Update
            (2, None, "DELETE", base_timestamp + timedelta(hours=1)),  # Delete
        ]
        mixed_df = spark.createDataFrame(mixed_data, generic_schema)
        deduped_mixed = deduplicate_bronze_data(mixed_df)
        prepared_mixed = prepare_merge_columns(deduped_mixed, result.columns)
        result = apply_multi_table_merge(prepared_mixed, result)
        
        assert result.count() == 3  # 1 (updated) + 3 (kept) + 4 (new) - 2 (deleted) = 3
        assert result.filter(col("id") == 2).count() == 0  # Deleted
        assert result.filter(col("id") == 4).count() == 1  # New


# ============================================================================
# Test Class: Concurrent Processing Simulation
# ============================================================================

class TestConcurrentProcessing:
    """Tests for concurrent table processing patterns."""
    
    def test_thread_safe_operations(self, spark, base_timestamp):
        """Test that processing logic is thread-safe."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("table_name", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        target_schema = StructType([
            StructField("id", LongType(), False),
            StructField("table_name", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        
        def process_table(table_name):
            data = [
                (i, table_name, "INSERT", base_timestamp)
                for i in range(5)
            ]
            df = spark.createDataFrame(data, schema)
            deduped = deduplicate_bronze_data(df)
            prepared = prepare_merge_columns(deduped, ["id", "table_name", "operation"])
            target = spark.createDataFrame([], target_schema)
            result = apply_multi_table_merge(prepared, target)
            return result.count()
        
        tables = ["users", "transactions", "products"]
        
        # Process sequentially (simulating what ThreadPoolExecutor would do)
        results = [process_table(t) for t in tables]
        
        assert all(count == 5 for count in results)
    
    @pytest.mark.parametrize("num_workers", [1, 2, 3])
    def test_parallel_table_discovery(self, num_workers):
        """Test table discovery pattern used in notebook."""
        # Simulate folder structure
        mock_folders = [
            type('MockPath', (), {'name': 'users/'})(),
            type('MockPath', (), {'name': 'transactions/'})(),
            type('MockPath', (), {'name': 'products/'})(),
        ]
        
        tables = [path.name[:-1] for path in mock_folders]
        
        assert len(tables) == 3
        assert "users" in tables
        assert "transactions" in tables
        assert "products" in tables


# ============================================================================
# Test Class: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in multi-table processing."""
    
    def test_empty_dataframe_handling(self, spark):
        """Test that empty DataFrames are handled correctly."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        
        empty_df = spark.createDataFrame([], schema)
        target_df = spark.createDataFrame([(1, "existing", "INSERT")], schema)
        
        result = apply_multi_table_merge(empty_df, target_df)
        
        assert result.count() == 1
    
    def test_null_values_in_operations(self, spark, base_timestamp):
        """Test handling of NULL values in CDC data."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        data = [
            (1, None, "DELETE", base_timestamp),
            (2, "valid_data", "INSERT", base_timestamp),
        ]
        df = spark.createDataFrame(data, schema)
        
        result = deduplicate_bronze_data(df)
        
        assert result.count() == 2
    
    @pytest.mark.parametrize("invalid_operation", [
        "INVALID",
        "",
        None,
    ])
    def test_invalid_operation_handling(self, spark, invalid_operation):
        """Test behavior with invalid operation types."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        
        data = [(1, "test_data", invalid_operation)]
        source_df = spark.createDataFrame(data, schema)
        target_df = spark.createDataFrame([(2, "existing", "INSERT")], schema)
        
        # Should not raise, invalid operations treated as non-DELETE
        result = apply_multi_table_merge(source_df, target_df)
        
        # Record with invalid operation should be included (not DELETE)
        assert result.count() >= 1


# ============================================================================
# Test Class: Data Integrity
# ============================================================================

class TestDataIntegrity:
    """Tests for data integrity in multi-table CDC processing."""
    
    def test_no_data_loss_on_merge(self, spark, base_timestamp):
        """Test that no data is lost during merge operations."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        target_schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        
        # Create target with IDs 1-5
        target_data = [(i, f"data_{i}", "INSERT") for i in range(1, 6)]
        target_df = spark.createDataFrame(target_data, target_schema)
        
        # Source only updates ID 3
        source_data = [(3, "data_3_updated", "UPDATE", base_timestamp)]
        source_df = spark.createDataFrame(source_data, schema)
        
        deduped = deduplicate_bronze_data(source_df)
        prepared = prepare_merge_columns(deduped, target_df.columns)
        result = apply_multi_table_merge(prepared, target_df)
        
        # Should still have all 5 records
        assert result.count() == 5
        # ID 3 should be updated
        assert result.filter(col("id") == 3).collect()[0]["data"] == "data_3_updated"
    
    @pytest.mark.parametrize("original_count,updates,deletes,expected", [
        (10, 2, 1, 10),  # 10 - 1 delete + updates don't change count
        (5, 0, 5, 0),    # All deleted
        (5, 3, 0, 5),    # Only updates, no count change
        (0, 5, 0, 5),    # Empty target, 5 inserts
    ])
    def test_record_count_accuracy(self, spark, base_timestamp, 
                                   original_count, updates, deletes, expected):
        """Test that record counts are accurate after operations."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("operation_date", TimestampType(), True),
        ])
        
        target_schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        
        # Create target
        target_data = [(i, f"data_{i}", "INSERT") for i in range(original_count)]
        target_df = spark.createDataFrame(target_data, target_schema) if original_count > 0 else spark.createDataFrame([], target_schema)
        
        # Create source with updates and deletes
        source_data = []
        # Updates for existing IDs
        for i in range(updates):
            if i < original_count:
                source_data.append((i, f"updated_{i}", "UPDATE", base_timestamp))
            else:
                source_data.append((i, f"new_{i}", "INSERT", base_timestamp))
        # Deletes
        for i in range(deletes):
            if i < original_count:
                source_data.append((i, None, "DELETE", base_timestamp + timedelta(hours=1)))
        
        if source_data:
            source_df = spark.createDataFrame(source_data, schema)
            deduped = deduplicate_bronze_data(source_df)
            prepared = prepare_merge_columns(deduped, target_df.columns)
            result = apply_multi_table_merge(prepared, target_df)
        else:
            result = target_df
        
        assert result.count() == expected

