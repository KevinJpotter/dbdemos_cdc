"""
Unit tests for 01-sql_cdc_pipeline.sql declarative pipeline.

This module tests the SQL-based CDC pipeline functionality including:
- Data quality expectations (valid_id, valid_operation, valid_json_schema)
- CDC operations handling (APPEND, UPDATE, DELETE)
- APPLY CHANGES merge logic simulation
- SCD Type 2 history tracking

Tests use pytest fixtures and follow the dbx_test framework patterns.
These tests validate the transformation logic independently of DLT runtime.

Note: Since DLT SQL statements can't be executed directly in pytest,
we test the underlying logic using equivalent PySpark transformations.
"""

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, row_number, when, lit, current_timestamp, 
    max as spark_max, expr
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)
from datetime import datetime, timedelta


# ============================================================================
# Helper Functions - Simulating SQL Pipeline Logic
# ============================================================================

def apply_data_quality_expectations(df: DataFrame) -> DataFrame:
    """
    Apply data quality expectations matching the SQL pipeline:
    - valid_id: id IS NOT NULL
    - valid_operation: operation IN ('APPEND', 'DELETE', 'UPDATE')
    - valid_json_schema: _rescued_data IS NULL
    
    Records violating any expectation are DROPPED (ON VIOLATION DROP ROW).
    
    Args:
        df: Input CDC DataFrame
        
    Returns:
        DataFrame with invalid records filtered out
    """
    valid_operations = ['APPEND', 'DELETE', 'UPDATE']
    
    return (df
            .filter(col("id").isNotNull())  # valid_id
            .filter(col("operation").isin(valid_operations))  # valid_operation
            .filter(col("_rescued_data").isNull()))  # valid_json_schema


def deduplicate_by_sequence(df: DataFrame, 
                            keys: list = ["id"],
                            sequence_col: str = "operation_date") -> DataFrame:
    """
    Deduplicate CDC data by keeping the latest record per key based on sequence.
    Simulates SEQUENCE BY operation_date in APPLY CHANGES.
    
    Args:
        df: Input DataFrame with CDC records
        keys: Columns to partition by (primary key)
        sequence_col: Column to order by for selecting latest record
        
    Returns:
        Deduplicated DataFrame with only the latest record per key
    """
    window_spec = Window.partitionBy(*keys).orderBy(col(sequence_col).desc())
    return (df
            .withColumn("_rank", row_number().over(window_spec))
            .filter(col("_rank") == 1)
            .drop("_rank"))


def apply_cdc_changes(source_df: DataFrame, 
                      target_df: DataFrame,
                      keys: list = ["id"],
                      delete_condition: str = "operation = 'DELETE'",
                      except_columns: list = None) -> DataFrame:
    """
    Simulate APPLY CHANGES INTO logic for testing.
    
    Implements:
    - DELETE: Remove matching records where delete_condition is true
    - UPDATE/INSERT: Upsert remaining records by key
    - EXCEPT columns: Drop specified columns from result
    
    Args:
        source_df: Source DataFrame with CDC operations
        target_df: Target DataFrame (current state)
        keys: Key columns for matching
        delete_condition: SQL expression for delete detection
        except_columns: Columns to exclude from final result
        
    Returns:
        DataFrame representing merged result
    """
    if except_columns is None:
        except_columns = ["operation", "operation_date", "_rescued_data"]
    
    # Extract delete IDs
    deletes = source_df.filter(expr(delete_condition))
    delete_ids = set(row["id"] for row in deletes.select(keys).collect())
    
    # Extract non-delete records
    upserts = source_df.filter(~expr(delete_condition))
    upsert_ids = set(row["id"] for row in upserts.select(keys).collect())
    
    # Filter target: remove deleted and to-be-updated records
    all_affected_ids = delete_ids | upsert_ids
    target_filtered = target_df.filter(~col("id").isin(list(all_affected_ids)))
    
    # Prepare upsert data (drop except columns)
    existing_except_cols = [c for c in except_columns if c in upserts.columns]
    upserts_prepared = upserts.drop(*existing_except_cols)
    
    # Ensure schema compatibility
    result_columns = [c for c in target_filtered.columns]
    upserts_final = upserts_prepared.select(
        *[col(c) if c in upserts_prepared.columns else lit(None).alias(c) 
          for c in result_columns]
    )
    
    # Union existing with upserts
    return target_filtered.union(upserts_final)


def apply_scd2_changes(source_df: DataFrame,
                       target_df: DataFrame,
                       keys: list = ["id"],
                       sequence_col: str = "operation_date") -> DataFrame:
    """
    Simulate APPLY CHANGES with STORED AS SCD TYPE 2.
    
    Creates historical records with __START_AT and __END_AT columns.
    
    Args:
        source_df: Source DataFrame with CDC operations
        target_df: Current SCD2 table (may be empty)
        keys: Key columns for matching
        sequence_col: Column containing change timestamp
        
    Returns:
        SCD2 DataFrame with history tracking columns
    """
    # Get current timestamp for __END_AT of old records
    current_ts = datetime.now()
    
    # For each new record, close the previous version and insert new
    # This is a simplified simulation - actual DLT handles this internally
    
    result_data = []
    
    # Collect source and target for processing
    source_records = source_df.collect()
    target_records = {row["id"]: row for row in target_df.collect()} if target_df.count() > 0 else {}
    
    for src_row in source_records:
        record_id = src_row["id"]
        change_ts = src_row[sequence_col]
        
        if src_row["operation"] == "DELETE":
            # Close the current version (set __END_AT)
            if record_id in target_records:
                closed_record = dict(target_records[record_id].asDict())
                closed_record["__END_AT"] = change_ts
                result_data.append(closed_record)
        else:
            # Close previous version if exists
            if record_id in target_records:
                prev = dict(target_records[record_id].asDict())
                prev["__END_AT"] = change_ts
                result_data.append(prev)
            
            # Add new version
            new_record = {
                "id": record_id,
                "name": src_row["name"],
                "address": src_row["address"],
                "email": src_row["email"],
                "__START_AT": change_ts,
                "__END_AT": None
            }
            result_data.append(new_record)
    
    # Create DataFrame from results
    if result_data:
        return target_df.sparkSession.createDataFrame(result_data, target_df.schema)
    return target_df


# ============================================================================
# Test Class: Data Quality Expectations
# ============================================================================

class TestDataQualityExpectations:
    """
    Tests for SQL pipeline data quality expectations:
    - CONSTRAINT valid_id EXPECT (id IS NOT NULL) ON VIOLATION DROP ROW
    - CONSTRAINT valid_operation EXPECT (operation IN (...)) ON VIOLATION DROP ROW
    - CONSTRAINT valid_json_schema EXPECT (_rescued_data IS NULL) ON VIOLATION DROP ROW
    """
    
    def test_valid_records_pass_all_expectations(self, spark, sample_customers_cdc):
        """
        Test that valid records pass all data quality expectations.
        
        Why: Ensures baseline functionality - good data should flow through unchanged.
        """
        # Arrange: sample_customers_cdc contains valid records
        input_df = sample_customers_cdc
        input_count = input_df.count()
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert: all records should pass
        assert result.count() == input_count, "Valid records should not be filtered"
    
    def test_null_id_records_are_dropped(self, spark, sample_cdc_with_null_ids):
        """
        Test that records with NULL id are dropped.
        
        Why: The pipeline requires valid IDs for deduplication and merge operations.
        SQL: CONSTRAINT valid_id EXPECT (id IS NOT NULL) ON VIOLATION DROP ROW
        """
        # Arrange: fixture has 1 record with NULL id
        input_df = sample_cdc_with_null_ids
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert: NULL id record should be dropped
        assert result.filter(col("id").isNull()).count() == 0
        assert result.count() == 2, "Only 2 valid records should remain"
    
    @pytest.mark.parametrize("invalid_operation", [
        "INVALID",
        "MERGE", 
        "INSERT",  # DLT uses APPEND, not INSERT
        "UPSERT",
        "",
        "append",  # Case sensitive
    ])
    def test_invalid_operations_are_dropped(self, spark, customers_cdc_schema, 
                                            base_timestamp, invalid_operation):
        """
        Test that records with invalid operation types are dropped.
        
        Why: Pipeline only handles APPEND, UPDATE, DELETE operations.
        SQL: CONSTRAINT valid_operation EXPECT (operation IN ('APPEND', 'DELETE', 'UPDATE'))
        """
        # Arrange
        data = [
            (1, "Alice", "123 Main St", "alice@email.com", 
             invalid_operation, base_timestamp, None),
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert
        assert result.count() == 0, f"Operation '{invalid_operation}' should be rejected"
    
    @pytest.mark.parametrize("valid_operation", ["APPEND", "UPDATE", "DELETE"])
    def test_valid_operations_are_accepted(self, spark, customers_cdc_schema,
                                           base_timestamp, valid_operation):
        """
        Test that all valid operation types are accepted.
        
        Why: Verify the complete set of valid operations work correctly.
        """
        # Arrange
        data = [
            (1, "Alice", "123 Main St", "alice@email.com",
             valid_operation, base_timestamp, None),
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert
        assert result.count() == 1, f"Operation '{valid_operation}' should be accepted"
    
    def test_rescued_data_records_are_dropped(self, spark, sample_cdc_with_rescued_data):
        """
        Test that records with non-null _rescued_data are dropped.
        
        Why: Non-null _rescued_data indicates schema mismatch in JSON parsing.
        SQL: CONSTRAINT valid_json_schema EXPECT (_rescued_data IS NULL) ON VIOLATION DROP ROW
        """
        # Arrange: fixture has 1 record with rescued_data
        input_df = sample_cdc_with_rescued_data
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert
        assert result.filter(col("_rescued_data").isNotNull()).count() == 0
        assert result.count() == 2, "Only 2 valid records should remain"
    
    def test_multiple_violations_in_same_batch(self, spark, customers_cdc_schema, base_timestamp):
        """
        Test handling of batch with multiple different violations.
        
        Why: Real data may have various quality issues mixed together.
        """
        # Arrange: records with different violations
        data = [
            # Valid
            (1, "Valid", "Address", "valid@email.com", "APPEND", base_timestamp, None),
            # NULL id
            (None, "No ID", "Address", "noid@email.com", "APPEND", base_timestamp, None),
            # Invalid operation
            (2, "Bad Op", "Address", "badop@email.com", "INVALID", base_timestamp, None),
            # Rescued data
            (3, "Bad Schema", "Address", "schema@email.com", "APPEND", base_timestamp, '{"extra": 1}'),
            # Another valid
            (4, "Also Valid", "Address", "alsovalid@email.com", "UPDATE", base_timestamp, None),
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        
        # Act
        result = apply_data_quality_expectations(input_df)
        
        # Assert: only 2 valid records
        assert result.count() == 2
        result_ids = [row["id"] for row in result.select("id").collect()]
        assert set(result_ids) == {1, 4}


# ============================================================================
# Test Class: Deduplication Logic
# ============================================================================

class TestDeduplicationLogic:
    """
    Tests for SEQUENCE BY operation_date deduplication.
    The pipeline keeps only the latest record per ID based on operation_date.
    """
    
    def test_keeps_latest_by_operation_date(self, spark, sample_cdc_with_duplicates):
        """
        Test that only the record with latest operation_date is kept.
        
        Why: Ensures out-of-order CDC events are correctly resolved.
        """
        # Arrange
        input_df = apply_data_quality_expectations(sample_cdc_with_duplicates)
        
        # Act
        result = deduplicate_by_sequence(input_df)
        
        # Assert: only 1 record per ID
        assert result.count() == 1
        row = result.collect()[0]
        assert row["name"] == "Alice V4 - FINAL", "Latest record (V4) should be kept"
    
    @pytest.mark.parametrize("num_versions", [2, 5, 10, 50])
    def test_deduplication_scales_with_many_versions(self, spark, customers_cdc_schema,
                                                     base_timestamp, num_versions):
        """
        Test deduplication with varying numbers of record versions.
        
        Why: Verify performance and correctness with many updates to same record.
        """
        # Arrange: create many versions of same record
        data = [
            (1, f"Version {i}", f"Address {i}", f"v{i}@email.com",
             "UPDATE", base_timestamp + timedelta(hours=i), None)
            for i in range(num_versions)
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        clean_df = apply_data_quality_expectations(input_df)
        
        # Act
        result = deduplicate_by_sequence(clean_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["name"] == f"Version {num_versions - 1}", "Latest version should be kept"
    
    def test_deduplication_preserves_multiple_ids(self, spark, customers_cdc_schema,
                                                  base_timestamp):
        """
        Test that deduplication works correctly across multiple IDs.
        
        Why: Ensure different records aren't incorrectly merged.
        """
        # Arrange: multiple IDs with duplicates
        data = [
            # ID 1: 2 versions
            (1, "Alice V1", "Addr1", "a1@email.com", "APPEND", base_timestamp, None),
            (1, "Alice V2", "Addr1", "a2@email.com", "UPDATE", 
             base_timestamp + timedelta(hours=1), None),
            # ID 2: 3 versions
            (2, "Bob V1", "Addr2", "b1@email.com", "APPEND", base_timestamp, None),
            (2, "Bob V2", "Addr2", "b2@email.com", "UPDATE",
             base_timestamp + timedelta(hours=1), None),
            (2, "Bob V3", "Addr2", "b3@email.com", "UPDATE",
             base_timestamp + timedelta(hours=2), None),
            # ID 3: 1 version
            (3, "Charlie", "Addr3", "c@email.com", "APPEND", base_timestamp, None),
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        clean_df = apply_data_quality_expectations(input_df)
        
        # Act
        result = deduplicate_by_sequence(clean_df)
        
        # Assert
        assert result.count() == 3, "Should have 1 record per ID"
        
        results_by_id = {row["id"]: row for row in result.collect()}
        assert results_by_id[1]["name"] == "Alice V2"
        assert results_by_id[2]["name"] == "Bob V3"
        assert results_by_id[3]["name"] == "Charlie"
    
    def test_same_timestamp_deterministic_ordering(self, spark, customers_cdc_schema,
                                                   base_timestamp):
        """
        Test behavior when multiple records have same operation_date.
        
        Why: Verify deterministic behavior for edge case scenarios.
        """
        # Arrange: same timestamp for all records
        data = [
            (1, "Version A", "Addr", "a@email.com", "APPEND", base_timestamp, None),
            (1, "Version B", "Addr", "b@email.com", "UPDATE", base_timestamp, None),
            (1, "Version C", "Addr", "c@email.com", "UPDATE", base_timestamp, None),
        ]
        input_df = spark.createDataFrame(data, customers_cdc_schema)
        clean_df = apply_data_quality_expectations(input_df)
        
        # Act
        result = deduplicate_by_sequence(clean_df)
        
        # Assert: should get exactly 1 record (deterministic but order undefined)
        assert result.count() == 1


# ============================================================================
# Test Class: APPLY CHANGES Logic
# ============================================================================

class TestApplyChangesLogic:
    """
    Tests for APPLY CHANGES INTO behavior.
    Simulates the SQL: APPLY CHANGES INTO customers FROM stream(customers_cdc_clean)
    """
    
    def test_append_inserts_new_records(self, spark, customers_silver_schema,
                                        customers_cdc_schema, base_timestamp):
        """
        Test that APPEND operations insert new records.
        
        Why: APPEND is the primary way new customers are added.
        """
        # Arrange
        source_data = [
            (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "456 Oak Ave", "bob@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 2
        assert set(row["id"] for row in result.collect()) == {1, 2}
    
    def test_update_modifies_existing_records(self, spark, customers_silver_schema,
                                              customers_cdc_schema, base_timestamp):
        """
        Test that UPDATE operations modify existing records.
        
        Why: Customer data changes over time (address, email, etc.).
        """
        # Arrange: existing record
        target_data = [(1, "Alice Old", "Old Address", "old@email.com")]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        # Update CDC record
        source_data = [
            (1, "Alice New", "New Address", "new@email.com", "UPDATE", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["name"] == "Alice New"
        assert row["address"] == "New Address"
        assert row["email"] == "new@email.com"
    
    def test_delete_removes_records(self, spark, customers_silver_schema,
                                    customers_cdc_schema, base_timestamp):
        """
        Test that DELETE operations remove records from target.
        
        Why: Customers may be removed from the system.
        SQL: APPLY AS DELETE WHEN operation = "DELETE"
        """
        # Arrange: two existing records
        target_data = [
            (1, "Alice", "123 Main St", "alice@email.com"),
            (2, "Bob", "456 Oak Ave", "bob@email.com"),
        ]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        # Delete CDC record for Alice
        source_data = [
            (1, None, None, None, "DELETE", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        assert result.filter(col("id") == 1).count() == 0, "Alice should be deleted"
        assert result.filter(col("id") == 2).count() == 1, "Bob should remain"
    
    def test_mixed_operations_in_batch(self, spark, customers_silver_schema,
                                       customers_cdc_schema, base_timestamp):
        """
        Test batch with INSERT, UPDATE, and DELETE in same CDC batch.
        
        Why: Real CDC streams contain mixed operations.
        """
        # Arrange: existing records
        target_data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
            (3, "Charlie", "Addr3", "charlie@email.com"),
        ]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        # Mixed CDC: update Alice, delete Bob, insert Diana
        source_data = [
            (1, "Alice Updated", "New Addr1", "alice.new@email.com", "UPDATE", base_timestamp, None),
            (2, None, None, None, "DELETE", base_timestamp, None),
            (4, "Diana", "Addr4", "diana@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 3  # Alice (updated), Charlie (unchanged), Diana (new)
        
        results_by_id = {row["id"]: row for row in result.collect()}
        assert results_by_id[1]["name"] == "Alice Updated"
        assert 2 not in results_by_id  # Bob deleted
        assert results_by_id[3]["name"] == "Charlie"
        assert results_by_id[4]["name"] == "Diana"
    
    def test_except_columns_are_removed(self, spark, customers_silver_schema,
                                        customers_cdc_schema, base_timestamp):
        """
        Test that COLUMNS * EXCEPT (operation, operation_date, _rescued_data) works.
        
        Why: The target table should not contain CDC metadata columns.
        """
        # Arrange
        source_data = [
            (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert "operation" not in result.columns
        assert "operation_date" not in result.columns
        assert "_rescued_data" not in result.columns
        assert set(result.columns) == {"id", "name", "address", "email"}


# ============================================================================
# Test Class: SCD Type 2 Logic
# ============================================================================

class TestSCD2Logic:
    """
    Tests for SCD Type 2 history tracking.
    SQL: STORED AS SCD TYPE 2
    """
    
    def test_initial_insert_sets_start_date(self, spark, scd2_schema,
                                            customers_cdc_schema, base_timestamp):
        """
        Test that initial insert sets __START_AT to operation_date.
        
        Why: SCD2 requires validity tracking from the start.
        """
        # Arrange
        source_data = [
            (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        target_df = spark.createDataFrame([], scd2_schema)
        
        # Act
        result = apply_scd2_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        row = result.collect()[0]
        assert row["__START_AT"] == base_timestamp
        assert row["__END_AT"] is None  # Current version has no end date
    
    def test_update_creates_history(self, spark, scd2_schema,
                                    customers_cdc_schema, base_timestamp):
        """
        Test that UPDATE closes previous version and creates new version.
        
        Why: SCD2 must preserve complete history of changes.
        """
        # Arrange: existing SCD2 record
        target_data = [
            (1, "Alice", "Old Address", "alice@email.com", base_timestamp, None),
        ]
        target_df = spark.createDataFrame(target_data, scd2_schema)
        
        # Update CDC
        update_ts = base_timestamp + timedelta(days=30)
        source_data = [
            (1, "Alice", "New Address", "alice@email.com", "UPDATE", update_ts, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        # Act
        result = apply_scd2_changes(source_df, target_df)
        
        # Assert: 2 records - old (closed) and new (current)
        assert result.count() == 2
        
        old_version = result.filter(col("address") == "Old Address").collect()[0]
        new_version = result.filter(col("address") == "New Address").collect()[0]
        
        assert old_version["__END_AT"] == update_ts, "Old version should be closed"
        assert new_version["__START_AT"] == update_ts
        assert new_version["__END_AT"] is None, "New version should be current"
    
    def test_multiple_updates_create_full_history(self, spark, scd2_schema,
                                                  sample_customers_scd2_history):
        """
        Test that multiple updates create complete history chain.
        
        Why: Verify full audit trail is maintained.
        """
        # Arrange
        source_df = apply_data_quality_expectations(sample_customers_scd2_history)
        target_df = spark.createDataFrame([], scd2_schema)
        
        # Process each change sequentially to build history
        changes = source_df.orderBy("operation_date").collect()
        
        for change in changes:
            change_df = spark.createDataFrame([change], source_df.schema)
            target_df = apply_scd2_changes(change_df, target_df)
        
        # Assert: 4 historical versions
        assert target_df.count() == 4
        
        # Only the last version should be current (__END_AT is NULL)
        current_versions = target_df.filter(col("__END_AT").isNull())
        assert current_versions.count() == 1
        
        current = current_versions.collect()[0]
        assert current["name"] == "Alice Smith"
    
    def test_delete_closes_record(self, spark, scd2_schema,
                                  customers_cdc_schema, base_timestamp):
        """
        Test that DELETE operation closes the current version.
        
        Why: Even deleted records should have complete history.
        """
        # Arrange
        target_data = [
            (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, None),
        ]
        target_df = spark.createDataFrame(target_data, scd2_schema)
        
        # Delete CDC
        delete_ts = base_timestamp + timedelta(days=30)
        source_data = [
            (1, None, None, None, "DELETE", delete_ts, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        # Act
        result = apply_scd2_changes(source_df, target_df)
        
        # Assert: record should be closed, not removed
        assert result.count() == 1
        row = result.collect()[0]
        assert row["__END_AT"] == delete_ts


# ============================================================================
# Test Class: Empty and Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_source_no_changes(self, spark, customers_silver_schema,
                                     customers_cdc_schema):
        """
        Test that empty source DataFrame produces no changes.
        
        Why: Empty batches should be handled gracefully.
        """
        # Arrange
        target_data = [(1, "Alice", "123 Main St", "alice@email.com")]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        source_df = spark.createDataFrame([], customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 1
        assert result.collect()[0]["id"] == 1
    
    def test_all_records_filtered_by_expectations(self, spark, customers_silver_schema,
                                                  customers_cdc_schema, base_timestamp):
        """
        Test when all source records fail quality expectations.
        
        Why: Bad data batches shouldn't corrupt target.
        """
        # Arrange
        target_data = [(1, "Alice", "123 Main St", "alice@email.com")]
        target_df = spark.createDataFrame(target_data, customers_silver_schema)
        
        # All invalid records
        source_data = [
            (None, "No ID", "Addr", "noid@email.com", "APPEND", base_timestamp, None),
            (2, "Bad Op", "Addr", "badop@email.com", "INVALID", base_timestamp, None),
        ]
        source_df = spark.createDataFrame(source_data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert: target unchanged
        assert result.count() == 1
    
    def test_null_values_in_non_key_columns(self, spark, customers_silver_schema,
                                            sample_cdc_with_nulls):
        """
        Test handling of NULL values in optional columns.
        
        Why: NULL values in name/address/email should be allowed.
        """
        # Arrange
        source_df = apply_data_quality_expectations(sample_cdc_with_nulls)
        source_df = deduplicate_by_sequence(source_df)
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert: all records inserted despite NULL values
        assert result.count() == 3
    
    def test_large_batch_processing(self, spark, customers_cdc_schema,
                                    customers_silver_schema, base_timestamp):
        """
        Test processing of large CDC batch.
        
        Why: Verify scalability with realistic data volumes.
        """
        # Arrange: 1000 records
        data = [
            (i, f"Customer {i}", f"Address {i}", f"customer{i}@email.com",
             "APPEND", base_timestamp + timedelta(seconds=i), None)
            for i in range(1000)
        ]
        source_df = spark.createDataFrame(data, customers_cdc_schema)
        source_df = apply_data_quality_expectations(source_df)
        source_df = deduplicate_by_sequence(source_df)
        
        target_df = spark.createDataFrame([], customers_silver_schema)
        
        # Act
        result = apply_cdc_changes(source_df, target_df)
        
        # Assert
        assert result.count() == 1000


# ============================================================================
# Test Class: End-to-End Pipeline Simulation
# ============================================================================

class TestEndToEndPipeline:
    """Integration tests simulating complete pipeline flow."""
    
    def test_complete_cdc_lifecycle(self, spark, customers_cdc_schema,
                                    customers_silver_schema, base_timestamp):
        """
        Test complete customer lifecycle: create, update, delete.
        
        Why: Validates full pipeline behavior for typical use case.
        """
        # Initialize empty target
        target = spark.createDataFrame([], customers_silver_schema)
        
        # Batch 1: Initial inserts
        batch1 = spark.createDataFrame([
            (1, "Alice", "Addr1", "alice@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "Addr2", "bob@email.com", "APPEND", base_timestamp, None),
        ], customers_cdc_schema)
        batch1 = apply_data_quality_expectations(batch1)
        batch1 = deduplicate_by_sequence(batch1)
        target = apply_cdc_changes(batch1, target)
        
        assert target.count() == 2
        
        # Batch 2: Updates
        batch2 = spark.createDataFrame([
            (1, "Alice Updated", "New Addr1", "alice.new@email.com", "UPDATE",
             base_timestamp + timedelta(hours=1), None),
        ], customers_cdc_schema)
        batch2 = apply_data_quality_expectations(batch2)
        batch2 = deduplicate_by_sequence(batch2)
        target = apply_cdc_changes(batch2, target)
        
        alice = target.filter(col("id") == 1).collect()[0]
        assert alice["name"] == "Alice Updated"
        
        # Batch 3: Delete
        batch3 = spark.createDataFrame([
            (2, None, None, None, "DELETE", base_timestamp + timedelta(hours=2), None),
        ], customers_cdc_schema)
        batch3 = apply_data_quality_expectations(batch3)
        batch3 = deduplicate_by_sequence(batch3)
        target = apply_cdc_changes(batch3, target)
        
        assert target.count() == 1
        assert target.filter(col("id") == 2).count() == 0
    
    @pytest.mark.parametrize("num_batches,records_per_batch", [
        (3, 10),
        (5, 5),
        (10, 3),
    ])
    def test_multiple_batch_processing(self, spark, customers_cdc_schema,
                                       customers_silver_schema, base_timestamp,
                                       num_batches, records_per_batch):
        """
        Test processing multiple sequential batches.
        
        Why: Verify incremental processing works correctly.
        """
        target = spark.createDataFrame([], customers_silver_schema)
        
        for batch_num in range(num_batches):
            batch_data = [
                (batch_num * records_per_batch + i,
                 f"Customer_{batch_num}_{i}",
                 f"Address_{batch_num}_{i}",
                 f"customer{batch_num}_{i}@email.com",
                 "APPEND",
                 base_timestamp + timedelta(hours=batch_num),
                 None)
                for i in range(records_per_batch)
            ]
            batch_df = spark.createDataFrame(batch_data, customers_cdc_schema)
            batch_df = apply_data_quality_expectations(batch_df)
            batch_df = deduplicate_by_sequence(batch_df)
            target = apply_cdc_changes(batch_df, target)
        
        expected_total = num_batches * records_per_batch
        assert target.count() == expected_total

