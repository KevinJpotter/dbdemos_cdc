"""
Unit tests for 01-CDC-CDF-simple-pipeline notebook.

This module tests the core CDC functionality including:
- Bronze layer data ingestion
- Silver layer merge/deduplication logic
- Gold layer CDF processing and upsert logic

Tests use pytest fixtures and parametrize for comprehensive coverage.
"""

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, row_number, dense_rank, regexp_replace, lit
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from datetime import datetime, timedelta
from delta.tables import DeltaTable


# ============================================================================
# Helper Functions (Extracted from notebook for testing)
# ============================================================================

def deduplicate_cdc_data(df: DataFrame, id_col: str = "id", 
                          order_col: str = "operation_date") -> DataFrame:
    """
    Deduplicate CDC data by keeping only the most recent record per ID.
    
    Args:
        df: Input DataFrame with CDC data
        id_col: Column to partition by for deduplication
        order_col: Column to order by (descending) for selecting latest record
        
    Returns:
        DataFrame with duplicates removed, keeping most recent record per ID
    """
    windowSpec = Window.partitionBy(id_col).orderBy(col(order_col).desc())
    return df.withColumn("rank", row_number().over(windowSpec)).where("rank = 1").drop("rank")


def apply_merge_logic(source_df: DataFrame, target_df: DataFrame, 
                      id_col: str = "id") -> DataFrame:
    """
    Apply CDC merge logic to combine source and target DataFrames.
    Simulates MERGE INTO behavior for testing purposes.
    
    Args:
        source_df: Source DataFrame with CDC operations
        target_df: Target DataFrame (current state)
        id_col: Column to match on for merge
        
    Returns:
        DataFrame representing the merged result
    """
    # Get columns for the result (excluding operation column)
    result_columns = [c for c in target_df.columns if c != "operation"]
    
    # Handle deletes - remove matching IDs where operation is DELETE
    delete_ids = (source_df
                  .filter(col("operation") == "DELETE")
                  .select(id_col)
                  .collect())
    delete_id_list = [row[id_col] for row in delete_ids]
    
    # Filter out deleted records from target
    target_filtered = target_df.filter(~col(id_col).isin(delete_id_list))
    
    # Handle updates and inserts
    updates = source_df.filter(col("operation") != "DELETE")
    
    # Remove matching IDs from target (will be replaced by updates)
    update_ids = [row[id_col] for row in updates.select(id_col).collect()]
    target_without_updates = target_filtered.filter(~col(id_col).isin(update_ids))
    
    # Combine: existing (non-updated) + updates
    # Select only columns that exist in target
    update_cols = [c for c in updates.columns if c in target_df.columns]
    result = target_without_updates.union(updates.select(update_cols))
    
    return result


def process_cdf_for_gold(cdf_df: DataFrame, id_col: str = "id") -> DataFrame:
    """
    Process CDF data for Gold layer, handling deduplication and filtering.
    
    Args:
        cdf_df: DataFrame with CDF changes including _change_type and _commit_version
        id_col: Column to deduplicate on
        
    Returns:
        Deduplicated DataFrame ready for Gold layer merge
    """
    windowSpec = Window.partitionBy(id_col).orderBy(col("_commit_version").desc())
    return (cdf_df
            .withColumn("rank", dense_rank().over(windowSpec))
            .where("rank = 1 and _change_type != 'update_preimage'")
            .drop("_commit_version", "rank"))


def clean_address(df: DataFrame, address_col: str = "address") -> DataFrame:
    """
    Clean address field by removing quotes.
    
    Args:
        df: Input DataFrame
        address_col: Name of address column to clean
        
    Returns:
        DataFrame with cleaned address
    """
    return df.withColumn(address_col, regexp_replace(col(address_col), "\"", ""))


# ============================================================================
# Test Class: Deduplication Logic
# ============================================================================

class TestDeduplication:
    """Tests for CDC data deduplication functionality."""
    
    @pytest.mark.parametrize("num_duplicates,expected_count", [
        (1, 1),   # Single record, no duplicates
        (3, 1),   # Three records for same ID, expect 1
        (5, 1),   # Five records for same ID, expect 1
    ])
    def test_deduplication_keeps_single_record(self, spark, cdc_schema, 
                                                base_timestamp, num_duplicates, 
                                                expected_count):
        """Test that deduplication keeps exactly one record per ID."""
        # Create test data with multiple records for same ID
        data = [
            (1, f"Version{i}", f"Address{i}", f"email{i}@test.com",
             base_timestamp + timedelta(hours=i), "UPDATE", None, f"file{i}.csv")
            for i in range(num_duplicates)
        ]
        df = spark.createDataFrame(data, cdc_schema)
        
        result = deduplicate_cdc_data(df)
        
        assert result.count() == expected_count
    
    def test_deduplication_keeps_latest_record(self, spark, cdc_schema, base_timestamp):
        """Test that deduplication keeps the most recent record."""
        data = [
            (1, "Old Name", "Old Address", "old@email.com", base_timestamp, "INSERT", None, "file1.csv"),
            (1, "New Name", "New Address", "new@email.com", 
             base_timestamp + timedelta(hours=2), "UPDATE", None, "file2.csv"),
            (1, "Middle Name", "Middle Address", "middle@email.com", 
             base_timestamp + timedelta(hours=1), "UPDATE", None, "file3.csv"),
        ]
        df = spark.createDataFrame(data, cdc_schema)
        
        result = deduplicate_cdc_data(df)
        row = result.collect()[0]
        
        assert row["name"] == "New Name"
        assert row["address"] == "New Address"
    
    @pytest.mark.parametrize("ids,expected_counts", [
        ([1, 1, 2, 2, 3], {1: 1, 2: 1, 3: 1}),  # Multiple IDs with duplicates
        ([1, 2, 3, 4, 5], {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}),  # Unique IDs
    ])
    def test_deduplication_multiple_ids(self, spark, cdc_schema, base_timestamp, 
                                        ids, expected_counts):
        """Test deduplication works correctly across multiple IDs."""
        data = [
            (id_val, f"Name{i}", f"Address{i}", f"email{i}@test.com",
             base_timestamp + timedelta(hours=i), "UPDATE", None, f"file{i}.csv")
            for i, id_val in enumerate(ids)
        ]
        df = spark.createDataFrame(data, cdc_schema)
        
        result = deduplicate_cdc_data(df)
        
        # Check count per ID
        for id_val, expected in expected_counts.items():
            actual = result.filter(col("id") == id_val).count()
            assert actual == expected, f"ID {id_val}: expected {expected}, got {actual}"


# ============================================================================
# Test Class: Merge Logic (Silver Layer)
# ============================================================================

class TestMergeLogic:
    """Tests for CDC merge/upsert logic in Silver layer."""
    
    @pytest.mark.parametrize("operation,expected_exists", [
        ("INSERT", True),
        ("UPDATE", True),
        ("DELETE", False),
    ])
    def test_operation_types(self, spark, silver_schema, base_timestamp, 
                             operation, expected_exists):
        """Test that different CDC operations are handled correctly."""
        # Create target table
        target_data = [(1, "Original", "Orig Address", "orig@email.com", "INSERT")]
        target_df = spark.createDataFrame(target_data, silver_schema)
        
        # Create source with operation
        source_schema = StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("address", StringType(), True),
            StructField("email", StringType(), True),
            StructField("operation", StringType(), True),
        ])
        source_data = [(1, "Modified", "New Address", "new@email.com", operation)]
        source_df = spark.createDataFrame(source_data, source_schema)
        
        result = apply_merge_logic(source_df, target_df)
        
        id_exists = result.filter(col("id") == 1).count() > 0
        assert id_exists == expected_exists
    
    def test_insert_new_record(self, spark, silver_schema):
        """Test INSERT operation adds new record."""
        target_data = [(1, "Existing", "Address1", "exist@email.com", "INSERT")]
        target_df = spark.createDataFrame(target_data, silver_schema)
        
        source_data = [(2, "New User", "Address2", "new@email.com", "INSERT")]
        source_df = spark.createDataFrame(source_data, silver_schema)
        
        result = apply_merge_logic(source_df, target_df)
        
        assert result.count() == 2
        assert result.filter(col("id") == 2).count() == 1
    
    def test_update_modifies_existing(self, spark, silver_schema):
        """Test UPDATE operation modifies existing record."""
        target_data = [(1, "Old Name", "Old Address", "old@email.com", "INSERT")]
        target_df = spark.createDataFrame(target_data, silver_schema)
        
        source_data = [(1, "New Name", "New Address", "new@email.com", "UPDATE")]
        source_df = spark.createDataFrame(source_data, silver_schema)
        
        result = apply_merge_logic(source_df, target_df)
        row = result.filter(col("id") == 1).collect()[0]
        
        assert row["name"] == "New Name"
        assert row["address"] == "New Address"
    
    def test_delete_removes_record(self, spark, silver_schema):
        """Test DELETE operation removes record."""
        target_data = [
            (1, "User1", "Address1", "user1@email.com", "INSERT"),
            (2, "User2", "Address2", "user2@email.com", "INSERT"),
        ]
        target_df = spark.createDataFrame(target_data, silver_schema)
        
        source_data = [(1, None, None, None, "DELETE")]
        source_df = spark.createDataFrame(source_data, silver_schema)
        
        result = apply_merge_logic(source_df, target_df)
        
        assert result.count() == 1
        assert result.filter(col("id") == 1).count() == 0
        assert result.filter(col("id") == 2).count() == 1


# ============================================================================
# Test Class: CDF Processing (Gold Layer)
# ============================================================================

class TestCDFProcessing:
    """Tests for Change Data Feed processing for Gold layer."""
    
    @pytest.mark.parametrize("change_types,expected_type", [
        (["insert"], "insert"),
        (["update_preimage", "update_postimage"], "update_postimage"),
        (["delete"], "delete"),
    ])
    def test_cdf_change_type_filtering(self, spark, base_timestamp, 
                                        change_types, expected_type):
        """Test that correct change type is selected after deduplication."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("_change_type", StringType(), True),
            StructField("_commit_version", LongType(), True),
        ])
        
        data = [(1, f"Name_{ct}", ct, i + 1) for i, ct in enumerate(change_types)]
        df = spark.createDataFrame(data, schema)
        
        result = process_cdf_for_gold(df)
        
        if expected_type != "update_preimage":
            assert result.count() == 1
            row = result.collect()[0]
            assert row["_change_type"] == expected_type
    
    def test_cdf_filters_preimage(self, spark):
        """Test that update_preimage records are filtered out."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("_change_type", StringType(), True),
            StructField("_commit_version", LongType(), True),
        ])
        
        data = [
            (1, "Before Update", "update_preimage", 2),
            (1, "After Update", "update_postimage", 2),
        ]
        df = spark.createDataFrame(data, schema)
        
        result = process_cdf_for_gold(df)
        
        # Should only have postimage
        assert result.count() == 1
        assert result.collect()[0]["name"] == "After Update"
    
    def test_cdf_keeps_latest_version(self, spark):
        """Test that only the latest commit version is kept."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("_change_type", StringType(), True),
            StructField("_commit_version", LongType(), True),
        ])
        
        data = [
            (1, "Version 1", "insert", 1),
            (1, "Version 2", "update_postimage", 2),
            (1, "Version 3", "update_postimage", 3),
        ]
        df = spark.createDataFrame(data, schema)
        
        result = process_cdf_for_gold(df)
        
        assert result.count() == 1
        assert result.collect()[0]["name"] == "Version 3"


# ============================================================================
# Test Class: Data Cleaning
# ============================================================================

class TestDataCleaning:
    """Tests for data cleaning transformations."""
    
    @pytest.mark.parametrize("input_address,expected_address", [
        ('"123 Main St"', '123 Main St'),
        ('456 Oak Ave', '456 Oak Ave'),
        ('"City, "State" 12345"', 'City, State 12345'),
        ('No quotes here', 'No quotes here'),
        ('""', ''),
    ])
    def test_address_cleaning(self, spark, input_address, expected_address):
        """Test that quotes are properly removed from addresses."""
        schema = StructType([
            StructField("id", LongType(), False),
            StructField("address", StringType(), True),
        ])
        df = spark.createDataFrame([(1, input_address)], schema)
        
        result = clean_address(df)
        actual_address = result.collect()[0]["address"]
        
        assert actual_address == expected_address


# ============================================================================
# Test Class: End-to-End Pipeline Simulation
# ============================================================================

class TestEndToEndPipeline:
    """Integration tests simulating full CDC pipeline flow."""
    
    def test_full_cdc_flow_insert_update_delete(self, spark, cdc_schema, 
                                                 silver_schema, base_timestamp):
        """Test complete CDC flow with INSERT, UPDATE, and DELETE operations."""
        # Initial inserts
        initial_data = [
            (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, "INSERT", None, "f1.csv"),
            (2, "Bob", "456 Oak Ave", "bob@email.com", base_timestamp, "INSERT", None, "f1.csv"),
            (3, "Charlie", "789 Pine Rd", "charlie@email.com", base_timestamp, "INSERT", None, "f1.csv"),
        ]
        initial_df = spark.createDataFrame(initial_data, cdc_schema)
        
        # Deduplicate and prepare for merge
        deduped = deduplicate_cdc_data(initial_df)
        
        # Create empty target
        empty_target = spark.createDataFrame([], silver_schema)
        
        # Apply initial inserts
        result = apply_merge_logic(
            deduped.select("id", "name", "address", "email", "operation"),
            empty_target
        )
        
        assert result.count() == 3
        
        # Apply updates
        update_data = [
            (1, "Alice Updated", "999 New St", "alice.new@email.com", 
             base_timestamp + timedelta(hours=1), "UPDATE", None, "f2.csv"),
        ]
        update_df = spark.createDataFrame(update_data, cdc_schema)
        deduped_update = deduplicate_cdc_data(update_df)
        
        result = apply_merge_logic(
            deduped_update.select("id", "name", "address", "email", "operation"),
            result
        )
        
        alice_row = result.filter(col("id") == 1).collect()[0]
        assert alice_row["name"] == "Alice Updated"
        assert result.count() == 3
        
        # Apply delete
        delete_data = [
            (2, None, None, None, base_timestamp + timedelta(hours=2), "DELETE", None, "f3.csv"),
        ]
        delete_df = spark.createDataFrame(delete_data, cdc_schema)
        deduped_delete = deduplicate_cdc_data(delete_df)
        
        result = apply_merge_logic(
            deduped_delete.select("id", "name", "address", "email", "operation"),
            result
        )
        
        assert result.count() == 2
        assert result.filter(col("id") == 2).count() == 0
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_multiple_batches(self, spark, cdc_schema, silver_schema, 
                              base_timestamp, batch_size):
        """Test processing multiple CDC batches of varying sizes."""
        result = spark.createDataFrame([], silver_schema)
        
        for batch_num in range(3):
            batch_data = [
                (batch_num * batch_size + i, 
                 f"User_{batch_num}_{i}", 
                 f"Address_{batch_num}_{i}",
                 f"user{batch_num}_{i}@email.com",
                 base_timestamp + timedelta(hours=batch_num),
                 "INSERT", None, f"file{batch_num}.csv")
                for i in range(batch_size)
            ]
            batch_df = spark.createDataFrame(batch_data, cdc_schema)
            deduped = deduplicate_cdc_data(batch_df)
            
            result = apply_merge_logic(
                deduped.select("id", "name", "address", "email", "operation"),
                result
            )
        
        expected_count = batch_size * 3
        assert result.count() == expected_count


# ============================================================================
# Test Class: Schema Handling
# ============================================================================

class TestSchemaHandling:
    """Tests for schema validation and handling."""
    
    def test_null_handling_in_delete(self, spark, cdc_schema, base_timestamp):
        """Test that NULL values in DELETE operations are handled correctly."""
        data = [
            (1, None, None, None, base_timestamp, "DELETE", None, "file.csv"),
        ]
        df = spark.createDataFrame(data, cdc_schema)
        
        result = deduplicate_cdc_data(df)
        
        assert result.count() == 1
        row = result.collect()[0]
        assert row["id"] == 1
        assert row["operation"] == "DELETE"
    
    @pytest.mark.parametrize("rescued_data", [
        None,
        '{"extra_field": "value"}',
        '',
    ])
    def test_rescued_data_handling(self, spark, cdc_schema, base_timestamp, rescued_data):
        """Test that _rescued_data column is handled correctly."""
        data = [
            (1, "Name", "Address", "email@test.com", base_timestamp, "INSERT", rescued_data, "file.csv"),
        ]
        df = spark.createDataFrame(data, cdc_schema)
        
        result = deduplicate_cdc_data(df)
        
        assert result.count() == 1

