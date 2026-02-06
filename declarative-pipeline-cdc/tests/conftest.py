"""
Pytest fixtures for Declarative CDC Pipeline unit tests.

This module provides:
- Spark session lifecycle management
- Schema definitions for CDC data
- Sample data fixtures for various test scenarios
- Mock utilities for external dependencies

The fixtures follow the dbx_test framework patterns and are designed
for fast, deterministic, isolated unit tests.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType, 
    DoubleType, IntegerType, BooleanType
)
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
import tempfile
import shutil


# ============================================================================
# Spark Session Fixture
# ============================================================================

@pytest.fixture(scope="session")
def spark():
    """
    Create a local Spark session for testing.
    
    The session is configured with:
    - Delta Lake support
    - In-memory warehouse for isolation
    - Minimal logging for cleaner test output
    
    This fixture is session-scoped to avoid the overhead of creating
    a new Spark session for each test.
    """
    # Check if running in Databricks (spark already exists)
    try:
        from pyspark.sql import SparkSession
        existing_spark = SparkSession.getActiveSession()
        if existing_spark is not None:
            return existing_spark
    except Exception:
        pass
    
    # Create local Spark session for testing
    builder = (SparkSession.builder
               .appName("declarative-pipeline-cdc-tests")
               .master("local[*]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.sql.warehouse.dir", tempfile.mkdtemp())
               .config("spark.driver.memory", "2g")
               .config("spark.sql.shuffle.partitions", "2")  # Reduce for faster tests
               .config("spark.ui.enabled", "false"))  # Disable UI for headless testing
    
    spark_session = builder.getOrCreate()
    spark_session.sparkContext.setLogLevel("WARN")
    
    yield spark_session
    
    # Cleanup
    spark_session.stop()


# ============================================================================
# Test Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_catalog():
    """Test catalog name - used to mock real catalog references."""
    return "test_catalog"


@pytest.fixture(scope="session")
def test_schema():
    """Test schema/database name - used to mock real schema references."""
    return "test_schema"


@pytest.fixture
def temp_path():
    """
    Create a temporary directory for test data.
    Automatically cleaned up after each test.
    """
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# Schema Fixtures - Matching Pipeline Definitions
# ============================================================================

@pytest.fixture(scope="module")
def customers_cdc_schema():
    """
    Schema for raw CDC customer data (Bronze layer).
    Matches the schema inferred by cloud_files with JSON format.
    """
    return StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation", StringType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("_rescued_data", StringType(), True),  # Autoloader schema rescue
    ])


@pytest.fixture(scope="module")
def customers_clean_schema():
    """
    Schema for cleaned CDC data (after expectations/quality checks).
    This is the schema after the cdc_clean view/table.
    """
    return StructType([
        StructField("id", LongType(), False),  # NOT NULL after expectation
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation", StringType(), False),  # Valid after expectation
        StructField("operation_date", TimestampType(), True),
    ])


@pytest.fixture(scope="module")
def customers_silver_schema():
    """
    Schema for materialized customer table (Silver layer).
    After APPLY CHANGES, excludes operation columns.
    """
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
    ])


@pytest.fixture(scope="module")
def scd2_schema():
    """
    Schema for SCD Type 2 table.
    Includes validity tracking columns added by APPLY CHANGES with SCD2.
    """
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("__START_AT", TimestampType(), True),
        StructField("__END_AT", TimestampType(), True),
    ])


@pytest.fixture(scope="module")
def generic_cdc_schema():
    """
    Generic CDC schema for multi-table pipeline tests.
    Used when testing the dynamic table creation in Python pipeline.
    """
    return StructType([
        StructField("id", LongType(), True),
        StructField("data", StringType(), True),
        StructField("operation", StringType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("_rescued_data", StringType(), True),
    ])


# ============================================================================
# Time Fixtures
# ============================================================================

@pytest.fixture
def base_timestamp():
    """Base timestamp for test data - provides consistent reference point."""
    return datetime(2024, 1, 1, 10, 0, 0)


@pytest.fixture
def timestamp_sequence(base_timestamp):
    """
    Factory fixture to generate a sequence of timestamps.
    
    Returns:
        Function that generates timestamps with specified intervals
    """
    def _generate(count: int, interval_hours: int = 1):
        return [base_timestamp + timedelta(hours=i * interval_hours) for i in range(count)]
    return _generate


# ============================================================================
# Sample Data Fixtures - Valid Data
# ============================================================================

@pytest.fixture
def sample_customers_cdc(spark, customers_cdc_schema, base_timestamp):
    """
    Sample valid CDC customer data with INSERT operations.
    Represents typical initial data load.
    """
    data = [
        (1, "Alice Johnson", "123 Main St, New York, NY", "alice@email.com", 
         "APPEND", base_timestamp, None),
        (2, "Bob Smith", "456 Oak Ave, Los Angeles, CA", "bob@email.com",
         "APPEND", base_timestamp, None),
        (3, "Charlie Brown", "789 Pine Rd, Chicago, IL", "charlie@email.com",
         "APPEND", base_timestamp, None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_customers_with_updates(spark, customers_cdc_schema, base_timestamp):
    """
    Sample CDC data including UPDATE operations.
    Tests the sequence deduplication and merge logic.
    """
    data = [
        # Initial inserts
        (1, "Alice Johnson", "123 Main St, New York, NY", "alice@email.com",
         "APPEND", base_timestamp, None),
        (2, "Bob Smith", "456 Oak Ave, Los Angeles, CA", "bob@email.com",
         "APPEND", base_timestamp, None),
        # Update for Alice - later timestamp should win
        (1, "Alice Johnson-Smith", "999 Updated Blvd, New York, NY", "alice.updated@email.com",
         "UPDATE", base_timestamp + timedelta(hours=2), None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_customers_with_deletes(spark, customers_cdc_schema, base_timestamp):
    """
    Sample CDC data including DELETE operations.
    Tests the delete handling in APPLY CHANGES.
    """
    data = [
        (1, "Alice Johnson", "123 Main St, New York, NY", "alice@email.com",
         "APPEND", base_timestamp, None),
        (2, "Bob Smith", "456 Oak Ave, Los Angeles, CA", "bob@email.com",
         "APPEND", base_timestamp, None),
        (3, "Charlie Brown", "789 Pine Rd, Chicago, IL", "charlie@email.com",
         "APPEND", base_timestamp, None),
        # Delete Bob
        (2, None, None, None, "DELETE", base_timestamp + timedelta(hours=1), None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_customers_scd2_history(spark, customers_cdc_schema, base_timestamp):
    """
    Sample CDC data for testing SCD2 history tracking.
    Multiple updates to same customer to verify history preservation.
    """
    data = [
        # Initial insert
        (1, "Alice Johnson", "123 Main St", "alice@email.com",
         "APPEND", base_timestamp, None),
        # First address change
        (1, "Alice Johnson", "456 New St", "alice@email.com",
         "UPDATE", base_timestamp + timedelta(days=30), None),
        # Email change
        (1, "Alice Johnson", "456 New St", "alice.new@email.com",
         "UPDATE", base_timestamp + timedelta(days=60), None),
        # Name change (marriage)
        (1, "Alice Smith", "456 New St", "alice.smith@email.com",
         "UPDATE", base_timestamp + timedelta(days=90), None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


# ============================================================================
# Sample Data Fixtures - Edge Cases
# ============================================================================

@pytest.fixture
def sample_cdc_with_nulls(spark, customers_cdc_schema, base_timestamp):
    """
    CDC data with NULL values in various columns.
    Tests handling of optional fields.
    """
    data = [
        (1, "Alice", None, "alice@email.com", "APPEND", base_timestamp, None),  # Null address
        (2, "Bob", "456 Oak Ave", None, "APPEND", base_timestamp, None),  # Null email
        (3, None, "789 Pine Rd", "charlie@email.com", "APPEND", base_timestamp, None),  # Null name
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_cdc_with_duplicates(spark, customers_cdc_schema, base_timestamp):
    """
    CDC data with duplicate records for the same ID.
    Tests deduplication by operation_date (latest wins).
    """
    data = [
        # Multiple versions of same customer, different timestamps
        (1, "Alice V1", "Address V1", "v1@email.com",
         "APPEND", base_timestamp, None),
        (1, "Alice V2", "Address V2", "v2@email.com",
         "UPDATE", base_timestamp + timedelta(hours=1), None),
        (1, "Alice V3", "Address V3", "v3@email.com",
         "UPDATE", base_timestamp + timedelta(hours=2), None),
        (1, "Alice V4 - FINAL", "Address V4", "v4@email.com",
         "UPDATE", base_timestamp + timedelta(hours=3), None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_cdc_empty(spark, customers_cdc_schema):
    """Empty CDC DataFrame for testing empty input handling."""
    return spark.createDataFrame([], customers_cdc_schema)


@pytest.fixture
def sample_cdc_with_rescued_data(spark, customers_cdc_schema, base_timestamp):
    """
    CDC data with non-null _rescued_data column.
    These records should be DROPPED by the data quality expectations.
    """
    data = [
        # Valid record
        (1, "Alice", "123 Main St", "alice@email.com",
         "APPEND", base_timestamp, None),
        # Invalid JSON schema - has rescued data (should be dropped)
        (2, "Bob", "456 Oak Ave", "bob@email.com",
         "APPEND", base_timestamp, '{"unexpected_field": "value"}'),
        # Another valid record
        (3, "Charlie", "789 Pine Rd", "charlie@email.com",
         "APPEND", base_timestamp, None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_cdc_with_invalid_operations(spark, customers_cdc_schema, base_timestamp):
    """
    CDC data with invalid operation types.
    These should be DROPPED by the valid_operation expectation.
    """
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        (2, "Bob", "456 Oak Ave", "bob@email.com", "INVALID_OP", base_timestamp, None),  # Invalid
        (3, "Charlie", "789 Pine Rd", "charlie@email.com", "UPDATE", base_timestamp, None),
        (4, "Diana", "101 Elm St", "diana@email.com", "MERGE", base_timestamp, None),  # Invalid
        (5, "Eve", "202 Birch Ln", "eve@email.com", "DELETE", base_timestamp, None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


@pytest.fixture
def sample_cdc_with_null_ids(spark, customers_cdc_schema, base_timestamp):
    """
    CDC data with NULL id values.
    These should be DROPPED by the valid_id expectation.
    """
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        (None, "Invalid User", "No Address", "invalid@email.com", "APPEND", base_timestamp, None),
        (2, "Bob", "456 Oak Ave", "bob@email.com", "APPEND", base_timestamp, None),
    ]
    return spark.createDataFrame(data, customers_cdc_schema)


# ============================================================================
# Multi-Table Pipeline Fixtures
# ============================================================================

@pytest.fixture
def multi_table_names():
    """List of table names for multi-table pipeline tests."""
    return ["customers", "orders", "products", "inventory"]


@pytest.fixture
def sample_multi_table_data(spark, generic_cdc_schema, base_timestamp, multi_table_names):
    """
    Factory fixture to create CDC data for multiple tables.
    
    Returns:
        Dictionary mapping table names to their CDC DataFrames
    """
    result = {}
    for i, table_name in enumerate(multi_table_names):
        data = [
            (j + 1, f"{table_name}_data_{j}", "APPEND", 
             base_timestamp + timedelta(hours=i), None)
            for j in range(3)
        ]
        result[table_name] = spark.createDataFrame(data, generic_cdc_schema)
    return result


# ============================================================================
# Mock Folder Structure Fixtures
# ============================================================================

@pytest.fixture
def mock_folder_listing():
    """
    Mock of dbutils.fs.ls() output for testing pipeline table discovery.
    Simulates the folder structure in /Volumes/{catalog}/{schema}/raw_data/
    """
    class MockFileInfo:
        """Mock FileInfo object matching Databricks FileInfo structure."""
        def __init__(self, name: str, path: str):
            self.name = name
            self.path = path
    
    return [
        MockFileInfo("customers/", "/Volumes/test_catalog/test_schema/raw_data/customers/"),
        MockFileInfo("orders/", "/Volumes/test_catalog/test_schema/raw_data/orders/"),
        MockFileInfo("products/", "/Volumes/test_catalog/test_schema/raw_data/products/"),
    ]


# ============================================================================
# Assertion Helper Fixtures
# ============================================================================

@pytest.fixture
def assert_dataframe_equal(spark):
    """
    Factory fixture providing DataFrame equality assertion.
    
    Compares:
    - Schema (column names and types)
    - Row count
    - Row-level data (order-independent by default)
    
    Usage:
        assert_dataframe_equal(expected_df, actual_df)
    """
    def _assert_equal(expected, actual, check_order=False, check_schema=True):
        # Schema comparison
        if check_schema:
            assert expected.schema == actual.schema, (
                f"Schema mismatch:\n"
                f"Expected: {expected.schema.simpleString()}\n"
                f"Actual: {actual.schema.simpleString()}"
            )
        
        # Row count comparison
        expected_count = expected.count()
        actual_count = actual.count()
        assert expected_count == actual_count, (
            f"Row count mismatch: expected {expected_count}, got {actual_count}"
        )
        
        # Data comparison
        if check_order:
            expected_data = expected.collect()
            actual_data = actual.collect()
            for i, (exp_row, act_row) in enumerate(zip(expected_data, actual_data)):
                assert exp_row == act_row, f"Row {i} mismatch:\nExpected: {exp_row}\nActual: {act_row}"
        else:
            # Order-independent comparison using set semantics
            expected_data = set(tuple(row) for row in expected.collect())
            actual_data = set(tuple(row) for row in actual.collect())
            assert expected_data == actual_data, (
                f"Data mismatch:\n"
                f"Expected: {expected_data}\n"
                f"Actual: {actual_data}"
            )
    
    return _assert_equal


@pytest.fixture
def assert_schema_contains(spark):
    """
    Factory fixture for partial schema assertion.
    Verifies that actual schema contains all expected columns.
    """
    def _assert_contains(df, expected_columns: List[str]):
        actual_columns = df.columns
        missing = set(expected_columns) - set(actual_columns)
        assert not missing, f"Missing columns: {missing}. Actual columns: {actual_columns}"
    
    return _assert_contains


# ============================================================================
# Streaming Test Fixtures (Optional - for streaming pipeline tests)
# ============================================================================

@pytest.fixture
def memory_stream_source(spark):
    """
    Factory fixture to create a MemoryStream for streaming tests.
    
    Usage:
        stream = memory_stream_source(schema)
        stream.addData([...])
        query = stream.toDF().writeStream...
    """
    def _create_stream(schema: StructType):
        from pyspark.sql.streaming import DataStreamReader
        return spark.readStream.format("rate").option("rowsPerSecond", 1)
    
    return _create_stream


# ============================================================================
# Delta Table Cleanup Fixture
# ============================================================================

@pytest.fixture
def temp_delta_tables(spark, temp_path):
    """
    Fixture that tracks temporary Delta tables and cleans them up.
    
    Usage:
        table_path = temp_delta_tables.create("my_table", df)
        # ... test code ...
        # Automatic cleanup after test
    """
    class DeltaTableManager:
        def __init__(self):
            self.tables = []
            self.base_path = temp_path
        
        def create(self, name: str, df, partition_by: List[str] = None):
            """Create a Delta table and track it for cleanup."""
            path = os.path.join(self.base_path, name)
            writer = df.write.format("delta").mode("overwrite")
            if partition_by:
                writer = writer.partitionBy(*partition_by)
            writer.save(path)
            self.tables.append(path)
            return path
        
        def cleanup(self):
            """Remove all tracked tables."""
            for path in self.tables:
                shutil.rmtree(path, ignore_errors=True)
    
    manager = DeltaTableManager()
    yield manager
    manager.cleanup()

