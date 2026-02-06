"""
Pytest fixtures for CDC Pipeline unit tests.
These fixtures provide mock data and Spark session setup for testing.
"""

import pytest
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType, IntegerType
)
from datetime import datetime, timedelta
from delta.tables import DeltaTable
import os


@pytest.fixture(scope="session")
def test_catalog(spark):
    """Get or create test catalog name."""
    return "test_cdc_catalog"


@pytest.fixture(scope="session")
def test_schema(spark):
    """Get or create test schema/database name."""
    return "test_cdc_schema"


# ============================================================================
# Schema Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def cdc_schema():
    """Schema for CDC input data (Bronze layer)."""
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("operation", StringType(), True),
        StructField("_rescued_data", StringType(), True),
        StructField("file_name", StringType(), True)
    ])


@pytest.fixture(scope="module")
def silver_schema():
    """Schema for Silver layer table."""
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation", StringType(), True)
    ])


@pytest.fixture(scope="module")
def gold_schema():
    """Schema for Gold layer table."""
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("gold_data", StringType(), True)
    ])


@pytest.fixture(scope="module")
def cdf_schema():
    """Schema for CDF (Change Data Feed) output."""
    return StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation", StringType(), True),
        StructField("_change_type", StringType(), True),
        StructField("_commit_version", LongType(), True),
        StructField("_commit_timestamp", TimestampType(), True)
    ])


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def base_timestamp():
    """Base timestamp for test data."""
    return datetime(2024, 1, 1, 10, 0, 0)


@pytest.fixture
def sample_cdc_data(spark, cdc_schema, base_timestamp):
    """Sample CDC data with INSERT, UPDATE, DELETE operations."""
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (2, "Bob", "456 Oak Ave", "bob@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (3, "Charlie", "789 Pine Rd", "charlie@email.com", base_timestamp, "INSERT", None, "file1.csv"),
    ]
    return spark.createDataFrame(data, cdc_schema)


@pytest.fixture
def sample_cdc_with_updates(spark, cdc_schema, base_timestamp):
    """Sample CDC data including UPDATE operations."""
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (2, "Bob", "456 Oak Ave", "bob@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (1, "Alice Updated", "999 New St", "alice.new@email.com", 
         base_timestamp + timedelta(hours=1), "UPDATE", None, "file2.csv"),
    ]
    return spark.createDataFrame(data, cdc_schema)


@pytest.fixture
def sample_cdc_with_deletes(spark, cdc_schema, base_timestamp):
    """Sample CDC data including DELETE operations."""
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (2, "Bob", "456 Oak Ave", "bob@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (2, None, None, None, base_timestamp + timedelta(hours=1), "DELETE", None, "file2.csv"),
    ]
    return spark.createDataFrame(data, cdc_schema)


@pytest.fixture
def sample_cdc_with_duplicates(spark, cdc_schema, base_timestamp):
    """Sample CDC data with duplicate IDs requiring deduplication."""
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, "INSERT", None, "file1.csv"),
        (1, "Alice V2", "456 New St", "alice2@email.com", 
         base_timestamp + timedelta(hours=1), "UPDATE", None, "file2.csv"),
        (1, "Alice V3", "789 Final St", "alice3@email.com", 
         base_timestamp + timedelta(hours=2), "UPDATE", None, "file3.csv"),
    ]
    return spark.createDataFrame(data, cdc_schema)


@pytest.fixture
def sample_silver_data(spark, silver_schema):
    """Sample Silver layer data."""
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", "INSERT"),
        (2, "Bob", "456 Oak Ave", "bob@email.com", "INSERT"),
        (3, "Charlie", "789 Pine Rd", "charlie@email.com", "INSERT"),
    ]
    return spark.createDataFrame(data, silver_schema)


@pytest.fixture
def sample_cdf_data(spark, base_timestamp):
    """Sample CDF (Change Data Feed) data for Gold layer tests."""
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("email", StringType(), True),
        StructField("gold_data", StringType(), True),
        StructField("_change_type", StringType(), True),
        StructField("_commit_version", LongType(), True),
    ])
    data = [
        (1, "Alice", "123 Main St", "alice@email.com", "Delta CDF is Awesome", "insert", 1),
        (2, "Bob", "456 Oak Ave", "bob@email.com", "Delta CDF is Awesome", "insert", 1),
        (1, "Alice", "123 Main St", "alice@email.com", "Delta CDF is Awesome", "update_preimage", 2),
        (1, "Alice Updated", "999 New St", "alice.new@email.com", "Delta CDF is Awesome", "update_postimage", 2),
    ]
    return spark.createDataFrame(data, schema)


# ============================================================================
# Table Management Fixtures
# ============================================================================

@pytest.fixture
def temp_table_cleanup(spark):
    """
    Fixture that yields table names and cleans them up after test.
    Usage: tables = temp_table_cleanup; tables.append("my_temp_table")
    """
    tables_to_cleanup = []
    yield tables_to_cleanup
    for table in tables_to_cleanup:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass


@pytest.fixture
def create_test_table(spark, temp_table_cleanup):
    """
    Factory fixture to create test Delta tables.
    Returns a function that creates tables and registers them for cleanup.
    """
    def _create_table(table_name, df, enable_cdf=False):
        # Write DataFrame as Delta table
        df.write.format("delta").mode("overwrite").saveAsTable(table_name)
        
        # Enable CDF if requested
        if enable_cdf:
            spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        
        temp_table_cleanup.append(table_name)
        return table_name
    
    return _create_table


# ============================================================================
# Multi-Table Test Fixtures
# ============================================================================

@pytest.fixture
def multi_table_config():
    """Configuration for multi-table CDC tests."""
    return {
        "users": {
            "columns": ["id", "name", "email", "operation", "operation_date"],
            "id_column": "id"
        },
        "transactions": {
            "columns": ["id", "user_id", "amount", "operation", "operation_date"],
            "id_column": "id"
        },
        "products": {
            "columns": ["id", "name", "price", "operation", "operation_date"],
            "id_column": "id"
        }
    }


@pytest.fixture
def sample_users_cdc(spark, base_timestamp):
    """Sample users CDC data for multi-table tests."""
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("operation", StringType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("_rescued_data", StringType(), True),
        StructField("file_name", StringType(), True)
    ])
    data = [
        (1, "User1", "user1@email.com", "INSERT", base_timestamp, None, "users.csv"),
        (2, "User2", "user2@email.com", "INSERT", base_timestamp, None, "users.csv"),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_transactions_cdc(spark, base_timestamp):
    """Sample transactions CDC data for multi-table tests."""
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("user_id", LongType(), True),
        StructField("amount", StringType(), True),
        StructField("operation", StringType(), True),
        StructField("operation_date", TimestampType(), True),
        StructField("_rescued_data", StringType(), True),
        StructField("file_name", StringType(), True)
    ])
    data = [
        (100, 1, "99.99", "INSERT", base_timestamp, None, "transactions.csv"),
        (101, 2, "149.99", "INSERT", base_timestamp, None, "transactions.csv"),
    ]
    return spark.createDataFrame(data, schema)

