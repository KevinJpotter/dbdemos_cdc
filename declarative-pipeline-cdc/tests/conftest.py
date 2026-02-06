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
# Delta Table Cleanup Fixture (Path-based)
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


# ============================================================================
# Test Schema/Database Management Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_database_name():
    """
    Generate a unique test database name.
    Uses timestamp to avoid conflicts between test runs.
    """
    import time
    return f"test_cdc_pipeline_{int(time.time())}"


@pytest.fixture(scope="module")
def test_database(spark, test_database_name):
    """
    Create a test database/schema for the test module.
    
    The database is created at module scope and cleaned up after all tests
    in the module complete. This allows multiple tests to share tables.
    
    Yields:
        str: The name of the created database
    """
    # Create the test database
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {test_database_name}")
    spark.sql(f"USE {test_database_name}")
    
    print(f"Created test database: {test_database_name}")
    
    yield test_database_name
    
    # Cleanup: drop all tables and the database
    try:
        tables = spark.sql(f"SHOW TABLES IN {test_database_name}").collect()
        for table in tables:
            table_name = table["tableName"]
            spark.sql(f"DROP TABLE IF EXISTS {test_database_name}.{table_name}")
        spark.sql(f"DROP DATABASE IF EXISTS {test_database_name} CASCADE")
        print(f"Cleaned up test database: {test_database_name}")
    except Exception as e:
        print(f"Warning: Failed to cleanup database {test_database_name}: {e}")


@pytest.fixture
def test_schema_manager(spark, test_database):
    """
    Factory fixture for creating and managing test schemas within a database.
    
    Provides methods to:
    - Create schemas
    - Drop schemas
    - List tables in schemas
    
    Usage:
        schema_name = test_schema_manager.create_schema("bronze")
        test_schema_manager.drop_schema(schema_name)
    """
    class SchemaManager:
        def __init__(self, database: str):
            self.database = database
            self.schemas_created = []
        
        def create_schema(self, schema_name: str) -> str:
            """Create a schema within the test database."""
            full_name = f"{self.database}.{schema_name}"
            # In non-Unity Catalog, schemas are databases
            # For Unity Catalog, this would be different
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            self.schemas_created.append(schema_name)
            return schema_name
        
        def drop_schema(self, schema_name: str, cascade: bool = True):
            """Drop a schema."""
            cascade_str = "CASCADE" if cascade else ""
            spark.sql(f"DROP SCHEMA IF EXISTS {schema_name} {cascade_str}")
            if schema_name in self.schemas_created:
                self.schemas_created.remove(schema_name)
        
        def list_tables(self, schema_name: str = None) -> List[str]:
            """List tables in a schema or the default database."""
            target = schema_name if schema_name else self.database
            tables = spark.sql(f"SHOW TABLES IN {target}").collect()
            return [t["tableName"] for t in tables]
        
        def cleanup(self):
            """Clean up all created schemas."""
            for schema in self.schemas_created[:]:
                self.drop_schema(schema, cascade=True)
    
    manager = SchemaManager(test_database)
    yield manager
    manager.cleanup()


# ============================================================================
# Managed Table Creation Fixtures
# ============================================================================

@pytest.fixture
def create_managed_table(spark, test_database):
    """
    Factory fixture to create managed Delta tables in the test database.
    
    Tables are automatically tracked for cleanup.
    
    Usage:
        table_name = create_managed_table(
            name="customers",
            df=sample_df,
            enable_cdf=True,
            partition_by=["date"]
        )
    """
    tables_created = []
    
    def _create_table(
        name: str,
        df,
        enable_cdf: bool = False,
        partition_by: List[str] = None,
        table_properties: Dict[str, str] = None
    ) -> str:
        """
        Create a managed Delta table.
        
        Args:
            name: Table name (will be created in test database)
            df: DataFrame to write
            enable_cdf: Enable Change Data Feed on the table
            partition_by: Columns to partition by
            table_properties: Additional table properties
            
        Returns:
            Full table name (database.table)
        """
        full_name = f"{test_database}.{name}"
        
        # Build writer
        writer = df.write.format("delta").mode("overwrite")
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        # Create table
        writer.saveAsTable(full_name)
        tables_created.append(full_name)
        
        # Apply table properties
        if enable_cdf:
            spark.sql(f"ALTER TABLE {full_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
        
        if table_properties:
            for key, value in table_properties.items():
                spark.sql(f"ALTER TABLE {full_name} SET TBLPROPERTIES ('{key}' = '{value}')")
        
        return full_name
    
    yield _create_table
    
    # Cleanup created tables
    for table in tables_created:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass


@pytest.fixture
def create_delta_table_with_cdf(spark, test_database):
    """
    Factory fixture specifically for creating Delta tables with CDF enabled.
    
    CDF (Change Data Feed) is required for tracking changes in Delta tables,
    which is essential for CDC pipeline testing.
    
    Usage:
        table_name = create_delta_table_with_cdf("silver_customers", initial_df)
        # Now can read changes via spark.read.format("delta").option("readChangeFeed", "true")
    """
    tables_created = []
    
    def _create_cdf_table(
        name: str,
        df,
        primary_keys: List[str] = None
    ) -> str:
        """
        Create a Delta table with Change Data Feed enabled.
        
        Args:
            name: Table name
            df: Initial data to write
            primary_keys: Optional primary key columns (for documentation)
            
        Returns:
            Full table name
        """
        full_name = f"{test_database}.{name}"
        
        # Create table
        df.write.format("delta").mode("overwrite").saveAsTable(full_name)
        tables_created.append(full_name)
        
        # Enable CDF
        spark.sql(f"""
            ALTER TABLE {full_name} 
            SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """)
        
        # Add comment if primary keys specified
        if primary_keys:
            pk_str = ", ".join(primary_keys)
            spark.sql(f"COMMENT ON TABLE {full_name} IS 'Primary Keys: {pk_str}'")
        
        return full_name
    
    yield _create_cdf_table
    
    # Cleanup
    for table in tables_created:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass


# ============================================================================
# CDC Pipeline Table Setup Fixtures
# ============================================================================

@pytest.fixture
def cdc_pipeline_tables(spark, test_database, customers_cdc_schema, 
                        customers_silver_schema, scd2_schema, base_timestamp):
    """
    Create the complete set of tables needed for CDC pipeline testing.
    
    Creates:
    - customers_cdc: Bronze layer raw CDC data
    - customers_cdc_clean: Cleaned CDC data (view simulation)
    - customers: Silver layer materialized table
    - SCD2_customers: SCD Type 2 history table
    
    This fixture sets up a realistic pipeline structure for integration tests.
    """
    tables = {}
    
    # Bronze layer: Raw CDC data
    bronze_data = [
        (1, "Alice", "123 Main St", "alice@email.com", "APPEND", base_timestamp, None),
        (2, "Bob", "456 Oak Ave", "bob@email.com", "APPEND", base_timestamp, None),
    ]
    bronze_df = spark.createDataFrame(bronze_data, customers_cdc_schema)
    bronze_name = f"{test_database}.customers_cdc"
    bronze_df.write.format("delta").mode("overwrite").saveAsTable(bronze_name)
    tables["bronze"] = bronze_name
    
    # Silver layer: Materialized customers
    silver_data = [
        (1, "Alice", "123 Main St", "alice@email.com"),
        (2, "Bob", "456 Oak Ave", "bob@email.com"),
    ]
    silver_df = spark.createDataFrame(silver_data, customers_silver_schema)
    silver_name = f"{test_database}.customers"
    silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_name)
    spark.sql(f"ALTER TABLE {silver_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
    tables["silver"] = silver_name
    
    # SCD2 layer: Historical tracking
    scd2_data = [
        (1, "Alice", "123 Main St", "alice@email.com", base_timestamp, None),
        (2, "Bob", "456 Oak Ave", "bob@email.com", base_timestamp, None),
    ]
    scd2_df = spark.createDataFrame(scd2_data, scd2_schema)
    scd2_name = f"{test_database}.scd2_customers"
    scd2_df.write.format("delta").mode("overwrite").saveAsTable(scd2_name)
    tables["scd2"] = scd2_name
    
    yield tables
    
    # Cleanup
    for table in tables.values():
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass


@pytest.fixture
def empty_cdc_pipeline_tables(spark, test_database, customers_cdc_schema,
                              customers_silver_schema, scd2_schema):
    """
    Create empty tables with correct schemas for CDC pipeline testing.
    
    Useful for testing initial data load scenarios where tables exist
    but have no data.
    """
    tables = {}
    
    # Create empty DataFrames with correct schemas
    empty_bronze = spark.createDataFrame([], customers_cdc_schema)
    empty_silver = spark.createDataFrame([], customers_silver_schema)
    empty_scd2 = spark.createDataFrame([], scd2_schema)
    
    # Bronze
    bronze_name = f"{test_database}.customers_cdc_empty"
    empty_bronze.write.format("delta").mode("overwrite").saveAsTable(bronze_name)
    tables["bronze"] = bronze_name
    
    # Silver with CDF
    silver_name = f"{test_database}.customers_empty"
    empty_silver.write.format("delta").mode("overwrite").saveAsTable(silver_name)
    spark.sql(f"ALTER TABLE {silver_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
    tables["silver"] = silver_name
    
    # SCD2
    scd2_name = f"{test_database}.scd2_customers_empty"
    empty_scd2.write.format("delta").mode("overwrite").saveAsTable(scd2_name)
    tables["scd2"] = scd2_name
    
    yield tables
    
    # Cleanup
    for table in tables.values():
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass


# ============================================================================
# Multi-Table Pipeline Setup Fixture
# ============================================================================

@pytest.fixture
def multi_table_pipeline_setup(spark, test_database, generic_cdc_schema, base_timestamp):
    """
    Create multiple tables simulating a multi-table CDC pipeline.
    
    Creates bronze and silver tables for:
    - customers
    - orders
    - products
    
    This matches the structure of the Python pipeline that loops over folders.
    """
    table_configs = {
        "customers": {
            "bronze_data": [
                (1, "Customer 1", "APPEND", base_timestamp, None),
                (2, "Customer 2", "APPEND", base_timestamp, None),
            ],
            "silver_schema": StructType([
                StructField("id", LongType(), False),
                StructField("data", StringType(), True),
            ])
        },
        "orders": {
            "bronze_data": [
                (100, "Order 100", "APPEND", base_timestamp, None),
                (101, "Order 101", "APPEND", base_timestamp, None),
            ],
            "silver_schema": StructType([
                StructField("id", LongType(), False),
                StructField("data", StringType(), True),
            ])
        },
        "products": {
            "bronze_data": [
                (1000, "Product A", "APPEND", base_timestamp, None),
                (1001, "Product B", "APPEND", base_timestamp, None),
            ],
            "silver_schema": StructType([
                StructField("id", LongType(), False),
                StructField("data", StringType(), True),
            ])
        }
    }
    
    created_tables = {}
    
    for table_name, config in table_configs.items():
        # Bronze table
        bronze_df = spark.createDataFrame(config["bronze_data"], generic_cdc_schema)
        bronze_full_name = f"{test_database}.{table_name}_cdc"
        bronze_df.write.format("delta").mode("overwrite").saveAsTable(bronze_full_name)
        
        # Silver table (empty initially)
        silver_df = spark.createDataFrame([], config["silver_schema"])
        silver_full_name = f"{test_database}.{table_name}"
        silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_full_name)
        spark.sql(f"ALTER TABLE {silver_full_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
        
        created_tables[table_name] = {
            "bronze": bronze_full_name,
            "silver": silver_full_name
        }
    
    yield created_tables
    
    # Cleanup all tables
    for table_info in created_tables.values():
        for table in table_info.values():
            try:
                spark.sql(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass


# ============================================================================
# Table Operation Utilities
# ============================================================================

@pytest.fixture
def table_operations(spark):
    """
    Utility fixture providing common table operations for testing.
    
    Provides methods for:
    - MERGE operations
    - INSERT/UPDATE/DELETE
    - Reading CDF (Change Data Feed)
    - Table history inspection
    """
    class TableOperations:
        def __init__(self):
            pass
        
        def merge_into(self, target_table: str, source_df, 
                       match_columns: List[str],
                       update_columns: List[str] = None,
                       delete_condition: str = None) -> int:
            """
            Perform a MERGE INTO operation.
            
            Args:
                target_table: Target table name
                source_df: Source DataFrame
                match_columns: Columns to match on
                update_columns: Columns to update (None = all non-key columns)
                delete_condition: Optional delete condition
                
            Returns:
                Number of rows affected
            """
            from delta.tables import DeltaTable
            
            target = DeltaTable.forName(spark, target_table)
            
            # Build match condition
            match_cond = " AND ".join([f"target.{c} = source.{c}" for c in match_columns])
            
            # Build update mapping
            if update_columns is None:
                update_columns = [c for c in source_df.columns if c not in match_columns]
            update_map = {c: f"source.{c}" for c in update_columns}
            
            # Build merge
            merge = (target.alias("target")
                    .merge(source_df.alias("source"), match_cond))
            
            if delete_condition:
                merge = merge.whenMatchedDelete(condition=delete_condition)
            
            merge = (merge
                    .whenMatchedUpdate(set=update_map)
                    .whenNotMatchedInsertAll())
            
            merge.execute()
            
            return source_df.count()
        
        def read_cdf(self, table_name: str, 
                     start_version: int = None,
                     start_timestamp: str = None) -> Any:
            """
            Read Change Data Feed from a Delta table.
            
            Args:
                table_name: Table name
                start_version: Starting version (optional)
                start_timestamp: Starting timestamp (optional)
                
            Returns:
                DataFrame with CDF data including _change_type column
            """
            reader = (spark.read
                     .format("delta")
                     .option("readChangeFeed", "true"))
            
            if start_version is not None:
                reader = reader.option("startingVersion", start_version)
            elif start_timestamp is not None:
                reader = reader.option("startingTimestamp", start_timestamp)
            else:
                reader = reader.option("startingVersion", 0)
            
            return reader.table(table_name)
        
        def get_table_history(self, table_name: str, limit: int = 10) -> Any:
            """Get Delta table history."""
            from delta.tables import DeltaTable
            return DeltaTable.forName(spark, table_name).history(limit)
        
        def get_table_version(self, table_name: str) -> int:
            """Get current version of a Delta table."""
            history = self.get_table_history(table_name, limit=1)
            return history.collect()[0]["version"]
        
        def insert_into(self, table_name: str, df) -> None:
            """Insert data into a table."""
            df.write.format("delta").mode("append").saveAsTable(table_name)
        
        def update_table(self, table_name: str, 
                         condition: str, 
                         update_values: Dict[str, str]) -> None:
            """Update rows in a table."""
            set_clause = ", ".join([f"{k} = {v}" for k, v in update_values.items()])
            spark.sql(f"UPDATE {table_name} SET {set_clause} WHERE {condition}")
        
        def delete_from(self, table_name: str, condition: str) -> None:
            """Delete rows from a table."""
            spark.sql(f"DELETE FROM {table_name} WHERE {condition}")
        
        def truncate_table(self, table_name: str) -> None:
            """Truncate a table (delete all rows)."""
            spark.sql(f"TRUNCATE TABLE {table_name}")
    
    return TableOperations()


# ============================================================================
# Table Assertion Fixtures
# ============================================================================

@pytest.fixture
def assert_table_equals(spark, assert_dataframe_equal):
    """
    Assert that a table contains expected data.
    
    Usage:
        assert_table_equals("my_table", expected_df)
    """
    def _assert_equals(table_name: str, expected_df, check_order: bool = False):
        actual_df = spark.table(table_name)
        assert_dataframe_equal(expected_df, actual_df, check_order=check_order)
    
    return _assert_equals


@pytest.fixture
def assert_table_row_count(spark):
    """
    Assert that a table has expected row count.
    
    Usage:
        assert_table_row_count("my_table", 100)
    """
    def _assert_count(table_name: str, expected_count: int):
        actual_count = spark.table(table_name).count()
        assert actual_count == expected_count, (
            f"Table {table_name} row count mismatch: "
            f"expected {expected_count}, got {actual_count}"
        )
    
    return _assert_count


@pytest.fixture
def assert_cdf_changes(spark, table_operations):
    """
    Assert expected changes in Change Data Feed.
    
    Usage:
        assert_cdf_changes(
            table_name="my_table",
            expected_inserts=2,
            expected_updates=1,
            expected_deletes=0
        )
    """
    def _assert_changes(table_name: str,
                        start_version: int = 0,
                        expected_inserts: int = None,
                        expected_updates: int = None,
                        expected_deletes: int = None):
        cdf = table_operations.read_cdf(table_name, start_version=start_version)
        
        if expected_inserts is not None:
            actual_inserts = cdf.filter("_change_type = 'insert'").count()
            assert actual_inserts == expected_inserts, (
                f"Insert count mismatch: expected {expected_inserts}, got {actual_inserts}"
            )
        
        if expected_updates is not None:
            # Updates show as update_preimage + update_postimage pairs
            actual_updates = cdf.filter("_change_type = 'update_postimage'").count()
            assert actual_updates == expected_updates, (
                f"Update count mismatch: expected {expected_updates}, got {actual_updates}"
            )
        
        if expected_deletes is not None:
            actual_deletes = cdf.filter("_change_type = 'delete'").count()
            assert actual_deletes == expected_deletes, (
                f"Delete count mismatch: expected {expected_deletes}, got {actual_deletes}"
            )
    
    return _assert_changes

