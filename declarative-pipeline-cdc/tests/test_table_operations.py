"""
Unit tests for CDC pipeline using actual Delta tables and schemas.

This module tests the declarative pipeline with real table creation,
including:
- Schema and database creation/cleanup
- Delta table operations (MERGE, INSERT, UPDATE, DELETE)
- Change Data Feed (CDF) functionality
- SCD Type 2 history tracking with actual tables
- Multi-table pipeline scenarios

These tests require a Spark session with Delta Lake support.
They create actual tables in a test database and clean up after.

Note: These tests are slower than DataFrame-only tests but provide
higher confidence that the pipeline will work in production.
"""

import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp, expr
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType
)
from datetime import datetime, timedelta
from typing import List, Dict


# ============================================================================
# Test Class: Database and Schema Setup
# ============================================================================

class TestDatabaseSetup:
    """
    Tests for database and schema creation.
    
    Why: Validates that test infrastructure properly creates and cleans up
    databases/schemas for isolated testing.
    """
    
    def test_database_is_created(self, spark, test_database):
        """
        Test that the test database is created successfully.
        
        Why: Test database must exist before any table operations.
        """
        # Act: Check if database exists
        databases = spark.sql("SHOW DATABASES").collect()
        database_names = [db["databaseName"] for db in databases]
        
        # Assert
        assert test_database in database_names, (
            f"Test database '{test_database}' not found in {database_names}"
        )
    
    def test_database_is_active(self, spark, test_database):
        """
        Test that the test database is set as the active database.
        
        Why: Tables should be created in the correct database context.
        """
        # Act
        current_db = spark.sql("SELECT current_database()").collect()[0][0]
        
        # Assert
        assert current_db == test_database
    
    def test_schema_manager_creates_schema(self, spark, test_schema_manager):
        """
        Test that schema manager can create schemas.
        
        Why: Multi-schema pipelines need isolated schemas.
        """
        # Arrange
        schema_name = "test_bronze_layer"
        
        # Act
        created_name = test_schema_manager.create_schema(schema_name)
        
        # Assert
        schemas = spark.sql("SHOW DATABASES").collect()
        schema_names = [s["databaseName"] for s in schemas]
        assert schema_name in schema_names
    
    def test_schema_manager_tracks_created_schemas(self, spark, test_schema_manager):
        """
        Test that schema manager tracks created schemas for cleanup.
        
        Why: Ensures proper cleanup after tests.
        """
        # Arrange
        schema1 = test_schema_manager.create_schema("bronze_test")
        schema2 = test_schema_manager.create_schema("silver_test")
        
        # Assert
        assert len(test_schema_manager.schemas_created) == 2
        assert "bronze_test" in test_schema_manager.schemas_created
        assert "silver_test" in test_schema_manager.schemas_created


# ============================================================================
# Test Class: Managed Table Creation
# ============================================================================

class TestManagedTableCreation:
    """
    Tests for creating managed Delta tables.
    
    Why: Validates table creation with various configurations
    including CDF enablement and partitioning.
    """
    
    def test_create_basic_table(self, spark, create_managed_table,
                                customers_silver_schema, base_timestamp):
        """
        Test creating a basic managed Delta table.
        
        Why: Core functionality for all table-based tests.
        """
        # Arrange
        data = [(1, "Alice", "123 Main St", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        
        # Act
        table_name = create_managed_table("test_customers", df)
        
        # Assert: Table exists and has data
        result = spark.table(table_name)
        assert result.count() == 1
    
    def test_create_table_with_cdf_enabled(self, spark, create_managed_table,
                                           customers_silver_schema):
        """
        Test creating a table with Change Data Feed enabled.
        
        Why: CDF is required for tracking changes in CDC pipelines.
        """
        # Arrange
        data = [(1, "Alice", "123 Main St", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        
        # Act
        table_name = create_managed_table("customers_with_cdf", df, enable_cdf=True)
        
        # Assert: Check table properties
        props = spark.sql(f"SHOW TBLPROPERTIES {table_name}").collect()
        prop_dict = {row["key"]: row["value"] for row in props}
        assert prop_dict.get("delta.enableChangeDataFeed") == "true"
    
    def test_create_partitioned_table(self, spark, create_managed_table,
                                      customers_cdc_schema, base_timestamp):
        """
        Test creating a partitioned Delta table.
        
        Why: Partitioning improves query performance for large tables.
        """
        # Arrange
        data = [
            (1, "Alice", "Addr1", "a@email.com", "APPEND", base_timestamp, None),
            (2, "Bob", "Addr2", "b@email.com", "UPDATE", base_timestamp, None),
        ]
        df = spark.createDataFrame(data, customers_cdc_schema)
        
        # Act
        table_name = create_managed_table(
            "partitioned_cdc", df, 
            partition_by=["operation"]
        )
        
        # Assert: Check partitioning
        desc = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0]
        assert "operation" in desc["partitionColumns"]
    
    def test_create_table_with_custom_properties(self, spark, create_managed_table,
                                                  customers_silver_schema):
        """
        Test creating a table with custom table properties.
        
        Why: Allows setting custom metadata and configurations.
        """
        # Arrange
        data = [(1, "Alice", "123 Main St", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        
        # Act
        table_name = create_managed_table(
            "custom_props_table", df,
            table_properties={
                "pipeline.name": "test_cdc_pipeline",
                "pipeline.layer": "silver"
            }
        )
        
        # Assert
        props = spark.sql(f"SHOW TBLPROPERTIES {table_name}").collect()
        prop_dict = {row["key"]: row["value"] for row in props}
        assert prop_dict.get("pipeline.name") == "test_cdc_pipeline"
        assert prop_dict.get("pipeline.layer") == "silver"


# ============================================================================
# Test Class: CDC Pipeline Tables
# ============================================================================

class TestCDCPipelineTables:
    """
    Tests for the pre-configured CDC pipeline tables fixture.
    
    Why: Validates the complete pipeline table structure is correctly set up.
    """
    
    def test_bronze_table_created(self, spark, cdc_pipeline_tables):
        """
        Test that bronze layer CDC table is created.
        
        Why: Bronze layer is the entry point for CDC data.
        """
        # Act
        result = spark.table(cdc_pipeline_tables["bronze"])
        
        # Assert
        assert result.count() == 2
        assert "operation" in result.columns
        assert "_rescued_data" in result.columns
    
    def test_silver_table_has_cdf_enabled(self, spark, cdc_pipeline_tables):
        """
        Test that silver layer table has CDF enabled.
        
        Why: CDF is required for downstream Gold layer processing.
        """
        # Act
        props = spark.sql(f"SHOW TBLPROPERTIES {cdc_pipeline_tables['silver']}").collect()
        prop_dict = {row["key"]: row["value"] for row in props}
        
        # Assert
        assert prop_dict.get("delta.enableChangeDataFeed") == "true"
    
    def test_scd2_table_has_history_columns(self, spark, cdc_pipeline_tables):
        """
        Test that SCD2 table has __START_AT and __END_AT columns.
        
        Why: These columns track record validity periods.
        """
        # Act
        result = spark.table(cdc_pipeline_tables["scd2"])
        
        # Assert
        assert "__START_AT" in result.columns
        assert "__END_AT" in result.columns
    
    def test_empty_pipeline_tables_have_correct_schema(self, spark,
                                                        empty_cdc_pipeline_tables,
                                                        customers_cdc_schema,
                                                        customers_silver_schema):
        """
        Test that empty pipeline tables have correct schemas.
        
        Why: Schema must be correct even before data is loaded.
        """
        # Act
        bronze_df = spark.table(empty_cdc_pipeline_tables["bronze"])
        silver_df = spark.table(empty_cdc_pipeline_tables["silver"])
        
        # Assert: row counts are 0
        assert bronze_df.count() == 0
        assert silver_df.count() == 0
        
        # Assert: schemas match
        assert set(bronze_df.columns) == set(customers_cdc_schema.fieldNames())
        assert set(silver_df.columns) == set(customers_silver_schema.fieldNames())


# ============================================================================
# Test Class: Table Operations
# ============================================================================

class TestTableOperations:
    """
    Tests for table CRUD operations using the table_operations fixture.
    
    Why: Validates core operations needed for CDC processing.
    """
    
    def test_insert_into_table(self, spark, create_managed_table,
                               table_operations, customers_silver_schema):
        """
        Test inserting data into a table.
        
        Why: INSERT is used for initial data loads.
        """
        # Arrange
        initial_data = [(1, "Alice", "Addr1", "alice@email.com")]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_managed_table("insert_test", initial_df)
        
        new_data = [(2, "Bob", "Addr2", "bob@email.com")]
        new_df = spark.createDataFrame(new_data, customers_silver_schema)
        
        # Act
        table_operations.insert_into(table_name, new_df)
        
        # Assert
        result = spark.table(table_name)
        assert result.count() == 2
    
    def test_update_table(self, spark, create_managed_table,
                          table_operations, customers_silver_schema):
        """
        Test updating rows in a table.
        
        Why: UPDATE is needed for CDC UPDATE operations.
        """
        # Arrange
        data = [(1, "Alice", "Old Address", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("update_test", df)
        
        # Act
        table_operations.update_table(
            table_name,
            condition="id = 1",
            update_values={"address": "'New Address'"}
        )
        
        # Assert
        result = spark.table(table_name).filter("id = 1").collect()[0]
        assert result["address"] == "New Address"
    
    def test_delete_from_table(self, spark, create_managed_table,
                               table_operations, customers_silver_schema):
        """
        Test deleting rows from a table.
        
        Why: DELETE is needed for CDC DELETE operations.
        """
        # Arrange
        data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
        ]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("delete_test", df)
        
        # Act
        table_operations.delete_from(table_name, condition="id = 2")
        
        # Assert
        result = spark.table(table_name)
        assert result.count() == 1
        assert result.filter("id = 2").count() == 0
    
    def test_merge_into_table(self, spark, create_managed_table,
                              table_operations, customers_silver_schema):
        """
        Test MERGE INTO operation (upsert).
        
        Why: MERGE is the core operation for APPLY CHANGES.
        """
        # Arrange: Initial data
        initial_data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
        ]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_managed_table("merge_test", initial_df)
        
        # Source data: update Alice, insert Charlie
        source_data = [
            (1, "Alice Updated", "New Addr1", "alice.new@email.com"),
            (3, "Charlie", "Addr3", "charlie@email.com"),
        ]
        source_df = spark.createDataFrame(source_data, customers_silver_schema)
        
        # Act
        table_operations.merge_into(
            target_table=table_name,
            source_df=source_df,
            match_columns=["id"],
            update_columns=["name", "address", "email"]
        )
        
        # Assert
        result = spark.table(table_name)
        assert result.count() == 3  # Bob + Alice (updated) + Charlie (new)
        
        alice = result.filter("id = 1").collect()[0]
        assert alice["name"] == "Alice Updated"
        
        charlie = result.filter("id = 3").collect()[0]
        assert charlie["name"] == "Charlie"
    
    def test_truncate_table(self, spark, create_managed_table,
                            table_operations, customers_silver_schema):
        """
        Test truncating a table.
        
        Why: Needed for reset/rebuild scenarios.
        """
        # Arrange
        data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
        ]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("truncate_test", df)
        
        # Act
        table_operations.truncate_table(table_name)
        
        # Assert
        result = spark.table(table_name)
        assert result.count() == 0


# ============================================================================
# Test Class: Change Data Feed (CDF)
# ============================================================================

class TestChangeDataFeed:
    """
    Tests for Change Data Feed functionality.
    
    Why: CDF is essential for tracking changes and building Gold layer.
    """
    
    def test_cdf_captures_inserts(self, spark, create_delta_table_with_cdf,
                                  table_operations, customers_silver_schema):
        """
        Test that CDF captures INSERT operations.
        
        Why: New records should appear as 'insert' in CDF.
        """
        # Arrange: Create table with initial data
        initial_data = [(1, "Alice", "Addr1", "alice@email.com")]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_delta_table_with_cdf("cdf_insert_test", initial_df)
        
        # Get initial version
        initial_version = table_operations.get_table_version(table_name)
        
        # Insert new row
        new_data = [(2, "Bob", "Addr2", "bob@email.com")]
        new_df = spark.createDataFrame(new_data, customers_silver_schema)
        table_operations.insert_into(table_name, new_df)
        
        # Act: Read CDF
        cdf = table_operations.read_cdf(table_name, start_version=initial_version + 1)
        
        # Assert
        inserts = cdf.filter("_change_type = 'insert'")
        assert inserts.count() == 1
        assert inserts.collect()[0]["name"] == "Bob"
    
    def test_cdf_captures_updates(self, spark, create_delta_table_with_cdf,
                                  table_operations, customers_silver_schema):
        """
        Test that CDF captures UPDATE operations.
        
        Why: Updates show as preimage/postimage pairs in CDF.
        """
        # Arrange
        initial_data = [(1, "Alice", "Old Address", "alice@email.com")]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_delta_table_with_cdf("cdf_update_test", initial_df)
        
        initial_version = table_operations.get_table_version(table_name)
        
        # Update row
        table_operations.update_table(
            table_name,
            condition="id = 1",
            update_values={"address": "'New Address'"}
        )
        
        # Act
        cdf = table_operations.read_cdf(table_name, start_version=initial_version + 1)
        
        # Assert: Should have preimage and postimage
        preimage = cdf.filter("_change_type = 'update_preimage'")
        postimage = cdf.filter("_change_type = 'update_postimage'")
        
        assert preimage.count() == 1
        assert postimage.count() == 1
        assert preimage.collect()[0]["address"] == "Old Address"
        assert postimage.collect()[0]["address"] == "New Address"
    
    def test_cdf_captures_deletes(self, spark, create_delta_table_with_cdf,
                                  table_operations, customers_silver_schema):
        """
        Test that CDF captures DELETE operations.
        
        Why: Deleted records appear as 'delete' in CDF.
        """
        # Arrange
        initial_data = [
            (1, "Alice", "Addr1", "alice@email.com"),
            (2, "Bob", "Addr2", "bob@email.com"),
        ]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_delta_table_with_cdf("cdf_delete_test", initial_df)
        
        initial_version = table_operations.get_table_version(table_name)
        
        # Delete row
        table_operations.delete_from(table_name, condition="id = 2")
        
        # Act
        cdf = table_operations.read_cdf(table_name, start_version=initial_version + 1)
        
        # Assert
        deletes = cdf.filter("_change_type = 'delete'")
        assert deletes.count() == 1
        assert deletes.collect()[0]["name"] == "Bob"


# ============================================================================
# Test Class: Multi-Table Pipeline
# ============================================================================

class TestMultiTablePipeline:
    """
    Tests for multi-table CDC pipeline scenarios.
    
    Why: Validates the dynamic multi-table processing pattern.
    """
    
    def test_all_tables_created(self, spark, multi_table_pipeline_setup):
        """
        Test that all tables in multi-table setup are created.
        
        Why: Pipeline dynamically creates tables for each folder.
        """
        # Assert: All three table pairs exist
        expected_tables = ["customers", "orders", "products"]
        
        for table_name in expected_tables:
            assert table_name in multi_table_pipeline_setup
            assert "bronze" in multi_table_pipeline_setup[table_name]
            assert "silver" in multi_table_pipeline_setup[table_name]
    
    def test_bronze_tables_have_data(self, spark, multi_table_pipeline_setup):
        """
        Test that bronze tables are populated with test data.
        
        Why: Bronze tables receive raw CDC data.
        """
        for table_name, tables in multi_table_pipeline_setup.items():
            bronze_df = spark.table(tables["bronze"])
            assert bronze_df.count() == 2, (
                f"Bronze table {table_name} should have 2 rows"
            )
    
    def test_silver_tables_are_empty_initially(self, spark, multi_table_pipeline_setup):
        """
        Test that silver tables start empty.
        
        Why: Silver tables are populated by APPLY CHANGES from bronze.
        """
        for table_name, tables in multi_table_pipeline_setup.items():
            silver_df = spark.table(tables["silver"])
            assert silver_df.count() == 0, (
                f"Silver table {table_name} should be empty initially"
            )
    
    def test_silver_tables_have_cdf_enabled(self, spark, multi_table_pipeline_setup):
        """
        Test that all silver tables have CDF enabled.
        
        Why: CDF is needed for downstream processing.
        """
        for table_name, tables in multi_table_pipeline_setup.items():
            props = spark.sql(f"SHOW TBLPROPERTIES {tables['silver']}").collect()
            prop_dict = {row["key"]: row["value"] for row in props}
            assert prop_dict.get("delta.enableChangeDataFeed") == "true", (
                f"Silver table {table_name} should have CDF enabled"
            )
    
    def test_process_multi_table_cdc(self, spark, multi_table_pipeline_setup,
                                     table_operations, generic_cdc_schema,
                                     base_timestamp):
        """
        Test processing CDC data for multiple tables.
        
        Why: Validates the multi-table processing workflow.
        """
        silver_schema = StructType([
            StructField("id", LongType(), False),
            StructField("data", StringType(), True),
        ])
        
        for table_name, tables in multi_table_pipeline_setup.items():
            # Read bronze data
            bronze_df = spark.table(tables["bronze"])
            
            # Filter for valid APPEND operations (simulating expectations)
            valid_data = bronze_df.filter(
                (col("operation") == "APPEND") & 
                (col("id").isNotNull())
            )
            
            # Extract columns for silver (excluding CDC metadata)
            silver_data = valid_data.select("id", "data")
            
            # Write to silver table
            silver_data.write.format("delta").mode("append").saveAsTable(tables["silver"])
        
        # Assert: All silver tables now have data
        for table_name, tables in multi_table_pipeline_setup.items():
            silver_df = spark.table(tables["silver"])
            assert silver_df.count() == 2


# ============================================================================
# Test Class: Table Assertions
# ============================================================================

class TestTableAssertions:
    """
    Tests for table assertion helper fixtures.
    
    Why: Validates the assertion utilities work correctly.
    """
    
    def test_assert_table_row_count_passes(self, spark, create_managed_table,
                                           assert_table_row_count,
                                           customers_silver_schema):
        """
        Test that row count assertion passes for correct count.
        """
        # Arrange
        data = [(1, "Alice", "Addr", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("count_test", df)
        
        # Act & Assert: Should not raise
        assert_table_row_count(table_name, 1)
    
    def test_assert_table_row_count_fails(self, spark, create_managed_table,
                                          assert_table_row_count,
                                          customers_silver_schema):
        """
        Test that row count assertion fails for incorrect count.
        """
        # Arrange
        data = [(1, "Alice", "Addr", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("count_fail_test", df)
        
        # Act & Assert
        with pytest.raises(AssertionError) as exc_info:
            assert_table_row_count(table_name, 5)
        
        assert "expected 5, got 1" in str(exc_info.value)
    
    def test_assert_table_equals_passes(self, spark, create_managed_table,
                                        assert_table_equals,
                                        customers_silver_schema):
        """
        Test that table equality assertion passes for matching data.
        """
        # Arrange
        data = [(1, "Alice", "Addr", "alice@email.com")]
        df = spark.createDataFrame(data, customers_silver_schema)
        table_name = create_managed_table("equals_test", df)
        
        expected_df = spark.createDataFrame(data, customers_silver_schema)
        
        # Act & Assert: Should not raise
        assert_table_equals(table_name, expected_df)


# ============================================================================
# Test Class: End-to-End Table Pipeline
# ============================================================================

class TestEndToEndTablePipeline:
    """
    End-to-end integration tests using actual tables.
    
    Why: Validates complete pipeline flow with real Delta tables.
    """
    
    def test_full_cdc_lifecycle_with_tables(self, spark, test_database,
                                            customers_cdc_schema,
                                            customers_silver_schema,
                                            table_operations,
                                            base_timestamp):
        """
        Test complete CDC lifecycle: create tables, process CDC, verify results.
        
        This test simulates a real pipeline execution:
        1. Create bronze and silver tables
        2. Insert CDC data into bronze
        3. Process CDC to populate silver
        4. Verify silver table state
        5. Apply updates and deletes
        6. Verify final state
        """
        # Step 1: Create bronze table
        bronze_name = f"{test_database}.e2e_customers_cdc"
        bronze_df = spark.createDataFrame([], customers_cdc_schema)
        bronze_df.write.format("delta").mode("overwrite").saveAsTable(bronze_name)
        
        # Step 2: Create silver table with CDF
        silver_name = f"{test_database}.e2e_customers"
        silver_df = spark.createDataFrame([], customers_silver_schema)
        silver_df.write.format("delta").mode("overwrite").saveAsTable(silver_name)
        spark.sql(f"ALTER TABLE {silver_name} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
        
        try:
            # Step 3: Insert initial CDC data (APPEND operations)
            initial_cdc = spark.createDataFrame([
                (1, "Alice", "Addr1", "alice@email.com", "APPEND", base_timestamp, None),
                (2, "Bob", "Addr2", "bob@email.com", "APPEND", base_timestamp, None),
                (3, "Charlie", "Addr3", "charlie@email.com", "APPEND", base_timestamp, None),
            ], customers_cdc_schema)
            initial_cdc.write.format("delta").mode("append").saveAsTable(bronze_name)
            
            # Step 4: Process CDC - filter valid records and merge to silver
            bronze_data = spark.table(bronze_name)
            valid_appends = (bronze_data
                           .filter(col("operation") == "APPEND")
                           .filter(col("id").isNotNull())
                           .select("id", "name", "address", "email"))
            
            table_operations.merge_into(
                target_table=silver_name,
                source_df=valid_appends,
                match_columns=["id"],
                update_columns=["name", "address", "email"]
            )
            
            # Verify: 3 customers in silver
            assert spark.table(silver_name).count() == 3
            
            # Step 5: Insert UPDATE CDC
            update_ts = base_timestamp + timedelta(hours=1)
            update_cdc = spark.createDataFrame([
                (1, "Alice Updated", "New Addr1", "alice.new@email.com", "UPDATE", update_ts, None),
            ], customers_cdc_schema)
            update_cdc.write.format("delta").mode("append").saveAsTable(bronze_name)
            
            # Process update
            update_data = (spark.table(bronze_name)
                          .filter(col("operation") == "UPDATE")
                          .filter(col("operation_date") == update_ts)
                          .select("id", "name", "address", "email"))
            
            table_operations.merge_into(
                target_table=silver_name,
                source_df=update_data,
                match_columns=["id"],
                update_columns=["name", "address", "email"]
            )
            
            # Verify: Alice updated
            alice = spark.table(silver_name).filter("id = 1").collect()[0]
            assert alice["name"] == "Alice Updated"
            
            # Step 6: Insert DELETE CDC
            delete_ts = base_timestamp + timedelta(hours=2)
            delete_cdc = spark.createDataFrame([
                (2, None, None, None, "DELETE", delete_ts, None),
            ], customers_cdc_schema)
            delete_cdc.write.format("delta").mode("append").saveAsTable(bronze_name)
            
            # Process delete
            delete_ids = (spark.table(bronze_name)
                         .filter(col("operation") == "DELETE")
                         .filter(col("operation_date") == delete_ts)
                         .select("id")
                         .collect())
            
            for row in delete_ids:
                table_operations.delete_from(silver_name, f"id = {row['id']}")
            
            # Verify: Bob deleted, only 2 customers remain
            final_silver = spark.table(silver_name)
            assert final_silver.count() == 2
            assert final_silver.filter("id = 2").count() == 0
            
            # Step 7: Verify CDF captured all changes
            cdf = table_operations.read_cdf(silver_name, start_version=0)
            
            inserts = cdf.filter("_change_type = 'insert'").count()
            updates = cdf.filter("_change_type = 'update_postimage'").count()
            deletes = cdf.filter("_change_type = 'delete'").count()
            
            assert inserts == 3  # Initial 3 customers
            assert updates == 1  # Alice update
            assert deletes == 1  # Bob delete
            
        finally:
            # Cleanup
            spark.sql(f"DROP TABLE IF EXISTS {bronze_name}")
            spark.sql(f"DROP TABLE IF EXISTS {silver_name}")
    
    @pytest.mark.parametrize("num_customers,num_updates,num_deletes", [
        (10, 3, 2),
        (50, 10, 5),
        (100, 20, 10),
    ])
    def test_scalable_cdc_processing(self, spark, create_managed_table,
                                     create_delta_table_with_cdf,
                                     table_operations,
                                     customers_cdc_schema,
                                     customers_silver_schema,
                                     base_timestamp,
                                     num_customers, num_updates, num_deletes):
        """
        Test CDC processing at various scales.
        
        Why: Validates that table operations scale correctly.
        """
        # Create initial data
        initial_data = [
            (i, f"Customer {i}", f"Address {i}", f"customer{i}@email.com")
            for i in range(num_customers)
        ]
        initial_df = spark.createDataFrame(initial_data, customers_silver_schema)
        table_name = create_delta_table_with_cdf(
            f"scale_test_{num_customers}", initial_df
        )
        
        initial_version = table_operations.get_table_version(table_name)
        
        # Apply updates
        for i in range(num_updates):
            table_operations.update_table(
                table_name,
                condition=f"id = {i}",
                update_values={"name": f"'Updated Customer {i}'"}
            )
        
        # Apply deletes (from end to avoid affecting update indices)
        for i in range(num_deletes):
            delete_id = num_customers - 1 - i
            table_operations.delete_from(table_name, f"id = {delete_id}")
        
        # Verify final state
        final_df = spark.table(table_name)
        expected_count = num_customers - num_deletes
        assert final_df.count() == expected_count
        
        # Verify CDF
        cdf = table_operations.read_cdf(table_name, start_version=initial_version + 1)
        assert cdf.filter("_change_type = 'update_postimage'").count() == num_updates
        assert cdf.filter("_change_type = 'delete'").count() == num_deletes

