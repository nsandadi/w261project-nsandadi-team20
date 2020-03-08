# Databricks notebook source
# MAGIC %md
# MAGIC This is an EDA on the 20GB Airlines dataset that Kyle prepped for us on 3/5/2020; Will also look at the weather dataset.

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

# Load the parquet file for the 20GB dataset
airlines = spark.read.option("header", "true").parquet("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet/*")
print("Number of Rows: " + str(airlines.count()))

# COMMAND ----------

airlines.groupBy("DISTANCE_GROUP").count().take(100)

# COMMAND ----------

# Cast columns and save as new parquet file with cast columns 
from pyspark.sql import types 
from pyspark.sql import functions as F

def CastVar(airlines, varName, castType='int'):
  return airlines.withColumn(varName, airlines[varName].cast(castType))
  
# Cast to ints
toIntVars = ['YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
             'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DEL15',
             'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DEL15',
             'CANCELLED', 'DIVERTED',
             'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME',
             'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP']
for var in toIntVars:
  airlines = CastVar(airlines, var, 'int')

# cast to floats
toFloatVars = ['DEP_DELAY', 'DEP_DELAY_NEW',
               'ARR_DELAY', 'ARR_DELAY_NEW']
for var in toFloatVars:
  airlines = CastVar(airlines, var, 'float')

# cast to string (no-op, since already strings)
toStrVars = ['FL_DATE', # Flight date
             'OP_CARRIER_AIRLINE_ID', 
             'ORIGIN_AIRPORT_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 
             'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM',
             'DEP_DELAY_GROUP' 'DEP_TIME_BLK',
             'ARR_DELAY_GROUP', 'ARR_TIME_BLK']

# COMMAND ----------

airlines

# COMMAND ----------

display(airlines.groupBy("Year", "Month").count().orderBy("Month", "Year"))

# COMMAND ----------

