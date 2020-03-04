# Databricks notebook source
DATAPATH = '/databricks-datasets/airlines'
all_files = dbutils.fs.ls(DATAPATH)
chunks = [c.path for c in all_files if "part" in c.path]
print("Number of parquet = {}".format(len(chunks)))
airlines0 = spark.read.option("header", "true").csv(chunks[0])
airlines_rest = spark.read.option("header", "false").csv(chunks[1:])
airlines = airlines0.union(airlines_rest)

# COMMAND ----------

# Update datatypes for more accurate aggregations
from pyspark.sql import types 

airlines = airlines.withColumn("Year", airlines["Year"].cast('int'))
airlines = airlines.withColumn("Month", airlines["Month"].cast('int'))
airlines = airlines.withColumn("DayOfMonth", airlines["DayOfMonth"].cast('int'))
airlines = airlines.withColumn("DayOfWeek", airlines["DayOfWeek"].cast('int'))
airlines = airlines.withColumn("DepTime", airlines["DepTime"].cast('int'))
airlines = airlines.withColumn("CRSDepTime", airlines["CRSDepTime"].cast('int'))
airlines = airlines.withColumn("ArrTime", airlines["ArrTime"].cast('int'))
airlines = airlines.withColumn("CRSArrTime", airlines["CRSArrTime"].cast('int'))
airlines = airlines.withColumn("ActualElapsedTime", airlines["ActualElapsedTime"].cast('int'))
airlines = airlines.withColumn("CRSElapsedTime", airlines["CRSElapsedTime"].cast('int'))
airlines = airlines.withColumn("AirTime", airlines["AirTime"].cast('int'))
airlines = airlines.withColumn("ArrDelay", airlines["ArrDelay"].cast('int'))
airlines = airlines.withColumn("DepDelay", airlines["DepDelay"].cast('int'))
airlines = airlines.withColumn("Distance", airlines["Distance"].cast('int'))
airlines = airlines.withColumn("ActualElapsedTime", airlines["ActualElapsedTime"].cast('int'))

airlines = airlines.withColumn("Cancelled", airlines["Cancelled"].cast('int'))
airlines = airlines.withColumn("Diverted", airlines["Diverted"].cast('int'))
airlines = airlines.withColumn("IsArrDelayed", airlines["IsArrDelayed"].cast('int'))
airlines = airlines.withColumn("IsDepDelayed", airlines["IsDepDelayed"].cast('int'))

# COMMAND ----------

print(airlines.columns)
display(airlines.take(1))

# COMMAND ----------



# COMMAND ----------

# Display number of unique flights by year
display(airlines.groupBy("Year").count())

# COMMAND ----------

# Determine distribution of Departure Times
display(airlines.groupBy("DepTime").count())


# COMMAND ----------

airlines.groupBy()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

