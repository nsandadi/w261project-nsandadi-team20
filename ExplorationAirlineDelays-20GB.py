# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC 2015 - 2019

# COMMAND ----------

# MAGIC %md ### Additioinal sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 1. http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html
# MAGIC 2. https://www.world-airport-codes.com/

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/new_airlines_data")

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/new_airlines_data/2015/")

# COMMAND ----------

airlines = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/data/datasets_final_project/new_airlines_data/2015/285206953_T_ONTIME_REPORTING_0.csv.gz")

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

# This dataset  has a trailing comma which creates an empty column. Let's delete this column.
airlines=airlines.drop('_c41')

# COMMAND ----------

display(airlines)

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_data")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
schema = StructType([StructField('01001099999.csv0100644 0000000 0000000 00012535623 13617404052 011523 0ustar000000000 0000000 "STATION"', StringType(), True), 
                      StructField('DATE', StringType(), True),
                      StructField('SOURCE', StringType(), True),
                      StructField('LATITUDE', StringType(), True),
                      StructField('LONGITUDE', StringType(), True),
                      StructField('ELEVATION', StringType(), True),
                      StructField('NAME', StringType(), True),
                      StructField('REPORT_TYPE', StringType(), True),
                      StructField('CALL_SIGN', StringType(), True),
                      StructField('QUALITY_CONTROL', StringType(), True),
                      StructField('WND', StringType(), True),
                      StructField('CIG', StringType(), True),
                      StructField('VIS', StringType(), True),
                      StructField('TMP', StringType(), True),
                      StructField('DEW', StringType(), True),
                      StructField('SLP', StringType(), True),
                      StructField('AA1', StringType(), True),
                      StructField('AA2', StringType(), True),
                      StructField('AJ1', StringType(), True),
                      StructField('AY1', StringType(), True),
                      StructField('AY2', StringType(), True),
                      StructField('GA1', StringType(), True),
                      StructField('GA2', StringType(), True),
                      StructField('GA3', StringType(), True),
                      StructField('GE1', StringType(), True),
                      StructField('GF1', StringType(), True),
                      StructField('IA1', StringType(), True),
                      StructField('KA1', StringType(), True),
                      StructField('KA2', StringType(), True),
                      StructField('MA1', StringType(), True),
                      StructField('MD1', StringType(), True),
                      StructField('MW1', StringType(), True),
                      StructField('OC1', StringType(), True),
                      StructField('OD1', StringType(), True),
                      StructField('SA1', StringType(), True),
                      StructField('UA1', StringType(), True),
                      StructField('REM', StringType(), True),
                      StructField('EQD', StringType(), True)
                    ])



# COMMAND ----------

weather = spark.read.option("header", "true")\
                    .schema(schema)\
                    .csv("dbfs:/mnt/mids-w261/data/datasets_final_project/new_weather_data/2019.tar.gz")

# COMMAND ----------

weather = weather.withColumnRenamed('01001099999.csv0100644 0000000 0000000 00012535623 13617404052 011523 0ustar000000000 0000000 "STATION"','STATION')

# COMMAND ----------

display(weather)

# COMMAND ----------

# MAGIC %md # Stations

# COMMAND ----------

stations = spark.read.option("header", "true").csv("dbfs:/mnt/mids-w261/data/DEMO8/gsod/stations.csv.gz")

# COMMAND ----------

display(stations)

# COMMAND ----------

from pyspark.sql import functions as f
stations.where(f.col('name').contains('JAN MAYEN NOR NAVY'))

# COMMAND ----------

stations.select('name').distinct().count()

# COMMAND ----------

display(stations.select('name').distinct())

# COMMAND ----------

weather.select('NAME').distinct().count()

# COMMAND ----------

display(weather.select('name').distinct())

# COMMAND ----------

