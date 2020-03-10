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
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
airlines_schema = StructType([ 
    # Variables related to time of year flight takes place
    StructField('YEAR',ShortType(),True),
    StructField('QUARTER',ShortType(),True),
    StructField('MONTH',ShortType(),True),
    StructField('DAY_OF_MONTH',ShortType(),True),
    StructField('DAY_OF_WEEK',ShortType(),True),

    StructField('FL_DATE',DateType(),True), # flight date

    StructField('OP_CARRIER_AIRLINE_ID',ShortType(),True), # Id of the flight's operating carrier

    # Origin Airport Info
    StructField('ORIGIN_AIRPORT_ID',ShortType(),True),
    StructField('ORIGIN',StringType(),True),
    StructField('ORIGIN_CITY_NAME',StringType(),True),
    StructField('ORIGIN_STATE_ABR',StringType(),True),
    StructField('ORIGIN_STATE_FIPS',ShortType(),True), # Numeric State Identifier
    StructField('ORIGIN_STATE_NM',StringType(),True), # full name

    # Destination Airport Info
    StructField('DEST_AIRPORT_ID',IntegerType(),True),
    StructField('DEST_AIRPORT_SEQ_ID',IntegerType(),True), # only destination has squence id: An identification number assigned by US DOT to identify 
                                                           # a unique airport at a given point of time. Airport attributes, such as airport name or 
                                                           # coordinates, may change over time.
    StructField('DEST',StringType(),True),
    StructField('DEST_CITY_NAME',StringType(),True),
    StructField('DEST_STATE_ABR',StringType(),True),
    StructField('DEST_STATE_FIPS',ShortType(),True),
    StructField('DEST_STATE_NM',StringType(),True),

    # Metrics related to departure time & delays
    StructField('CRS_DEP_TIME',StringType(),True), # scheduled departure time; CRS Departure Time (local time: hhmm)
    StructField('DEP_TIME',StringType(),True), # actual time; Actual Departure Time (local time: hhmm) (most of time this is CRS_DEP_TIME + DEP_DALY)
    StructField('DEP_DELAY',IntegerType(),True), # Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. 
    StructField('DEP_DELAY_NEW',IntegerType(),True),
    StructField('DEP_DEL15',IntegerType(),True), # Departure Delay Indicator, 15 Minutes or More (1=Yes)
    StructField('DEP_DELAY_GROUP',IntegerType(),True), # Departure Delay intervals, every (15 minutes from <-15 to >180)
    StructField('DEP_TIME_BLK',StringType(),True), # CRS Departure Time Block, Hourly Intervals

    # Metrics related to arrival time & delays
    StructField('CRS_ARR_TIME',StringType(),True),
    StructField('ARR_TIME',StringType(),True),
    StructField('ARR_DELAY',IntegerType(),True),
    StructField('ARR_DELAY_NEW',IntegerType(),True),
    StructField('ARR_DEL15',IntegerType(),True),
    StructField('ARR_DELAY_GROUP',IntegerType(),True),
    StructField('ARR_TIME_BLK',StringType(),True),

    # Indicators for cancelled & diverted flights (true/false)
    StructField('CANCELLED',BooleanType(),True),
    StructField('DIVERTED',BooleanType(),True),

    # Metrics related to elapsed time of flight
    StructField('CRS_ELAPSED_TIME',IntegerType(),True),
    StructField('ACTUAL_ELAPSED_TIME',IntegerType(),True),

    StructField('FLIGHTS',ShortType(),True), # should be number of flights, but this is always "1"

    # Metrics related to distance of flight
    StructField('DISTANCE',IntegerType(),True),
    StructField('DISTANCE_GROUP',ShortType(),True)
  ])


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*a.parquet")
display(airlines.sample(False, 0.00001))

mini_airlines = airlines.sample(False, 0.00001)

# COMMAND ----------

# Explore whether CRS_DEP_TIME + DEP_DELAY = DEP_TIME
res = mini_airlines.where(airlines.DEP_TIME != (airlines.CRS_DEP_TIME + airlines.DEP_DELAY)) \
             .select('CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', (airlines.CRS_DEP_TIME + airlines.DEP_DELAY).alias('Sum')).collect()
  
res = airlines.select((airlines.CRS_DEP_TIME.isNull()).alias("IsCrsDepTimeNull"), \
                      (airlines.DEP_TIME.isNull()).alias("IsDepTimeNull"), \
                      (airlines.DEP_DELAY.isNull()).alias("IsDepDelayNull"), \
                      (airlines.DEP_TIME != (airlines.CRS_DEP_TIME + airlines.DEP_DELAY)).alias("Expression")) \
              .groupBy('IsCrsDepTimeNull', 'IsDepTimeNull', 'IsDepDelayNull', 'Expression').count().orderBy('Expression')

# Note that when all values are defined, (last two rows), the condition doesn't always hold
display(res)

# COMMAND ----------

# Explore Dep_Delay_New (are all negative Dep_Delays set to 0 and all else the same?)
res = airlines.select((airlines.DEP_DELAY.isNull()).alias("IsDepDelayNull"), \
                      (airlines.DEP_DELAY_NEW.isNull()).alias("IsDepDelayNewNull"), \
                     (((airlines.DEP_DELAY > 0) & (airlines.DEP_DELAY == airlines.DEP_DELAY_NEW)) | \
                      ((airlines.DEP_DELAY <= 0) & (airlines.DEP_DELAY_NEW == 0))).alias("Expression")) \
              .groupBy('IsDepDelayNull', 'IsDepDelayNewNull', 'Expression').count()
  
# Note that the expression is always true for the entire dataset except null entries
display(res)

# COMMAND ----------

# Total number of records in airlines:
print("Number of records = " + str(airlines.count()))

# COMMAND ----------

# Summary statistics for entire dataset
desc = airlines.describe()
display(desc)

# COMMAND ----------

# Evaluate distribution of records across time (how many records for every month of every year?)
res = airlines.groupBy("YEAR", "MONTH").count().orderBy("MONTH", "YEAR")

# Still have certain months with missing data: especially in 2017... (guess we'll have to live with it)
display(res)

# COMMAND ----------

# Evaluate distribution of records across time (how many records for every quarter of every year?)
res = airlines.groupBy("YEAR", "QUARTER").count().orderBy("QUARTER", "YEAR")

# Almost every quarter has data--only 2017 quarter 2 doesn't have data, but have a lot more in Q1 so might be ok?
display(res)

# COMMAND ----------

# General EDA to check for unique values/distribution of values/presence of nulls
varName = 'Origin_City_'
display(airlines.groupBy(varName).count().orderBy(airlines[varName].asc()))

# COMMAND ----------

# count distinct number of values
airlines.select(varName).distinct().count()

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_weather_data")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType
schema = StructType([StructField('STATION', StringType(), True), 
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
                      .parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_weather_data/201*a.parquet")
weather.count()


# COMMAND ----------

display(weather.where('DATE =="DATE"'))

# COMMAND ----------

display(weather.describe())

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

