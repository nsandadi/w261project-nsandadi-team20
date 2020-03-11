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

  
    StructField('OP_UNIQUE_CARRIER',ShortType(),True), # Id of the flight's operating carrier

  
    # Origin Airport Info
    StructField('ORIGIN_AIRPORT_ID',ShortType(),True),
    StructField('ORIGIN_AIRPORT_SEQ_ID',IntegerType(),True), # An identification number assigned by US DOT to identify a unique airport at a given point of time
                                                             # Airport attributes, such as airport name or coordinates, may change over time.
    StructField('ORIGIN_CITY_MARKET_ID', StringType(), True), # City Market ID is an identification number assigned by US DOT to identify a city market. 
                                                              # Use this field to consolidate airports serving the same city market.
    StructField('ORIGIN',StringType(),True),
    StructField('ORIGIN_CITY_NAME',StringType(),True),
    StructField('ORIGIN_STATE_ABR',StringType(),True),
    StructField('ORIGIN_STATE_FIPS',ShortType(),True), # Numeric State Identifier
    StructField('ORIGIN_STATE_NM',StringType(),True), # full name
    StructField('ORIGIN_WAC', StringType(), True), # Origin Airport, World Area Code

  
    # Destination Airport Info
    StructField('DEST_AIRPORT_ID',IntegerType(),True),
    StructField('DEST_AIRPORT_SEQ_ID',IntegerType(),True), # An identification number assigned by US DOT to identify a unique airport at a given point of time.
                                                           # Airport attributes, such as airport name or coordinates, may change over time.
    StructField('DEST_CITY_MARKET_ID',StringType(),True), # City Market ID is an identification number assigned by US DOT to identify a city market.
                                                          # Use this field to consolidate airports serving the same city market.
    StructField('DEST',StringType(),True),
    StructField('DEST_CITY_NAME',StringType(),True),
    StructField('DEST_STATE_ABR',StringType(),True),
    StructField('DEST_STATE_FIPS',ShortType(),True),
    StructField('DEST_STATE_NM',StringType(),True),
    StructField('DEST_WAC', StringType(), True), # Destination Airport, World Area Code
  
    # Metrics related to departure time & delays
    StructField('CRS_DEP_TIME',StringType(),True), # scheduled departure time; CRS Departure Time (local time: hhmm)
    StructField('DEP_TIME',StringType(),True), # actual time; Actual Departure Time (local time: hhmm) (most of time this is CRS_DEP_TIME + DEP_DALY)
    StructField('DEP_DELAY',IntegerType(),True), # Difference in minutes between scheduled and actual departure time. Early departures show negative numbers. 
    StructField('DEP_DELAY_NEW',IntegerType(),True),
    StructField('DEP_DEL15',IntegerType(),True), # Departure Delay Indicator, 15 Minutes or More (1=Yes)
    StructField('DEP_DELAY_GROUP',IntegerType(),True), # Departure Delay intervals, every (15 minutes from <-15 to >180)
    StructField('DEP_TIME_BLK',StringType(),True), # CRS Departure Time Block, Hourly Intervals

    
    # Taxi & Wheels Info
    StructField('TAXI_OUT', DoubleType(), True), # Taxi Out Time, in Minutes
    StructField('WHEELS_OFF', IntegerType(), True), # Wheels Off Time (local time: hhmm)
    StructField('WHEELS_ON', IntegerType(), True), # Wheels On Time (local time: hhmm)
    StructField('TAXI_IN', DoubleType(), True), # Taxi In Time, in Minutes
  
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
    StructField('AIR_TIME', DoubleType(), True),
  
  
    StructField('FLIGHTS',ShortType(),True), # should be number of flights, but this is always "1"

  
    # Metrics related to distance of flight
    StructField('DISTANCE',IntegerType(),True),
    StructField('DISTANCE_GROUP',ShortType(),True)
  
    
    # Delay groups
    StructField('CARRIER_DELAY', DoubleType(), True), # Carrier Delay, in Minutes
    StructField('WEATHER_DELAY', DoubleType(), True), # Weather Delay, in Minutes
    StructField('NAS_DELAY', DoubleType(), True), # National Air System Delay, in Minutes
    StructField('SECURITY_DELAY', DoubleType(), True), # Security Delay, in Minutes
    StructField('LATE_AIRCRAFT_DELAY', DoubleType(), True) # Late Aircraft Delay, in Minutes	
  ])


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")
display(airlines.sample(False, 0.00001))

mini_airlines = airlines.sample(False, 0.00001)

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

# Save a few copies of this datset (just in case....)
# airlines.write.format("parquet").save("/dbfs/user/team20/airlines-backup3-3-10.parquet")

# COMMAND ----------

# backup files at our disposal -- all the same data as what Luis shared Tuesday 8:36PM PST
display(dbutils.fs.ls("dbfs/user/team20"))

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

# General EDA to check for unique values/distribution of values/presence of nulls
varName = 'late_aircraft_delay'
#display(airlines.groupBy(varName).count().orderBy(airlines[varName].asc()))
print("Number of distinct values: " + str(airlines.select(varName).distinct().count()))
print("          Number of nulls: " + str(airlines.filter(airlines[varName].isNull()).count()))

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
varName = 'Distance_Group'
display(airlines.groupBy(varName).count().orderBy(airlines[varName].asc()))

# COMMAND ----------

# count distinct number of values
airlines.select(varName).distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA Summary
# MAGIC ##### Time-of-year data
# MAGIC * `Year` - ranges 2015-2019
# MAGIC * `Quarter` -- data well represented across quarters & years
# MAGIC * `Month` -- data well represented across months & years
# MAGIC * `Day_of_Month` - ranges 1-31, fewer records on 31st
# MAGIC * `Day_of_week` - ranges 1-7
# MAGIC * `FL_Date` - flight data (as a string)
# MAGIC 
# MAGIC ##### Airline info
# MAGIC * `Op_Unique_Carrier` - airline Id, total of 19 distinct airlines
# MAGIC 
# MAGIC ##### Origin Airport Info
# MAGIC * `Origin_Airport_Id` - 371 distinct numeric values
# MAGIC * `Origin_Airport_Seq_Id` - 688 distinct numeric values
# MAGIC * `Origin_City_Market_Id` - 344 distinct values
# MAGIC * `Origin` - 371 distinct alphanumeric values (e.g. "SEA")
# MAGIC * `Origin_city_name` - 362 distinct values (e.g. "Seattle, WA") ----- seems odd 362 < 371...
# MAGIC * `Origin_state_abr` - 53 distinct values (e.g. "WA")
# MAGIC * `Origin_state_FIPS` - 53 distinct values (ranges 1-78)
# MAGIC * `Origin_state_nm` - 53 distinct values (e.g. "Washington")
# MAGIC * `Origin_WAC` - 53 distinct values (ranges 1-93)
# MAGIC 
# MAGIC ##### Destination Airport Info
# MAGIC * `Dest_Airport_id` - 369 distinct values (numeric)
# MAGIC * `Dest_Airort_Seq_id` - 686 distinct value (numeric)
# MAGIC * `Dest_City_Market_Id` - 343 distinct values
# MAGIC * `Dest` - 369 distinct values (e.g. "SEA") 
# MAGIC * `Dest_City_Name` - 361 distinct values (e.g. "Seattle, WA") ------ seems odd 361 < 369...
# MAGIC * `Dest_State_Abr` - 53 distinct values (e.g. "WA")
# MAGIC * `Dest_State_FIPS` - 53 distinct values (ranges 1-78)
# MAGIC * `Dest_State_Nm` - 53 distinct values (e.g. "Washington")
# MAGIC * `Dest_WAC` - 53 distinct values (ranges 1-93)
# MAGIC 
# MAGIC ##### Departure-Related Info
# MAGIC * `CRS_Dep_Time` - ranges 0001 - 2359, 1433 values
# MAGIC * `Dep_Time` - 472,320 nulls, ranges 0001 - 2400, 1441 values
# MAGIC * `Dep_Delay` - 477,296 nulls, ranges -234 to 2755, 1749 values
# MAGIC * `Dep_Delay_New` - 477,296 nulls, ranges 0 to 2755, 1655 values (zero-ed out all negatives)
# MAGIC * `Dep_Del15` - 477,296 nulls, 25576004 with "0" value, 5693541 with "1" value
# MAGIC * `Dep_Delay_Group` - 477,296 nulls, ranges -2 to 12, 16 values
# MAGIC * `Dep_Time_Blk` - hour time blocks, no nulls, 19 values (e.g. 2300-2359); aggregated variable
# MAGIC 
# MAGIC ##### Taxi & Wheels Info
# MAGIC * `Taxi_out` - 486,417 nulls, ranges from 0 to 227, 195 distinct values
# MAGIC * `Wheels_Off` - 486,412 nulls, ranges from 1 to 2400, 1441 distinct values
# MAGIC * `Wheels_On` - 501,924 nulls, ranges from 1 to 2400, 1441 distinct values
# MAGIC * `Taxi_in` - 501,924 nulls, ranges from 0 to 414, 291 distinct values
# MAGIC 
# MAGIC ##### Arrival-Related Info
# MAGIC * `CRS_Arr_Time` - ranges 0001 - 2400, 1440 values
# MAGIC * `Arr_Time` - 501,922 nulls, ranges 0001-2400, 1441 values
# MAGIC * `Arr_Delay` - 570,640 nulls, ranges -1238 to 2695, 1761 values
# MAGIC * `Arr_Delay_New` - 570,640 nulls, ranges 0 to 2695, 1641 values
# MAGIC * `Arr_Del15` - 570,640 nulls, 25377086 with "0" value, 5799115 with "1" value
# MAGIC * `Arr_Delay_Group` - 570,640 nulls, ranges -2 to 12, 16 values, seems to follow a normal distribution
# MAGIC * `Arr_Time_Blk` - hour time blocks, no nulls, 19 values (e.g. 2300-2359); aggregated variable`
# MAGIC 
# MAGIC ##### Cancelled/Diverted Indicators
# MAGIC * `Cancelled` - 31,256,894 not cancelled; 489,947 cancelled
# MAGIC * `Diverted` - 31,668,733 not diverted; 78,108 diverted
# MAGIC 
# MAGIC ##### Elapsed Time Info
# MAGIC * `CRS_Elapsed_Time` - 164 null values, range -99 to 948, 633 distinct values
# MAGIC * `Acutal_Elapsed_Time` - 568,042 null values, range 14 to 1604, 739 values
# MAGIC * `Air_Time` - 568,042 null values, ranges 4 to 1557, 705 values
# MAGIC 
# MAGIC ##### Flights
# MAGIC * `Flights` - number of flights, which is always 1
# MAGIC 
# MAGIC ##### Distance Info
# MAGIC * `Distance` - ranges from 21 to 5095, 1689 distinct values
# MAGIC * `Distance_Group` - aggregated variable, ranges 1-11, 11 total values
# MAGIC 
# MAGIC ##### Delay Info
# MAGIC * `Carrier_Delay` - 25,947,727 nulls, ranges 0 to 2695, 1597 distinct values
# MAGIC * `Weather_Delay` - 25,947,727 nulls, ranges 0 to 2692, 1241 distinct values
# MAGIC * `NAS_Delay` - 25,947,727 nulls, ranges 0 to 1848, 1249 distinct values
# MAGIC * `Security_Delay` - 25,947,727 nulls, ranges 0 to 1078, 315 distinct values
# MAGIC * `Late_Aircraft_Delay` - 25,947,727 nulls, ranges 0 to 2454, 1262 distinct values

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

