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
    StructField('CRS_ELAPSED_TIME',IntegerType(),True),  # CRS Elapsed Time of Flight, in Minutes
    StructField('ACTUAL_ELAPSED_TIME',IntegerType(),True),  # Elapsed Time of Flight, in Minutes		
    StructField('AIR_TIME', DoubleType(), True),  # Flight Time, in Minutes
  
  
    StructField('FLIGHTS',ShortType(),True), # should be number of flights, but this is always "1"

  
    # Metrics related to distance of flight
    StructField('DISTANCE',IntegerType(),True),  # Distance between airports (miles)		
    StructField('DISTANCE_GROUP',ShortType(),True)  # Distance Intervals, every 250 Miles, for Flight Segment
  
    
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

# MAGIC %md
# MAGIC 
# MAGIC #### Features & What we want to do with them -- Task 2
# MAGIC 
# MAGIC * **Features**
# MAGIC   - **Departure Day & Departure Time**
# MAGIC       - Use variables `Day_Of_Week` (categorical variable), `CRS_Dep_Time` (group aggregation)
# MAGIC       - interaction between the two terms
# MAGIC   - **Using `Dep_Delay` for estimating Arrival Delay**
# MAGIC       - normalize the `Dep_Delay` variable
# MAGIC   - **Name of Airline**
# MAGIC       - Use `Unique_Carrier` as categorical variable
# MAGIC 
# MAGIC * **Outcome Variables:**
# MAGIC   - Model 1: Predicting Departure Delay (based on `Dep_Del***` (need to pick which, likely `Dep_Delay`))
# MAGIC   - Model 2: Predicting Arrival Delay (based on `Arr_Del***` (need to pick which, likely `Arr_Delay`))
# MAGIC   
# MAGIC   
# MAGIC * **Priorities (in-order)**
# MAGIC   - Predict models as 0/1 indicators of departure/arrival delay, using only basic airlines dataset
# MAGIC   - Incorporate Holidays into models
# MAGIC   - Predict models with depature/arrival delay amounts using only basic airlines dataset
# MAGIC   - Incorporate Weather data (& station data)
# MAGIC   - Incorporating airlines dataset metadata from ingesting dataset in graph (e.g. to get congestion data)
# MAGIC   - Predicting specific categories of delay (e.g. Carrir_Delay, Weather_Delay)

# COMMAND ----------

# MAGIC %md # Weather
# MAGIC https://data.nodc.noaa.gov/cgi-bin/iso?id=gov.noaa.ncdc:C00532
# MAGIC 
# MAGIC Full Dataset Description: https://www1.ncdc.noaa.gov/pub/data/noaa/isd-format-document.pdf

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

# Save a few copies of this datset (just in case....)
#weather.write.format("parquet").save("/dbfs/user/team20/weather-backup3-3-10.parquet")

# backup files at our disposal -- all the same data as what Luis shared Tuesday 8:36PM PST
display(dbutils.fs.ls("dbfs/user/team20"))

# COMMAND ----------

display(weather.where('DATE =="DATE"'))

# COMMAND ----------

mini_weather = weather.sample(False, 0.0000001)

dispaly(mini_weather)

# COMMAND ----------

display(weather.describe())

# COMMAND ----------

# General EDA to check for unique values/distribution of values/presence of nulls
varName = 'REM'
#display(weather.groupBy(varName).count().orderBy(weather[varName].desc()))
print("Number of distinct values: " + str(weather.select(varName).distinct().count()))
print("          Number of nulls: " + str(weather.filter(weather[varName].isNull()).count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA Summary
# MAGIC ##### Record Metadata
# MAGIC * `Station` - weather station identifier, 15,194 distinct values; 
# MAGIC * `Date` - format: "2019-12-31T23:59:00", minute granularity; ranges from 2015 to 2019, down to the minute granularity for multiple (but not all) locations
# MAGIC * `Source` - alphanumeric (1, 2, 4, 6, 7, 8, I, K, 0), total of 9 distinct values
# MAGIC * `Latitutde` - ranges -0.0166667 to 9.993861, 24,416 distinct values, no missing values indicators
# MAGIC * `Longitude` - ranges -0.005456 to 99.9666666, 34,609 distinct values, no missing values indicators
# MAGIC * `Elevation` - ranges +0000 to 999.1, 5338 distinct values
# MAGIC * `Name` - 4,715,523 null values, 14,857 distinct values (e.g. "068 BAFFIN BAY POINT OF ROCKS TX, TX US")
# MAGIC * `Report_Type` - 14 distinct values (e.g. "FM-12")
# MAGIC * `Call_Sign` - 30 are blank values; 449,370,610 take on value 99999 (mising values), 2211 distinct values (e.g. "K1A6 ")
# MAGIC * `Quality_Control` - 2 Distinct values "V030" (26651613 records) and "V020" (600342723 records)
# MAGIC 
# MAGIC ##### Weather Data
# MAGIC * `WND` - (e.g. "001,1,9,9999,9"  "999,9,V,9999,9"); entries with multiple "9"'s are likely mising values, but need to check with documentation. This variable likely holds mulitple metrics relate to wind direction & speed that are comma-separated, and need to refer to documentation for clarification here (need to do data cleanup); 172927 distinct values
# MAGIC * `CIG` - (e.g. "00000,1,9,9"  "99999,9,W,N"); 5715 distinct values
# MAGIC * `VIS` - (e.g. "000000,1,9,9"  "999999,9,V,A")
# MAGIC * `TMP` - (e.g. "+0000,1"  "-0913,1")
# MAGIC * `DEW` - (e.g. "+0000,1"  "-0929,1")
# MAGIC * `SLP` - (e.g. "08600,1"  "99999,9")
# MAGIC * `AA1` - (e.g. "+000,1"  "SYN14716128 46/// ///// 1//// 2/////39671 4////05711/ 333 55195 2///// 555/2//// 3//00 4//// 5//// 70114 1/125 /011103111/ 401// 50010 60010 7//// 2/133=")
# MAGIC * `AA2` - (e.g. ""MET08811/28/16 12:48:02 METAR KEHA 281848Z 26005KT 12/M07 RMK AO1 ,""a1~M/PK WND 44 M T01171072""  "SYN16416435 46/// /3603 10127 20083 30101 40256 5//// 333 55/// 2//// 555 20130 3//61 40202 51301 61815 7///1 8///2 9///3 1/112 21111 31111 411// 51111 61111 7//// 3/198=")
# MAGIC * `AJ1` - (e.g. ""MET10704/23/15 14:48:02 METAR KEHA 232048Z      KT RMK   %s AMOS 71/51/1514/M PK W""u5ZN 9 Uu 2s#}R8(9 ?q V ERm000""  "SYN15256132 46/// ///// 1//// 2//// 30139 4//// 52009 333 55000 20000 555 2//// 3//60 4//// 5//// 61305 89922 99923 1/129 21111 31111 411// 50010 60010 7////=")
# MAGIC * `AY1` - (e.g. "+000,1"  "SYN99940607 42960 02702 10125 20101 02801 10108 10260 20100 20177 33304 40131 40620 40689 42560 42960 57002 83100 333 83830 40605 41560 60701 10148 20108 40136 58002 70152 86800 333 82828 84630 40675 42560 52806 10202 20165 40130 58001 85600 333 85725 40606 41530 82901 10120 20108 40135 52001 72065 888XX 333 83830 85632 40638 41550 80000 10168 20143 40127 50008 71022 885// 333 88630 40689 42560 33304 10260 20177 40131 57002 83100 333 83830 40616 42460 73601 10150 20123 40130 52002 87300 333 83918 83822 85625 40656 42560 63505 10175 20138 40146 50002 86500 333 86628 40633 41558 62404 10175 20130 40130 50008 70501 84830 333 84825 84360 40672 42660 43104 10198 20154 40150 50005 83201 333 83835 40670 42560 73003 10177 20140 40146 51002 87500 333 87630 40662 41650 73302 10181 20152 40136 50006 71011 87500 333 87635 40664 41558 03004 10192 20155 40134 51005 70500 84200 333 84830 40666 41558 32705 10210 20177 40135 51005 71000 83200 333 83830 40674 42560 43207 10200 20137 40149 56002 84200 333 848")
# MAGIC * `AY2` - (e.g. "+296,1"  "SYN25362600 32970 03608 10275 20065 00220 03413 03619 10290 10310 10319 10360 10362 10418 10430 20010 20040 20139 20240 20243 20245 20255 30059 31956 31962 32970 39747 39853 39927 40090 40109 40118 40133 580009 59005 59005 59007 62640 62641 62650 70700 70700=")
# MAGIC * `GA1` - (e.g. "+003,1"  "SYN44891234 32474 81306 10306 20255 30074 40076 83201 333 563/9 59004 83819 88075 555 92500 91329 11474 80906 10289 20267 30166 40169 60014 70282 83307 333 562/9 58007 70013 83918 88275 555 92500 91336 32474 80715 10294 20244 30155 40158 83281 333 56299 59002 83818 88075 555 92500 91339 32474 80405 10300 20257 30091 40097 8422/ 333 56199 58000 84818 88463 555 92500 91320 32474 80106 10306 20261 30095 40100 83101 333 562/9 58000 83816 85074 555 92500=")
# MAGIC * `GA2` - (e.g. "+000,1"  "SYN82040607 42960 01601 10227 20065 02103 10304 10315 20059 20110 40112 40117 40662 40674 41540 41907 57015 57016 70600 70722 71508 87800 333 83830 86635 40606 41920 82302 10250 20062 40118 57001 70622 8801X 333 88360 40646 41920 43309 10265 20117 70711 84020 333 84460 40670 41506 50000 10257 20110 40140 58001 70640 84402 333 84628 40605 41957 01801 10258 20086 40111 58006 70500 40657 41610 52703 10250 20131 40109 58001 72161 85500 333 85635 40622 41558 83603 10175 20090 40105 52009 79562 8692X 333 82920 85624 86460 40665 42960 30906 10320 20092 40114 57022 83030 333 83360 40680 41958 30604 10336 20181 40126 58011 70600 81032 333 81360 40623 41559 82102 10178 20128 40164 50001 76066 8452X 333 84624 88460 40611 41559 81803 10182 20127 40149 56003 76022 888XX 333 83825 86628 40691 42960 00608 10328 20176 40158 58014=")
# MAGIC * `GA3` - (e.g. "+00000,1,0"  "SYN87762650 32970 73512 10420 20050 03205 10310 10315 10360 10365 10370 10370 10383 10385 10400 10400 10430 20040 20090 20140 20140 20160 20180 20183 20190 20190 20240 20258 31765 32770 32770 32770 32770 32770 32770 32970 32970 32970 32970 39/// 39393 39426 39455 39461 39534 39578 39623 39632 39662 39685 39815 4//// 40057 40057 40067 40068 40068 40071 40075 40076 40084 40093 40120 5//// 56009 56029 56099 56099 56409 56599 56609 56909 56909 56909 56909 57892 57942 58002 58003 58005 58005 58008 58009 58010 58011 58011 58015 59007 62721 62730 62751 62752 62762 627680 62771 62781 62790 62805 62810 70722 71804 71805 72103 72103 72202 72206 72504 72713 72803 73003 80007 81147 81850 81950 82107 82107 82107 82365 82650 82650 82850 82856 82856 82856 83047 83365 83507 83650 83850 84047 84047 84365 84365 84807 85307 87275 87275 87275 87275 87275 87275 87275 87275 87275 87275 87275=")
# MAGIC * `GE1` - (e.g. "+000,1"  "SYN99962660 31968 03609 10300 20030 03203 03204 10270 10280 10288 10290 10305 10306 10306 10315 10315 10318 10330 10334 10340 10340 10380 10385 10392 10395 10405 10406 10410 10410 10412 10418 10430 10430 10430 10435 10457 12970 12970 20075 20150 20152 20170 20170 20174 20190 20198 20200 20204 20210 20230 20245 20250 20250 20255 20264 20270 20275 20280 20285 20285 20287 20287 20290 20300 20310 21040 31910 31930 31962 32970 32970 32970 32970 32970 32970 32970 32970 32970 39411 39443 39471 39501 39554 39587 39619 39620 39630 39642 39655 39669 39670 39712 39840 40057 40083 40083 40094 40098 40100 40102 40102 40111 40124 40128 40134 40137 40140 40154 5//// 56009 56009 56009 56009 56009 56009 56009 56009 56090 56099 56099 56099 58000 58002 58002 58003 58006 58006 58010 58013 58013 58018 59001 59005 59009 59012 60064 60254 62680 62721 62730 62750 62751 62752 62762 62771 62772 62781 62790 62795 62805 62809 70112 70600 70600 70722 70722 71804 71806 72203 72207 72215 72303 72306 72312 72502 72505 729")
# MAGIC * `GF1` - (e.g. "+00000,1,0"  "SYN52242409 32996 20000 10182 20151 40121 81160 333 20142 58000/82 42408 NIL 42407 32997 20000 10140 20120 40141 82002 333 10244 58005/87 42413 31994 00000 10140 20114 40216 742// 333 20100 59002/84/ 42522 NIL 42523 31997 00000 10110 20098 40179 70004 333 20084 58001/92 42414 32996 01102 10140 20122 40177 333 20/// 58000 555 10001/89 42420 31796 00000 10130 20122 40153 705// 333 20/// 59009/RH 87 42317 31595 03303 10122 20103 30017 4//// 74044 333 20065 59018/88 42529 31595 00000 10134 20109 40108 333 20058 59002/85 42727=")
# MAGIC * `IA1` - (e.g. "+0000,1,0,+0007,1,0,00002,1,0,00001,1,0"  "SYN99962600 31965 03109 10320 20135 01307 10200 10212 10215 10220 10225 10225 10230 10230 10230 10240 10240 10250 10270 10310 10330 10340 10341 10342 10344 10345 10345 10348 10356 10366 10368 10380 10380 10385 10400 10400 10435 10440 10460 10460 10470 11/05 11568 11669 11670 11770 12570 12670 12970 12970 12970 2-0200 2//// 2//// 20115 20150 20160 20180 20185 20190 20195 20195 20196 20200 20200 20200 20202 20205 20206 20210 20210 20210 20210 20210 20214 20215 20217 20224 20230 20236 20260 20270 20304 20310 20310 20320 30002 31930 31958 31965 32670 32770 32970 32970 39218 39290 39439 39472 39477 39544 39623 39631 39655 39670 39695 39695 39698 39710 39785 39789 39841 40041 40048 40052 40053 40074 40086 40097 40101 40103 40105 40108 40110 40119 40120 40128 40131 40134 40134 56009 56049 56090 56090 56090 56090 56090 56909 56990 56990 56990 56990 56990 56990 56990 57992 58006 58015 58018 58022 58027 58030 58033 58037 58045 58047 58050 58050 58050 58050 58051 58051 59006 59010 60044 60044 60114 60")
# MAGIC * `KA1` - (e.g. "+000,1"  "SYN99940691 11658 70602 10138 20086 40228 51008 69902 78022 87800 333 20115 83835 84640 40620 12960 00000 10075 20014 60002 333 20017 40607 12960 00201 10067 21010 60002 333 20039 40665 12957 20101 10102 20058 40214 52010 60002 70500 80001 333 20066 820// 40638 12960 20302 10062 20038 40212 52010 60002 82030 333 20020 82360 40662 12560 60901 10100 20043 40208 54000 86800 333 20066 82830 84635 40674 11958 31203 10102 20048 40213 52012 60002 70500 80001 333 20082 830// 40666 12560 70000 10110 20063 40209 54000 60002 87800 333 20058 83830 84633 40672 12960 03602 10094 20051 40220 52012 60002 333 20065 40676 12560 40403 10120 20045 40212 52013 60002 83530 333 20106 83625 40670 12960 00000 10095 20042 40216 52018 60002 333 20066 40675 12960 50902 10096 20046 40202 52006 60002 85030 333 20074 85360 40650 11640 23102 10066 20040 40218 52012 60002 71000 81130 333 20037 81840 40632 12960 00000 10071 20041 40220 58004 60002 20035 40657 12960 13502 10072 20045 40206 52010 60002 80001 333 20054 810// 4")
# MAGIC * `KA2` - (e.g. "+0000,1,0,+0000,1,0,00000,1,0,00000,1,0"  "SYN99963330 41575 13404 10248 20125 37785 40251 70500 81100 333 59008 81820/63331 41575 52003 10236 20155 38088 40238 71722 85900 333 59008 81920 84624/63332 42580 41302 10246 20178 38220 40078 83230 333 5900783825/63333 41560 30801 10302 20085 38153 40162 70510 83830 333 59004 81626 83827/63334 41575 61802 10188 20131 37636 40098 70522 86300 333 59004 81925 83826 83627/63340 41570 63102 10220 20176 37953 40177 71722 86900 333 59018 84823 83624 81925/ 63402 42580 70000 10182 20165 38380 40056 83270 333 59002 83826 86360/63403 41560 62701 10212 20154 70522 81130 333 81825 86360/63450 42580 71704 10216 20110 37739 40077 87800 333 59010 85624 84825/63451 41556 30804 10256 20138 38130 40059 7050/ 83100 333 59012 83826/63453 41560 22702 10364 20112 39027 70500 82100 333 5901082825/63460 42580 61402 10246 20163 84230 333 84824 82358/63471 41559 11102 10336 20149 38830 40041 70500 81100 333 58009 81827/63474 42580 60302 10224 20102 86800 333 84821 83622/63478 41578 42308 10314 2//// 39794 40173 70")
# MAGIC * `MA1` - (e.g. "+000,1"  "SYN99963332 42980 02502 10240 20114 38224 40081 333 58002 //O/P CLASS/// STYLE//////O/P///SPAN///DIV//DIV CLASS/// STYLE/// ID//YUI/3/16/0/1/1420608922258/49883///SPAN STYLE//FONT-SIZE/12. 0PT/LINE-HEIGHT/115// CLASS/// ID//YUI/3/16/0/1/142 0608922258/49882//63333 42580 30206 10248 20104 38180 40135 83100 333 58006 83827 //O/P CLASS/// STYLE//////O/P///SPAN///DIV//DIV CLASS/// STYLE/// ID//YUI/3/16/0/1/1420608922258/49881///SPAN STYLE//FONT-SIZE/12. 0PT/LINE-HEIGHT/115// CLASS/// ID//YUI/3/16/0/1/142 0608922258/49880//63334 42580 11903 10228 20024 37634 40055 81100 333 59007 81826 //O/P CLASS/// STYLE//////O/P///SPAN///DIV//DIV CLASS/// STYLE/// ID//YUI/3/16/0/1/1420608922258/49879///SPAN STYLE//FONT-SIZE/12. 0PT/LINE-HEIGHT/115// CLASS/// ID//YUI/3/16/0/1/142 0608922258/49878//63340 42580 13102 10240 20112 37961 40188 81100 333 58011 81827 //O/P CLASS/// STYLE//////O/P///SPAN///DIV//DIV CLASS/// STYLE/// ID//YUI/3/16/0/1/1420608922258/49999///SPAN STYLE//FONT-SIZE/12. 0PT/LINE-HEIGHT/115//")
# MAGIC * `MD1` - (e.g. "+000,1"  "SYN99984542 31670 40000 10170 20098 / / / / / / / / / // // //VFR //VFR //VFR /VFR/ /VFR/ /VFR/ /VFR/ /VFR/ /VFR/ 01 050/ 050/ 050/ 080/ 080/ 1/2///BKN040/ 1/2///SCT030/ 100/ 120/ 20G25KT/ 20G25KT/ 22N 23N///BKN025/ 23N///SCT025 24N/ 32N 37350 400MB/ 57W///CARIBBEAN///GULF 70 75W///BKN030 75W///BKN030/ ALL AND AND ATLANTIC ATLC BARRANQUILLA BKN030 BKN060/ BKN060/ BKN080/ BTN COLD CONDS/ DOMINGO E END FIR FIR FIR FIR FIR FIR FIR// FIR///CURACAO FIR///NRN FIR///PORT/AU/PRINCE FL200/ FL200/ GTR HISPANIOLA/ ICE IFR IMPLY ISLAND JUAN LLWS LYRD LYRD MAIQUETIA MEXICO MIAMI MVFR N NEW NLY NRN NRN NWD///BKN025 NWLY NWRN OCNL OF OF OF OR OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK/ OTLK///VFR///FRQ OVC060/ OVC100/ OVR OVR PANAMA PART PIARCO RDG S S SAN SANTO SCT SCT SCT SCT SCT SCT025 SERN SEV SEV SFC SFC/ SHRA/ SHRA/ SHRA/ SHRA/ SHRA/ SKC/ SRN STNR SWRN SYNOPSIS///LRG TO TO TO TOPS TOPS TOPS TOPS TOPS TOPS TOPS TOPS TOPS TS TURB VIS W W WDLY WDLY WND WND WND/ WND/ WND/ WNDS WRN WTRS///BKN03")
# MAGIC * `MW1` - (e.g. "5924 HR PRECIPITATION (IN): 0.00 24 HR MAX TEMP (F): 8024 HR MIN TEMP (F): 72PEAK WIND SPEED (MPH): 26 PEAK WIND DIR (DEG): 70 PEAK WIND TIME (LST):      FASTEST 2MIN SPEED (MPH): 22FASTEST 2MIN DIR (DEG): 60 FASTEST 2MIN TIME (LST):     AVERAGE WIND SPEED (MPH):13.6 24 HR AVG TEMP (F): 76DEPART FROM NORMAL:  3  HEATING DEGREE DAYS:  0 COOLING DEGREE DAYS: 11"  "SYN99980009 21560 23505 10354 20241 30071 40076 70500 81801 333 56090 59007 60007 81820 80022 21462 53506 10330 20255 30063 40065 70521 82231 333 56190 59014 60007 82815 83070 80028 21460 30103 10322 20270 30043 40078 70511 82900 333 569// 57922 59019 60007 83615 82920 80035 21560 1//// 10344 20245 30078 40083 70500 81500 333 562// 59002 60007 81620 80036 21560 40505 10359 20213 70511 82202 333 562/9 60007 82820 80063 21565 30000 10356 20255 83400 333 5632/ 60007 83420 80084 22470 43204 10319 20253 82202 333 569/7 57833 60007 82817 83080 80091 21464 60000 10324 20260 70522 82875 70522 82275 333 56099 60007 82810 83070 80094 21468 73204 10248 20216 38786 70522 84875 333 56745 59022 60007 84815 84358 80097 22570 61308 10306 20189 39707 4//// 83431 333 56399 59001 60007 83622 83359 80099 21362 80302 10259 20243 39908 40050 70522 82672 333 56232 58013 60007 82609 85459 80110 22570 61006 10285 20150 38467 81974 333 56245 57992 59006 60007 81930 85070 80112 21570 60904 10208 20129 37888 70522 822")
# MAGIC * `OC1` - (e.g. ""MET07309/13/17 14:56:02 METAR KNBT 131956Z VRB04KT 10SM A2991 RMK AO2 SLP1fA ""j""  "SYN99980035 02570 11011 10337 20200 30102 40107 60001 81100 333 562// 59014 60007 81820 80036 01565 10407 10340 20188 60001 70500 81800 333 562// 60007 81820 80063 01562 30000 10324 20250 60001 83200 333 561// 60007 83820 80084 02560 43206 10311 20237 60001 83110 333 5683/ 60007 83820 80091 01566 40702 10324 20229 70500 83101 333 567/9 60001 83820 80094 01565 32805 10266 20171 38799 60001 70510 83100 333 567// 59009 60007 83820 80097 02570 41810 10323 20185 3//// 4//// 60001 83102 333 564/9 5//// 60007 83822 80099 02566 70802 10314 20231 39922 40066 60001 84830 333 5613/ 59006 60007 84827 85360 80110 01670 20910 10295 20087 38475 60001 70500 81101 333 562/1 59007 60007 82840 80112 02570 41207 10222 20106 37893 60001 83101 333 563/4 59008 60007 83825 80139 02570 10608 10314 20225 60001 81100 333 56200 60007 81823 80144 01457 53104 10292 20251 69931 70552 85205 333 567/5 60007 85817 83070 80210 01570 50502 10272 20166 38632 60001 70511 82831 333 56120 57893 59003 60007 82820 83358 80211 02570")
# MAGIC * `OD1` - (e.g. "+000,1"  "SYN99980144 01350 71502 10244 20241 60074 7102/ 85770 333 20240 31/// 55022 5699/ 60062 85705 80210 01462 61003 10200 20184 38659 60014 70522 83431 333 20190 31/// 50300 55066 56220 59005 60007 83615 83070 80211 01565 60000 10200 20192 38795 60004 7022/ 86500 333 20186 30/// 50430 55078 568// 59011 60008 86620 80214 01570 72702 10228 20196 60004 702// 82150 333 20206 30/// 50510 55070 5634/ 60002 82830 86360 80222 01458 80301 10110 20096 37538 60054 75052 8547/ 333 20096 31/// 55/// 5649/ 59007 69907 85615 85457 80234 01450 80000 10222 20211 39643 40140 69934 7152/ 8477/ 333 20213 30/// 50270 55031 56030 59012 69917 84715 86359 80259 02662 71701 10207 20207 39049 60004 81535 333 20200 30/// 55048 56449 59006 60007 81640 85362 80315 01766 61804 10248 20176 39606 40108 69904 7152/ 81170 333 20245 30/// 50560 55039 5644/ 59011 60002 81850 86360 80342 02570 50000 10158 20110 38219 60004 702// 85500 333 20148 30/// 50590 55092 563// 59002 60009 85624 80370 01470 70000 10090 20063 37175 69934 702")
# MAGIC * `SA1` - (e.g. ""MET08901/02/18 15:56:02 METAR KNBT 022056Z 35009KT A3048 RMK ""r   -       J    *S   327 56014 $""  "SYN99980063 21565 40000 10340 20252 30061 40075 70521 84400 333 5676/ 58001 60007 84620 80084 21570 40000 10312 20244 82106 333 56309 60007 82823 84280 80091 22564 62102 10342 20235 39919 40062 81202 333 566/9 59009 60007 82820 84070 80094 22470 53203 10261 20195 38803 83852 333 56899 59006 60007 83817 83080 80097 22570 40105 10332 20216 39697 40091 82131 333 56199 59018 60007 82822 80099 22466 52202 10298 20247 39900 40042 83806 333 563/9 59013 60007 83815 83270 80110 21665 40805 10297 20105 38485 70521 84101 333 562/9 59008 60007 84835 80112 22570 41404 10227 20076 37894 82901 333 563/0 57992 59002 60007 82920 83070 80139 21460 82502 10246 20238 30038 40074 79599 8697/ 333 5660/ 58005 60017 86915 80144 21568 62303 10327 20245 70522 84205 333 569/9 60007 84820 80210 21460 73304 10284 20174 38632 71522 83271 333 56780 57891 59003 60007 83817 84358 80211 22565 70000 10276 20192 38770 83181 333 56352 59001 60007 83825 85360 80214 21570 71304 10285 20186 70222 84870 333 5633/ 60007 84827 87460")
# MAGIC * `UA1` - (e.g. "+000,1"  "SYN99980139 02470 73302 10268 20235 30047 40083 60101 83570 333 5670/ 59005 60007 83617 84357 80144 01458 51603 10294 20250 60001 71522 84230 333 5671/ 60007 84815 80210 01460 70202 20202 38652 60031 72165 83276 333 56110 57892 58010 60031 83815 85358 80211 02565 72002 10270 20192 38792 60001 85170 333 5625/ 58009 60007 85825 83360 80214 01570 61303 10291 20188 60001 70122 82150 333 5634/ 60007 82830 85360 80222 01566 71307 10178 20094 37535 6//// 71552 85875 333 56399 59006 6///7 71552 85875 80234 01460 80000 10266 20223 39639 40130 60121 71565 85571 333 56339 59006 60097 85615 83359 80259 01660 71702 102809 20184 39045 60001 71522 8317/ 333 5649/ 58010 6007 83830 87462 80315 01662 81605 10289 20184 39609 40104 60001 71522 8117/ 333 5642/ 59008 60007 81840 88361 80342 02570 60504 10218 20120 38227 60001 86800 333 563// 58012 60007 86825 80370 02470 81304 10116 20070 37182 60001 888// 333 564// 58002 60007 88616 80398 02570 71502 10309 20234 30037 40132 60001 84130 333 5629/ 59004 60007 848")
# MAGIC * `REM` - (e.g. ""MET10909/28/17 07:50:02 METAR PALJ 281650Z 00000KT 10SM SCT005 BKN020 OVC040 06/05 A2949 RMK EST PASS CLSD NOSPECI","R01    067TMP046R02    057DPT046""  "SYN99980035 21560 20407 10285 20251 30090 40095 70510 82500 333 562// 58002 60007 81518 80036 21468 60506 10330 20206 70500 83232 333 56135 57892 60007 83815 80063 21565 23103 10320 20235 30059 40073 70521 81210 333 5675/ 58000 60007 81823 80084 22565 23301 10306 20246 82100 333 568// 60007 82820 80091 21466 20902 10308 20244 39924 40067 70500 82101 333 567/9 59003 60007 82810 80094 21465 63304 10233 20209 38795 70511 83870 333 5673/ 59005 60007 83810 83357 80097 21464 80104 10268 20222 39704 40098 8387/ 333 5619/ 58002 60007 83818 83359 80099 21462 73602 10268 20254 39877 40018 70566 86502 333 568/2 57882 58018 60137 82812 85620 80110 21565 60204 10236 20159 38483 71522 83380 333 5614/ 57992 58000 60007 83833 85358 80112 21465 70603 10182 20136 37882 70522 83251 333 56399 57893 58007 60007 83817 83358 80139 21560 70000 10310 20237 71522 83275 333 56100 60007 83820 83358 80144 21460 61806 10306 20222 71522 86200 333 564// 60007 86816 80210 21462 60905 10226 20184 38632 71522 83871 333 56990")
# MAGIC * `EQD` - (e.g. "+000,1"  "SYN79663881 32480 40000 10143 20094 38236 48/// 81101 333 10252 58000 81816 84070//G///////////////////////////////////// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// ////////////////////////////////////////////////////////////////// /// /// CHECK TEXT NEW ENDING ADDED HTDAYFYX=")

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

