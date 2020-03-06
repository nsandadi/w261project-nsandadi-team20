# Databricks notebook source
# Load full dataset into airlines dataframe
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
from pyspark.sql import functions as F

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
#airlines = airlines.withColumn("AirTime", airlines["AirTime"].cast('int'))
airlines = airlines.withColumn("ArrDelay", airlines["ArrDelay"].cast('int'))
airlines = airlines.withColumn("DepDelay", airlines["DepDelay"].cast('int'))
airlines = airlines.withColumn("Distance", airlines["Distance"].cast('int'))
airlines = airlines.withColumn("ActualElapsedTime", airlines["ActualElapsedTime"].cast('int'))
airlines = airlines.withColumn("LateAircraftDelay", airlines["LateAircraftDelay"].cast('int'))
airlines = airlines.withColumn("SecurityDelay", airlines["SecurityDelay"].cast('int'))
airlines = airlines.withColumn("NASDelay", airlines["NASDelay"].cast('int'))
airlines = airlines.withColumn("WeatherDelay", airlines["WeatherDelay"].cast('int'))
airlines = airlines.withColumn("CarrierDelay", airlines["CarrierDelay"].cast('int'))
airlines = airlines.withColumn("TaxiOut", airlines["TaxiOut"].cast('int'))
airlines = airlines.withColumn("TaxiIn", airlines["TaxiIn"].cast('int'))


airlines = airlines.withColumn("Cancelled", airlines["Cancelled"].cast('int'))
airlines = airlines.withColumn("Diverted", airlines["Diverted"].cast('int'))
#airlines = airlines.withColumn("IsArrDelayed", F.when(F.contains(F.col("IsArrDelayed"), "YES"), 1).otherwhise(0))
                               # airlines["IsArrDelayed"].cast('int'))
#airlines = airlines.withColumn("IsDepDelayed", airlines["IsDepDelayed"].cast('int'))

# COMMAND ----------

#airlines = airlines.withColumn("AirTime", airlines["AirTime"].cast('int'))

print(airlines.columns)

display(airlines.take(1))

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
AIRLINES_path = userhome + "/AIRLINES/" 
AIRLINES_path_open = '/dbfs' + AIRLINES_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(AIRLINES_path)

# COMMAND ----------

# Save airlines as parquet
airlines.write.mode('overwrite').format("parquet").save(AIRLINES_path+"airlines.parquet")

# COMMAND ----------

#########################################
############ START HERE!!! ##############
#########################################

# read from parquet
airlines = spark.read.parquet(AIRLINES_path+"airlines.parquet")

# COMMAND ----------

# Display number records
airlines.count()

# COMMAND ----------

# Take a random sample to get ~100K rows
airlines_mini = airlines.sample(False, 0.001, 1).cache()

# COMMAND ----------

# get summary stats for the mini datset
display(airlines_mini.describe())

# COMMAND ----------

# do basic variable EDA
varName = 'Distance'

display(airlines_mini.groupBy(varName).count().orderBy(varName))

# COMMAND ----------

# Get summary stats for full dataset
display(airlines.describe())

# COMMAND ----------

# Figure out departure delays and arrival delay statistics grouped by Year and Month
display(airlines_mini.groupBy("Year", "Month").agg({'ArrDelay': 'mean'}).orderBy("Year", "Month"))

# COMMAND ----------

display(airlines_mini.groupBy("Year", "Month").agg({'DepDelay': 'mean'}).orderBy("Year", "Month"))

# COMMAND ----------

# Figure out departure delays and arrival delay statistics grouped by Month and DayOfMonth
display(airlines_mini.groupBy("DayOfWeek", "Month").agg({'ArrDelay': 'mean'}).orderBy("DayOfWeek"))

# COMMAND ----------

# Figure out departure delays and arrival delay statistics grouped by Month and DayOfMonth
display(airlines_mini.groupBy("DayOfWeek", "Month").agg({'DepDelay': 'mean'}).orderBy("DayOfWeek"))

# COMMAND ----------

display(airlines_mini.select("DepDelay"))

# COMMAND ----------

display(airlines_mini.select("ArrDelay"))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Determine distribution of Departure Times
res = airlines.select("DepTime", "ArrDelay", "DepDelay").map(lambda x: (math.round((x[0] / 100)), x[1], x[2])) #.groupBy("DepTime").mean()
type(res)


# COMMAND ----------

import matplotlib.pyplot as plt

for column in airlines:
    plt.figure()
    airlines.boxplot([column])

# COMMAND ----------

airlines_mini["DepTimeHr"] = airlines_mini.withColumn("DepTime", F.round(airlines_mini["DepTime"]))
display(airlines_mini.groupBy("DepTimeHr"))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

