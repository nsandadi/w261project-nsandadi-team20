# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Airlines EDA using Plotly
# MAGIC 
# MAGIC This is a notebook for doing some basic airlines EDA using plotly (like Shaji's plots)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initial Data Prep

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# Filter to datset with entries where diverted != 1, cancelled != 1, dep_delay != Null, and arr_delay != Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print("Number of records in full dataset:", airlines.count())

def SplitDataset(model_name):
  # Split airlines data into train, dev, test
  test = airlines.where('Year = 2019') # held out
  train, val = airlines.where('Year != 2019').randomSplit([7.0, 1.0], 6)

  # Select a mini subset for the training dataset (~2000 records)
  mini_train = train.sample(fraction=0.0001, seed=6)

  print("train_" + model_name + " size = " + str(train.count()))
  print("mini_train_" + model_name + " size = " + str(mini_train.count()))
  print("val_" + model_name + " size = " + str(val.count()))
  print("test_" + model_name + " size = " + str(test.count()))
  
  return (mini_train, train, val, test) 

mini_train, train, val, test = SplitDataset("")

# COMMAND ----------

# Create the full data (not including test) for EDA
full_data = train.union(val)

# COMMAND ----------

# save full data as parquet for easier analysis
full_data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/full_training_data_airlines.parquet")
display(dbutils.fs.ls("dbfs/user/team20"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Start Here!

# COMMAND ----------

# Read in data for faster querying
full_data = spark.read.option("header", "true").parquet(f"dbfs/user/team20/full_training_data_airlines.parquet")

# COMMAND ----------

outcomeName = 'Dep_Del15'
nfeatureNames = [
  # 0        1            2              3              4                5                6               7               8 
  'Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group'
]
#                         9              10       11
cfeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']

# Filter full data to just relevant columns
full_data_dep = full_data.select([outcomeName] + nfeatureNames + cfeatureNames)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install Dependencies

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install plotly --upgrade

# COMMAND ----------

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt
import plotly.express as px

import plotly.io as pio

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

from pyspark.sql.functions import col, countDistinct

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basic Initial EDA

# COMMAND ----------

print("Expected Num Records:  23903381")
print("  Actual Num Records: ", full_data_dep.count())

# COMMAND ----------

display(full_data_dep.sample(False, 0.00001))

# COMMAND ----------

# Get number of distinct values for each column in full training dataset
display(full_data_dep.agg(*(countDistinct(col(c)).alias(c) for c in full_data_dep.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Helper Functions for EDA

# COMMAND ----------

# Helper Function for Group 1 graphs plotting distinct values of feature on X and number of flights on Y, categorized
# by outocme variable

def MakeRegBarChart(full_data_dep, outcomeName, var, orderBy, barmode, xtype):
  if (orderBy):
      d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(orderBy).toPandas()
  else:
    d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().toPandas()

  t1 = go.Bar(
    x = d[d[outcomeName] == 0.0][var],
    y = d[d[outcomeName] == 0.0]["count"],
    name=outcomeName + " = " + str(0.0)
  )
  t2 = go.Bar(
    x = d[d[outcomeName] == 1.0][var],
    y = d[d[outcomeName] == 1.0]["count"],
    name=outcomeName + " = " + str(1.0)
  )

  l = go.Layout(
    barmode=barmode, 
    title="Flight Counts by " + var + " & " + outcomeName,
    xaxis=dict(title=var, type=xtype),
    yaxis=dict(title="Number of Flights")
  )
  fig = go.Figure(data=[t1, t2], layout=l)
  fig.show()

# COMMAND ----------

# Helper function for Group 3 graphs that plot the probability of outcome on the x axis, the number of flights on the x axis
# With entries for each distinct value of the feature as separate bars.

def MakeProbBarChart(full_data_dep, outcomeName, var, xtype, numDecimals):
  # Filter out just to rows with delays or no delays
  d_delay = full_data_dep.select(var, outcomeName).filter(col(outcomeName) == 1.0).groupBy(var, outcomeName).count().orderBy("count")
  d_nodelay = full_data_dep.select(var, outcomeName).filter(col(outcomeName) == 0.0).groupBy(var, outcomeName).count().orderBy("count")

  # Join tables to get probabilities of departure delay for each table
  probs = d_delay.join(d_nodelay, d_delay[var] == d_nodelay[var]) \
             .select(d_delay[var], (d_delay["count"]).alias("DelayCount"), (d_nodelay["count"]).alias("NoDelayCount"), \
                     (d_delay["count"] / (d_delay["count"] + d_nodelay["count"])).alias("Prob_" + outcomeName))

  # Join back with original data to get 0/1 labeling with probablities of departure delay as attribute of airlines
  d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count()
  d = d.join(probs, full_data_dep[var] == probs[var]) \
       .select(d[var], d[outcomeName], d["count"], probs["Prob_" + outcomeName]) \
       .orderBy("Prob_" + outcomeName, outcomeName).toPandas()
  d = d.round({'Prob_' + outcomeName: numDecimals})

  t1 = go.Bar(
    x = d[d[outcomeName] == 0.0]["Prob_" + outcomeName],
    y = d[d[outcomeName] == 0.0]["count"],
    name=outcomeName + " = " + str(0.0),
    text=d[d[outcomeName] == 0.0][var]
  )
  t2 = go.Bar(
    x = d[d[outcomeName] == 1.0]["Prob_" + outcomeName],
    y = d[d[outcomeName] == 1.0]["count"],
    name=outcomeName + " = " + str(1.0),
    text=d[d[outcomeName] == 1.0][var]
  )

  l = go.Layout(
    barmode='stack', 
    title="Flight Counts by " + "Prob_" + outcomeName + " & " + outcomeName + " for each " + var,
    xaxis=dict(title="Prob_" + outcomeName + " (Note: axis type = " + xtype + ")", type=xtype),
    yaxis=dict(title="Number of Flights")
  )
  fig = go.Figure(data=[t1, t2], layout=l)
  fig.show()
  
  return d

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Group 1 Plots

# COMMAND ----------

# Plot Year and outcome
var = "Year"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(var).toPandas()
display(d)

# COMMAND ----------

# Plot Year and outcome
var = "Year"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='stack', xtype='category')

# COMMAND ----------

# Plot Month & outcome
var = "Month"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(var).toPandas()
display(d)

# COMMAND ----------

# Plot Month and outcome
var = "Month"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='stack', xtype='category')

# COMMAND ----------

# Plot that demonstrates the probability of a departure delay, given the day of year (interaction of month & day of month)
var = "Day_Of_Year"
d = full_data_dep.select("Month", "Day_Of_Month", outcomeName) \
                 .withColumn(var, f.concat(f.col('Month'), f.lit('-'), f.col('Day_Of_Month'))) \
                 .groupBy(var, "Month", "Day_Of_Month", outcomeName).count() \
                 .orderBy("Month", "Day_Of_Month") \
                 .toPandas()
display(d)

# COMMAND ----------

# Plot Day_Of_Month interacted with Month and outcome
var = "Day_Of_Year"
d = full_data_dep.select("Month", "Day_Of_Month", outcomeName) \
                 .withColumn(var, f.concat(f.col('Month'), f.lit('-'), f.col('Day_Of_Month'))) \
                 .groupBy(var, "Month", "Day_Of_Month", outcomeName).count() \
                 .orderBy("Month", "Day_Of_Month") \
                 .toPandas()

t1 = go.Bar(
  x = d[d[outcomeName] == 0.0][var],
  y = d[d[outcomeName] == 0.0]["count"],
  name=outcomeName + " = " + str(0.0)
)
t2 = go.Bar(
  x = d[d[outcomeName] == 1.0][var],
  y = d[d[outcomeName] == 1.0]["count"],
  name=outcomeName + " = " + str(1.0)
)

l = go.Layout(
  barmode='stack', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var, type='category'),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------

# Plot Day of Week and outcome
var = "Day_Of_Week"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(var).toPandas()
display(d)

# COMMAND ----------

# Plot Day of Week and outcome
var = "Day_Of_Week"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='group', xtype='category')

# COMMAND ----------

# Effectively demonstrates the probability of a departure delay, given the distance group
var = "Distance_Group"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(var).toPandas()
display(d)

# COMMAND ----------

# Plot Distance Group and outcome
var = "Distance_Group"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='group', xtype='category')

# COMMAND ----------

# Effectively demonstrates the probability of a departure delay, given the carrier
# Airline Codes to Airlines: https://www.bts.gov/topics/airlines-and-airports/airline-codes
var = "Op_Unique_Carrier"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy("count").toPandas()
display(d)

# COMMAND ----------

# Plot Carrier and outcome
# Airline Codes to Airlines: https://www.bts.gov/topics/airlines-and-airports/airline-codes
var = "Op_Unique_Carrier"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy='count', barmode='group', xtype='category')

# COMMAND ----------

# Plot Origin and outcome
# Airport Codes: https://www.bts.gov/topics/airlines-and-airports/world-airport-codes
var = "Origin"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy='count', barmode='group', xtype='category')

# COMMAND ----------

# Plot Destination and outcome
# Airport Codes: https://www.bts.gov/topics/airlines-and-airports/world-airport-codes
var = "Dest"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy='count', barmode='group', xtype='category')

# COMMAND ----------

# Plot CRS_Elapsed_Time and outcome
var = "CRS_Elapsed_Time"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='stack', xtype='linear')

# COMMAND ----------

# Plot CRS_Dep_Time and outcome
var = "CRS_Dep_Time"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='stack', xtype='linear')

# COMMAND ----------

# Plot CRS_Arr_Time and outcome
var = "CRS_Arr_Time"
MakeRegBarChart(full_data_dep, outcomeName, var, orderBy=var, barmode='stack', xtype='linear')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Group 2 Plots

# COMMAND ----------

# Make helper code for bucketizing (binning) values
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import udf
from pyspark.sql.types import *

def BinValues(df, var, splits, labels):
  bucketizer = Bucketizer(splits=splits, inputCol=var, outputCol=var + "_bin")
  df_buck = bucketizer.setHandleInvalid("keep").transform(df)
  
  bucketMaps = {}
  bucketNum = 0
  for l in labels:
    bucketMaps[bucketNum] = l
    bucketNum = bucketNum + 1
    
  def newCols(x):
    return bucketMaps[x]
  
  callnewColsUdf = udf(newCols, StringType())
    
  return df_buck.withColumn(var + "_binlabel", callnewColsUdf(f.col(var + "_bin")))


# COMMAND ----------

var = 'CRS_Dep_Time'
d = full_data_dep.select(var, outcomeName)
d = BinValues(d, var, splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
              labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm',
                        '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

MakeRegBarChart(d, outcomeName, var + "_bin", orderBy=var + "_bin", barmode='group', xtype='category')

# COMMAND ----------

var = 'CRS_Dep_Time'
d = full_data_dep.select(var, outcomeName)
d = BinValues(d, var, splits = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
              labels = [str(i) for i in range(0, 24)])

MakeRegBarChart(d, outcomeName, var + "_bin", orderBy=var + "_bin", barmode='stack', xtype='category')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##### Group 3 Plots

# COMMAND ----------

# Plot Carrier and outcome
# Airline Codes to Airlines: https://www.bts.gov/topics/airlines-and-airports/airline-codes
var = "Op_Unique_Carrier"

# Filter out just to rows with delays or no delays
d_delay = full_data_dep.select(var, outcomeName).filter(col(outcomeName) == 1.0).groupBy(var, outcomeName).count().orderBy("count")
d_nodelay = full_data_dep.select(var, outcomeName).filter(col(outcomeName) == 0.0).groupBy(var, outcomeName).count().orderBy("count")

# Join tables to get probabilities of departure delay for each table
probs = d_delay.join(d_nodelay, d_delay[var] == d_nodelay[var]) \
           .select(d_delay[var], (d_delay["count"]).alias("DelayCount"), (d_nodelay["count"]).alias("NoDelayCount"), \
                   (d_delay["count"] / (d_delay["count"] + d_nodelay["count"])).alias("Prob_" + outcomeName))

# Join back with original data to get 0/1 labeling with probablities of departure delay as attribute of airlines
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count()
d = d.join(probs, full_data_dep[var] == probs[var]) \
     .select(d[var], d[outcomeName], d["count"], probs["Prob_" + outcomeName]) \
     .orderBy("Prob_" + outcomeName, outcomeName).toPandas()
d = d.round({'Prob_' + outcomeName: 4})

display(d)

# COMMAND ----------

# Plot Carrier and outcome with bar plots of probability on x axis
# Airline Codes to Airlines: https://www.bts.gov/topics/airlines-and-airports/airline-codes
var = "Op_Unique_Carrier"  
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='linear', numDecimals=4)

# COMMAND ----------

# Plot Origin airport and outcome with bar plots of probability on x axis
# Airport Codes: https://www.bts.gov/topics/airlines-and-airports/world-airport-codes
var = "Origin"
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='category', numDecimals=4)

# COMMAND ----------

# Plot Destination airport and outcome with bar plots of probability on x axis
# Airport Codes: https://www.bts.gov/topics/airlines-and-airports/world-airport-codes
var = "Dest"
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='category', numDecimals=4)

# COMMAND ----------

# Plot distance group and outcome with bar plots of probability on x axis
var = "Distance_Group"
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='linear', numDecimals=4)

# COMMAND ----------

# Plot Month and outcome with bar plots of probability on x axis
var = "Month"
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='linear', numDecimals=4)

# COMMAND ----------

# Plot Day_Of_Year and outcome with bar plots of probability on x axis?
var = "Day_Of_Year"
d = full_data_dep.withColumn(var, f.concat(f.col('Month'), f.lit('-'), f.col('Day_Of_Month')))
d = MakeProbBarChart(d, outcomeName, var, xtype='category', numDecimals=10)

# COMMAND ----------

# Plot CRS_Elapsed_Time and outcome with bar plots of probability on x axis
var = "CRS_Elapsed_Time"
MakeProbBarChart(full_data_dep, outcomeName, var, xtype='category', numDecimals=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Group 4 Plots

# COMMAND ----------

def PlotBasicBoxPlots(full_data_dep, var, outcomeName):
  d = full_data_dep.select(var, outcomeName).sample(True, 0.0005, 6).toPandas()
  fig = px.box(d, y=var, points="all", color=outcomeName, title="Boxplots of " + var + " by " + outcomeName)
  fig.show()

# COMMAND ----------

var = "Distance"
PlotBasicBoxPlots(full_data_dep, var, outcomeName)

# COMMAND ----------

def PlotBinnedBoxPlots(full_data_dep, var, binnedVar, outcomeName):
  d = full_data_dep.select(var, outcomeName, binnedVar).sample(True, 0.0005, 6).toPandas()
  fig = px.box(d, x=binnedVar, y=var, points="all", color=outcomeName)
  fig.show()

# COMMAND ----------

var = "Distance"
binnedVar = var + "_Group"
PlotBinnedBoxPlots(full_data_dep, var, binnedVar, outcomeName)

# COMMAND ----------

var = "CRS_Elapsed_Time"
PlotBasicBoxPlots(full_data_dep, var, outcomeName)

# COMMAND ----------

var = "CRS_Elapsed_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [-100, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720], 
              labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'])

PlotBinnedBoxPlots(d, var, binnedVar, outcomeName)

# COMMAND ----------

# Plot CRS_Elapsed_Time and outcome as porbability chart
var = "CRS_Elapsed_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [-100, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720], 
              labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'])
MakeRegBarChart(d, outcomeName, binnedVar, orderBy=binnedVar, barmode='stack', xtype='category')

# COMMAND ----------

MakeProbBarChart(d, outcomeName, binnedVar + "label", xtype='category', numDecimals=4)

# COMMAND ----------

var = "CRS_Dep_Time"
PlotBasicBoxPlots(full_data_dep, var, outcomeName)

# COMMAND ----------

var = "CRS_Dep_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
              labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

PlotBinnedBoxPlots(d, var, binnedVar, outcomeName)

# COMMAND ----------

MakeRegBarChart(d, outcomeName, binnedVar, orderBy=binnedVar, barmode='stack', xtype='category')

# COMMAND ----------

MakeProbBarChart(d, outcomeName, binnedVar + "label", xtype='category', numDecimals=4)

# COMMAND ----------

var = "CRS_Dep_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 
                        1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400], 
              labels = ['12am-1am', '1am-2am', '2am-3am', '3am-4am', '4am-5am', '5am-6am', 
                        '6am-7am', '7am-8am', '8am-9am', '9am-10am', '10am-11am', '11am-12pm', 
                        '12pm-1pm', '1pm-2pm', '2pm-3pm', '3pm-4pm', '4pm-5pm', '5pm-6pm', 
                        '6pm-7pm', '7pm-8pm', '8pm-9pm', '9pm-10pm', '10pm-11pm', '11pm-12am'])

PlotBinnedBoxPlots(d, var, binnedVar, outcomeName)

# COMMAND ----------

MakeRegBarChart(d, outcomeName, binnedVar, orderBy=binnedVar, barmode='stack', xtype='category')

# COMMAND ----------

MakeProbBarChart(d, outcomeName, binnedVar + "label", xtype='category', numDecimals=4)

# COMMAND ----------

var = "CRS_Arr_Time"
PlotBasicBoxPlots(full_data_dep, var, outcomeName)

# COMMAND ----------

var = "CRS_Arr_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
              labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

PlotBinnedBoxPlots(d, var, binnedVar, outcomeName)

# COMMAND ----------

MakeRegBarChart(d, outcomeName, binnedVar, orderBy=binnedVar, barmode='stack', xtype='category')

# COMMAND ----------

MakeProbBarChart(d, outcomeName, binnedVar + "label", xtype='category', numDecimals=4)

# COMMAND ----------

var = "CRS_Arr_Time"
binnedVar = var + "_bin"

d = BinValues(full_data_dep, var, 
              splits = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 
                        1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400], 
              labels = ['12am-1am', '1am-2am', '2am-3am', '3am-4am', '4am-5am', '5am-6am', 
                        '6am-7am', '7am-8am', '8am-9am', '9am-10am', '10am-11am', '11am-12pm', 
                        '12pm-1pm', '1pm-2pm', '2pm-3pm', '3pm-4pm', '4pm-5pm', '5pm-6pm', 
                        '6pm-7pm', '7pm-8pm', '8pm-9pm', '9pm-10pm', '10pm-11pm', '11pm-12am'])

PlotBinnedBoxPlots(d, var, binnedVar, outcomeName)

# COMMAND ----------

MakeRegBarChart(d, outcomeName, binnedVar, orderBy=binnedVar, barmode='stack', xtype='category')

# COMMAND ----------

MakeProbBarChart(d, outcomeName, binnedVar + "label", xtype='category', numDecimals=4)

# COMMAND ----------

