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


import plotly.io as pio

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Basic Initial EDA

# COMMAND ----------

full_data_dep.count()

# COMMAND ----------

display(full_data_dep.sample(False, 0.00001))

# COMMAND ----------

# Get number of distinct values for each column in full training dataset
from pyspark.sql.functions import col, countDistinct
display(full_data_dep.agg(*(countDistinct(col(c)).alias(c) for c in full_data_dep.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Group 1 Plots

# COMMAND ----------

# Plot Year and outcome
var = "Year"
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
  barmode='stack', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------

# Plot Month and outcome
var = "Month"
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
  barmode='stack', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------



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
  barmode='group', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var, type='category'),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------

# Plot Day_Of_Month interacted with Month and outcome
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
  barmode='group', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------

# Effectively demonstrates the probability of a departure delay, given the distance group
var = "Distance_Group"
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy(var).toPandas()
display(d)

# COMMAND ----------

# Plot Distance Group and outcome
var = "Distance_Group"
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
  barmode='group', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

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
d = full_data_dep.select(var, outcomeName).groupBy(var, outcomeName).count().orderBy("count").toPandas()

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
  barmode='group', 
  title="Flight Counts by " + var + " & " + outcomeName,
  xaxis=dict(title=var),
  yaxis=dict(title="Number of Flights")
)
fig = go.Figure(data=[t1, t2], layout=l)
fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding Plotly

# COMMAND ----------

fig = {
    "data": [{"type": "bar",
              "x": [1, 2, 3],
              "y": [1, 3, 2]}],
    "layout": {"title": {"text": "A Bar Chart"}}
}

# To display the figure defined by this dict, use the low-level plotly.io.show function
import plotly.io as pio
pio.show(fig)

# COMMAND ----------

import plotly.graph_objects as go
fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(
        title=go.layout.Title(text="A Bar Chart")
    )
)
fig.show()

# COMMAND ----------

