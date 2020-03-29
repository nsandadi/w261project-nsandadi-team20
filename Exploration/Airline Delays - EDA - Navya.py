# Databricks notebook source
# MAGIC %md
# MAGIC # Airline Delays EDA

# COMMAND ----------

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# COMMAND ----------

# Load the data into dataframe
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# COMMAND ----------

# Filter to datset with entries where diverted != 1, cancelled != 1, dep_delay != Null, and arr_delay != Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print(airlines.count())

# Type of dataset
print(type(airlines)) # It is a pyspark dataframe

# COMMAND ----------

# Helper function to split the dataset into train and test (by Diana)
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

# Converting the spark dataframe to a pandas dataframe
mini_train_pd = mini_train.toPandas()

# Print the first few rows of pandas dataframe
mini_train_pd.head()

# COMMAND ----------

train_pd = train[['DAY_OF_WEEK', 'CRS_DEP_TIME', 'DEP_DEL15', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME']].toPandas()
train_pd.head()


# COMMAND ----------

print(train_pd.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model for Prediction Departure Delay 
# MAGIC - Variables to predict Departure Delay (1/0) - `Dep_Del15`
# MAGIC - Inference Time: 6 hours before CRS_Dep_Time
# MAGIC 
# MAGIC ##### Year, Month, Day of week, Day of Month
# MAGIC - Day of Month -- include when we join on holidays
# MAGIC - Year by itself -- continuous variable
# MAGIC - Month by itself -- categorical 
# MAGIC - Day of week -- categorical
# MAGIC 
# MAGIC ##### Unique_Carrer
# MAGIC - categorical
# MAGIC 
# MAGIC ##### Origin-attribute
# MAGIC - categorical
# MAGIC 
# MAGIC ##### Destination-attribute
# MAGIC - categorical
# MAGIC 
# MAGIC ##### CRS_Dep_Time, CRS_Arr_Time
# MAGIC - If continuous: minutes after midnight
# MAGIC - If categorical: groups of 15 minutes, 30 minutes, or 1 hr (binning)
# MAGIC - can use continuous and/or categorical
# MAGIC - Interaction of Day of week with CRS_Dep_Time (by hr)
# MAGIC - Interaction of Day of week with CRS_Arr_Time (by hr) -- might not be useful, but can eval with L1 Norm
# MAGIC 
# MAGIC ##### CRS_Elapsed_Time
# MAGIC - If continuous: minutes after midnight
# MAGIC - If categorical: groups of 15 minutes, 30 minutes, or 1 hr (binning)
# MAGIC - can use continuous and/or categorical
# MAGIC 
# MAGIC ##### Distance & Distance_Group
# MAGIC - experiment with using either or
# MAGIC - have both categorical & continuous depending on which we want to use
# MAGIC 
# MAGIC ##### Outcome: Boolean(`Dep_Delay > 15` === `Dep_Del15 = 1`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA Tasks
# MAGIC 
# MAGIC ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group', 'Op_Unique_Carrier', 'Origin', 'Dest']
# MAGIC Outcome = Dep_Del15
# MAGIC 
# MAGIC #### Groups:
# MAGIC - 1: Shobha
# MAGIC - 2: Navya
# MAGIC - 3: Diana
# MAGIC - 4: Shaji
# MAGIC - 5: EC -- whoever gets to it first
# MAGIC 
# MAGIC #### Group 1:
# MAGIC Bar Plot of counts for each outcome vs. feature (or interaction of features)
# MAGIC * Year vs. Dep_Del15
# MAGIC * Month vs. Dep_Del15
# MAGIC * Day_Of_Month interacted with Month vs. Dep_Del15
# MAGIC * Day_of_week vs. Dep_Del15
# MAGIC * Distance_Group vs. Dep_Del15
# MAGIC * Op_Unique_Carrier vs. Dep_Del15
# MAGIC 
# MAGIC #### Group 2:
# MAGIC Bar Plot of counts for each outcome vs. feature (or interaction of features)
# MAGIC * Day_of_week interacted with binned CRS_DepTime  vs. Dep_Del15
# MAGIC * Day_of_week interacted with binned CRS_ArrTime  vs. Dep_Del15
# MAGIC * Day_of_week interacted with binned CRS_ElapsedTime  vs. Dep_Del15
# MAGIC 
# MAGIC #### Group 3:
# MAGIC Bar Plot of counts of lights on y axis, probability of departure delay on x axis for a carrier, having each bar be a carrier
# MAGIC * Op_Unique_Carrier vs. Dep_Del15
# MAGIC * Origin vs. Dep_Del15
# MAGIC * Dest vs. Dep_Del15
# MAGIC * Distance_Group vs. Dep_Del15
# MAGIC 
# MAGIC #### Group 4:
# MAGIC Plotting Distribution of continuous vars vs. outcome vars (histograms, bar plots, box plots, etc)
# MAGIC * CRS_Dep_Time vs. Dep_Del15
# MAGIC * CRS_Arr_Time vs. Dep_Del15
# MAGIC * CRS_Elapsed_time vs. Dep_Del15
# MAGIC * Distance vs. Dep_Del15
# MAGIC From these plots, understand reasonable binning (to suggest splits to the models via Bucketizer)
# MAGIC 
# MAGIC #### Group 5:
# MAGIC For arrival delay, have additional variables:
# MAGIC Dep_Time, Dep_Delay, Taxi_out, Carrier_Delay, Late_Aircraft_Delay (plane arrived late for flight), Security_Delay

# COMMAND ----------

# MAGIC %md
# MAGIC ### Group 2 - Mini Train Set

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) Day_of_week interacted with binned CRS_DepTime vs. Dep_Del15

# COMMAND ----------

dayOfWeek_depTime_df = mini_train_pd[['DAY_OF_WEEK', 'CRS_DEP_TIME', 'DEP_DEL15']]

dayOfWeek_depTime_df['crs_dep_binned'] = pd.cut(x = dayOfWeek_depTime_df['CRS_DEP_TIME'], bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

dayOfWeek_depTime_df['dayOfWeek_binned'] = pd.cut(x = dayOfWeek_depTime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dow_crsdep_binned = dayOfWeek_depTime_df[['crs_dep_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_dep_binned', 'dayOfWeek_binned'], as_index = False).count()
print(dow_crsdep_binned.shape)

# Databricks bar chart
display(dow_crsdep_binned)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) Day_of_week interacted with binned CRS_ArrTime vs. Dep_Del15

# COMMAND ----------

dayOfWeek_arrTime_df = mini_train_pd[['DAY_OF_WEEK', 'CRS_ARR_TIME', 'DEP_DEL15']]

dayOfWeek_arrTime_df['crs_arr_binned'] = pd.cut(x = dayOfWeek_arrTime_df['CRS_ARR_TIME'],  bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

dayOfWeek_arrTime_df['dayOfWeek_binned'] = pd.cut(x = dayOfWeek_arrTime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dow_crsarr_binned = dayOfWeek_arrTime_df[['crs_arr_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_arr_binned', 'dayOfWeek_binned'], as_index = False).count()
print(dow_crsarr_binned.shape)

# Databricks barchart
display(dow_crsarr_binned)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) Day_of_week interacted with binned CRS_Elapsed_Time vs. Dep_Del15

# COMMAND ----------

dayOfWeek_elapTime_df = mini_train_pd[['DAY_OF_WEEK', 'CRS_ELAPSED_TIME', 'DEP_DEL15']]
print(dayOfWeek_elapTime_df.describe())

# COMMAND ----------

dayOfWeek_elapTime_df['crs_elap_binned'] = pd.cut(x = dayOfWeek_elapTime_df['CRS_ELAPSED_TIME'], bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720], labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'])

dayOfWeek_elapTime_df['dayOfWeek_binned'] = pd.cut(x = dayOfWeek_elapTime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dayOfWeek_crselap_binned = dayOfWeek_elapTime_df[['crs_elap_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_elap_binned', 'dayOfWeek_binned'], as_index = False).count()
print(dow_crselap_binned.shape)

# Databricks bar chart
display(dow_crselap_binned)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Group 2 - Full Training Set

# COMMAND ----------

# MAGIC %md
# MAGIC #### EDA using Plotly

# COMMAND ----------

# Install plotly
%sh 
pip install plotly --upgrade

# COMMAND ----------

import plotly
plotly.__version__

# COMMAND ----------

import pandas as pd
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime as dt

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) Day_of_week interacted with binned CRS_DepTime vs. Dep_Del15

# COMMAND ----------

dow_dtime_df = train_pd[['DAY_OF_WEEK', 'CRS_DEP_TIME', 'DEP_DEL15']]
print(dow_dtime_df.shape)

dow_dtime_df['crs_dep_binned'] = pd.cut(x = dow_dtime_df['CRS_DEP_TIME'], bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

dow_dtime_df['dayOfWeek_binned'] = pd.cut(x = dow_dtime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dow_dtime_binned_df = dow_dtime_df[['crs_dep_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_dep_binned', 'dayOfWeek_binned'], as_index = False).count()
# dow_dtime_binned_df

# Databricks barchart
# display(dow_dtime_binned_df)

# COMMAND ----------

import plotly.offline as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Monday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Monday'
    
)
trace2 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Tuesday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Tuesday'
)

trace3 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Wednesday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Wednesday'
    
)
trace4 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Thursday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Thursday'
)

trace5 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Friday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Friday'
    
)

trace6 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Saturday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Saturday'
)

trace7 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_dtime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Sunday'].groupby('crs_dep_binned').sum()['DEP_DEL15'].tolist(),
    name='Sunday'
)

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    barmode='group'  #'stack' for stacked bars
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) Day_of_week interacted with binned CRS_ArrTime vs. Dep_Del15

# COMMAND ----------

dow_atime_df = train_pd[['DAY_OF_WEEK', 'CRS_ARR_TIME', 'DEP_DEL15']]
print(dow_atime_df.shape)

dow_atime_df['crs_arr_binned'] = pd.cut(x = dow_atime_df['CRS_ARR_TIME'], bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])

dow_atime_df['dayOfWeek_binned'] = pd.cut(x = dow_atime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dow_atime_binned_df = dow_atime_df[['crs_arr_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_arr_binned', 'dayOfWeek_binned'], as_index = False).count()
# dow_atime_binned_df

# Databricks barchart
# display(dow_dtime_binned_df)

# COMMAND ----------

import plotly.offline as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Monday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Monday'
    
)
trace2 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Tuesday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Tuesday'
)

trace3 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Wednesday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Wednesday'
    
)
trace4 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Thursday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Thursday'
)

trace5 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Friday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Friday'
    
)

trace6 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Saturday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Saturday'
)

trace7 = go.Bar(
    x=['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'],
    y= dow_atime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Sunday'].groupby('crs_arr_binned').sum()['DEP_DEL15'].tolist(),
    name='Sunday'
)

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    barmode='group'  #'stack' for stacked bars
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) Day_of_week interacted with binned CRS_Elapsed_Time vs. Dep_Del15

# COMMAND ----------

# Summary statistics
dow_elapTime_df = train_pd[['DAY_OF_WEEK', 'CRS_ELAPSED_TIME', 'DEP_DEL15']]
print(dow_elapTime_df.describe())

# COMMAND ----------

dow_elapTime_df['crs_elap_binned'] = pd.cut(x = dow_elapTime_df['CRS_ELAPSED_TIME'], bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720], labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'])

dow_elapTime_df['dayOfWeek_binned'] = pd.cut(x = dow_elapTime_df['DAY_OF_WEEK'], bins = [0, 1, 2, 3, 4, 5 , 6, 7], labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

dow_elapTime_binned_df = dow_elapTime_df[['crs_elap_binned', 'dayOfWeek_binned', 'DEP_DEL15']].groupby(['crs_elap_binned', 'dayOfWeek_binned'], as_index = False).count()
# dow_elapTime_binned_df

# Databricks bar chart
# display(dow_crselap_binned)

# COMMAND ----------

import plotly.offline as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Monday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Monday'
    
)
trace2 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Tuesday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Tuesday'
)

trace3 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Wednesday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Wednesday'
    
)
trace4 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Thursday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Thursday'
)

trace5 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Friday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Friday'
    
)

trace6 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Saturday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Saturday'
)

trace7 = go.Bar(
    x=['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12 hours'],
    y= dow_elapTime_binned_df[dow_dtime_binned_df['dayOfWeek_binned']=='Sunday'].groupby('crs_elap_binned').sum()['DEP_DEL15'].tolist(),
    name='Sunday'
)

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(
    barmode='group'  #'stack' for stacked bars
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plotly Examples

# COMMAND ----------

# Basic Bar Chart
import plotly.offline as py
import plotly.graph_objs as go

data = [go.Bar(
            x=['giraffes', 'orangutans', 'monkeys'],
            y=[20, 14, 23]
    )]

py.iplot(data)

# COMMAND ----------

# Grouped bar chart
import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[20, 14, 23],
    name='SF Zoo'
)
trace2 = go.Bar(
    x=['giraffes', 'orangutans', 'monkeys'],
    y=[12, 18, 29],
    name='LA Zoo'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group' #'stack' for stacked bar chart
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')

# COMMAND ----------

# username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# userhome = 'dbfs:/user/' + username
# print(userhome)
# scratch_path = userhome + "/scratch/" 
# scratch_path_open = '/dbfs' + scratch_path.split(':')[-1] # for use with python open()
# dbutils.fs.mkdirs(scratch_path)
# scratch_path

# COMMAND ----------

# Shaji's code
flights_df_raw = airlines.groupBy("OP_UNIQUE_CARRIER").count().orderBy(["count"], ascending=[0])
airline_codes_df = spark.read.option("header", "true").csv("dbfs:/user/shajikk@ischool.berkeley.edu/scratch/" + 'airlines.csv')
flights_df = flights_df_raw.join(airline_codes_df, flights_df_raw["OP_UNIQUE_CARRIER"] == airline_codes_df["Code"] ).toPandas()

fig = go.Figure()

color = [i for i in range(flights_df.shape[0])]
cmax = len(color)

y = flights_df['count'].to_list()
x = flights_df['Description'].to_list()

fig.add_trace(go.Bar(
    x=x,
    y=y,
    name='Crap',
    marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
    #colorscale="Viridis"
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=45)
#fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.update_layout(barmode='stack', 
                  xaxis=dict(categoryorder='total descending', title='Months'), 
                  title=go.layout.Title(text="Flight counts for the dataset"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

