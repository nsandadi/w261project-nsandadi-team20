# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC 2015 - 2019

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Additioinal sources
# MAGIC This might be useful in matching station codes to airports:
# MAGIC 
# MAGIC http://dss.ucar.edu/datasets/ds353.4/inventories/station-list.html  
# MAGIC https://www.world-airport-codes.com/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up user directories

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
scratch_path = userhome + "/scratch/" 
scratch_path_open = '/dbfs' + scratch_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(scratch_path)
scratch_path

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install libraries (Plotly)

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install plotly --upgrade

# COMMAND ----------

import pandas as pd
import numpy as np
from plotly.offline import plot
#from plotly.graph_objs import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
airlines_schema = StructType([ StructField('YEAR',ShortType(),True),
                      StructField('QUARTER',ShortType(),True),
                      StructField('MONTH',ShortType(),True),
                      StructField('DAY_OF_MONTH',ShortType(),True),
                      StructField('DAY_OF_WEEK',ShortType(),True),
                      StructField('FL_DATE',DateType(),True),
                      StructField('OP_CARRIER_AIRLINE_ID',ShortType(),True),
                      StructField('ORIGIN_AIRPORT_ID',ShortType(),True),
                      StructField('ORIGIN',StringType(),True),
                      StructField('ORIGIN_CITY_NAME',StringType(),True),
                      StructField('ORIGIN_STATE_ABR',StringType(),True),
                      StructField('ORIGIN_STATE_FIPS',ShortType(),True),
                      StructField('ORIGIN_STATE_NM',StringType(),True),
                      StructField('DEST_AIRPORT_ID',IntegerType(),True),
                      StructField('DEST_AIRPORT_SEQ_ID',IntegerType(),True),
                      StructField('DEST',StringType(),True),
                      StructField('DEST_CITY_NAME',StringType(),True),
                      StructField('DEST_STATE_ABR',StringType(),True),
                      StructField('DEST_STATE_FIPS',ShortType(),True),
                      StructField('DEST_STATE_NM',StringType(),True),
                      StructField('CRS_DEP_TIME',StringType(),True),
                      StructField('DEP_TIME',StringType(),True),
                      StructField('DEP_DELAY',IntegerType(),True),
                      StructField('DEP_DELAY_NEW',IntegerType(),True),
                      StructField('DEP_DEL15',IntegerType(),True),
                      StructField('DEP_DELAY_GROUP',IntegerType(),True),
                      StructField('DEP_TIME_BLK',StringType(),True),
                      StructField('CRS_ARR_TIME',StringType(),True),
                      StructField('ARR_TIME',StringType(),True),
                      StructField('ARR_DELAY',IntegerType(),True),
                      StructField('ARR_DELAY_NEW',IntegerType(),True),
                      StructField('ARR_DEL15',IntegerType(),True),
                      StructField('ARR_DELAY_GROUP',IntegerType(),True),
                      StructField('ARR_TIME_BLK',StringType(),True),
                      StructField('CANCELLED',BooleanType(),True),
                      StructField('DIVERTED',BooleanType(),True),
                      StructField('CRS_ELAPSED_TIME',IntegerType(),True),
                      StructField('ACTUAL_ELAPSED_TIME',IntegerType(),True),
                      StructField('FLIGHTS',ShortType(),True),
                      StructField('DISTANCE',IntegerType(),True),
                      StructField('DISTANCE_GROUP',ShortType(),True)
                    ])

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data"))

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*a.parquet")
display(airlines.sample(False, 0.00001))

# COMMAND ----------

display(airlines.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### List the Carriers by number of flights operated.

# COMMAND ----------

flights_df_raw = airlines.groupBy("OP_CARRIER_AIRLINE_ID").count().orderBy(["count"], ascending=[0])
airline_codes_df = spark.read.option("header", "true").csv(scratch_path + '/airline_data/OP_CARRIER_AIRLINE_ID.csv')
flights_df_mod = flights_df_raw.join(airline_codes_df, flights_df_raw["OP_CARRIER_AIRLINE_ID"] == airline_codes_df["Code"] )
flights_df = (flights_df_mod.toPandas().assign(
                 Description=  (lambda d : list(map(lambda x: x.split(':')[0], list(d.Description)))))
             )

# COMMAND ----------

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

# MAGIC %md
# MAGIC 
# MAGIC ### Kernel density estimation for flight carriers

# COMMAND ----------

df_sample_raw = airlines.sample(False, 0.01, 1)
df_sample = (df_sample_raw.join(airline_codes_df, df_sample_raw["OP_CARRIER_AIRLINE_ID"] == airline_codes_df["Code"] )
             .toPandas()
             .assign(Description = (lambda d : list(map(lambda x: x.split(':')[0], list(d.Description)))))      
            )

display(df_sample.head(2))

# COMMAND ----------

from scipy.stats import gaussian_kde

min_value = -300
max_value = 300
band_width = 2

# Main function to generate KDE
def gen_kde_df (test) :

    def process_data(key, df): 
        ll = df['DELAY'].astype(float).to_list()
        kde = gaussian_kde(list(ll), bw_method=band_width)
        # Evenly space x values
        x = np.linspace(min_value, max_value, 100)
        # Evaluate pdf at every value of x
        y = kde.pdf(x)
        return pd.DataFrame({'xs': x, 'ys': y})       

    test2 = (
     (
      test
      .loc[(test.DELAY.astype(float) > min_value) & (test.DELAY.astype(float) < max_value)]
     )
    .groupby('Description')
    .apply(lambda x: process_data(x.name, x))
    .reset_index()
    .drop(columns=['level_1'])
    )

    def test_fn(x):
        return pd.DataFrame({
                           'xs': [x['xs'].to_list()], 
                           'ys': [x['ys'].to_list()],
                           'Description' : x['Description'].iloc[0]
                            })  

    test3 = (test2
    .groupby('Description')
    .apply(lambda x: test_fn(x))
    )
    
    return test3
# ++++++++++++++++++ Subplot 
fig = make_subplots(rows=1, cols=2, subplot_titles=("Arrival delay", "Departure delay"))



# ++++++++++++++++++ Arrival delay
kde_df_arrival = (df_sample
.loc[df_sample.ARR_DELAY != 'NA',]     
.loc[:,["Description", "ARR_DELAY"]]
.rename(columns = {'ARR_DELAY':'DELAY'})
)

# Call Main function to generate KDE
kde_df_arrival = gen_kde_df(kde_df_arrival)

color = [i for i in range(kde_df_arrival.shape[0])]
cmax = len(color)


#fig1 = go.Figure()

for i, (index, row) in enumerate(kde_df_arrival.iterrows()):
    fig.add_trace(go.Scatter(
      mode = 'lines',
      line_shape='spline',
      x = row['xs'],
      y = row['ys'],
      #hovertext=df_holidays.holiday.to_list(),
      #showlegend = False,
      name = row['Description'],
      #marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
      marker_color=px.colors.qualitative.Dark24[i]

    ),     
    row=1, col=1)

# ++++++++++++++++++ Departure delay

kde_df_dep = (df_sample
.loc[df_sample.DEP_DELAY != 'NA',]     
.loc[:,["Description", "DEP_DELAY"]]
.rename(columns = {'DEP_DELAY':'DELAY'})
)

# Call Main function to generate KDE
kde_df_dep = gen_kde_df(kde_df_dep)
   
# fig2 = go.Figure()

for i, (index, row) in enumerate(kde_df_dep.iterrows()):
    fig.add_trace(go.Scatter(
      mode = 'lines',
      line_shape='spline',
      x = row['xs'],
      y = row['ys'],
      #hovertext=df_holidays.holiday.to_list(),
      #showlegend = False,
      name = row['Description'], 
      marker_color=px.colors.qualitative.Dark24[i]

    ),
    row=1, col=2)

fig.update_yaxes(title_text="Probablity of delay", showgrid=False, row=1, col=1)
fig.update_yaxes(title_text="Probablity of delay", showgrid=False, row=1, col=2)
fig.update_yaxes(title_text="Actual delay", showgrid=False, row=1, col=1)
fig.update_yaxes(title_text="Actual delay", showgrid=False, row=1, col=2)

fig.update_layout(height=800, width=1200, title_text="KDE ")
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Distribution of delays

# COMMAND ----------

import calendar
df = (airlines.groupBy("YEAR","MONTH").count().toPandas()
      .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
      .assign(month_name = (lambda d : list(map(lambda x: calendar.month_abbr[x], d.MONTH))))
     )

# COMMAND ----------

fig = go.Figure()


for i, (yr, g) in enumerate(df.groupby(['YEAR'])):
    y = g['count'].to_list()
    x = g.month_name
    marker_color=px.colors.sequential.Rainbow[i]  
  
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name=yr,
        marker_color=marker_color
    ))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45)
#fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.update_layout(barmode='stack', 
                  xaxis=dict(categoryorder='total descending', title='Months'), 
                  title=go.layout.Title(text="A Bar Chart"),  
                  yaxis=dict(title='stacked Year with month'))
fig.show()

# COMMAND ----------


fig = go.Figure()


# for i, (q, g) in enumerate(df.groupby(['quarter'])):
#   print(q)
#   print(g.groupby(['YEAR']).agg(["sum"]))

for i, (q, g) in enumerate(df.groupby(['quarter'])):
    t = g.groupby(['YEAR']).agg(["sum"])
                                
    y = t['count']['sum'].to_list()
    x = list(t.index)
    marker_color=px.colors.sequential.RdBu[i]  
  
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name='Quarter ' + str(q),
        marker_color=marker_color
    ))

fig.update_layout(barmode='group', xaxis_tickangle=-45,   height=600, width=600)
fig.update_layout(barmode='stack', 
                  xaxis=dict(categoryorder='total descending', title='Months'), 
                  title=go.layout.Title(text="A Bar Chart"),  
                  yaxis=dict(title='stacked Year with month'))
fig.show()


# COMMAND ----------

import datetime
import re 
date = datetime.datetime(2018,1,1)
all_dates = list()
h  = list()

Holidays = dict()

Holidays['01-01'] = "New year"
Holidays['01-02'] = "After New year"
Holidays['01-15'] = "MLK day"
Holidays['02-19'] = "President's day"
Holidays['05-13'] = "Mother's day"
Holidays['05-28'] = "Memorial day"
Holidays['06-17'] = "Father's day"
Holidays['07-04'] = "Independence day"
Holidays['09-03'] = "Labor day"
Holidays['10-08'] = "Columbus day"
Holidays['11-12'] = "Veterans day"
Holidays['11-22'] = "Thanksgiving"
Holidays['11-13'] = "Thanksgiving"
Holidays['12-24'] = "Christmas eve"
Holidays['12-25'] = "Christmas"
Holidays['12-26'] = "Dec holiday"
Holidays['12-28'] = "Christmas"
Holidays['12-29'] = "Dec holiday"
Holidays['12-30'] = "Dec holiday"
Holidays['12-31'] = "New year's eve"

def chop_date(d) :
  word = str(d)
  res = re.split(' |-', word)
  return res[1]+'-'+res[2]
  

def check_holiday(d) :
  word = str(d)
  res = re.split(' |-', word)
  d = chop_date(d)
  if d in Holidays:
    return(Holidays[d])
  return "NA"

for i in range(365): 
    h.append(check_holiday(date))
    all_dates.append(date)
    date += datetime.timedelta(days=1)

val = [1] * 365
day_of_year = [i+1 for i in range(0, 365)]

data = np.array([all_dates, h, val, day_of_year])
df = pd.DataFrame(data=data.T, index=np.array(range(0, 365)), columns = ["day", "holiday", "val", "day_of_year"])

df.join(departure_delay_df.set_index('day_of_year'), on='day_of_year')
df.day_of_year = df.day_of_year.astype(int)
departure_delay_df.day_of_year = departure_delay_df.day_of_year.astype(int)
df = pd.merge(departure_delay_df, df, on='day_of_year', how='outer', indicator=True)
df.rename(columns={'avg(DEP_DELAY)':'delay'}, inplace=True)

# COMMAND ----------

import plotly.graph_objects as go

df = df.sort_values("day_of_year")
df_holidays = df.loc[df.holiday != 'NA',].sort_values("day_of_year")
df_regular = df.loc[df.holiday == 'NA',].sort_values("day_of_year")

fig = go.Figure()


fig.add_trace(go.Scatter(
  mode = 'markers',
  x = df_holidays.day.to_list(),
  y = df_holidays.delay.to_list(),
  hovertext=df_holidays.holiday.to_list(),
  marker = dict(
   color = 'rgb(17, 157, 255)',
   size = 12,
   line = dict(
     color = 'rgb(231, 99, 250)',
     width = 2
   )
  ),
  showlegend = False,
  name = '', 
  
))

fig.add_trace(go.Scatter(
  mode = 'markers',
  x = df_regular.day.to_list(),
  y = df_regular.delay.to_list(),
  marker = dict(
   color = 'rgb(17, 157, 255)',
   size = 3,
  ),
  showlegend = False,
  name = ''
))

fig.add_trace(go.Scatter(
  mode = 'lines',
  line_shape='spline',
  x = df.day.to_list(),
  y = df.delay.to_list(),
  hovertext=df_holidays.holiday.to_list(),
  #showlegend = False,
  #name = '', 
  
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45)
#fig.update_traces(mode='lines+markers')
fig.update_layout( 
                  xaxis=dict(categoryorder='total descending', title='Day of the year'), 
                  title=go.layout.Title(text="Airline delay overlyed on holidays"),  
                  yaxis=dict(title='Airline delays'))
fig.show()

# COMMAND ----------

#airlines_2017.limit(2).toPandas()

# COMMAND ----------

from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import col
from pyspark.sql.functions import date_format

departure_delay = (airlines
       .withColumn("FL_DATE", to_timestamp(col("FL_DATE")))
       .withColumn("day_of_year", date_format(col("FL_DATE"), "D"))
       .filter('DEP_DELAY > 0')
       .groupBy("day_of_year")
       .agg({"DEP_DELAY":"avg"})
       .orderBy(["avg(DEP_DELAY)"], ascending=[1])
      )

#tmp = df1.filter('DEP_DELAY > 0').groupBy("day_of_year").agg({"DEP_DELAY":"avg"})

# COMMAND ----------

departure_delay_df = departure_delay.toPandas()

# COMMAND ----------

#departure_delay_df

# COMMAND ----------

df = spark.createDataFrame([(1, 'John', 1.79, 28,'M', 'Doctor'),
                        (2, 'Steve', 1.78, 45,'M', None),
                        (3, 'Emma', 1.75, None, None, None),
                        (4, 'Ashley',1.6, 33,'F', 'Analyst'),
                        (5, 'Olivia', 1.8, 54,'F', 'Teacher'),
                        (6, 'Hannah', 1.82, None, 'F', None),
                        (7, 'William', 1.7, 42,'M', 'Engineer'),
                        (None,None,None,None,None,None),
                        (8,'Ethan',1.55,38,'M','Doctor'),
                        (9,'Hannah',1.65,None,'F','Doctor')]
                       , ['Id', 'Name', 'Height', 'Age', 'Gender', 'Profession'])

df.groupBy("Profession").agg({"Age":"avg"}).show()


# COMMAND ----------

