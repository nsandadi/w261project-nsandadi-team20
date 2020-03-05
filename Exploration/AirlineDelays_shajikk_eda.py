# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC ~140GB

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

# MAGIC %md
# MAGIC ### Read the data, convert into dataframe

# COMMAND ----------

DATAPATH = '/databricks-datasets/airlines'
all_files = dbutils.fs.ls(DATAPATH)
chunks = [c.path for c in all_files if "part" in c.path]
print("Number of parquet = {}".format(len(chunks)))

airlines0 = spark.read.option("header", "true").csv(chunks[0])
airlines_rest = spark.read.option("header", "false").csv(chunks[1:])
airlines = airlines0.union(airlines_rest)

# COMMAND ----------

print(airlines.columns)
airlines.printSchema()

# COMMAND ----------

airlines.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %timeit
# MAGIC airlines.select('Year').distinct().collect()

# COMMAND ----------

# MAGIC %timeit
# MAGIC airlines.groupBy("Year").count().show()
# MAGIC airlines.groupBy("Month").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Subset the 2007 data, save the data and re-load

# COMMAND ----------

airlines_2007 = airlines.filter('Year == 2007')
# Save the df3 DataFrame in Parquet format
airlines_2007.write.parquet(scratch_path + '/airlines_2007.parquet', mode='overwrite')

# COMMAND ----------

dbutils.fs.ls(scratch_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### List the Carriers by number of flights operated.

# COMMAND ----------

airlines_2007 = spark.read.parquet(scratch_path + '/airlines_2007.parquet').cache()
airlines_2007.limit(2).toPandas()

# COMMAND ----------

flights_df = airlines_2007.groupBy("UniqueCarrier").count().orderBy(["count"], ascending=[0]).toPandas()
flights_df

# COMMAND ----------

from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np


# Instead of simply calling plot(...), store your plot as a variable and pass it to displayHTML().
# Make sure to specify output_type='div' as a keyword argument.
# (Note that if you call displayHTML() multiple times in the same cell, only the last will take effect.)

color = [i for i in range(flights_df.shape[0])]
cmax = len(color)

trace1 = Histogram(
            histfunc = "sum",
            x=flights_df['UniqueCarrier'].to_list(), 
            y=flights_df['count'].to_list(),
            marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
)


layout = Layout(
    title='Flight counts for 2007',
    xaxis=dict(
        title='Airlines'
    ),
    yaxis=dict(
        title='Number of flights'
    ),
    bargap=0.2,
    bargroupgap=0.1,
    autosize=True
)

p = plot({
    'data': [trace1],
    'layout': layout
  },
  output_type='div'
)


displayHTML(p)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Kernel density estimation for flight carriers (Y2007)

# COMMAND ----------

df_sample = airlines_2007.sample(False, 0.0005, 1).toPandas()

# COMMAND ----------

from scipy.stats import gaussian_kde

min_value = -300
max_value = 300
band_width = 2

def process_data(key, df): 
    ll = df['ArrDelay'].astype(int).to_list()
    kde = gaussian_kde(list(ll), bw_method=band_width)
    # Evenly space x values
    x = np.linspace(min_value, max_value, 100)
    # Evaluate pdf at every value of x
    y = kde.pdf(x)

    return pd.DataFrame({
                         'xs': x, 
                         'ys': y, 
                        })       


test = (df_sample
.loc[df_sample.ArrDelay != 'NA',]     
.loc[:,["UniqueCarrier", "ArrDelay"]]
)

test1 = test.loc[(test.ArrDelay.astype(int) > min_value) & (test.ArrDelay.astype(int) < max_value)]

test2 = (test1
.groupby('UniqueCarrier')
.apply(lambda x: process_data(x.name, x))
.reset_index()
.drop(columns=['level_1'])
)

test2.head(5)

# COMMAND ----------

kde_scatter = []

def test_fn(x):
    d =  pd.DataFrame({
                       'xs': [x['xs'].to_list()], 
                       'ys': [x['ys'].to_list()],
                       'UniqueCarrier' : x['UniqueCarrier'].iloc[0]
                        })
    
    return d    
  

test3 = (test2
.groupby('UniqueCarrier')
.apply(lambda x: test_fn(x))
)
    
for index, row in test3.iterrows():
    kde_scatter.append(
        Scatter(
        x=row['xs'],
        y=row['ys'],
        name=row['UniqueCarrier']
        )
    )

# COMMAND ----------

layout = Layout(
    xaxis=dict(
    range=[-300, 300],
    title='Airline delays'
  ),
  title='Kernel density estimation for flight carriers (Y2007)',
  yaxis=dict(
      title='Probablity'
  ),
)
  
p = plot({
    'data': kde_scatter,
    'layout': layout
  },
  output_type='div'
)


displayHTML(p)

# COMMAND ----------

