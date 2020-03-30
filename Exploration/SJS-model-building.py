# Databricks notebook source
display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data"))

# COMMAND ----------

# read the parquet files into airlines as parquet
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# COMMAND ----------

# Filter to datset with entries where diverted != 1, cancelled != 1, dep_delay != Null, and arr_delay != Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print(airlines.count())

# COMMAND ----------

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

display(train.groupBy('DEP_DEL15').count())

# COMMAND ----------

# Prep Datasets for all 4 Models
mini_train_lr, train_lr, val_lr, test_lr = SplitDataset("lr") # For Shobha
mini_train_nb, train_nb, val_nb, test_nb = SplitDataset("nb") # For Navya
mini_train_dt, train_dt, val_dt, test_dt = SplitDataset("dt") # For Diana
mini_train_svm, train_svm, val_svm, test_svm = SplitDataset("svm") # For Shaji

# COMMAND ----------

type(mini_train_lr)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer


# COMMAND ----------

# Convert to rdd (for now, might not use)
mini_train_rdd = mini_train_lr.rdd
val_rdd = val_lr.rdd

# COMMAND ----------

indexers = [StringIndexer(inputCol=column, outputCol=column+"_INDEX") for column in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST'] ]
assembler = VectorAssembler(inputCols = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP'], outputCol = "features")

lr = LogisticRegression(featuresCol="features", labelCol="DEP_DEL15", maxIter=100, regParam=0.1, elasticNetParam=1)
# elasticNetParam: For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.

pipeline = Pipeline(stages=indexers + [assembler, lr])

# Fit the model
lr_model = pipeline.fit(mini_train)

# COMMAND ----------

predictions = lr_model.transform(val)

# COMMAND ----------

display(predictions.select("DEP_DEL15", "prediction", "probability"))


# COMMAND ----------

# Model Evaluation

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Model Evaluation")
print("----------------")

# Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: ", accuracy)

# Recall
evaluator = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction",
                                              metricName="weightedRecall")
recall = evaluator.evaluate(predictions)
print("Recall: ", recall)

# Precision
evaluator = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction",
                                              metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)
print("Precision: ", precision)

# F1
evaluator = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", predictionCol="prediction",
                                              metricName="f1")
f1 = evaluator.evaluate(predictions)
print("F1: ", f1)



# COMMAND ----------

from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
indexers = [StringIndexer(inputCol=column, outputCol=column+"_INDEX") for column in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST'] ]
assembler = VectorAssembler(inputCols = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK','CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP'], outputCol = "features")

lr = LogisticRegressionWithLBFGS(featuresCol="features", labelCol="DEP_DEL15", maxIter=100, regParam=0.1, elasticNetParam=0, regType='l2')
# elasticNetParam: For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.

pipeline = Pipeline(stages=indexers + [assembler, lr])

# Fit the model
lr_model = pipeline.fit(mini_train)

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
scratch_path = userhome + "/scratch/" 
scratch_path_open = '/dbfs' + scratch_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(scratch_path)
scratch_path

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

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data"))

# COMMAND ----------


display(airlines.sample(False, 0.00001))

# COMMAND ----------

flights_df_raw = airlines.groupBy("OP_UNIQUE_CARRIER").count().orderBy(["count"], ascending=[0])
airline_codes_df = spark.read.option("header", "true").csv("dbfs:/user/shajikk@ischool.berkeley.edu/scratch/" + 'airlines.csv')
flights_df = flights_df_raw.join(airline_codes_df, flights_df_raw["OP_UNIQUE_CARRIER"] == airline_codes_df["Code"] ).toPandas()

# COMMAND ----------

import calendar
df = (airlines.groupBy("YEAR","MONTH").count().toPandas()
      .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
      .assign(month_name = (lambda d : list(map(lambda x: calendar.month_abbr[x], d.MONTH))))
     )


# COMMAND ----------

display(df.head(50))

# COMMAND ----------

import calendar
df = (airlines.groupBy("YEAR","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.DEP_DEL15))))
#       .assign(month_name = (lambda d : list(map(lambda x: calendar.month_abbr[x], d.DEP_DEL15))))
     )

# COMMAND ----------

display(df.head(50))

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['YEAR'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

fig.add_trace(go.Bar(
    x=x,
    y=y,
    name='Crap',
    marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
    #colorscale="Viridis"
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
# fig.update_layout(barmode='group', xaxis_tickangle=45)
#fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.update_layout(barmode='group', 
                  xaxis=dict(categoryorder='total descending', title='YEAR'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

#Bar Plot of counts for each outcome vs. feature (or interaction of features)
# * Year vs. Dep_Del15
# * Month vs. Dep_Del15
# * Day_Of_Month interacted with Month vs. Dep_Del15
# * Day_of_week vs. Dep_Del15
# * Distance_Group vs. Dep_Del15
# * Op_Unique_Carrier vs. Dep_Del15

# COMMAND ----------

df = (airlines.groupBy("MONTH","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.DEP_DEL15))))
      .assign(month_name = (lambda d : list(map(lambda x: calendar.month_abbr[x], d.MONTH))))
     )


# COMMAND ----------

df.shape

# COMMAND ----------

display(df.head(50))

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['MONTH'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

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
                  xaxis=dict(categoryorder='total descending', title='MONTH'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

df = (airlines.groupBy("DAY_OF_WEEK","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
      .assign(week_day = (lambda d : list(map(lambda x: calendar.day_abbr[x-1], d.DAY_OF_WEEK))))
     )

# COMMAND ----------

df.shape

# COMMAND ----------

display(df.head(14))

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['DAY_OF_WEEK'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

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
                  xaxis=dict(categoryorder='total descending', title='DAY_OF_WEEK'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

df = (airlines.groupBy("DISTANCE_GROUP","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
#       .assign(week_day = (lambda d : list(map(lambda x: calendar.day_abbr[x-1], d.DAY_OF_WEEK))))
     )

# COMMAND ----------

df.shape

# COMMAND ----------

display(df.head(22))

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['DISTANCE_GROUP'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

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
                  xaxis=dict(categoryorder='total descending', title='DISTANCE_GROUP'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

df = (airlines.groupBy("OP_UNIQUE_CARRIER","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
#       .assign(week_day = (lambda d : list(map(lambda x: calendar.day_abbr[x-1], d.DAY_OF_WEEK))))
     )

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['OP_UNIQUE_CARRIER'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

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
                  xaxis=dict(categoryorder='total descending', title='OP_UNIQUE_CARRIER'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

airlines_1 = airlines.where('DEP_DEL15 == 1')
df = (airlines_1.groupBy("DAY_OF_MONTH","DEP_DEL15").count().orderBy(["DEP_DEL15"], ascending=[0]).toPandas()
#       .assign(quarter = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
#       .assign(week_day = (lambda d : list(map(lambda x: calendar.day_abbr[x-1], d.DAY_OF_WEEK))))
     )

# COMMAND ----------

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['DAY_OF_MONTH'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

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
                  xaxis=dict(categoryorder='total descending', title='DAY_OF_MONTH'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

airlines_1 = airlines.where('DEP_DEL15 == 1')
df = (airlines_1.groupBy("MONTH","DAY_OF_MONTH","DEP_DEL15").count().orderBy(["MONTH"], ascending=[1]).orderBy(["DAY_OF_MONTH"], ascending=[1]).toPandas()
      .assign(day = (lambda d: list(map(lambda x: (x-1) // 3 + 1, d.MONTH))))
#       .assign(week_day = (lambda d : list(map(lambda x: calendar.day_abbr[x-1], d.DAY_OF_WEEK))))
     )

# COMMAND ----------

df.shape

# COMMAND ----------

display(df.head(366))

# COMMAND ----------

# import plotly.plotly as py
# import plotly.graph_objs as go

# trace1 = go.Bar(
#     x=['giraffes', 'orangutans', 'monkeys'],
#     y=[20, 14, 23],
#     name='SF Zoo'
# )
# trace2 = go.Bar(
#     x=['giraffes', 'orangutans', 'monkeys'],
#     y=[12, 18, 29],
#     name='LA Zoo'
# )

# data = [trace1, trace2]
# layout = go.Layout(
#     barmode='group'
# )

# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='grouped-bar')

fig = go.Figure()

color = [i for i in range(df.shape[0])]
cmax = len(color)

y = df['count'].to_list()
x = df['MONTH'].to_list()

# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')

fig.add_trace(go.Bar(
    x=x,
    y=y,
    name='MONTH',
    marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
    #colorscale="Viridis"
))

data = df['DAY_OF_MONTH'].to_list()
# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=45)
#fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.update_layout(barmode='group', 
                  xaxis=dict(categoryorder='total descending', title='MONTH'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

# df = pd.DataFrame()
# line = pd.DataFrame({"count": 2, "MONTH": 'Jan'}, index=[1])
# df = df.append(line, ignore_index=False)
# line = pd.DataFrame({"count": 3, "MONTH": 'Jan'}, index=[2])
# df = df.append(line, ignore_index=False)
# line = pd.DataFrame({"count": 4, "MONTH": 'Jan'}, index=[3])
# df = df.append(line, ignore_index=False)
# # ++++++++++ comment above and uncomment below
df = pd.DataFrame()
line = pd.DataFrame({"count": 2, "MONTH": 'Jan'}, index=[1])
df = df.append(line, ignore_index=False)
line = pd.DataFrame({"count": 3, "MONTH": 'Feb'}, index=[2])
df = df.append(line, ignore_index=False)
line = pd.DataFrame({"count": 4, "MONTH": 'March'}, index=[3])
df = df.append(line, ignore_index=False)
# # ++++++++++
fig = go.Figure()
color = [i for i in range(df.shape[0])]
cmax = len(color)
y = df['count'].to_list()
x = df['MONTH'].to_list()
# fig = px.bar(df, x="YEAR", y="count", color='DEP_DEL15')
fig.add_trace(go.Bar(
    x=x,
    y=y,
    name='MONTH',
    marker=dict(colorscale='Viridis', cmax=cmax, cmin=0, color=color)
    #colorscale="Viridis"
))
fig.update_layout(xaxis_tickangle=45)
fig.update_layout(barmode='group', 
                  xaxis=dict(categoryorder='total descending', title='MONTH'), 
                  title=go.layout.Title(text="Departure Delay of 15 minutes or more"),  
                  yaxis=dict(title='Number of flights'))
fig.show()

# COMMAND ----------

