# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays - Final Notebook
# MAGIC ###### W261 Spring 2020 
# MAGIC ###### Presentation Date: April 16th, 2020
# MAGIC ###### Team 20: Diana Iftimie, Shaji K Kunjumohamed, Navya Sandadi, & Shobha Sankar

# COMMAND ----------

# MAGIC %md 
# MAGIC ## I. Question Formulation & Introduction
# MAGIC 
# MAGIC As we've all probably experienced at some point in our lives, air travel is never easy. Whether you're the person getting on a flight traveling around the world, the folks in the air traffic control towers orchestrating incoming and outgoing flights, or the airports and airlines trying their best to effectively coordinate flights at every hour of every day of every year, so much can go wrong. The delays alone are enough to completely derail anyone's plans and trigger a cascading effect of consequences down the line as delays continue to stack up on top of each other over the course of time. And the biggest problem is that these delays often occur when we least expect them and at the worst possible times.
# MAGIC 
# MAGIC To attempt to solve this problem, we introduce the *Airline Delays* dataset, a dataset of US domestic flights from 2015 to 2019 collected by the Bureau of Transportation Statistics for the purpose of studying airline delays. For this analysis, we will primarily use this dataset to study the nature of airline delays in the United States over the last few years, with the ultimate goal of developing models for predicting significant flight departure delays (30 minutes or more) in the United States. 
# MAGIC 
# MAGIC In developing such models, we seek to answer the core question, **"Given known information prior to a flight's departure, can we predict departure delays and identify the likely causes of such delays?"**. In the last few years, about 11% of all US domestic flights resulted in significant delays, and answering these questions can truly help us to understand why such delays happen. In doing so, not only can airlines and airports start to identify likely causes and find ways to mitigate them and save both time and money, but air travelers also have the potential to better prepare for likely delays and possibly even plan for different flights in order to reduce their chance of significant delay. 
# MAGIC 
# MAGIC To effectively investigate this question and produce a practically useful model, we will aim to develop a model that performs better than a baseline model that predicts the majority class of 'no delay' every time (this would have an accuracy of 89%). Having said that, we have been informed by our instructors that the state of the art is 85% accuracy, but will proceed to also prioritize model interpretability along side model performance metrics to help address our core question. Given the classification nature of this problem, we will concentrate on improving metrics such as precision, recall, F1, area under ROC, and area under PR curve, over our baseline model. We will also concentrate on producing models that can explain what features of flights known prior to departure time can best predict departure delays and from these, attempt to best infer possible causes of departure delays. 

# COMMAND ----------

# DBTITLE 1,Import Pyspark ML Dependencies (Hide)
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.functions import udf
from pyspark.sql import Window

import pyspark.ml.pipeline as pl
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorIndexer, VectorAssembler, OneHotEncoderEstimator

from pyspark.mllib.evaluation import MulticlassMetrics

from dateutil.relativedelta import relativedelta, SU, MO, TU, WE, TH, FR, SA
import pandas as pd
import datetime as dt

# COMMAND ----------

# DBTITLE 1,Import Plotly Dependencies (Hide)
# MAGIC %sh 
# MAGIC pip install plotly --upgrade

# COMMAND ----------

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. EDA & Discussion of Challenges
# MAGIC 
# MAGIC ### Dataset Introduction
# MAGIC The Bureau of Transporation Statistics provides us with a wide variety of features relating to each flight in the *Airline Delays* dataset. These features range from features about the scheduled flight such as the planned departure, arrival, and elapsed times, the planned distance, the carrier and airport information, information regarding the causes of certain delays for the entire flight, as well as the amounts of delay (for both flight departure and arrival), among many other features. 
# MAGIC 
# MAGIC Given that for this analysis, we will be concentrating on predicting and identify the likely causes of departure delays before any such delay happens, we will primarily concentrate our EDA, feature engineering, and model development using features of flights that would be known at inference time. We will choose the inference time to be 6 hours prior to the scheduled departure time of a flight. Realistically speaking, providing someone with a notice that a flight will likely be delayed 6 hours in advance is likely a sufficient amount of time to let people prepare for such a delay to reduce the cost of the departure delay, if it occurs. Such features that fit this criterion include those that are related to:
# MAGIC 
# MAGIC * **Time of year / Flight Date** 
# MAGIC     - `Year`: The year in which the flight occurs (range: [2015, 2019])
# MAGIC     - `Month`: A numerical indicator for the month in which the flight occurs (range: [1, 12], 1 corresponds to January)
# MAGIC     - `Day_Of_Month`: The day of the month in which the flight occurs (range: [1, 31])
# MAGIC     - `Day_Of_Week`: A numerical indiciator for the day of the week in which the flight occurs (range: [1, 7], 1 corresponds to Monday)
# MAGIC * **Scheduled Departure & Arrival Times**
# MAGIC     - `CRS_Dep_Time`: The scheduled departure time of the flight (range: (0, 2400], 100 corresponds to 1AM departure time)
# MAGIC     - `CRS_Arr_Time`: The scheduled arrival time of the flight (range: (0, 2400], 100 corresonds to 1AM arrival time)
# MAGIC * **Planned Elapsed Times & Distances**
# MAGIC     - `CRS_Elapsed_Time`: The scheduled elapsed time (in minutes) of the flight (continuous variable, 60 corresponds to 1 hour elapsed time)
# MAGIC     - `Distance`: The planned distance (in miles) for the flight distance from origin to destination airports (continuous variable, e.g. 2475 miles)
# MAGIC     - `Distance_Group`: A binned version of the `Distance` variable into integer bins (range: [1, 11], e.g. 2475 miles maps to a distance group of 10)
# MAGIC * **Airline Carrier**
# MAGIC     - `Op_Unique_Carrier`: A shortcode denoting the airline carrier that operated the flight (categorical, 19 distinct carriers, e.g. 'AS' corresponds to Alaska Airlines, more mappings of airlines codes can be found here: https://www.bts.gov/topics/airlines-and-airports/airline-codes) 
# MAGIC * **Origin & Destination Airports** 
# MAGIC     - `Origin`: A shortcode denoting the origin airport from which the plane took off (categorical, 364 distinct airports, e.g. 'SFO' corresponds to San Francisco International Airport, more mappings of airport codes can be found here: https://www.bts.gov/topics/airlines-and-airports/world-airport-codes)
# MAGIC     - `Dest`: A shortcode denoting the destination airport at which the plane landed (categorical, 364 distinct airports, same in construct as `Origin`)
# MAGIC 
# MAGIC We will prioritize working with these features in the next few sections. Additionally, we will use the variable `Dep_Delay` (which describes the amount of departure delay in minutes) to define our outcome variable for "significiant" departure delays (i.e. delays of 30 minutes or more). These significant delays will be encoded in the variable `Dep_Del30`, a 0/1 indicator for whether the flight was delayed, which we will append to the dataset below. Finally, we will focus our analysis to the subset of flights that are not diverted, are not cancelled, and have non-null values for departure delays to ensure that we can accurately predict departure delays for flights. This subset will still leave us with a significant number of records to work with for the purpose of training and model development. 
# MAGIC 
# MAGIC Below are a few example flights taken from the *Airline Delays* dataset that we will use for our analysis.

# COMMAND ----------

# DBTITLE 1,Examples Records from Airline Delays Dataset (Hide code)
def LoadAirlineDelaysData():
  # Read in original dataset
  airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

  # Filter to datset with entries where diverted != 1, cancelled != 1, and dep_delay != Null
  airlines = airlines.where('DIVERTED != 1') \
                     .where('CANCELLED != 1') \
                     .filter(airlines['DEP_DELAY'].isNotNull()) 
  return airlines

# Generate other Departure Delay outcome indicators for n minutes
def CreateNewDepDelayOutcome(data, thresholds):
  for threshold in thresholds:
    if ('Dep_Del' + str(threshold) in data.columns):
      print('Dep_Del' + str(threshold) + " already exists")
      continue
    data = data.withColumn('Dep_Del' + str(threshold), (data['Dep_Delay'] >= threshold).cast('integer'))
  return data  

# Load data & define outcome variable
airlines = LoadAirlineDelaysData()
airlines = CreateNewDepDelayOutcome(airlines, [30])

print(airlines.columns)

# Filter dataset to variables of interest
outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group']
contNumFeatureNames = ['CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
joiningFeatures = ['FL_Date'] # Features needed to join with the holidays dataset--not needed for training
airlines = airlines.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames + joiningFeatures)

# Display a small sample of flight records
display(airlines.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames).sample(fraction=0.0001, seed=6).take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that because we are interested in predicting departure delays for future flights, we will define our test set to be the entirety of flights from the year 2019 and use the years 2015-2018 for feature engineering and model development (training & validation sets). This way, we will simulate the conditions for training a model that will predict departure delays for future flights. This will leave about 23% of the data for testing and the remaining 77% for training & validation. 
# MAGIC 
# MAGIC Additionally, in this section and the next section on feature engineering, we'll mainly be operating on distinct columns of the dataset at any given time. In order to help with scalability of our explorations and preprocessing, we will save these dataset splits to parquet files in the cluster, as parquet is optimized for column-wise storage and thus will help improve the performance of our column-wise analysis of the *Airlines Delays* dataset. We will also focus our EDA on the union of training & validation sets, to ensure our decisions are not influenced by the test set.

# COMMAND ----------

# DBTITLE 1,Split, Save, & Reload Dataset from parquet (hide code & result)
def SplitDataset(airlines):
  # Split airlines data into train, dev, test
  test = airlines.where('Year = 2019') # held out
  train, val = airlines.where('Year != 2019').randomSplit([7.0, 1.0], 6)

  # Select a mini subset for the training dataset (~2000 records)
  mini_train = train.sample(fraction=0.0001, seed=6)

  print("mini_train size = " + str(mini_train.count()))
  print("train size = " + str(train.count()))
  print("val size = " + str(val.count()))
  print("test size = " + str(test.count()))
  
  return (mini_train, train, val, test) 

# Write train & val data to parquet for easier EDA
def WriteAndRefDataToParquet(data, dataName):
  # Write data to parquet format (for easier EDA)
  data.write.mode('overwrite').format("parquet").save("dbfs:/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet("dbfs:/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

# Split dataset into training/validation/test; use mini_train to help with quick testing
mini_train, train, val, test = SplitDataset(airlines)

# Save and reload datasets for more efficient EDA & feature engineering
mini_train = WriteAndRefDataToParquet(mini_train, 'mini_train')
train = WriteAndRefDataToParquet(train, 'train')
val = WriteAndRefDataToParquet(val, 'val')
test = WriteAndRefDataToParquet(test, 'test')
train_and_val = WriteAndRefDataToParquet(train.union(val), 'train_and_val')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prospective Models for the Departure Delay Classification Task
# MAGIC Before we go into detail on our core EDA tasks, we'd like to introduce the models we will be considering in section IV to motivate our discussion. At a high level, we will be considering the set of models that include the following:
# MAGIC * Decision Trees
# MAGIC * Logistic Regression
# MAGIC * Naive Bayes
# MAGIC * Support Vector Machines
# MAGIC 
# MAGIC Given that our task for this analysis is to classify flights as delayed (1) or not delayed (0), we want to ensure that the models we consider are well suited for classification tasks, which all four of these models are good at. Additionally, since we'll be interested in looking at explainable models to help inform why certain flights are delayed over others, we consider Decision Trees, Logistic Regression, and Naive Bayes explicitly for this task, whereas Support Vector Machines are not as well-suited for explainability. We will discuss these models in more detail in section IV when we discuss our algorithm exploration. With that said, we'll keep these models in mind as we explore our core EDA tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #1: Binning Continuous Features
# MAGIC Among the features that we have to consider for predicting departure delays, we have 4 that are relatively continuous and can take on thousands of different values: `CRS_Dep_Time`, `CRS_Arr_Time`, `CRS_Elapsed_Time`, and `Distance`. Let's primarily focus on the variable `CRS_Elapsed_Time` to motivate this discussion and we'll generalize it to the remaining 3 variables. Below is a plot showing the bulk of the distribution for the feature `CRS_Elapsed_Time`.

# COMMAND ----------

# Helper Function for plotting distinct values of feature on X and number of flights on Y, categorized by outocme variable
def MakeRegBarChart(full_data_dep, outcomeName, var, orderBy, barmode, xtype, xrange=None, yrange=None):
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
    xaxis=dict(title=var, type=xtype, range=xrange),
    yaxis=dict(title="Number of Flights", range=yrange)
  )
  fig = go.Figure(data=[t1, t2], layout=l)
  fig.show()
  return fig

var = 'CRS_Elapsed_Time'
fig = MakeRegBarChart(train_and_val, outcomeName, var, orderBy=var, barmode='stack', xtype='linear', xrange=[0, 450])

# COMMAND ----------

# MAGIC %md
# MAGIC From the plot above, we can see the continuous nature of `CRS_Elapsed_Time`, with the majortiy of flights ranging between 50 to 200 minutes. Note that each of the taller bars correspond to flight durations at the 5-minute markers (60, 65, 70, etc), as these are more "convenient" flight durations that airlines tend to define when scheduling their flights. 
# MAGIC 
# MAGIC Now, if we consider the possible values that `CRS_Elapsed_Time` can intrinsically take on, it can be any number of minutes that the flight is scheduled to take; these could be values such as 60 minutes, 65 minutes, 120 minutes, even going onto 400 minutes in some cases, as evident above. Conceptually speaking, some of these times are drasticaly different (60 minutes vs. 400 minutes), while others are similar enough that they would be considered the same flight duration (60 minutes vs. 65 minutes). Because of this, these flight times could be aggregated into bins in order to group similar flight durations. For example, we could aggregate the `CRS_Elapsed_Time` into 1-hour blocks, which will produce a distribution such as the one shown below:

# COMMAND ----------

# Augments the provided dataset for the given variable with binned/bucketized
# versions of that variable, as defined by splits parameter
# Column name suffixed with '_bin' will be the bucketized column
# Column name suffixed with '_binlabel' will be the nicely-named version of the bucketized column
def BinValuesForEDA(df, var, splits, labels):
  # Bin values
  bucketizer = Bucketizer(splits=splits, inputCol=var, outputCol=var + "_bin")
  df_buck = bucketizer.setHandleInvalid("keep").transform(df)
  
  # Add label column for binned values
  bucketMaps = {}
  bucketNum = 0
  for l in labels:
    bucketMaps[bucketNum] = l
    bucketNum = bucketNum + 1
  
  callnewColsUdf = udf(lambda x: bucketMaps[x], StringType())  
  return df_buck.withColumn(var + "_binlabel", callnewColsUdf(F.col(var + "_bin")))

# Make plot with binned version of CRS_Elapsed_Time
d = BinValuesForEDA(train_and_val, var, 
                    splits = [float("-inf"), 60, 120, 180, 240, 300, 360, float("inf")], 
                    labels = ['0-1 hours', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', '5-6 hours', '6+ hours'])
fig = MakeRegBarChart(d, outcomeName, var + "_binlabel", orderBy=var + "_binlabel", barmode='stack', xtype='category')

# COMMAND ----------

# MAGIC %md
# MAGIC By doing this kind of binning, we can see that the same general shape of the distribution is preserved, albeit at a coarser level, which removes some of the extra information that was present in the original variable. But doing this kind of aggregation has its benefits when it comes to modeling. 
# MAGIC 
# MAGIC In the case of Logistic Regression, if we had referred to the original `CRS_Elapsed_Time`, we would estimate a single coefficient for the variable, which would tell us the effect that adding 1 minute to the elapsed time would have on the probability that a flight is delayed. However, if we were to bin this variable and treat the result as a categorical variable in our regression, the model would estimate a coefficient for all but one of the bins, which would tell us more detailed information about the effect of a flight having a certain kind of duration (e.g. 1-2 hours). By comparison to the coefficient we'd estimate on the raw `CRS_Elapsed_Time`, this would be a much more meaningful estimate to use to understand the underlying factors for departure delays. This will require the algorithm to have to learn more coefficients than if the original `CRS_Elapsed_Time` were to be used, but to answer our core question, it seems to be a worthwhile cost if we proceed with Logistic Regression as our model of choice.
# MAGIC 
# MAGIC For the case of Decision Trees, binning the `CRS_Elapsed_Time` will actually help to improve the scalability of the algorithm. If we are to use the raw `CRS_Elapsed_Time`, the decision tree algorithm will have to consider every possible split for the feature (as we'll see later, there are in fact 646 distinct values for the `CRS_Elapsed_Time` feature, meaning that the algorithm would have to consider 645 different splits when finding the best split). However, if we bin the feature as shown above, the number of splits the algorithm needs to consider drops to just 6, which is a large reduction in the amount of work the algorithm needs to do to find the best split, which will help with the scalability of the algorithm.
# MAGIC 
# MAGIC The benefits that come with binning `CRS_Elapsed_Time` can really extend to all our continuous variables. The `Distance_Group` already takes care of this for the feature `Distance` and in section III, we will take care bin the remaining continuous variable depending on the situation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #2: Ordering Categorical Features
# MAGIC Among the features we have to consider, there are three that are categorical in nature, namely `Op_Unique_Carrier`, `Origin`, and `Dest`. While some features such as `Op_Unique_Carrier` have only a few distinct values (19 in total), others such as `Origin` and `Dest` have many more distinct values (364 each). While these categorical features are meaningful to us, they can introduce some added difficulties to our models for training.
# MAGIC 
# MAGIC In the case of Support Vector Machines, each of these categorical features will have to be encoded with 1-hot encoding, where we generate a sparse vector of a length equal to the number of unique values in the categorical feature. For `Op_Unique_Carrier`, this would lead to the algorithm to have to consider a 19-dimensional space. Take it to `Origin` or `Dest` and these features alone will require considering a 364-dimensional space, which makes it difficult for the algorithm to scale.
# MAGIC 
# MAGIC Similarly for Logistic Regression, for a categorical feature with \\(n\\) distinct values, this will require estimating \\(n-1\\) unique coefficients--while these coefficients can be meaningful to us, the sheer number can be overwhelming to estimate from a scalability perspective. 
# MAGIC 
# MAGIC And with Decision Trees, the issue comes when considering the number of splits. If we take the `Op_Unique_Carrier` feature, because there are 19 values with no implicit order to them, when the Decision Tree algorithm considers all possible splits, it will have to consider not 18 splits, but \\(2^{k-1}-1 = 2^{19-1}-1 = 262,143\\) possible splits. Go even further to a categorical variable like `Origin` and the number becomes massive (\\(1.87 * 10^{109}\\) different splits to consider), which is computationally prohibitive to the algorithm as it will need to consider every single possible split to find the best split for the feature.
# MAGIC 
# MAGIC In order to address these issues, we'd really want to provide some sort of ordering and thus a ranking for each distinct value of our categorical features. Let's consider this in the context of our 'smaller' categorical variable, `Op_Unique_Carrier`. In its raw form, the distinct categories are in no way comparable (how does one compare 'DL' (Delta Airlines) to 'AS' (Alaska Airlines) in a meaningful way?). However, these categories could be compared using some intrinsic property of the category. Given that we're interested in using the information in `Op_Unique_Carrier` to predict our outcome `Dep_Del30`, we can evaluate what our average outcome is (or really the probability of a significant departure delay) for each variable and use this measure as a means of ordering the distinct categories. This ordering is shown in the plot below:

# COMMAND ----------

# Helper function that plot the probability of outcome on the x axis, the number of flights on the y axis
# With entries for each distinct value of the feature as separate bars.
def MakeProbBarChart(full_data_dep, var, xtype, numDecimals):
  # Filter out just to rows with delays or no delays
  d_delay = full_data_dep.select(var, outcomeName).filter(F.col(outcomeName) == 1.0).groupBy(var, outcomeName).count().orderBy("count")
  d_nodelay = full_data_dep.select(var, outcomeName).filter(F.col(outcomeName) == 0.0).groupBy(var, outcomeName).count().orderBy("count")

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
  return fig
  
# Plot Carrier and outcome with bar plots of probability on x axis
fig = MakeProbBarChart(airlines, "Op_Unique_Carrier", xtype='category', numDecimals=5)

# COMMAND ----------

# MAGIC %md
# MAGIC By ordering the airline carriers by this average outcome (`Prob_Dep_Del30`), we can not only begin to compare the airlies (Alaska Airlines seems to be better than Delta Airlines by a small margin), but we can actually strongly reduce the number of splits to consider from 262,143 possible splits to just 18 for `Op_Unique_Carrier` when it comes to the Decision Tree algorithm. Even further, if we assign numerical ranks, we have the potential to convert this categorical feature into a numerical feature (by assigning 1 to the highest ranked airline, 'HA' (Hawaiian Airlines), and 19 to the lowest ranked airline 'B6' (Jet Blue)), which helps to reduce the workload for both Logistic Regression and Support Vector Machines. 
# MAGIC 
# MAGIC This data transformation essentially describes the application of Breiman's Theorem, which we can apply to all our categorical features, even the ones we feature engineer. Note that any ranking we generate for our categorical features will need to be generated based on our training set and separately applied to the test set, to ensure the ranking isn't in any way influenced by the data in our test set. We will proceed to apply this to all our categorical features in section III. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #3: Balancing the Dataset
# MAGIC For our final EDA task, let's consider our outcome variable `Dep_Del30`. To ensure that our model is able to adequately predict whether a flight will be delayed, we need to make sure there is a good representation of both classes in the training set. However, as is evident in the pie chart below, we clearly have an over-representation of the no-delay category (`Dep_Del30 == 0`).

# COMMAND ----------

# Look at balance for outcome in training
display(train_and_val.groupBy(outcomeName).count())

# COMMAND ----------

# MAGIC %md
# MAGIC What this chart tells us is that a simple model that always predicts no departure delay will have a high accuracy of around 89% (assuming our test set has a similar distribution). But of course, simply predicting the majority class is meaningless when it comes to answering our core question--all it really tells us is that most flights are not delayed.
# MAGIC 
# MAGIC But the problem extends further. Even if we develop a model that doesn't always predict no departure delay as a simple baseline model would, there is still a major problem of bias if we leave the dataset unbalanced. Namely, if the dataset is unbalanced, the model will still favor the majority class and learn features of majority class well at the expense of the minority class, which will come at a cost to model performance. Regardless of which of our four algorithms we'll concentrate on, this data imbalance is a concern that we'll need to address. There are a variety of methods that we can use to deal with this data imbalance (including majority class undersampling, minority class oversampling, SMOTE, majority class splitting), which we will discuss the theory, implementation, and scalability concerns at the end of section III. Any method that will balance the dataset will help to reduce bias, coming at a cost to inducing more variance in the model. However, for the purpose of developing a model that can help inform the likely causes of departure delays, this is a worthwhile tradeoff.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Further EDA
# MAGIC During the investigation of our dataset, we did a deep dive to examine all of our in-scope variables from the original dataset to understand the nature of the dataset and help inform our decision making. To see more plots and discussion, please refer to our more extensive EDA linked below. We will also explore some more of these plots in the next section.
# MAGIC 
# MAGIC https://dbc-b1c912e7-d804.cloud.databricks.com/?o=7564214546094626#notebook/3895804345790408/command/3895804345790409

# COMMAND ----------

# MAGIC %md
# MAGIC ## III. Feature Engineering
# MAGIC With the EDA discussion from the previous section in mind, we now proceed with applying the feature engineering described in the previous section. We will first look at summary statistics and check for missing values for each of our features for predicting departure delays. We'll then look to binning our numerical features, adding additional features via interactions, bringing in additional datasets, and looking at aggregated statistics. For the categorical features in the dataset (both original and those developed from an feature transformations), we will apply Breiman's Theorem to order these features and transform them into numerical features. Finally, we'll take a closer look at the data preprocessing techniques we will explore for balancing our dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics & Missing Values Assessment
# MAGIC In the table shown below, we can see a set of summary statistics for each base feature, including general summary statistics (count, mean, standard deviation, min, max), as well as the number of distinct values and the number of null/missing values in the training & validation set. Based on the results shown below, all values appear to fall into the expected ranges based on the definitions of these features.

# COMMAND ----------

def GetStats(df):
  allCols = [outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames

  # Get summary stats for the full training dataset
  summaryStats = df.select(allCols).describe().select(['summary'] + allCols)

  # Get number of distinct values for each column in full training dataset
  distinctCount = df.select(allCols) \
                               .agg(*(F.countDistinct(F.col(c)).alias(c) for c in allCols)) \
                               .withColumn('summary', F.lit('distinct count')) \
                               .select(['summary'] + allCols)

  # Get number of nulls for each column in full training dataset
  nullCount = df.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in allCols]) \
                .withColumn('summary', F.lit('null count')) \
                .select(['summary'] + allCols)

  # Union all statistics to single display
  res = summaryStats.union(distinctCount.union(nullCount))
  display(res)
  
GetStats(train_and_val)

# COMMAND ----------

# MAGIC %md
# MAGIC The only odd value seems to be the minimum value for `CRS_Elapsed_Time` that takes on a value of -99.0. Upon closer inspection, there's no true indication for why this datapoint is negative, except that it is likely a mistake in the dataset, since the difference in the departure and arrival times is 76 minutes, which should be the actual value of `CRS_Elapsed_Time`. For this reason, we will correct the `CRS_Elapsed_Time` for this flight record.

# COMMAND ----------

# Correct CRS_ElapsedTime = -99
airlines = airlines.withColumn("CRS_Elapsed_Time", when(train_and_val["CRS_Elapsed_Time"] == -99, 76.0).otherwise(train_and_val["CRS_Elapsed_Time"]))

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, note that we do not appear to have any null values in our training and validation data and thus we will not need to handle missing values for the purpose of training. However, there is a potential that our test data has missing values and or the features of the test data take on values that were not seen in the training data. Because this is always a possibility at inference time, we will need to make sure our data transformations are robust to such cases--we will evaluate this on a case-by-case basis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Binning Continuous Numerical Features
# MAGIC As discussed in task #1 of the EDA in section II, one of the transformations we'd like to apply to our continuous numerical features is a binning transformation. In doing so, we can reduce the continuous variables to meaningful increments that will help with interpretability in Logistic Regression and help to reduce the number of splits that needs to be considered by the Decision Tree algorithm. In order to determine reasonable split points, let's evaluate the distributions of each of the continuous variables `CRS_Dep_Time`, `CRS_Arr_Time`, and `CRS_Elapsed_Time` (note that the continuous feature `Distance` has already been binned via the variable `Distance_Group`, so we will not examine this feature). These distribution are shown below:

# COMMAND ----------

fig1 = MakeRegBarChart(train_and_val, outcomeName, 'CRS_Dep_Time', orderBy='CRS_Dep_Time', barmode='stack', xtype='linear')

# COMMAND ----------

fig2 = MakeRegBarChart(train_and_val, outcomeName, 'CRS_Arr_Time', orderBy='CRS_Arr_Time', barmode='stack', xtype='linear')

# COMMAND ----------

# MAGIC %md
# MAGIC From the distributions of values for `CRS_Dep_Time` and `CRS_Arr_Time`, we can see clusters of flights defined by --00 to --59 blocks (these clusters are a remnant of the structure of the 2400-clock, since there are no valid times from --60 to --99). With that said, by analyzing any one of these clusters, it's clear that most flights are scheduled at 5-minute markers (such as 1200, 1205, 1210, etc) for both departure and arrival times. Since there really isn't a big difference between times separated by a couple of 5 minute intervals, we will choose to bin these values by 10-minute increments to ensure we still have enough data granularity to differentiate between populate times such as 1200 and 1230 but not too much granularity to have too many splits to consider (reducing to 10-minute granularity alone reduces the number of splits for a Decision Tree algorithm to consider from at most 1439 to just about 144 split points).

# COMMAND ----------

fig3 = MakeRegBarChart(train_and_val, outcomeName, 'CRS_Elapsed_Time', orderBy='CRS_Elapsed_Time', barmode='stack', xtype='linear')

# COMMAND ----------

# MAGIC %md
# MAGIC For the `CRS_Elapsed_Time`, as we saw in the EDA Task 1, we can still capture the general distribution of the scheduled flight duration even when we bin by 1 hour durations, which leaves us with meaningful groups and much fewer splits to consider. For this reason, we will bin `CRS_Elapsed_Time` to 1 hour increments. 
# MAGIC 
# MAGIC All three of the described binning transformations are applied with the code shown below and appended to our original *Airline Delays* dataset. Note that this binning operation is independently applied to each record in the original dataset, using the binning we decided based on the training dataset. Also note that by using the `Bucketizer` 'keep' flag, we explicitly choose to bin any invalid values into a special bucket, which will allow our models to be resiliant even in the event of encountering values that go beyond the limits defined. Note that for simplicity, we extend the `CRS_Elapsed_Time` bins to negative and positive infinity to ensure all values are binned properly. All new binned features will be suffixed with `_bin`, except for `Distance_Group`, which is already defined. A few examples of the new columns are shown below.

# COMMAND ----------

# Augments the provided dataset for the given feature/variable with a binned version
# of that variable, as defined by splits parameter
# Column name suffixed with '_bin' will be the binned column
def BinFeature(df, featureName, splits):
  if (featureName + "_bin" in df.columns):
    print("Variable '" + featureName + "_bin' already exists")
    return df
    
  # Generate binned column for feature
  bucketizer = Bucketizer(splits=splits, inputCol=featureName, outputCol=featureName + "_bin")
  df_bin = bucketizer.setHandleInvalid("keep").transform(df)
  df_bin = df_bin.withColumn(featureName + "_bin", df_bin[featureName + "_bin"].cast(IntegerType()))    
  return df_bin

# Bin numerical features in entire airlines dataset
# Note that splits are not based on test set but are applied to test set (as would be applied at inference time)
airlines = BinFeature(airlines, 'CRS_Dep_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
airlines = BinFeature(airlines, 'CRS_Arr_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
airlines = BinFeature(airlines, 'CRS_Elapsed_Time', splits = [float("-inf")] + [i for i in range(0, 660 + 1, 60)] + [float("inf")]) # 1-hour blocks
binFeatureNames = ['CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin']

display(airlines.select(contNumFeatureNames + binFeatureNames + ['Distance_Group']).take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interacting Features
# MAGIC During our in-depth EDA, we investigated a few interactions based on intuition. From our own experiences, we believed that certain temporal interaction terms and airport-related interactions would intuitively be indicative of a flight that is more likely to experience a departure delay.
# MAGIC 
# MAGIC Let's consider the features `Month` and `Day_Of_Month`. On their own, we may be able to capture aggregated information about flights delays on a particular day of the month or a particular month on its own. But if we interact the two, we get the concept of `Day_Of_Year`, and upon closer inspection, there do appear to be certain days of the year that experience more traffic and more delays. Take the stacked bar chart comparing the ratio of delayed and no-delay flights shown below for `Day_Of_Year`, which is the concatenation of `Month` and `Day_Of_Month`. From this chart, we can see that there is a higher probability of departure delays for dates in the summer, as well as near the end of December and beginning of January, but not on holidays in that time frame (e.g. December 25th - Christmas Day). Because there does appear to be some relationship between `Day_Of_Year` and the likelihood of a departure delay, we choose to add this as one of our interaction terms.

# COMMAND ----------

# Plot that demonstrates the probability of a departure delay, given the day of year (interaction of month & day of month)
var = "Day_Of_Year"
d = train_and_val.select("Month", "Day_Of_Month", outcomeName) \
                 .withColumn(var, F.concat(F.col('Month'), F.lit('-'), F.col('Day_Of_Month'))) \
                 .groupBy(var, "Month", "Day_Of_Month", outcomeName).count() \
                 .orderBy("Month", "Day_Of_Month") \
                 .toPandas()
display(d)

# COMMAND ----------

# MAGIC %md
# MAGIC Along a similar vein of thinking, we decided to add the interaction of `CRS_Dep_Time` and `Day_Of_Week` along with the interaction of `CRS_Arr_Time` and `Day_Of_Week` to produce `Dep_Time_Of_Week` and `Arr_Time_Of_Week` respectively. We discovered that there are certain times of day that encounter a higher likelihood of departure delays, depending on the day of the week these times fall on. But to ensure there are not too many categories in this new categorical feature, we will interact the binned times with the `Day_Of_Week` for each interaction.
# MAGIC 
# MAGIC Finally, we considered that there may be some inherent relationship between the origin airport and the destination airport with respect to departure delays that may be well captured via an interaction between the two features. Namely, during our EDA, we saw that more popular flights (such as between SFO (San Francisco) and LAX (Los Angeles)) also saw higher levels of departure delays, whereas less popular flights (such as between SEA (Seattle) and PHX (Pheonix)) saw lower levels of departure delays. Thus, we chose to interact `Origin` and `Dest` to form the categorical feature `Origin_Dest` for our dataset. All the interactions discussed are added using the provided code and a few examples are shown below:

# COMMAND ----------

# Given a tuple specifying the two features to interact and the new name of the feature,
# Creates an interaction term corresponding to each tuple
def AddInteractions(df, featurePairsAndNewFeatureNames):
  for (feature1, feature2, newName) in featurePairsAndNewFeatureNames:
    if (newName in df.columns):
      print("Variable '" + newName + "' already exists")
      continue
    
    # Generate interaction feature (concatenation of two features)
    df = df.withColumn(newName, F.concat(F.col(feature1), F.lit('-'), F.col(feature2)))
    
  return df

# Make interaction features on airlines dataset
# Note that interactions are independentently defined for each record
interactions = [('Month', 'Day_Of_Month', 'Day_Of_Year'), 
                ('Origin', 'Dest', 'Origin_Dest'),
                ('Day_Of_Week', 'CRS_Dep_Time_bin', 'Dep_Time_Of_Week'),
                ('Day_Of_Week', 'CRS_Arr_Time_bin', 'Arr_Time_Of_Week')]
intFeatureNames = [i[2] for i in interactions]
airlines = AddInteractions(airlines, interactions)

display(airlines.select(['Month', 'Day_Of_Month', 'Day_Of_Year', 'Day_Of_Week', 'CRS_Dep_Time_bin', 'Dep_Time_Of_Week', 'Day_Of_Week', 'CRS_Arr_Time_bin', 'Arr_Time_Of_Week', 'Origin', 'Dest', 'Origin_Dest']).take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding a Holiday Feature
# MAGIC With the `Day_Of_Year` variable in mind, we'd noted during the EDA that the likelihood of a departure delay, as well as the number of flights occuring on a given day of the year seemed to have some relationship with the commonly celebrated holidays in the United States. Take Christmas Day, which always falls on December 25th. If we examine the plot shown previously, there generally appeared to be a lower probability of delay on Christmas Day and Christmas Eve, with higher probabilities immediately before and after those two days. Intuitively, one would expect a higher probability of departure delay as people try to get home for the holidays or leave promptly after the holidays are over; but on the day of holidays, people generally stay home (and thus experience less probability of departure delay). 
# MAGIC 
# MAGIC There are a variety of cases where holidays and days near holidays seem to reflect something about the likelihood of a flight experiencing a departure delay. For this reason, we attempted to capture this information by joining a dataset of government holidays to our original *Airlines Delays* dataset and construct the `Holiday` feature. This feature is a categorical feature indicating whether a flight occurs before, after, or on a holiday, or is not in any way related to a holiday. Because the *Holidays* dataset is much smaller compared to the *Airline Delays* dataset and will fit in memory, we will join this dataset to *Airline Delays* via a broadcast join after it has been prepared with before and after limits for specific government holidays. Note that the limits for whether a flight occurs before or after a holiday is dependent on the day of the week in addition to government holidays that are near the flight date. This feature construction is done in the following code with a few examples shown below with new feature:

# COMMAND ----------

def AddHolidayFeature(df):
  if ('Holiday' in df.columns):
      print("Variable 'Holiday' already exists")
      return df
  
  # Import dataset of government holidays
  holiday_df_raw = spark.read.csv("dbfs:/user/shajikk@ischool.berkeley.edu/scratch/" + 'holidays.csv').toPandas()
  holiday_df_raw.columns = ['ID', 'FL_DATE', 'Holiday']

  # Get limits for a given date when we'll likely see "near holiday" conditions hold
  # This is more ofr the purpose of helping to capture likely long-weekend conditions
  # that could influence departure delays near holidays
  def get_limits(date):
    given_date = dt.datetime.strptime(date,'%Y-%m-%d')
    this_day = given_date.strftime('%a')

    lastSun = given_date + relativedelta(weekday=SU(-1))
    lastMon = given_date + relativedelta(weekday=MO(-1))
    lastTue = given_date + relativedelta(weekday=TU(-1))
    lastWed = given_date + relativedelta(weekday=WE(-1))
    lastThu = given_date + relativedelta(weekday=TH(-1))
    lastFri = given_date + relativedelta(weekday=FR(-1))
    lastSat = given_date + relativedelta(weekday=SA(-1))
    thisSun = given_date + relativedelta(weekday=SU(1))
    thisMon = given_date + relativedelta(weekday=MO(1))
    thisTue = given_date + relativedelta(weekday=TU(1))
    thisWed = given_date + relativedelta(weekday=WE(1))
    thisThu = given_date + relativedelta(weekday=TH(1))
    thisFri = given_date + relativedelta(weekday=FR(1))
    thisSat = given_date + relativedelta(weekday=SA(1)) 

    if this_day == 'Sun' : prev, nxt = lastFri, thisMon
    if this_day == 'Mon' : prev, nxt = lastFri, thisMon
    if this_day == 'Tue' : prev, nxt = lastFri, thisTue
    if this_day == 'Wed' : prev, nxt = lastFri, thisWed
    if this_day == 'Thu' : prev, nxt = lastWed, thisSun
    if this_day == 'Fri' : prev, nxt = lastThu, thisMon
    if this_day == 'Sat' : prev, nxt = lastFri, thisMon

    return prev.strftime("%Y-%m-%d"), prev.strftime('%a'), nxt.strftime("%Y-%m-%d"), nxt.strftime('%a')

  # Construct a holiday dataframe for days that fall before, after, or on a holiday
  # Consider holidays for all years 2015-2019 individually
  holiday_df = pd.DataFrame()
  for index, row in holiday_df_raw.iterrows():
      prev, prev_day, nxt, nxt_day = get_limits(row['FL_DATE'])
      line = pd.DataFrame({"ID": 0, "date": prev, "holiday" : 'before'}, index=[index-0.5])
      holiday_df = holiday_df.append(line, ignore_index=False)

      line = pd.DataFrame({"ID": 0, "date": row['FL_DATE'], "holiday" : 'holiday'}, index=[index])
      holiday_df = holiday_df.append(line, ignore_index=False)

      line = pd.DataFrame({"ID": 0, "date": nxt, "holiday" : 'after'}, index=[index+0.5])
      holiday_df = holiday_df.append(line, ignore_index=False)

  holiday_df = holiday_df.sort_index().reset_index(drop=True).drop("ID",  axis=1)

  # Convert holidays pandas dataframe to a pyspark dataframe for easier joining
  schema = StructType([StructField("FL_DATE", StringType(), True), StructField("Holiday", StringType(), True)])
  holiday = spark.createDataFrame(holiday_df, schema)
  
  # Add new holiday/no holiday column to dataset
  return df.join(F.broadcast(holiday), df.FL_Date == holiday.FL_DATE, how='left').drop(holiday.FL_DATE).na.fill("not holiday")

# Add holidays indicator to airlines dataset
# Note that holidays are known well in advance of a flight and are not specific to the training dataset
holFeatureNames = ['Holiday']
airlines = AddHolidayFeature(airlines)
display(airlines.select('Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Holiday').sample(fraction=0.0001, seed=9).sample(fraction=0.01, seed=6).take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding an Origin Activity Feature
# MAGIC Another intuition based feature that we believed would affect the likelihood of departure delay would be the amount of airline traffic an airport was experiencing around the time that departure was scheduled. Given that busier airports are likely to have busier air traffic control towers and more planes lined up to depart (not to mention more traffic inside of the airport itself as people try to get to their flights), it seemed likely that the amount of activity happening at the origin airport would be indicative of the likelihood of a departure delay. With the `Origin_Activity` feature, we attempt to define a numerical metric that aggregates the "amount of traffic" occuring at a given airport during a specific scheduled departure time block on a given day of the year. This aggregated dataset can then be joined back to the *Airline Delays* dataset via a broadcast join, which gives each flight record a count of the scheduled flight activity that would be occuring in the same departure time block and origin airport. This feature is generated below and a few examples are displayed below:

# COMMAND ----------

def AddOriginActivityFeature(df):
  if ('Origin_Activity' in df.columns):
      print("Variable 'Origin_Activity' already exists")
      return df
  
  # Construct a flight bucket attribute to group flights occuring on the same day in the same time block originating from the same airport
  # Compute aggregated statistics for these flight buckets
  df = df.withColumn("flightbucket", F.concat_ws("-", F.col("Year"), F.col("Month"), F.col("Day_Of_Month"), F.col("CRS_Dep_Time_bin"), F.col("Origin")))
  originActivityAgg = df.groupBy("flightbucket").count()
  
  # Join aggregated statistics back to original dataframe
  df = df.join(F.broadcast(originActivityAgg), df.flightbucket == originActivityAgg.flightbucket, how='left') \
         .drop(originActivityAgg.flightbucket) \
         .drop('flightbucket') \
         .withColumnRenamed('count', 'Origin_Activity')
  return df

# Add OriginActivity feature
# Note that the scheduled origin activity is known well in advance of a flight and is not specific to the training dataset
orgFeatureNames = ['Origin_Activity']
airlines = AddOriginActivityFeature(airlines)
display(airlines.select('Year', 'Month', 'Day_Of_Month', 'CRS_Dep_Time_bin', 'Origin', 'Origin_Activity').sample(fraction=0.0001, seed=7).take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying Breiman's Theorem to Categorical Features

# COMMAND ----------

# MAGIC %md
# MAGIC As discussed in the previous secition in EDA task #2, in our original *Airline Delays* dataset, we have three categorical features to consider, and with the addition of our interaction terms and our holiday feature, we have a total of 8 categorical features to consider for training our model. While some of these categorical features have few distinct values, some of them, especially our interaction terms like `Origin_Dest`, can have a very large number of distinct values. Depending on the model we choose following our algorithm exploration section, these large number of distinct values can be cause for concern with respect to the scalability of our algoirthms. For SVMs, this can lead to very large one-hot encoded vectors. For Logistic Regression, this can lead to many (if not too many) unique coefficients to estimate. And with Decision Trees, too many categories can lead to an inordinate number of possible splits for the algorithm to consider and finding the "best" split would be computationally prohibitive.
# MAGIC 
# MAGIC However, as we saw in our EDA task, we do have a way of ordering the categories in each categorical feature in a more meaningful way by applying Breiman's Theorem to each of our categorical features. Let's consider again one of our original categorical features `Op_Unique_Carrier` that we'd explored in EDA task #2. The cateogries by themselves, do not have any implicit ordering. Yet, uing these distinct categories, we can develop aggregated statistics on the outcome variable `Dep_Del30` to understand how some categories compare to others and rank them--this is the idea behind Breiman's Theorem. Below, we define a function for generating such "Breiman Ranks" given a training dataset, with the example shown for the `Op_Unique_Carrier` feature.

# COMMAND ----------

# Applies Breiman's Theorem to the categorical feature
# Generates the ranking of the categories in the provided categorical feature
# Orders the categories by the average outcome ascending, from integer 1 to n
# Note that this should only be run on the training data
def GenerateBreimanRanks(df, catFeatureName, outcomeName):
  window = Window.orderBy('avg(' + outcomeName + ')')
  breimanRanks = df.groupBy(catFeatureName).avg(outcomeName) \
                   .sort(F.asc('avg(' + outcomeName + ')')) \
                   .withColumn(catFeatureName + "_brieman", F.row_number().over(window))
  return breimanRanks

# Generate example Breiman ranks for Op_Unique_Carrier
exBreimanRanks = GenerateBreimanRanks(train_and_val, 'Op_Unique_Carrier', outcomeName)
display(exBreimanRanks)

# COMMAND ----------

# MAGIC %md
# MAGIC As evident in the output above, we can generate aggregated statistics on the outcome variable for each distinct carrier based ont the sample records present in the dataset for that carrier. Using these aggregated statistics, we can order the categories and assign a ranking from 1 to \\(n\\), where \\(n\\) is the number of distinct categories (in this case 19). This ranking is saved in the new feature `Op_Unique_Carrier_brieman`. Given that the table of Breiman ranks is small and can fit in memory (even for "larger" categorical features like our interaction terms), we can take these training-set-generated Breiman ranks and independently apply them to the entire *Airline Delays* dataset via a broadcast join, similar to the ones shown previously. This code is shown below. In doing so, we effectively transform our categorical feature into a numerical feature, which reduces the need for 1-hot encoding in SVM as well as the number of coefficients to estimate and splits to consider in the Logistic Regression and Decision Tree algorithms respectively. Note that to properly handle unseen values, we will encode unseen categorical features with a `-1` Brieman rank, as is common practice when transforming categorical features to numerical form.

# COMMAND ----------

# Using the provided Breiman's Ranks, applies Breiman's Theorem to the categorical feature
# and creates a column in the original table using the mapping in breimanRanks variable
# Note that this effectively transforms the categorical feature to a numerical feature
# The new column will be the original categorical feature name, suffixed with '_breiman'
def ApplyBreimansTheorem(df, breimanRanks, catFeatureName, outcomeName):
  if (catFeatureName + "_brieman" in df.columns):
    print("Variable '" + catFeatureName + "_brieman" + "' already exists")
    return df
  
  res = df.join(F.broadcast(breimanRanks), df[catFeatureName] == breimanRanks[catFeatureName], how='left') \
          .drop(breimanRanks[catFeatureName]) \
          .drop(breimanRanks['avg(' + outcomeName + ')']) \
          .fillna(-1, [catFeatureName + "_brieman"])
  return res

# COMMAND ----------

# MAGIC %md
# MAGIC With these two functions, we can apply Breiman's Theorem to each of our 8 categorical features separately and generate new columns numerical that will be suffixed with `_brieman`. Note that the Breiman ranks will always be generated by referring to the training dataset only. Once these are generated, they can be applied to any dataset, as the ranks are depending on the training dataset and should not be influenced by our test set. This is done below (note that we'll split the dataset again to ensure that our training dataset has all new features that were generated in the previous sections).

# COMMAND ----------

# Regenerate splits for Breiman ranking prep (so have context of new 
# features added that require application of Breiman's Theorem)
mini_train, train, val, test = SplitDataset(airlines)

# COMMAND ----------

# Apply breiman ranking to all datasets, based on ranking developed from training data
breimanRanksDict = {} # save to apply later as needed
featuresToApplyBreimanRanks = catFeatureNames + intFeatureNames + holFeatureNames
for feature in featuresToApplyBreimanRanks:
  # Get ranks for feature, based on training data only
  ranksDict = GenerateBreimanRanks(train, feature, outcomeName)
  breimanRanksDict[feature] = ranksDict
  
  # Apply Breiman's theorem & do feature transformation for all datasets
  mini_train = ApplyBreimansTheorem(mini_train, ranksDict, feature, outcomeName)
  train = ApplyBreimansTheorem(train, ranksDict, feature, outcomeName)
  val = ApplyBreimansTheorem(val, ranksDict, feature, outcomeName)
  test = ApplyBreimansTheorem(test, ranksDict, feature, outcomeName)
  airlines = ApplyBreimansTheorem(airlines, ranksDict, feature, outcomeName)
  
briFeatureNames = [entry + "_brieman" for entry in breimanRanksDict]

# Show examples of Breiman features
selectCols = []
for feature in featuresToApplyBreimanRanks:
  selectCols.append(feature)
  selectCols.append(feature + "_brieman")
display(train.select(selectCols).take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC With the application of Breiman's Theorem to our dataset, we now have a series of new features that we will leverage in section V for our chosen algorithm implementation, which are listed below. Note that we have checkpointed this work and saved the updated dataset to parquet format to be read from prior to model training (to avoid needing to re-compute these features).

# COMMAND ----------

print("All Available Features:")
print("------------------------")
print(" - Numerical Features: \t\t", numFeatureNames) # numerical features
print(" - Cont. Numerical Features:\t", contNumFeatureNames) # numerical features, but have binned versions in binFeatureNames
print(" - Categorical Features: \t", catFeatureNames) # categorical features
print(" - Binned Features: \t\t", binFeatureNames) # numerical features
print(" - Interaction Features: \t", intFeatureNames) # categorical features
print(" - Holiday Feature: \t\t", holFeatureNames) # categorical features
print(" - Origin Activity Feature: \t", orgFeatureNames) # numerical features
print(" - Breiman Ranked Features: \t", briFeatureNames) # numerical features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Balancing the Training Dataset
# MAGIC In the EDA task #3 from the previous section, we saw that our entire *Airline Delays* dataset is highly imbalanced, with only 11% of the flight data constituting flights with departure delays. As discussed previously, in order to develop a useful model that isn't biased by the majority class in the training dataset. To address this dataset imbalance issue, we will concentrate on two primary methods: SMOTE and Majority Class Splitting.

# COMMAND ----------

# MAGIC %md
# MAGIC #### SMOTE (Synthetic Minority Over-sampling Technique) 
# MAGIC 
# MAGIC A dataset is imbalanced if the classes are not approximately equally represented. Training a machine learning model with an imbalanced dataset causes the model to develop a certain bias towards the majority class. To tackle the issue of class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was introduced by Chawla et al. in 2002.(Chawla, Nitesh V., et al. “SMOTE: synthetic minority over-sampling technique.” Journal of artificial intelligence research16 (2002): 321–357).
# MAGIC 
# MAGIC Under-sampling of the majority class or/and over-sampling of the minority class have been proposed as a good means of increasing the sensitivity of a classifier to the minority class. However, under-sampling the majority class samples could potentially lead to loss of important information. Also, over-sampling the minority class could lead to overfitting. The reason is fairly straightforward. Consider the effect on the decision regions in feature space when minority over-sampling is done by replication (sampling with replacement). With replication, the decision region that results in a classification decision for the minority class can actually become smaller and more specific as the minority samples in the region are replicated. This is the opposite of the desired effect. 
# MAGIC 
# MAGIC SMOTE provides a new approach to over-sampling. It is an over-sampling approach in which the minority class is over-sampled by creating “synthetic” examples rather than by over-sampling with replacement. This approach is inspired by a technique that proved successful in handwritten character recognition (Ha & Bunke, 1997). They created extra training data by performing certain operations on real data. In their case, operations like rotation and skew were natural ways to perturb the training data. SMOTe generates synthetic examples in a less application-specific manner, by operating in “feature space” rather than “data space”. The minority class is over-sampled by taking each minority class sample and introducing synthetic examples along the line segments joining the k nearest neighbors. Our implementation currently uses seven nearest neighbors.
# MAGIC 
# MAGIC Synthetic samples are generated in the following way: 
# MAGIC - Take the difference between the feature vector (of the sample) under consideration and the feature vector of its nearest neighbor. 
# MAGIC - Multiply this difference by a random number between 0 and 1 to scale the difference.
# MAGIC - Add the scaled difference to the feature vector under consideration. 
# MAGIC 
# MAGIC The diagrams below highlight the steps of capturing the region of k-nearest neighbors for a given datapoint (in orange), connecting the datapoint under consideration to is k-nearest neighbors (also in orange) via the blue dotted lines in feature space, and generating a white synthetic datapoint along these blue dotted lines. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Visualizing SMOTE
# MAGIC <img src="https://github.com/nsandadi/Images/blob/master/Visualizing%20SMOTE.png?raw=true" width=100%>
# MAGIC 
# MAGIC Source: https://www.youtube.com/watch?v=FheTDyCwRdE

# COMMAND ----------

# MAGIC %md
# MAGIC By following these steps, we can generate a new random point along the line segment between two specific feature vectors to be a new synthetic datapoint. This approach effectively forces the decision region of the minority class to become more general. The synthetic examples cause the classifier to create larger and less specific decision regions (that contain nearby minority class points), rather than smaller and more specific regions. More general regions are now learned for the minority class samples rather than those being subsumed by the majority class samples around them. SMOTE provides more related minority class samples to learn from, thus allowing a learner to carve broader decision regions, leading to more coverage of the minority class. The effect is that models, in theory, will generalize better.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Deviations from the original paper
# MAGIC 
# MAGIC 1. The original paper shows that a combination of our method of over-sampling the minority class and under-sampling the majority class can achieve better classifier performance (in ROC space) than only under-sampling the majority class. However, for this project we only created "synthetic data" from minority samples without under-sampling the majority class, since under-sampling could cause potential loss of information useful for prediction.
# MAGIC 
# MAGIC 2. The number of minority class samples from our training set were approximately two million. However, running K Nearest Neighbors on each of these ~2M samples is not scalable as KNN needs to store the list of ~2M feature vectors in memory. We considered and implemented two approaches to address this scalability challenge:
# MAGIC   > i. Find KNN of each minority sample (feature vector) from a random sample (0.005%) of all the minority sample (feature vectors). This will produce a list of feature vectors small enough to fit in memory.
# MAGIC   
# MAGIC   > ii. Create clusters of minority sample data using K-means algorithm, run KNN on each cluster in parallel and generate synthetic data for each cluster. This approach uses the entire training data. We split the data into 1000 clusters.
# MAGIC 
# MAGIC   > Out of the above two approaches, we found the second approach took less time to run. Also, when we compared the distribution of minority samples from the original training set vs. all the minority samples after applying SMOTE, the data generated by the second approach matched the original feature distributions of the training set better than the data generated by the first approach. Thus for the remainder of this analysis, we will proceed with the second approach for balancing our dataset using SMOTE. 
# MAGIC 
# MAGIC We have provided an additional notebook with the full implementation of SMOTE for this project, which can be found here:
# MAGIC 
# MAGIC https://dbc-b1c912e7-d804.cloud.databricks.com/?o=7564214546094626#notebook/2791835342809045/revision/1586673126000

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Implementing SMOTE at Scale
# MAGIC The code below provides a summary of the functions we generated for SMOTE-ing our training dataset. These functions are documented below for reference and have been applied via the notebook mentioned previously.

# COMMAND ----------

# HELPER FUNCTIONS

from pyspark.ml.clustering import KMeans

# Train a k-means model on the minority samples (feature vectors)
def kmeans_model(featureVect, k):
  """
  Function to run k-means algorithm.
    arg:
        featureVect - (dataframe) feature vectors of minority samples
        k - (int) number of clusters
    returns:
        smote_rdd - (dataframe) predictions of k-means for each minority sample
  """
  kmeans = KMeans().setK(k).setSeed(1)
  model = kmeans.fit(featureVect)
  predict = model.transform(featureVect)
  return predict

# Calculate the Euclidean distance between two feature vectors
def euclidean_distance(row1, row2):
  """
  Function to calculate Euclidean distance.
   arg:
       row1, row2 - (list) feature vector
   returns:
       distance - (float) euclidean distance between two feature vectors
  """
  distance = 0.0
  for i in range(len(row1)-1):
      distance += (row1[i] - row2[i])**2
  return math.sqrt(distance)
  
# Locate the nearest neighbors
def get_neighbors(train, test_row, num_neighbors):
  """
  Function to calculate nearest neighbors.
   arg:
       train - (list) list of feature vectors from which to find nearest neighbors
       test_row - (list) feature vector under consideration whose nearest neighbors must be found
       num_neighbors - (int) number of nearest neighbors
   returns:
       neighbors - (list) nearest neighbors
  """
  distances = list()
  for train_row in train:
      dist = euclidean_distance(test_row, train_row)
      distances.append((train_row, dist))
  distances.sort(key=lambda tup: tup[1])
  neighbors = list()
  for i in range(num_neighbors):
      neighbors.append(distances[i+1][0])
  return neighbors
  
# Generate synthetic records
def synthetic(list1, list2):
  """
  Function to generate synthetic data.
   arg:
       list1, list2 - (list) feature vectors from which synthetic data point is generated
   returns:
       synthetic_records - (list) synthetic data
  """
  synthetic_records = []
  for i in range(len(list1)):
    synthetic_records.append(round(list1[i] + ((list2[i]-list1[i])*random.uniform(0, 1))))
  return synthetic_records

  
# RUNNING SMOTE AT SCALE

# Convert the k-means predictions dataframe into rdd, find nearest neighbors and generate synthetic data
def SmoteSampling(predict, k):
  """
  Function to create an rdd of sunthetic data.
    arg:
        predict - (dataframe) k-means predictions
        k - (int) number of nearest neighbors
    returns:
        smote_rdd - (rdd) synthetic data in the form of rows of feature vectors with label 1 or minority class
  """
  smote_rdd = predict.rdd.map(lambda x: (x[0], [list(x[1])])) \
                         .reduceByKey(lambda x,y: x+y) \
                         .flatMap(lambda x: [(n, get_neighbors(x[1], n, k)) for n in x[1]]) \
                         .flatMap(lambda x: [synthetic(x[0],n) for n in x[1]]) \
                         .map(lambda x: Row(features = DenseVector(x), label = 1)) \
                         .cache()
  return smote_rdd


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Using SMOTE-d Training Data
# MAGIC We applied our version of the SMOTE algorithm on the original subset of the training data we have worked with preivously. Since the feature engineering described in earlier in this section has not been applied to our SMOTE-d training data, we will apply the same feature engineering steps here as an additional step before being able to use the SMOTE-d data for training. The result will also be saved to parquet format to help with training moving forward. 

# COMMAND ----------

# Apply Feature Engineering described above to smoted training data
# Provided Breiman ranks dict should be based on unsmoted training data
def ApplyFeatureEngineeringToSmotedTrainingData(df, breimanRanksDict):
  # Select features to ensure features use pascal casing
  outcomeName = 'Dep_Del30'
  numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group']
  contNumFeatureNames = ['CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance']
  catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
  df = df.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames)
  
  
  # Bin Continuous Features
  df = BinFeature(df, 'CRS_Dep_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
  df = BinFeature(df, 'CRS_Arr_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
  df = BinFeature(df, 'CRS_Elapsed_Time', splits = [float("-inf")] + [i for i in range(0, 660 + 1, 60)] + [float("inf")]) # 1-hour blocks

  # Add Interaction Features
  interactions = [('Month', 'Day_Of_Month', 'Day_Of_Year'), 
                  ('Origin', 'Dest', 'Origin_Dest'),
                  ('Day_Of_Week', 'CRS_Dep_Time_bin', 'Dep_Time_Of_Week'),
                  ('Day_Of_Week', 'CRS_Arr_Time_bin', 'Arr_Time_Of_Week')]
  df = AddInteractions(df, interactions)
  intFeatureNames = [i[2] for i in interactions]

  # Add FL_Date column (for holiday feature generation)
  df = df.withColumn('FL_Date', F.concat_ws("-", F.col("Year"), F.col("Month"), F.col("Day_Of_Month")).cast(DateType()))
  
  # Add Holiday Feature
  df = AddHolidayFeature(df)
  holFeatureNames = ['Holiday']

  # Add Origin Activity Feature
  df = AddOriginActivityFeature(df)

  # Apply Breiman Ranking
  featuresToApplyBreimanRanks = catFeatureNames + intFeatureNames + holFeatureNames
  for feature in featuresToApplyBreimanRanks:
    # Apply Breiman's method & do feature transformation for all datasets
    df = ApplyBreimansMethod(df, breimanRanksDict[feature], feature, outcomeName)
    
  return df

# Read prepared data from parquet for training
def ReadDataFromParquet(dataName):
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

# Prep Smoted training data
train_smoted = ReadDataFromParquet('smoted_train_kmeans')
train_smoted = ApplyFeatureEngineeringToSmotedTrainingData(train_smoted, breimanRanksDict)
train_smoted = WriteAndRefDataToParquet(train_smoted, 'augmented_smoted_train_kmeans')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Majority Class Splitting
# MAGIC Another method for balancing the training dataset is with the use of we call the "Majority Class Splitting" technique, which is described in the paper "New Applications of Ensembles of Classifiers" by Barandela et. al (http://marmota.dlsi.uji.es/WebBIB/papers/2003/paa-2.pdf). Referring to the paper, this is a dataset balancing approach where instead of oversampling the minority class to balance the dataset (similar to what we did with SMOTE), we take an majority lass undersampling approach to get a data subset that is balanced between majority and minority classes. 
# MAGIC 
# MAGIC However, unlike the traditional majority class undersampling techniques, majority class splitting will generate multiple balanced subsets of the original dataset. Let's consider a hypothetical case where the ratio of majority to minority classes is 3:1. With the majority class splitting approach, we will take the majority class and randomly split the majority class into 3 parts, where each part is approximately the same size as the minority class. For each of these majority class samples, we will join the sample back with the full minority class. This will in turn generate 3 balanced subsets of the original dataset, each of which contains the full minority class. This technique is depicted in the diagram below:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/nsandadi/Images/blob/master/Majority%20Class%20Splitting.png?raw=true" width=50%>

# COMMAND ----------

# MAGIC %md
# MAGIC In the case of the training dataset for the *Airline Delays*, we have a ratio of 7:1 for the majority and minority class, which will generate 7 subsets of data using the majority splitting technique. While this is a possible dataset balance approach for us to use in model training, this approach is best-suited for ensemble approaches, where each model in the ensemble is assigned one balanced subset of data for training. With this approach, each model in the ensemble will have a balanced dataset to learn from, reducing bias in the indiviual models, but no majority class data is lost as would be the case for traditional undersampling techniques. We will explore the use of majority class splitting when we explore ensemble approaches to predictin departure dleays in section V of the report. 

# COMMAND ----------

# DBTITLE 1,Helper Code to Load All Data for Model Training (TODO: remove)
# Read prepared data from parquet for training
def ReadDataFromParquet(dataName):
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

airlines = ReadDataFromParquet('augmented')
mini_train = ReadDataFromParquet('augmented_mini_train')
train = ReadDataFromParquet('augmented_train')
val = ReadDataFromParquet('augmented_val')
test = ReadDataFromParquet('augmented_test')
train_smoted = ReadDataFromParquet('augmented_smoted_train_kmeans')

###########################################
# Define all variables for easy reference #
###########################################

# Numerical Variables to use for training
outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group'] ##
contNumFeatureNames = ['CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
binFeatureNames = ['CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin'] ##
intFeatureNames = ['Day_Of_Year', 'Origin_Dest', 'Dep_Time_Of_Week', 'Arr_Time_Of_Week']
holFeatureNames = ['Holiday']
orgFeatureNames = ['Origin_Activity'] ##
briFeatureNames = ['Op_Unique_Carrier_brieman', 'Origin_brieman', 'Dest_brieman', 'Day_Of_Year_brieman', 'Origin_Dest_brieman', 'Dep_Time_Of_Week_brieman', 'Arr_Time_Of_Week_brieman', 'Holiday_brieman'] ##

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## IV. Algorithm Exploration
# MAGIC - Apply 2-3 Algorithms
# MAGIC     - Logistic Regression
# MAGIC     - SVM
# MAGIC     - Naive Bayes
# MAGIC     - Decision Tree
# MAGIC - Expectations / Tradeoffs
# MAGIC     - All able to do classificaiton (Delay/No Delay)
# MAGIC     - LR: Interpretable (good), get estimate for effect of each variable, manual feature selection required, need to deal with multi-collinearity, among other things
# MAGIC     - SVM: needed to transform & one-hot encode categorical variables
# MAGIC     - DT: Not a lot of hyper parameter tuning/feature selection (most automated), 
# MAGIC - Results
# MAGIC 
# MAGIC -- See Task 3 Submission :) --
# MAGIC https://docs.google.com/document/d/1IaGOgYWSRCH-WgDzJ7N2Lw6y5h8zxpTQ8HHg9AqpML8/edit?usp=sharing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Condiderations for Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC For the algorithm exploration the important considerations were as follows:       
# MAGIC - Algorithm must scale 
# MAGIC - it must be suitable for classification
# MAGIC - it must handle continuous and categorical variables
# MAGIC - model must be interpretable / explainable
# MAGIC 
# MAGIC We performed minimal EDA and feature engineering for this exploration. The outcome variable Dep_Del30 marked the flight as "delayed" if the departure delay was 30 minutes or more. Given this dataset roughly had 10% of the true positives, it was a very imbalanced dataset. This means by taking a baseline model which simply assumes the outcome as 'no delay' it achieved an accuracy of ~90%. But practically this would be a useless model as it fails to identify any true positives. Instead we need to consider a model that will improve the repdiction of more true positives/ reduce false negatives.  
# MAGIC 
# MAGIC Based on the modeling results we observed the following:     
# MAGIC - Both logistic regression and SVM had high accuracy but failed to predict any of the true positives. Both of these models also failed to predict any false positives. Based on this we concluded both the models were predicting "no delay" all the time.
# MAGIC - Naive Bayes identified the most number of true positives, but it also had roughly 5 times as many false positives. It had an accuracy slightly better than a coin toss.
# MAGIC - Decision tree identified some true positives, it identified a similar set of false positives. It also demonstrated good accuracy and precision. Its recall rate wasn't very high. Overall emperically, this model looked much more balanced fit for our further analysis.
# MAGIC 
# MAGIC Theoretically logistic regression is very easy to model, interpret and train. The main limitation of logistic regression is the **multi-collinearity** among the features. Logisitc regression is also susceptible to overfitting due to imbalanced data. Support verctor machines on the other hand is not suitable for large datasets. As these algorithms use hyperplanes it is much harder to interpret them. Larger data sets take a long time to process in relation to other models. With Naive Bayes, they are easy to process and scale really well. Like logistic regression it makes an assumption that features are independent of each other which is not true in many cases.
# MAGIC 
# MAGIC On the other hand, decision trees also seemed theoretically promising. Compared to other algorithms desicion trees require less data preparation during pre-processing. It also doesn't require normalization or scaling and can handle null values gracefully. It is also easily interpretable. They require higher time to train the model. But based on the keypoints discussed able we picked the decision tree as our algorithm to move forward.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset Preparation for Algorithm Exploration

# COMMAND ----------


#subset the dataset to the features in numFeatureNames, contNumFeatureNames  & catFeatureNames
mini_train_algo = mini_train.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames)
train_algo = train.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames)
val_algo = val.select([outcomeName] + numFeatureNames + contNumFeatureNames + catFeatureNames)

# Define outcome & features to use in model development
# numFeatureNames are continuous features
# catFeatureNames are categorical features

outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group']
contNumFeatureNames = ['CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']

# COMMAND ----------

# this function train the model 
# Only support vector machines (svm) use one hot encoding for the categorical variable 

def train_model(df,algorithm,categoricalCols,continuousCols,labelCol,svmflag):

    indexers = [ StringIndexer(inputCol=cat, outputCol= cat + "_indexed", handleInvalid="keep") for cat in categoricalCols ]
  
    # If it is svm do hot encoding
    if svmflag == True:
      encoder = OneHotEncoderEstimator(inputCols=[indexer.getOutputCol() for indexer in indexers],
                                 outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers])
      assembler = VectorAssembler(inputCols=encoder.getOutputCols() + continuousCols, outputCol="features")
    # else skip it
    else:
      assembler = VectorAssembler(inputCols = continuousCols + [cat + "_indexed" for cat in categoricalCols], outputCol = "features")
      
    # choose the appropriate model regression  
    if algorithm == 'lr':
      lr = LogisticRegression(labelCol = outcomeName, featuresCol="features", maxIter=100, regParam=0.1, elasticNetParam=0)
      pipeline = Pipeline(stages=indexers + [assembler,lr])

    elif algorithm == 'dt':
      dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = 8, maxBins=366)
      pipeline = Pipeline(stages=indexers + [assembler,dt])
      
    elif algorithm == 'nb':
      # set the CRS_Elapsed_Time to 0 if its negative
      df = df.withColumn('CRS_Elapsed_Time', when(df['CRS_Elapsed_Time'] < 0, 0.0).otherwise(df['CRS_Elapsed_Time']))

      nb = NaiveBayes(labelCol = outcomeName, featuresCol = "features", smoothing = 1)
      pipeline = Pipeline(stages=indexers + [assembler,nb])
      
    elif algorithm == 'svm':
      svc = LinearSVC(labelCol = outcomeName, featuresCol = "features", maxIter=50, regParam=0.1)
      pipeline = Pipeline(stages=indexers + [encoder,assembler,svc])
      
    else:
      pass
    
    tr_model=pipeline.fit(df)

    return tr_model

# COMMAND ----------

# MAGIC  %md
# MAGIC  ### Model Evaluation Function

# COMMAND ----------

# Model Evaluation
# This function takes predictions dataframe and outcomeName and calculates the scores.
# If the returnval is true it will return the values otherwise it will print it.
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

# Evaluates model predictions for the provided predictions
# Predictions must have two columns: prediction & label
def EvaluateModelPredictions(predict_df, dataName=None, outcomeName='label', ReturnVal=False):
    
    if not ReturnVal :
      print("Model Evaluation - " + dataName)
      print("-----------------------------")
      
    evaluation_df = (predict_df \
                     .withColumn('metric', F.concat(F.col(outcomeName), F.lit('-'), F.col("prediction"))).groupBy("metric") \
                     .count() \
                     .toPandas() \
                     .assign(label = lambda df: df.metric.map({'1-1.0': 'TP', '1-0.0': 'FN', '0-0.0': 'TN', '0-1.0' : 'FP'})))
    metric = evaluation_df.set_index("label").to_dict()['count']
    
    # Default missing metrics to 0
    for key in ['TP', 'TN', 'FP', 'FN']:
      metric[key] = metric.get(key, 0)
      
	# Compute metrics
    total = metric['TP'] + metric['FN'] + metric['TN'] + metric['FP']
    accuracy = 0 if (total == 0) else (metric['TP'] + metric['TN'])/total
    precision = 0 if ((metric['TP'] + metric['FP']) == 0) else metric['TP'] /(metric['TP'] + metric['FP'])
    recall = 0 if ((metric['TP'] + metric['FN']) == 0) else metric['TP'] /(metric['TP'] + metric['FN'])
    fscore = 0 if (precision+recall) == 0 else 2 * precision * recall /(precision+recall)
    
    # Compute Area under ROC & PR Curve
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=outcomeName)
    areaUnderROC = evaluator.evaluate(predict_df, {evaluator.metricName: "areaUnderROC"})
    areaUnderPRC = evaluator.evaluate(predict_df, {evaluator.metricName: "areaUnderPR"})
    
    if ReturnVal : return {'Accuracy' : round(accuracy, 7), 
                           'Precision' : round(precision, 7), 
                           'Recall' : round(recall, 7), 
                           'f-score' : round(fscore, 7), 
                           'areaUnderROC' : round(areaUnderROC, 7), 
                           'AreaUnderPRC' : round(areaUnderPRC, 7),
                           'metric' : metric}
    
    print("Accuracy = {}, Precision = {}, Recall = {}, f-score = {}, AreaUnderROC = {}, AreaUnderPRC = {}".format(
        round(accuracy, 7), round(precision, 7),round(recall, 7),round(fscore, 7), round(areaUnderROC, 7), round(areaUnderPRC, 7))),
    print("\nConfusion Matrix:\n", pd.DataFrame.from_dict(metric, orient='index', columns=['count']), '\n')

# This function takes a trained model and predicts outcome.
# And calls model evaluation function
def PredictAndEvaluate(model, data, dataName, outcomeName):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Algorithm Exploration - Logistic Regression, Decision Tress, Naive Bayes & Support Vector Machines

# COMMAND ----------

# Train the model using the "train" dataset and test against the "val" dataset
dataName = 'val'
algorithms = ['lr','dt','nb','svm']
for algorithm in algorithms:
  newnumFeatureNames = numFeatureNames + contNumFeatureNames
  titleName = dataName+ ' with ' + algorithm
  # if svm the train_model need to apply one-hot encoding
  if algorithm == 'svm':
    tr_model = train_model(train_algo,algorithm,catFeatureNames,newnumFeatureNames,outcomeName,svmflag=True)
    PredictAndEvaluate(tr_model, val_algo, titleName, outcomeName)
  else:
    tr_model = train_model(train_algo,algorithm,catFeatureNames,newnumFeatureNames,outcomeName,svmflag=False)
    PredictAndEvaluate(tr_model, val_algo, titleName, outcomeName)


# COMMAND ----------

# MAGIC %md
# MAGIC ## V. Algorithm Implementation
# MAGIC - Toy example likely with a decision tree on a mini_mini_train dataset (like 10 rows)
# MAGIC - Walk through training the model and doing inference
# MAGIC - Show baseline "dummy" model results on train/validation/test that only predicts 0 (hardcode all predictions to be 0)
# MAGIC - Show basic decision tree and how it performs with unstacked data & discuss dataset imbalance
# MAGIC - Show how decision tree functions by comparison if we stack/smote the data (maybe with just a single stack)
# MAGIC - Move to ensemble of Decision Trees with stacked approach (maybe smote)
# MAGIC - Move to ensemble of Random Forests with stacked approach (maybe smote)
# MAGIC - Also do GBT & ensemble of GBT (find a good explanation for why)
# MAGIC - try to parallelize training of ensembles

# COMMAND ----------

##################
## START HERE!! ##
##################

# Re-read augmented data in parquet & avro formats
def ReadDataInParquetAndAvro(dataName):
  data_parquet = spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  data_avro = spark.read.format("avro").load(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".avro")
  return (data_parquet, data_avro)

mini_train, mini_train_avro = ReadDataInParquetAndAvro('augmented_mini_train')
train, train_avro = ReadDataInParquetAndAvro('augmented_train')
val, val_avro = ReadDataInParquetAndAvro('augmented_val')
test, test_avro = ReadDataInParquetAndAvro('augmented_test')

# Redefine feature names in one place
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
binFeatureNames = ['CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin']
intFeatureNames = ['Day_Of_Year', 'Origin_Dest', 'Dep_Time_Of_Week', 'Arr_Time_Of_Week']
holFeatureNames = ['Holiday']
orgFeatureNames = ['Origin_Activity']
briFeatureNames = ['Op_Unique_Carrier_brieman', 'Origin_brieman', 'Dest_brieman', 'Day_Of_Year_brieman', 'Origin_Dest_brieman', 'Dep_Time_Of_Week_brieman', 'Arr_Time_Of_Week_brieman', 'Holiday_brieman']
outcomeName = 'Dep_Del30'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Toy Example: Decision Trees

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dataset

# COMMAND ----------

## Build the toy example dataset from the cleaned and transformed mini training set
# toy_dataset = mini_train.select(['Dep_Del30', 'Day_Of_Week', 'Origin', 'Op_Unique_Carrier']) \
#                         .filter(mini_train['Dep_Del30'] == 0) \
#                         .sample(False, 0.0039, 8)

# toy_dataset = toy_dataset.union(mini_train.select(['Dep_Del30', 'Day_Of_Week', 'Origin', 'Op_Unique_Carrier']) \
#                          .filter(mini_train['Dep_Del30'] == 1) \
#                          .sample(False, 0.025, 8))

## Save the toy example dataset
# toy_dataset = WriteAndRefDataToParquet(toy_dataset, 'toy_dataset')

# COMMAND ----------

# Load the toy example dataset
toy_dataset = spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_toy_dataset.parquet")
display(toy_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Introduction to Decision Trees
# MAGIC Decision trees predict the label (or class) by evaluating a set of rules that follow an IF-THEN-ELSE pattern. The questions are the nodes, and the answers (true or false) are the branches in the tree to the child nodes. A decision tree model estimates the minimum number of true/false questions needed, to assess the probability of making a correct decision. 
# MAGIC 
# MAGIC We use CART decision tree algorithm (Classification and Regression Trees). Decision Tree is a greedy algorithm that considers all features to select the best feature for the split. Initially, we have a root node for the tree. The root node receives the entire training set as input and all subsequent nodes receive a subset of rows as input. Each node asks a true/false question about one of the features using a threshold and in response, the dataset is split into two subsets. The subsets become input to the child nodes added to the tree for the next level of splitting. The goal is to produce the purest distribution of labels at each node.
# MAGIC 
# MAGIC If a node contains examples of only a single type of label, it has 100% purity and becomes a leaf node. The subset doesn't need to be split any further. On the other hand, if a node still contains mixed labels, the decision tree chooses another question and threshold, based on which the dataset is split further. The trick to building an effective tree is to decide which feature to select and at which node. To do this, we need to quantify how well a feature and threshold can split the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Entropy
# MAGIC Entropy is a measure of disorder in the dataset. It characterizes the (im)purity of an arbitrary collection of examples. In decision trees, at each node, we split the data and try to group together samples that belong in the same class. The objective is to maximize the purity of the groups each time a new child node of the tree is created. The goal is to decrease the entropy as much as possible at each split. Entropy ranges between 0 and 1. Entropy of 0 indicates a pure set (i.e), group of observations containing only one label. 
# MAGIC 
# MAGIC ##### Gini impurity and Information gain
# MAGIC 
# MAGIC We quantify the amount of uncertainity at a single node by a metric called the gini impurity. We can quantify how much a split reduces the uncertainity by using a metric called the information gain. Information gain is the expected reduction in entropy caused by partitioning the examples according to a given feature. These two metrics are used to select the best feature and threshold at each split point. The best feature reduces the uncertainity the most. Given the feature and threshold, the algorithm recursively buils the tree at each of the new child nodes. This process continues until all the nodes are pure or we reach a stopping criteria (such as a minimum number of examples).

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Mathematical definition of entropy
# MAGIC 
# MAGIC The general formula for entropy is:
# MAGIC $$ E = \sum_i -p_i {\log_2 p_i} $$ where \\(p_i\\) is the frequentist probability of elements in class \\(i\\).
# MAGIC 
# MAGIC Since our dataset has a binary classification, all the observations fall into one of two classes. Suppose we have N observations in the dataset. Let's assume that n observations belong to label 1 and m = N - n observations belong to label 0. p and q, the ratios of elements of each label in the dataset are given by:
# MAGIC 
# MAGIC $$p = \frac{n}{N}$$ $$q = \frac{m}{N} = 1-p $$
# MAGIC 
# MAGIC Entropy is given by the following equation:
# MAGIC $$E = -p {\log_2 (p)} -q {\log_2 (q)}$$

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Entropy at Level 0
# MAGIC 
# MAGIC In our toy dataset, we have ten observations. Four of them have label 1 and six of them have label 0. Thus, entropy at the root node is given by:
# MAGIC 
# MAGIC $$ Entropy = -\frac{4}{10} {\log_2 (\frac{4}{10})} -\frac{6}{10} {\log_2 (\frac{6}{10})} = 0.966 $$
# MAGIC 
# MAGIC Entropy is close to 1 as we have a distribution close to 50/50 for the observations belonging to each class.

# COMMAND ----------

# MAGIC %md <img src="https://github.com/nsandadi/Images/blob/master/Decision_Tree_toy_example.jpg?raw=true" width=70%>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Entropy at Level 1
# MAGIC The data set that goes down each branch of the tree has its own entropy value. We can calculate the expected entropy for each possible attribute. This is the degree to which the entropy would change if we branch on a particular feature. We calculate the weighted entropy of the split by adding the entropies of the two child nodes, weighted by the proportion of examples from the parent node that ended up at that child.
# MAGIC 
# MAGIC ##### Weighted entropy calculations
# MAGIC $$ E(DayOfWeek) = -\frac{6}{10} {\log_2 (0.9042)} -\frac{4}{10} {\log_2 (1)} = 0.94 $$
# MAGIC $$ E(Carrier) = -\frac{6}{10} {\log_2 (0.9042)} -\frac{4}{10} {\log_2 (0)} = 0.54 $$
# MAGIC $$ E(Origin) = -\frac{5}{10} {\log_2 (0.72)} -\frac{5}{10} {\log_2 (0)} = 0.36 $$
# MAGIC 
# MAGIC ##### Information Gain at Level 1
# MAGIC Information gain gives the number of bits of information gained about the dataset by choosing a specific feature and threshold as the first branch of the decision tree, and is calculated as:
# MAGIC $$ G = Entropy (Parent) - Weighted Entropy (Child Nodes) $$
# MAGIC 
# MAGIC ##### Based on the information gain from the diagram above, the highest information gain is 0.606. Thus, the best feature to split on is Origin.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling Helpers
# MAGIC * Evaluation functions
# MAGIC * Decision Tree PrintModel Function

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Encodes a string column of labels to a column of label indices
# Set HandleInvalid to "keep" so that the indexer adds new indexes when it sees new labels (could also do "error" or "skip")
# Apply string indexer to categorical, binned, and interaction features (all string formatted), as applicable
# Docs: https://spark.apache.org/docs/latest/ml-features#stringindexer
def PrepStringIndexer(stringfeatureNames):
  return [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in stringfeatureNames]

# Use VectorAssembler() to merge our feature columns into a single vector column, which will be passed into the model. 
# We will not transform the dataset just yet as we will be passing the VectorAssembler into our ML Pipeline.
def PrepVectorAssembler(numericalFeatureNames, stringFeatureNames):
  return VectorAssembler(inputCols = numericalFeatureNames + [f + "_idx" for f in stringFeatureNames], outputCol = "features")

def TrainDecisionTreeModel(trainingData, stages, outcomeName, maxDepth, maxBins):
  # Train Model
  dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = maxDepth, maxBins=maxBins) 
  pipeline = Pipeline(stages = stages + [dt])
  dt_model = pipeline.fit(trainingData)
  return dt_model

def TrainRandomForestModel(trainingData, stages, outcomeNames, maxDepth, maxBins, maxTrees):
  # Train Model
  rf = RandomForestClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = maxDepth, maxBins=maxBins, maxTrees=maxTrees) 
  pipeline = Pipeline(stages = stages + [rf])
  rf_model = pipeline.fit(trainingData)
  return rf_model

import ast
import random

# Visualize the decision tree model that was trained in text form
# Note that the featureNames need to be in the same order they were provided
# to the vector assembler prior to training the model
def PrintDecisionTreeModel(model, featureNames):
  lines = model.toDebugString.split("\n")
  featuresUsed = set()
  print("\n")
  
  for line in lines:
    parts = line.split(" ")

    # Replace "feature #" with feature name
    if ("feature" in line):
      featureNumIdx = parts.index("(feature") + 1
      featureNum = int(parts[featureNumIdx])
      parts[featureNumIdx] = featureNames[featureNum] # replace feature number with actual feature name
      parts[featureNumIdx - 1] = "" # remove word "feature"
      featuresUsed.add(featureNames[featureNum])
      
    # For cateogrical features, summarize sets of values selected for easier reading
    if ("in" in parts):
      setIdx = parts.index("in") + 1
      vals = ast.literal_eval(parts[setIdx][:-1])
      vals = list(vals)
      numVals = len(vals)
      if (len(vals) > 5):
        newVals = random.sample(vals, 5)
        newVals = [str(int(d)) for d in newVals]
        newVals.append("...")
        vals = newVals
      parts[setIdx] = str(vals) + " (" + str(numVals) + " total values)"
      
    line = " ".join(parts)
    print(line)
    
  print("\n", "Provided Features: ", featureNames)
  print("\n", "    Used Features: ", featuresUsed)
  print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Decision Tree on Smoted (Balanced) Training Dataset

# COMMAND ----------

featureNames = numFeatureNames + binFeatureNames + orgFeatureNames + briFeatureNames
va_base = PrepVectorAssembler(numericalFeatureNames = featureNames, stringFeatureNames = [])
dt_model = TrainDecisionTreeModel(train_smoted, [va_base], outcomeName, maxDepth=10, maxBins=200)
PrintDecisionTreeModel(dt_model.stages[-1], featureNames)

# COMMAND ----------

PredictAndEvaluate(dt_model, train_smoted, 'train_smoted', outcomeName)
PredictAndEvaluate(dt_model, train, 'train', outcomeName)
PredictAndEvaluate(dt_model, val, 'val', outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Random Forest on Smoted (Balanced) Training Dataset

# COMMAND ----------

featureNames = numFeatureNames + binFeatureNames + orgFeatureNames + briFeatureNames
va_base = PrepVectorAssembler(numericalFeatureNames = featureNames, stringFeatureNames = [])
rf_model = TrainRandomForestModel(train_smoted, [va_base], outcomeName, maxDepth=10, maxBins=200, maxTrees=20)
# show feature improtance for RF model

# COMMAND ----------

eval = EvaluateModelPredictions(ensemble_test_prediction, dataName=data_name, ReturnVal=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Ensemble with Smoted (Balanced) Training Dataset

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Ensemble with Majority Class Splitted (Balanced) Training Dataset

# COMMAND ----------

# MAGIC %md 
# MAGIC Ensamble approach can be used to address the imbalanced training data problem. The approaches such as SMORT has scalability issues and ensamble approach try to overcome practical issues. The negative effects of imbalance can be avoided with out replicating the minority class and with out discarding information from majority class. In this approach, individual components of the ensemble with a balanced learning sample. Working in this way, it is possible to appropriately handle the difficulties of the imbalance, while avoiding the drawbacks inherent to the oversampling and undersampling techniques.
# MAGIC 
# MAGIC Stacking is a general framework that involves training a learning algorithm to combine the predictions of several other learning algorithms to make a final prediction. Stacking can be used to handle an imbalanced dataset. The steps can be outlined as below.  
# MAGIC (a) Group the **training** data into majority and minority class.   
# MAGIC (b) Split the  majority class into \\(N + 1\\) groups, each group containing same number of data points as that in minority class.    
# MAGIC (c) Create \\(N + 1\\) datasets for training by combining the each group from (b) with minority class from (a). Each of these groups will be balanced.  
# MAGIC (d) Use \\(N\\) datasets from (c) to train the first level classifier. Once the models are generated, use the remaing one dataset from (c) to generate predictions for each of these models. These \\(N\\) predictions are the features for the second level classifier. The target/label value for the second level classifier is the target/label value of this remaining dataset.  
# MAGIC (e) Train the second level classifier. A final pipeline can be created by combining the models.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform data for Ensemble
# MAGIC 
# MAGIC `mini_train`, `train`, `val` and `test` is transformed using VectorAssmebler into a feature vector - label format. The feature indexer is also calculated. 

# COMMAND ----------

def TransformDataForEnsemble(mini_train_data, train_data, val_data, test_data, train_smoted_data) :
  target       = ["Dep_Del30"]
  #all_features = ['Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Elapsed_Time', 'Distance',  'CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'Origin_Activity', 'Op_Unique_Carrier_brieman', 'Origin_brieman', 'Dest_brieman', 'Day_Of_Year_brieman', 'Origin_Dest_brieman', 'Dep_Time_Of_Week_brieman', 'Arr_Time_Of_Week_brieman', 'Holiday_brieman']
  all_features = numFeatureNames + binFeatureNames + orgFeatureNames + briFeatureNames
  
  assembler = VectorAssembler(inputCols=all_features, outputCol="features")
  ensemble_pipeline = Pipeline(stages=[assembler])

  tmp_mini_train, tmp_train, tmp_val, tmp_test, tmp_train_smoted = (ensemble_pipeline.fit(mini_train_data).transform(mini_train_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(train_data).transform(train_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(val_data).transform(val_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(test_data).transform(test_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(train_smoted_data).transform(train_smoted_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"))
  
  featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=400).fit(tmp_train
                                                                                                          .union(tmp_val)
                                                                                                          .union(tmp_test))
  featureIndexer_smorted = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=400).fit(tmp_train_smoted
                                                                                                        .union(tmp_val)
                                                                                                        .union(tmp_test))
  
  return all_features, featureIndexer, featureIndexer_smorted, tmp_mini_train, tmp_train, tmp_val, tmp_test, tmp_train_smoted


all_ensemble_features, ensemble_featureIndexer, ensemble_featureIndexer_smoted, ensemble_mini_train, ensemble_train, ensemble_val, ensemble_test, ensemble_train_smoted = TransformDataForEnsemble(mini_train, train, val, test, train_smoted)

print(all_ensemble_features)
ensemble_mini_train.show(2)

# COMMAND ----------

if False :
  ensemble_val.write.mode('overwrite').format("parquet").save("dbfs:/user/team20/finalnotebook/ensemble_val.v1.parquet")
  ensemble_test.write.mode('overwrite').format("parquet").save("dbfs:/user/team20/finalnotebook/ensemble_test.v1.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Balance the dataset, partition for further training
# MAGIC 
# MAGIC Balanced dataset is generated. We will generate a set of 10 datasets. 9 will be used for training the level one classifiers. The last one will be used to train the level two classfier. 

# COMMAND ----------

def PrepareDatasetForStacking(train, outcomeName, majClass = 0, minClass = 1):
  # Determine distribution of dataset for each outcome value (zero & one)
  ones, zeros = train.groupBy(outcomeName).count().sort(train[outcomeName].desc()).toPandas()["count"].to_list()

  # Set number of models & number of datasets (3 more than ratio majority to minority class)
  # last split use to train level 2 classifier
  num_splits = int(zeros/ones) + 3
  print("Number of splits : " + str(num_splits))
  
  # Split dataset for training individual modesl and for training the voting (ensemble) model
  zero_df = train.filter(outcomeName + ' == ' + str(majClass))
  one_df  = train.filter(outcomeName + ' == ' + str(minClass))

  # get number of values in minority class
  one_df_count = one_df.count()
  
  zeros_array = zero_df.randomSplit([1.0] * num_splits, 1)
  zeros_array_count = [s.count() for s in zeros_array]
  ones_array = [one_df.sample(False, min(0.999999999999, r/one_df_count), 1) for r in zeros_array_count]
  ones_array_count = [s.count() for s in ones_array]

  # Array of `num_models` datasets
  # below resampling (shuffling) may not be necessary for random forest.
  # Need to remove it in case of performance issues
  train_group = [a.union(b).sample(False, 0.999999999999, 1) for a, b in zip(zeros_array[0:-1], ones_array[0:-1])]
  
  # Construct dataset for voting (ensemble) model
  train_combiner = zeros_array[-1].union(ones_array[-1]).sample(False, 0.999999999999, 1) # Shuffle
  
  return (train_combiner, train_group)

# Prepare datasets for stacking
train_combiner, train_group = PrepareDatasetForStacking(ensemble_train, 'label')

# COMMAND ----------

# MAGIC %md 
# MAGIC Check for data set balancing, note that number of majority and minority classes are close enough.

# COMMAND ----------

print([[d.groupBy('label').count().toPandas()["count"].to_list()] for d in train_group], 
 train_combiner.groupBy('label').count().toPandas()["count"].to_list())

# COMMAND ----------

# MAGIC %md
# MAGIC Check for data set balancing for smoted data

# COMMAND ----------

print(ensemble_train_smoted.groupBy('label').count().toPandas()["count"].to_list())
smoted_splits = ensemble_train_smoted.randomSplit([1.0] * 10, 1)
train_combiner_smoted, train_group_smoted = smoted_splits[-1], smoted_splits[0:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train first-level classifiers

# COMMAND ----------

# MAGIC %md
# MAGIC Each of the first level classifier can be trained parallelly. Concurrency is obtained by using Python's ThreadPool utility, which triggers training of various models over many workers.

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from multiprocessing.pool import ThreadPool

# allow up to 10 concurrent threads
pool = ThreadPool(10)

# You can increase the timeout for broadcasts. Default is 300 s
spark.conf.set('spark.sql.broadcastTimeout', '900000ms')
spark.conf.get('spark.sql.broadcastTimeout')

# COMMAND ----------

def TrainEnsembleModels_parallel(en_train, featureIndexer, classifier) :
  job = []
  for num, _ in enumerate(en_train):
      print("Create ensemble model : " + str(num))      
      # Chain indexer and classifier in a Pipeline 
      job.append(Pipeline(stages=[featureIndexer, classifier]))
      
  return pool.map(lambda x: x[0].fit(x[1]), zip(job, en_train))

# The training is still done serially (code commented below) to avoid dabricks error.

# Parallel training is not done in databricks environment.      
# ensemble_model = TrainEnsembleModels_parallel(train_group, ensemble_featureIndexer, 
#                     # Type of model we can use.
#                     RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=5, numTrees=5, impurity='gini')
#                    )
# print("Training done")

def TrainEnsembleModels(en_train, featureIndexer, classifier) :
  model = []
  for num, train in enumerate(en_train):
      print("Create ensemble model : " + str(num))      
      model.append(Pipeline(stages=[featureIndexer, classifier]).fit(train))
  return model
      
ensemble_model = TrainEnsembleModels(train_group, ensemble_featureIndexer, 
                    # Type of model we can use.
                    #RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=5, numTrees=5, impurity='gini')
                    #RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=6, numTrees=25, impurity='gini')
                    # Works best
                    RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=8, numTrees=50, impurity='gini')
                   )

# COMMAND ----------

if False : 
  for i, model in enumerate(ensemble_model):
    model.save("dbfs:/user/team20/finalnotebook/ensemble_model" + str(i) +  ".v1.model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize feature importance for individual ensembles
# MAGIC 
# MAGIC Each feature's importance is the average of its importance across all trees in the ensemble. The importance vector is normalized to sum to 1.  
# MAGIC Reference : https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/classification/RandomForestClassifier.scala

# COMMAND ----------

from collections import defaultdict

def makedict(em, columns, features):
    plot = defaultdict(dict)
    rows = int(len(em)/columns)
    for num, m in enumerate(em):
        plot[num]['importance'] = list(m.stages[-1].featureImportances.toArray())
        plot[num]['features']   = features
        plot[num]['x_pos']      = int(num/columns)+1
        plot[num]['y_pos']      = num%columns+1
        plot[num]['title']      = "ensemble model {}".format(num)
    return plot, rows, columns

plt, rows, columns = makedict(ensemble_model, columns=3, features=all_ensemble_features)


fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple([plt[key]['title'] for key, value in plt.items()]))

for key, value in plt.items() :
    fig.add_trace(go.Bar(
      x=plt[key]['features'],
      y=plt[key]['importance'],
      #marker_color=list(map(lambda x: px.colors.sequential.thermal[x%12], range(0,len(plt[key]['features'])))),
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x] if x < 12 else px.colors.sequential.RdBu[x%12] , 
                      range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=1200, width=1200, title_text="Feature importance for individual ensembles (Majority class splitted)")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Construct a new data set based on the output of base classifiers
# MAGIC 
# MAGIC Do inference on first level classifiers. using the remaining balanced dataset. Collect the predictions. Use these predictions as features for level to voting model.

# COMMAND ----------

def do_ensemble_prediction(em, train_en) :
    prediction_array = []
    for num, m in enumerate(em) :
        predictions = em[num].transform(train_en)
        if num == 0 : prediction_array.append(
            predictions.select("label").withColumn('ROW_ID', F.monotonically_increasing_id())
        )      
        prediction_array.append(predictions
                                .select(F.col("prediction").alias("prediction_" + str(num)))
                                .withColumn('ROW_ID', F.monotonically_increasing_id())
        )
    return prediction_array

ensemble_prediction = do_ensemble_prediction(ensemble_model, train_combiner)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Assemble and transform data for second level training

# COMMAND ----------

# MAGIC %md
# MAGIC Join together the individual dataframes to create the final training dataset for level two model.

# COMMAND ----------

from functools import reduce

def assemble_dataframe(prediction_array) :
    def do_reduce(df1, df2): return df1.join(df2, "ROW_ID")
    return reduce(do_reduce, prediction_array).drop("ROW_ID")

def do_transform_final(df) :
    ensemble_columns = df.schema.names
    en_target, en_features = ensemble_columns[0], ensemble_columns[1:]

    assembler = VectorAssembler(inputCols=en_features, outputCol="features")
    pipeline = Pipeline(stages=[assembler])
    return (pipeline
            .fit(df)
            .transform(df)
            .select(["features"] + [en_target])
    )
  
reduced_df = assemble_dataframe(ensemble_prediction)
#reduced_df.show(2)
ensemble_transformed = do_transform_final(reduced_df)
#ensemble_transformed.show(2) 

# COMMAND ----------

reduced_df.show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Learn a second-level classifier based on training set from first-level.
# MAGIC 
# MAGIC Train the second level classifier. Use Logistic regression, Support vector machines and random forest for training.

# COMMAND ----------

def TrainCombiner(data, featureIndexer, classifier):
  # Chain indexer and forest in a Pipeline
  pipeline_ensemble = Pipeline(stages=[featureIndexer, classifier])

  # Train model.  This also runs the indexer.
  return pipeline_ensemble.fit(data)

# Set up VectorIndexer for second level training
ensemble_featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(ensemble_transformed)

# COMMAND ----------

# MAGIC %md
# MAGIC Discussion for Logistic Regression Voting Model: Want to use L2 regularization, not L1 for the voting model. If we use L1, this will effectively zero-out some of the coefficients for some of the models in the ensemble, which will be same as removing entire subsets of the data, which is not something we want--in this case, it is better to use L2.

# COMMAND ----------

# Logistic Regression
model_trained_ensemble_lr = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              LogisticRegression(featuresCol="indexedFeatures", maxIter=10, regParam=0.2))

# COMMAND ----------

# Linear SVM
model_trained_ensemble_svm = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              LinearSVC(featuresCol="indexedFeatures", maxIter=10, regParam=0.1))

# COMMAND ----------

# Random forest
model_trained_ensemble_rf = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              RandomForestClassifier(featuresCol="indexedFeatures", maxBins=20, maxDepth=5, numTrees=5, impurity='gini'))

# COMMAND ----------

if False : 
  model_trained_ensemble_lr.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_lr.v1.model")
  model_trained_ensemble_svm.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_svm.v1.model")
  model_trained_ensemble_rf.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_rf.v1.model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create final ensemble pipeline
# MAGIC 
# MAGIC Stitch together the final pipeline. The pipelne accepts below arguments :
# MAGIC * The level one model
# MAGIC * Level two voting model
# MAGIC * data for running inference.

# COMMAND ----------

def FinalEnsmblePipeline(model_comb, model_group, data) :
  return model_comb.transform(
    do_transform_final(
      assemble_dataframe(
        do_ensemble_prediction(model_group, data)
      )
    )
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model weights of ensembles
# MAGIC 
# MAGIC Below shows, Logistic regression and SV gives equal importance to all level one modes as opposed to Random forest.

# COMMAND ----------

from collections import defaultdict

def makedict(em, columns, features, info):
    plot = defaultdict(dict)
    rows = int(len(em)/columns)
    for num, m in enumerate(em):
        plot[num]['importance'] = em[num]
        plot[num]['features']   = features
        plot[num]['x_pos']      = int(num/columns)+1
        plot[num]['y_pos']      = num%columns+1
        plot[num]['title']      = info[num]
    return plot, rows, columns

plt, rows, columns = makedict([
  list(model_trained_ensemble_lr.stages[-1].coefficients.toArray()), 
  list(model_trained_ensemble_svm.stages[-1].coefficients.toArray()), 
  list(model_trained_ensemble_rf.stages[-1].featureImportances.toArray())], 
  columns=3, 
  features=[ "model {}".format(s) for s in range(0, 9)],
  info=[
    "Logistic regression - weights of ensembles",
    "SVM - weights of ensembles",
    "Random forest - weights of ensembles",
  ]
)

fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple([plt[key]['title'] for key, value in plt.items()]))

for key, value in plt.items() :
    fig.add_trace(go.Bar(
      x=plt[key]['features'],
      y=plt[key]['importance'],
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x], range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=400, width=1200, title_text="Feature importance for individual ensembles (Majority class splitted)")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the same steps with smorted data

# COMMAND ----------

# Train first-level classifiers using random forest.
ensemble_model_smoted = TrainEnsembleModels(train_group_smoted, ensemble_featureIndexer_smoted, 
                    RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=8, numTrees=50, impurity='gini')
                   )

# COMMAND ----------

from collections import defaultdict

def makedict(em, columns, features, info):
    plot = defaultdict(dict)
    rows = int(len(em)/columns)
    for num, m in enumerate(em):
        plot[num]['importance'] = em[num]
        plot[num]['features']   = features
        plot[num]['x_pos']      = int(num/columns)+1
        plot[num]['y_pos']      = num%columns+1
        plot[num]['title']      = info[num]
    return plot, rows, columns

plt, rows, columns = makedict([
  list(model_trained_ensemble_lr_smoted.stages[-1].coefficients.toArray()), 
  list(model_trained_ensemble_svm_smoted.stages[-1].coefficients.toArray()), 
  list(model_trained_ensemble_rf_smoted.stages[-1].featureImportances.toArray())], 
  columns=3, 
  features=[ "model {}".format(s) for s in range(0, 9)],
  info=[
    "Logistic regression - weights of ensembles",
    "SVM - weights of ensembles",
    "Random forest - weights of ensembles",
  ]
)

fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple([plt[key]['title'] for key, value in plt.items()]))

for key, value in plt.items() :
    fig.add_trace(go.Bar(
      x=plt[key]['features'],
      y=plt[key]['importance'],
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x], range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=400, width=1200, title_text="Feature importance for individual ensembles (Smoted)")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

from collections import defaultdict

def makedict(em, columns, features):
    plot = defaultdict(dict)
    rows = int(len(em)/columns)
    for num, m in enumerate(em):
        plot[num]['importance'] = list(m.stages[-1].featureImportances.toArray())
        plot[num]['features']   = features
        plot[num]['x_pos']      = int(num/columns)+1
        plot[num]['y_pos']      = num%columns+1
        plot[num]['title']      = "ensemble model {}".format(num)
    return plot, rows, columns

plt, rows, columns = makedict(ensemble_model_smoted, columns=3, features=all_ensemble_features)


fig = make_subplots(rows=rows, cols=columns, subplot_titles=tuple([plt[key]['title'] for key, value in plt.items()]))

for key, value in plt.items() :
    fig.add_trace(go.Bar(
      x=plt[key]['features'],
      y=plt[key]['importance'],
      #marker_color=list(map(lambda x: px.colors.sequential.thermal[x%12], range(0,len(plt[key]['features'])))),
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x] if x < 12 else px.colors.sequential.RdBu[x%12] , 
                      range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=1200, width=1200, title_text="Feature importance for individual ensembles (Smoted)")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

if False : 
  for i, model in enumerate(ensemble_model_smoted):
    model.save("dbfs:/user/team20/finalnotebook/ensemble_model_smoted" + str(i) +  ".v1.model")

# COMMAND ----------

# Construct a new data set based on the output of base classifiers
ensemble_prediction_smoted = do_ensemble_prediction(ensemble_model_smoted, train_combiner_smoted)

# COMMAND ----------

# Assemble and transform data for second level training
reduced_df_smoted = assemble_dataframe(ensemble_prediction_smoted)
ensemble_transformed_smoted = do_transform_final(reduced_df_smoted)

# COMMAND ----------

ensemble_featureIndexer_smoted = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(ensemble_transformed_smoted)

# COMMAND ----------

# Learn a second-level classifier based on training set from first-level.

# Logistic Regression
model_trained_ensemble_lr_smoted = TrainCombiner(ensemble_transformed_smoted, ensemble_featureIndexer_smoted, 
              LogisticRegression(featuresCol="indexedFeatures", maxIter=10, regParam=0.2))

# COMMAND ----------

# Linear SVM
model_trained_ensemble_svm_smoted = TrainCombiner(ensemble_transformed_smoted, ensemble_featureIndexer_smoted, 
              LinearSVC(featuresCol="indexedFeatures", maxIter=10, regParam=0.1))

# COMMAND ----------

# Random forest
model_trained_ensemble_rf_smoted = TrainCombiner(ensemble_transformed_smoted, ensemble_featureIndexer_smoted, 
              RandomForestClassifier(featuresCol="indexedFeatures", maxBins=20, maxDepth=5, numTrees=5, impurity='gini'))

# COMMAND ----------

if False : 
  model_trained_ensemble_lr_smoted.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_lr_smoted.v1.model")
  model_trained_ensemble_svm_smoted.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_svm_smoted.v1.model")
  model_trained_ensemble_rf_smoted.save("dbfs:/user/team20/finalnotebook/model_trained_ensemble_rf_smoted.v1.model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model evaluation
# MAGIC 
# MAGIC Run model evaluation with test set and validation set.

# COMMAND ----------

print("Loading model_trained_ensemble_lr_smoted.v1.model")
model_trained_ensemble_lr_smoted_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_lr_smoted.v1.model")
print("Loading model_trained_ensemble_svm_smoted.v1.model")
model_trained_ensemble_svm_smoted_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_svm_smoted.v1.model")
print("Loading model_trained_ensemble_rf_smoted.v1.model")
model_trained_ensemble_rf_smoted_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_rf_smoted.v1.model")
print("Loading model_trained_ensemble_lr.v1.model")
model_trained_ensemble_lr_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_lr.v1.model")
print("Loading model_trained_ensemble_svm.v1.model")
model_trained_ensemble_svm_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_svm.v1.model")
print("Loading model_trained_ensemble_rf.v1.model")
model_trained_ensemble_rf_load = pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/model_trained_ensemble_rf.v1.model")

# COMMAND ----------

ensemble_model_load = []
for i in range(0,9) :
  print("Loading ensemble_model " + str(i))
  ensemble_model_load.append(pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/ensemble_model" + str(i) + ".v1.model"))

# COMMAND ----------

ensemble_model_smoted_load = []
for i in range(0,9) :
  print("Loading ensemble_model " + str(i))
  ensemble_model_smoted_load.append(pl.PipelineModel.load("dbfs:/user/team20/finalnotebook/ensemble_model_smoted" + str(i) + ".v1.model"))

# COMMAND ----------

print("Loading ensemble_test.v1.parquet")
ensemble_test_load = spark.read.option("header", "true").parquet("dbfs:/user/team20/finalnotebook/ensemble_test.v1.parquet")
print("Loading ensemble_val.v1.parquet")
ensemble_val_load = spark.read.option("header", "true").parquet("dbfs:/user/team20/finalnotebook/ensemble_val.v1.parquet")

# COMMAND ----------

model_eval_smoted = []
for (l2_name, l2_model, l1_model)  in [
  ("LR-smoted", model_trained_ensemble_lr_smoted_load, ensemble_model_smoted_load), 
  ("SVM-smoted", model_trained_ensemble_svm_smoted_load, ensemble_model_smoted_load), 
  ("RF-smoted", model_trained_ensemble_rf_smoted_load, ensemble_model_smoted_load),
] :
  for data_name, data in [("test set", ensemble_test_load), ("validation", ensemble_val_load)] :
      print("Level 2 model type = {}, running on {}".format(l2_name,data_name))
      ensemble_test_prediction = FinalEnsmblePipeline(l2_model, l1_model, data)
      eval = EvaluateModelPredictions(ensemble_test_prediction, dataName=data_name, ReturnVal=True)
      model_eval_smoted.append({ 'l2_name' : l2_name, 'data_name' : data_name, 'result' : eval})

# COMMAND ----------

model_eval = model_eval_smoted
headerColor  = 'lightgrey'
rowEvenColor = 'lightgrey'
rowOddColor  = 'white'

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Run type</b>','<b>Accuracy</b>','<b>Precision</b>','<b>Recall</b>','<b>f-score</b>', 
            '<b>AUROC</b>', '<b>AUPRC</b>', '<b>TP</b>', '<b>TN</b>', '<b>FP</b>', '<b>FN</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='black', size=13)
  ),
  cells=dict(
    values=[
      [ev['l2_name'] + '<br>(' + ev['data_name'] + ')' for ev in model_eval],
      [ev['result']['Accuracy'] for ev in model_eval],
      [ev['result']['Precision'] for ev in model_eval],
      [ev['result']['Recall'] for ev in model_eval],
      [ev['result']['f-score'] for ev in model_eval],
      [ev['result']['areaUnderROC'] for ev in model_eval],
      [ev['result']['AreaUnderPRC'] for ev in model_eval],
      [ev['result']['metric']['TP'] for ev in model_eval],
      [ev['result']['metric']['TN'] for ev in model_eval],
      [ev['result']['metric']['FP'] for ev in model_eval],
      [ev['result']['metric']['FN'] for ev in model_eval],
    ],
    line_color='darkslategray',
    align = ['left'],
    font = dict(color = 'darkslategray', size = 13)
    ))
])
fig.update_layout(width=1400, height=600, title="Model evaluation results (smoted)")
fig.show()

# COMMAND ----------

model_eval_regular = []
for (l2_name, l2_model, l1_model)  in [
  ("LR", model_trained_ensemble_lr_load, ensemble_model_load), 
  ("SVM", model_trained_ensemble_svm_load, ensemble_model_load), 
  ("RF", model_trained_ensemble_rf_load, ensemble_model_load),
] :
  for data_name, data in [("test set", ensemble_test_load), ("validation", ensemble_val_load)] :
      print("Level 2 model type = {}, running on {}".format(l2_name,data_name))
      ensemble_test_prediction = FinalEnsmblePipeline(l2_model, l1_model, data)
      eval = EvaluateModelPredictions(ensemble_test_prediction, dataName=data_name, ReturnVal=True)
      model_eval_regular.append({ 'l2_name' : l2_name, 'data_name' : data_name, 'result' : eval})
      

# COMMAND ----------

model_eval = model_eval_regular
headerColor  = 'lightgrey'
rowEvenColor = 'lightgrey'
rowOddColor  = 'white'

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>Run type</b>','<b>Accuracy</b>','<b>Precision</b>','<b>Recall</b>','<b>f-score</b>', 
            '<b>AUROC</b>', '<b>AUPRC</b>', '<b>TP</b>', '<b>TN</b>', '<b>FP</b>', '<b>FN</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='black', size=13)
  ),
  cells=dict(
    values=[
      [ev['l2_name'] + '<br>(' + ev['data_name'] + ')' for ev in model_eval],
      [ev['result']['Accuracy'] for ev in model_eval],
      [ev['result']['Precision'] for ev in model_eval],
      [ev['result']['Recall'] for ev in model_eval],
      [ev['result']['f-score'] for ev in model_eval],
      [ev['result']['areaUnderROC'] for ev in model_eval],
      [ev['result']['AreaUnderPRC'] for ev in model_eval],
      [ev['result']['metric']['TP'] for ev in model_eval],
      [ev['result']['metric']['TN'] for ev in model_eval],
      [ev['result']['metric']['FP'] for ev in model_eval],
      [ev['result']['metric']['FN'] for ev in model_eval],
    ],
    line_color='darkslategray',
    align = ['left'],
    font = dict(color = 'darkslategray', size = 13)
    ))
])
fig.update_layout(width=1400, height=600, title="Model evaluation results (Not smoted)")
fig.show()

# COMMAND ----------

import numpy as np

fig = make_subplots(rows=rows, cols=columns, subplot_titles=("Majority class splitted","Smoted"))

def normalize_vec(x) :
    vec = np.stack(x).sum(axis=0) 
    return(vec/np.linalg.norm(vec))

for (row, col), value in [((1, 1), [m.stages[-1].featureImportances.toArray() for m in ensemble_model]), 
                          ((1, 2), [m.stages[-1].featureImportances.toArray() for m in ensemble_model_smoted])] :
    fig.add_trace(go.Bar(
      x = all_ensemble_features,
      y = normalize_vec(value),
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x] if x < 12 else px.colors.sequential.RdBu[x%12] , 
                            range(0,len(all_ensemble_features)))),
      name = '',
      showlegend = False,
    ), row=row, col=col)
    
    fig.update_xaxes(categoryorder='total descending', row=row, col=col)
    fig.update_xaxes(categoryorder='total descending', row=row, col=col)
    fig.update_xaxes(tickangle=-60)
    
fig.update_layout(height=600, width=1200, title_text="<b>Feature importance for individual ensembles</b>")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## VI. Conclusions
# MAGIC * Visualize Model Scores:
# MAGIC     - Confusion Matrix!!
# MAGIC     - precision-recall curve?

# COMMAND ----------

# MAGIC %md
# MAGIC ## VII. Applications of Course Concepts
# MAGIC - bias-variance tradeoff (in dataset balancing discussion)        
# MAGIC   During algorithm performance evaluation of Decision trees it became clear that this algorithm due to the higher complexity and low bias tended to overfit to the given training set.  Because of that there was high variance between training and validation sets. (???) To overcome the over-fitting and high variance we used random forests and ensembles of random forests. The hyperparameter tuning using random forests helped us to get to the optimal solution balancing both bias and variance.       
# MAGIC 
# MAGIC - 1-hot encoding for SVM's?        
# MAGIC   While using Support Vector Machines classifier, it had to deal with categorical features that didn’t necessarily have an ordering. In such cases instead of converting them to integer codes, we used one hot-encoding.
# MAGIC   
# MAGIC - assumptions (for different algorithms - for example OLS vs Trees)        
# MAGIC   During algorithm exploration we selected a set of variety of algorithms to pick the most suitable one for this particular airline dataset. The algorithms like logistic regression and Naive Bayes tend me very simple in its modeling. These simple models relatively insensitive to variance to different training datasets. But they tend to be highly biased. This problem seem to compound when the data is imbalanced. Algorithms like decision trees and support vector machines are much more complex and as we increase complexity they tend to less and less biased but has a tendency to show a lot of variance between training sets. In other words they seem to overfit to the given training set. In our case we chose the complex model which overfits and used additional methods like random forest, ensembles and hyper paramter tuning to reduce the overfitting of the model.
# MAGIC   
# MAGIC - Breiman's Theorem (for ordering categorical variables)           
# MAGIC   We applied Breiman's Method to some of the categorical features to generate a ranking of within each categorical feature and ordered the categories based on the ranking obtained from the calculation of the average outcome. This method helped us convert categorical features to ranked numerical features.
# MAGIC 
# MAGIC 
# MAGIC - how data is stored on cluster - parquet
# MAGIC - Scalability & sampling (for SMOTE)
# MAGIC - broadcasting (for SMOTE, Breiman's Theorem, Holiday feature)
# MAGIC - Distributing the problem to multiple workers via ensembles?? (idk if this is a course concept, but easily parallelized)

# COMMAND ----------

