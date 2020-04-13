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
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorIndexer, VectorAssembler, OneHotEncoderEstimator

from pyspark.mllib.evaluation import MulticlassMetrics

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
  data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

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

var = 'CRS_Elapsed_Time'
MakeRegBarChart(train_and_val, outcomeName, var, orderBy=var, barmode='stack', xtype='linear', xrange=[0, 450])

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
MakeRegBarChart(d, outcomeName, var + "_binlabel", orderBy=var + "_binlabel", barmode='stack', xtype='category')

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
  
# Plot Carrier and outcome with bar plots of probability on x axis
MakeProbBarChart(airlines, "Op_Unique_Carrier", xtype='category', numDecimals=5)

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
# MAGIC ## III. Feature Engineering
# MAGIC - Cleaning
# MAGIC     - Remove cancelled/diverted flights (don't have full data)
# MAGIC     - Remove flights with null outcomes (can't estimate)
# MAGIC - Transformations
# MAGIC     - Binning (elapsed time, departure/arrival time, )
# MAGIC     - Forming Indiciator Variables (e.g. departure/arrival delay > 30 min (true/false))
# MAGIC - Interaction Terms
# MAGIC     - Interacting variables to make new features ("Day of Year", "Time of week")
# MAGIC     - Binning interaction terms
# MAGIC     - Joining interaction terms to make new features
# MAGIC          - Day of month with month ("Day of Year") -> Holidays indiciator (or near holiday indicator)
# MAGIC          - Departure/Arrival time day of week ("Time of week")
# MAGIC - Treatment of Categorical Variables
# MAGIC     - Categorical variables: Time of Year Variables, Distance Group, Carrier, origin/destination airports
# MAGIC     - Order  Breiman's Theorem (rank by volume of flights, probability of departure delay, etc)
# MAGIC     - Make broader categories for categorical variables (not something we've tried yet, but we can do this)
# MAGIC     - one-hot encoding (svms)
# MAGIC - Treatment for categorical / binned variables
# MAGIC     - computing # of flights per category
# MAGIC     - computing probability of delay per category
# MAGIC     - Try a model that includes these to see if the model selects them (Decision tree)

# COMMAND ----------

# MAGIC %md
# MAGIC More visualizations from our in-depth EDA can be found in notebook here:
# MAGIC 
# MAGIC https://dbc-b1c912e7-d804.cloud.databricks.com/?o=7564214546094626#notebook/3895804345790408/command/3895804345790409

# COMMAND ----------

# MAGIC %md
# MAGIC ### Show columns of interest summarized with counts of null values & summary stats
# MAGIC * Clearly explain & define each variable
# MAGIC * Justify missing values & how will handle
# MAGIC * Addres feature distributions

# COMMAND ----------

# Get number of distinct values for each column in full training dataset
display(train_and_val.agg(*(F.countDistinct(F.col(c)).alias(c) for c in train_and_val.columns)))

# COMMAND ----------

# get summary stats for the full training dataset
display(train_and_val.describe())

# COMMAND ----------

# get number of null values for each column (none!)
display(train_and_val.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in [outcomeName] + numFeatureNames + catFeatureNames]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cover the following
# MAGIC * General EDA of vars
# MAGIC * Binning
# MAGIC * Interaction Terms
# MAGIC * Ordering of Categorical Variables (Breiman's Theorem)
# MAGIC * only do modifications to training split
# MAGIC * save result to cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bin Numerical Features (reducing potential splits)

# COMMAND ----------

from pyspark.ml.feature import Bucketizer
from pyspark.sql.types import IntegerType

# Augments the provided dataset for the given feature/variable with a binned version
# of that variable, as defined by splits parameter
# Column name suffixed with '_bin' will be the binned column
# Column name suffixed with '_binlabel' will be the nicely-named version of the binned column, using provided labels
def BinFeature(df, featureName, splits, labels=None):
  if (featureName + "_bin" in df.columns):
    print("Variable '" + featureName + "_bin' already exists")
    return df
    
  # Generate binned column for feature
  bucketizer = Bucketizer(splits=splits, inputCol=featureName, outputCol=featureName + "_bin")
  df_bin = bucketizer.setHandleInvalid("keep").transform(df)
  df_bin = df_bin.withColumn(featureName + "_bin", df_bin[featureName + "_bin"].cast(IntegerType()))
  
  if (labels is not None):
    # Map bucket number to binned feature label
    bucketMaps = {}
    bucketNum = 0
    for l in labels:
      bucketMaps[bucketNum] = l
      bucketNum = bucketNum + 1

    # Generate new column with binned feature label (human-readable)
    def newCols(x):
      return bucketMaps[x]
    callnewColsUdf = udf(newCols, StringType())
    df_bin = df_bin.withColumn(featureName + "_binlabel", callnewColsUdf(F.col(featureName + "_bin")))
    
  return df_bin

# Bin numerical features in entire airlines dataset
# Note that splits are not based on test set but are applied to test set (as would be applied at inference time)
# airlines = BinFeature(airlines, 'CRS_Dep_Time', splits = [i for i in range(0, 2400 + 1, 200)]) # 2-hour blocks
# airlines = BinFeature(airlines, 'CRS_Arr_Time', splits = [i for i in range(0, 2400 + 1, 200)]) # 2-hour blocks
airlines = BinFeature(airlines, 'CRS_Dep_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
airlines = BinFeature(airlines, 'CRS_Arr_Time', splits = [i for i in range(0, 2400 + 1, 10)]) # 10 min blocks
airlines = BinFeature(airlines, 'CRS_Elapsed_Time', splits = [float("-inf")] + [i for i in range(0, 660 + 1, 60)] + [float("inf")]) # 1-hour blocks
binFeatureNames = ['CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin']

display(airlines.take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Interaction Features

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
display(airlines.take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Holidays Feature

# COMMAND ----------

from dateutil.relativedelta import relativedelta, SU, MO, TU, WE, TH, FR, SA
import pandas as pd
import datetime as dt

def AddHolidayFeature(df):
  if ('Holiday' in df.columns):
      print("Variable 'Holiday' already exists")
      return df
  
  # Import dataset of government holidays
  holiday_df_raw = spark.read.csv("dbfs:/user/shajikk@ischool.berkeley.edu/scratch/" + 'holidays.csv').toPandas()
  holiday_df_raw.columns = ['ID', 'FL_DATE', 'Holiday']

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

    return prev.strftime("%Y-%m-%d"), prev.strftime('%a'),  nxt.strftime("%Y-%m-%d"), nxt.strftime('%a')

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
display(airlines.take(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Origin Activity Feature

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
display(airlines.take(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Breiman's Theorem to Categorical Features

# COMMAND ----------

# Regenerate splits for Breiman ranking prep
mini_train, train, val, test = SplitDataset(airlines)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

# Applies Breiman's Method to the categorical feature
# Generates the ranking of the categories in the provided categorical feature
# Orders the categories by the average outcome ascending, from integer 1 to n
# Note that this should only be run on the training data
def GenerateBreimanRanks(df, catFeatureName, outcomeName):
  window = Window.orderBy('avg(' + outcomeName + ')')
  breimanRanks = df.groupBy(catFeatureName).avg(outcomeName) \
                   .sort(F.asc('avg(' + outcomeName + ')')) \
                   .withColumn(catFeatureName + "_brieman", F.row_number().over(window))
  return breimanRanks

# Using the provided Breiman's Ranks, applies Breiman's Method to the categorical feature
# and creates a column in the original table using the mapping in breimanRanks variable
# Note that this effectively transforms the categorical feature to a numerical feature
# The new column will be the original categorical feature name, suffixed with '_breiman'
def ApplyBreimansMethod(df, breimanRanks, catFeatureName, outcomeName):
  if (catFeatureName + "_breiman" in df.columns):
    print("Variable '" + catFeatureName + "_brieman" + "' already exists")
    return df
  
  res = df.join(F.broadcast(breimanRanks), df[catFeatureName] == breimanRanks[catFeatureName], how='left') \
          .drop(breimanRanks[catFeatureName]) \
          .drop(breimanRanks['avg(' + outcomeName + ')']) \
          .fillna(-1, [catFeatureName + "_brieman"])
  return res

# COMMAND ----------

# Apply breiman ranking to all datasets, based on ranking developed from training data
breimanRanksDict = {} # save to apply later as needed
featuresToApplyBreimanRanks = catFeatureNames + intFeatureNames + holFeatureNames
for feature in featuresToApplyBreimanRanks:
  # Get ranks for feature, based on training data only
  ranksDict = GenerateBreimanRanks(train, feature, outcomeName)
  breimanRanksDict[feature] = ranksDict
  
  # Apply Breiman's method & do feature transformation for all datasets
  mini_train = ApplyBreimansMethod(mini_train, ranksDict, feature, outcomeName)
  train = ApplyBreimansMethod(train, ranksDict, feature, outcomeName)
  val = ApplyBreimansMethod(val, ranksDict, feature, outcomeName)
  test = ApplyBreimansMethod(test, ranksDict, feature, outcomeName)
  airlines = ApplyBreimansMethod(airlines, ranksDict, feature, outcomeName)
  
briFeatureNames = [entry + "_brieman" for entry in breimanRanksDict]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write and Reference augmented airlines

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
# MAGIC ### Dataset Balancing
# MAGIC - Describe SMOTE, show visuals from slides, show core parts of SMOTE algorithm (get neighbors, generate synthetic, the RDD-based code) but don't run the code; just load the data
# MAGIC     - Link to notebook where the smoted dataset was actually generated
# MAGIC - Describe Majority Class splitting (high-level with pictures from slides), say we'll use this in for stacking in ensemble approach later on

# COMMAND ----------

# DBTITLE 1,Helper Code to Load All Data for Model Training (TODO: remove)
# Read prepared data from parquet for training
def ReadDataFromParquet(data, dataName):
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

airlines = ReadDataFromParquet(airlines, 'augmented')
mini_train = ReadDataFromParquet(mini_train, 'augmented_mini_train')
train = ReadDataFromParquet(train, 'augmented_train')
val = ReadDataFromParquet(val, 'augmented_val')
test = ReadDataFromParquet(test, 'augmented_test')

###########################################
# Define all variables for easy reference #
###########################################

# Numerical Variables to use for training
outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group']
contNumFeatureNames = ['CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
binFeatureNames = ['CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin']
intFeatureNames = ['Day_Of_Year', 'Origin_Dest', 'Dep_Time_Of_Week', 'Arr_Time_Of_Week']
holFeatureNames = ['Holiday']
orgFeatureNames = ['Origin_Activity']
briFeatureNames = ['Op_Unique_Carrier_brieman', 'Origin_brieman', 'Dest_brieman', 'Day_Of_Year_brieman', 'Origin_Dest_brieman', 'Dep_Time_Of_Week_brieman', 'Arr_Time_Of_Week_brieman', 'Holiday_brieman']

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

# - Numerical Features: 		 ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
# - Categorical Features: 	 ['Op_Unique_Carrier', 'Origin', 'Dest']

#subset the dataset to the features in numFeatureNames & catFeatureNames
mini_train_algo = mini_train.select([outcomeName] + numFeatureNames + catFeatureNames)
train_algo = train.select([outcomeName] + numFeatureNames + catFeatureNames)
val_algo = val.select([outcomeName] + numFeatureNames + catFeatureNames)

# Define outcome & features to use in model development
# numFeatureNames are continuous features
# catFeatureNames are categorical features
outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
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
      df = df.withColumn("CRS_Elapsed_Time", \
              when(df["CRS_Elapsed_Time"] < 0, 0.0).otherwise(df["CRS_Elapsed_Time"]))
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

# this function run the prediction in the model
# And calculate prediction metrics like accuracy, recall, precision ...etc
def PredictAndEvaluate(model, data, dataName, outcomeName, algorithm):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName, algorithm)

# COMMAND ----------

# Model Evaluation

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

def PredictAndEvaluate(model, data, dataName, outcomeName):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName)

# COMMAND ----------

dataName = 'val'
algorithms = ['lr','dt','nb','svm']
for algorithm in algorithms:
  if algorithm == 'svm':
    titleName = dataName+ ' with ' + algorithm
    tr_model = train_model(train_algo,algorithm,catFeatureNames,numFeatureNames,outcomeName,svmflag=True)
    PredictAndEvaluate(tr_model, val_algo, dataName, outcomeName)
  else:
    titleName = dataName+ ' with ' + algorithm
    tr_model = train_model(train_algo,algorithm,catFeatureNames,numFeatureNames,outcomeName,svmflag=False)
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

# Build the toy example dataset from the cleaned and transformed mini training set
# toy_dataset = mini_train.select(['Dep_Del30', 'Day_Of_Week', 'Origin', 'Op_Unique_Carrier']) \
#                         .filter(mini_train['Dep_Del30'] == 0) \
#                         .sample(False, 0.0039, 8)

# toy_dataset = toy_dataset.union(mini_train.select(['Dep_Del30', 'Day_Of_Week', 'Origin', 'Op_Unique_Carrier']) \
#                          .filter(mini_train['Dep_Del30'] == 1) \
#                          .sample(False, 0.025, 8))

# display(toy_dataset)

# COMMAND ----------

# Save the toy example dataset
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
# MAGIC $$ G = Entropy (parent) - Entropy (child) $$
# MAGIC 
# MAGIC ##### Based on the information gain from the diagram above, the best feature to split on is Origin.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling Helpers
# MAGIC * Evaluation functions
# MAGIC * Decision Tree PrintModel Function

# COMMAND ----------

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

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

def TrainDecisionTreeModel(trainingData, stages, outcomeName, maxDepth, maxBins):
  # Train Model
  dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = maxDepth, maxBins=maxBins) 
  pipeline = Pipeline(stages = stages + [dt])
  dt_model = pipeline.fit(trainingData)
  return dt_model

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
# MAGIC ### Baseline Model: Predicting the Majority Class

# COMMAND ----------

features_base = (numFeatureNames + binFeatureNames + orgFeatureNames, # numerical features
                 catFeatureNames + intFeatureNames + holFeatureNames) # string features
si_base = PrepStringIndexer(features_base[1])
va_base = PrepVectorAssembler(numericalFeatureNames = features_base[0], stringFeatureNames = features_base[1])

# COMMAND ----------

model_base = TrainDecisionTreeModel(train, si_base + [va_base], outcomeName, maxDepth=1, maxBins=6653)
PrintDecisionTreeModel(model_base.stages[-1], features_base[0] + features_base[1])

# COMMAND ----------

PredictAndEvaluate(model_base, val, "val", outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### First Decision Tree with no fine-tuning/dataset balancing

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Ensemble training & Balancing dataset

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
# MAGIC First, the airline data is transformed and feature indexer is calculated.  Do assembler transformation before train test split for `VectorIndexer` to work 

# COMMAND ----------

# def TransformDataForEnsemble(full_dataset):

#   # Convert strings to index
#   str_indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in ["Op_Unique_Carrier",	"Origin", "Dest", "Holiday"]]
#   target       = ["Dep_Del30"]
#   all_features = ["Month", "Day_Of_Month", "Day_Of_Week", "Distance", 'CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin', "Op_Unique_Carrier_index", "Origin_index", "Dest_index", "Holiday_index", "Origin_Activity"]
  
#   assembler = VectorAssembler(inputCols=all_features, outputCol="features")
#   ensemble_pipeline = Pipeline(stages=str_indexers + [assembler])
  
#   transformed_data =  (ensemble_pipeline
#           .fit(full_dataset)
#           .transform(full_dataset)
#           .select(["Year", "features"] + target)
#           .withColumnRenamed("Dep_Del30", "label"))
  
#   # Here, create automatic Categories. All variables > 400 Categories will be treated as Continuous.
#   # else will be treated as Categories by random forest.
#   featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=400).fit(transformed_data)
  
#   return all_features, featureIndexer, transformed_data

#all_ensemble_features, ensamble_featureIndexer, ensemble_transformed_data = TransformDataForEnsemble(airlines)

def TransformDataForEnsemble(mini_train_data, train_data, val_data, test_data) :
  target       = ["Dep_Del30"]
  all_features = ['Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Elapsed_Time', 'Distance',  'CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'Origin_Activity', 'Op_Unique_Carrier_brieman', 'Origin_brieman', 'Dest_brieman', 'Day_Of_Year_brieman', 'Origin_Dest_brieman', 'Dep_Time_Of_Week_brieman', 'Arr_Time_Of_Week_brieman', 'Holiday_brieman']
  
  assembler = VectorAssembler(inputCols=all_features, outputCol="features")
  ensemble_pipeline = Pipeline(stages=[assembler])

  tmp_mini_train, tmp_train, tmp_val, tmp_test = (ensemble_pipeline.fit(mini_train_data).transform(mini_train_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(train_data).transform(train_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(val_data).transform(val_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"),
                                                  ensemble_pipeline.fit(test_data).transform(test_data).select(["features"] + target).withColumnRenamed("Dep_Del30", "label"))
  
  featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=400).fit(tmp_train
                                                                                                          .union(tmp_val)
                                                                                                          .union(tmp_test))
  
  return all_features, featureIndexer, tmp_mini_train, tmp_train, tmp_val, tmp_test


all_ensemble_features, ensamble_featureIndexer, ensemble_mini_train, ensemble_train, ensemble_val, ensemble_test = TransformDataForEnsemble(mini_train, train, val, test)

print(all_ensemble_features)
ensemble_mini_train.show(2)

# COMMAND ----------

# print(all_ensemble_features) # Show all features that are used in Ensemble.
# # Print the transformed dataframe.
# ensemble_transformed_data.show(4, truncate=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC Split data into train and test, after appling VectorIndexer.

# COMMAND ----------

#ensemble_mini_train, ensemble_train, ensemble_val, ensemble_test = SplitDataset(ensemble_transformed_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Balance the dataset, partition for further training

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
# MAGIC Check for data set balancing

# COMMAND ----------

([[d.groupBy('label').count().toPandas()["count"].to_list()] for d in train_group], 
 train_combiner.groupBy('label').count().toPandas()["count"].to_list())

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

# Parallel training is not done in databricks environment.      
# ensemble_model = TrainEnsembleModels_parallel(train_group, ensamble_featureIndexer, 
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
      
ensemble_model = TrainEnsembleModels(train_group, ensamble_featureIndexer, 
                    # Type of model we can use.
                    RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=5, numTrees=5, impurity='gini')
                   )

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
      marker_color=list(map(lambda x: px.colors.sequential.thermal[x%12], range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=1200, width=1200, title_text="Feature importance for individual ensembles")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Construct a new data set based on the output of base classifiers

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
        #EvaluateModelPredictions(m, predictions, str(num))
    return prediction_array

ensemble_prediction = do_ensemble_prediction(ensemble_model, train_combiner)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Assemble and transform data for second level training

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting array of dataframes is reduced into a single dataframe by iteratively joining over.

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

# COMMAND ----------

def TrainCombiner(data, featureIndexer, classifier):
  # Chain indexer and forest in a Pipeline
  pipeline_ensemble = Pipeline(stages=[featureIndexer, classifier])

  # Train model.  This also runs the indexer.
  return pipeline_ensemble.fit(data)

# Set up VectorIndexer for second level training
ensemble_featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=3).fit(ensemble_transformed)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
# Logistic Regression
model_trained_ensemble_lr = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              LogisticRegression(featuresCol="indexedFeatures", maxIter=10, regParam=0.2))

# Linear SVM
from pyspark.ml.classification import LinearSVC
model_trained_ensemble_svm = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              LinearSVC(featuresCol="indexedFeatures", maxIter=10, regParam=0.1))

# Random forest
model_trained_ensemble_rf = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
              RandomForestClassifier(featuresCol="indexedFeatures", maxBins=20, maxDepth=5, numTrees=5, impurity='gini'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create final ensemble pipeline

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
    
fig.update_layout(height=400, width=1200, title_text="Feature importance for individual ensembles")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model evaluation

# COMMAND ----------

model_eval = []
for (l2_name, l2_model)  in [("Logistic regression", model_trained_ensemble_lr), ("SVM", model_trained_ensemble_svm), ("Random Forest", model_trained_ensemble_rf)] :
  for data_name, data in [("test set", ensemble_test), ("validation", ensemble_val)] :
      print("Level 2 model type = {}, running on {}".format(l2_name,data_name))
      ensemble_test_prediction = FinalEnsmblePipeline(l2_model, ensemble_model, data)
      eval = EvaluateModelPredictions(ensemble_test_prediction, dataName=data_name, ReturnVal=True)
      model_eval.append({ 'l2_name' : l2_name, 'data_name' : data_name, 'result' : eval})

# COMMAND ----------

model_eval

# COMMAND ----------

# MAGIC %md
# MAGIC ## VI. Conclusions
# MAGIC * Visualize Model Scores:
# MAGIC     - Confusion Matrix!!
# MAGIC     - precision-recall curve?

# COMMAND ----------

# MAGIC %md
# MAGIC ## V. Applications of Course Concepts
# MAGIC - bias-variance tradeoff
# MAGIC - 1-hot encoding for SVM's? 
# MAGIC - Breiman's method
# MAGIC - how data is stored on cluster
# MAGIC - Distributing the problem to multiple workers via ensembles?? (idk if this is a course concept, but easily parallelized)

# COMMAND ----------

