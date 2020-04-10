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
# MAGIC To attempt to solve this problem, we introduce the *Airline Delays* dataset, a dataset of US domestic flights from 2015 to 2019 collected by the Bureau of Transportation Statistics for the purpose of studying airline delays. For this analysis, we will primarily use this dataset to study the nature of airline delays in the united states over the last few years, with the ultimate goal of developing models for predicting significant flight departure delays (30 minutes or more) in the United States. 
# MAGIC 
# MAGIC In developing such models, we seek to answer the core question, **"Given known information prior to a flight's departure, can we predict departure delays and identify the likely causes of such delays?"**. In the last few years, about 11% of all US domestic flights resulted in significant delays, and answering these questions can truly help us to understand why such delays happen. In doing so, not only can airlines and airports start to identify likely causes and find ways to mitigate them and save both time and money, but air travelers also have the potential to better prepare for likely delays and possibly even plan for different flights in order to reduce their chance of significant delay. 
# MAGIC 
# MAGIC To effectively investigate this question and produce a practically useful model, we will aim to develop a model that performs better than a baseline model that predicts the majority class of 'no delay' 89% of the time (the equivalent of random guessing, which would have an accuracy of 89%). Given the classification nature of this problem, we will concentrate on improving metrics such as precision, recall and F1 over our baseline model. We will also concentrate on producing models that can explain what features of flights known prior to departure time can best predict departure delays and from these, attempt to best infer possible causes of departure delays. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing Dependencies

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install plotly --upgrade

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

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
# MAGIC The Bureau of Transporation Statistics provides us with a wide variety of features relating to each flight in the *Airline Delays* dataset, ranging from features about the scheduled flight such as the planned departure, arrival, and elapsed times, the planned distance, the carrier and airport information, information regarding the causes of certain delays for the entire flight, as well as the amounts of delay (for both flight departure and arrival), among many other features. 
# MAGIC 
# MAGIC Given that for this analysis, we will be concentrating on predicting and identify the likely causes of departure delays before any such delay happens, we will primarily concentrate our EDA and model development using features of flights that would be known at inference time. We will choose the inference time to be 6 hours prior to the scheduled departure time of a flight. Realistically speaking, providing someone with a notice that a flight will likely be delayed 6 hours in advance is likely a sufficient amount of time to let people prepare for such a delay to reduce the cost of the departure delay, if it occurs. Such features that fit this criterion include those that are related to:
# MAGIC 
# MAGIC * **Time of year** (e.g. `Year`, `Month`, `Day_Of_Month`, `Day_Of_Week`)
# MAGIC * **Airline Carrier** (e.g. `Op_Unique_Carrier`)
# MAGIC * **Origin & Destination Airports** (e.g. `Origin`, `Dest`)
# MAGIC * **Scheduled Departure & Arrival Times** (e.g. `CRS_Dep_Time`, `CRS_Arr_Time`)
# MAGIC * **Planned Elapsed Times & Distances** (e.g. `CRS_Elapsed_Time`, `Distance`, `Distance_Group`)
# MAGIC 
# MAGIC Additionally, we will use the variable `Dep_Delay` to define our outcome variable for "significiant" departure delays (i.e. delays of 30 minutes or more). Finally, we will focus our analysis to the subset of flights that are not diverted, are not cancelled and have non-null values for departure delays to ensure that we can accurately predict departure delays for flights. this will still leave us with a significant number of records to work with for the purpose of training and model development. Below are a few example flights taken from the *Airline Delays* dataset that we will use for our analysis.

# COMMAND ----------

def LoadAirlineDelaysData():
  # Read in original dataset
  airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")
  print("Number of records in original dataset:", airlines.count())

  # Filter to datset with entries where diverted != 1, cancelled != 1, and dep_delay != Null
  airlines = airlines.where('DIVERTED != 1') \
                     .where('CANCELLED != 1') \
                     .filter(airlines['DEP_DELAY'].isNotNull()) 

  print("Number of records in reduced dataset: ", airlines.count())
  return airlines

airlines = LoadAirlineDelaysData()

# COMMAND ----------

# Print examples of flights
display(airlines.take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that because we are interested in predicting departure delays for future flights, we will define our test set to be the entirty of flights from the year 2019 and use the years 2015-2018 for training. This way, we will simulate the conditions for training a model that will predict departure delays for future flights. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explanation for why we chose the variables we chose (rationale for why not all variables are available at inference time)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes for II. EDA & Discussion of Challenges
# MAGIC - 2-3 EDA Tasks to help make decisions on how we implement the algorithm to be scalable
# MAGIC     - Binning could be a way to scale decision tree algorithms (when defining splits, e.g. CRS_Dep_Time)
# MAGIC     - More customized binning for the origin-destination to fit some custom criterion (to make new features)
# MAGIC     - Brieman's Theorem can help with binning on categorical variables -- we could do an EDA task to rank some categorical variables (reduce number of possible splits -- bring order to unordered cateogrical variables)
# MAGIC - Challenges anticipated based on EDA
# MAGIC     - Cleaning/joining datasets?
# MAGIC     - Feature Selection (or feature augmentation)
# MAGIC     - gradient descent for logistic regression?
# MAGIC     - sharing clusters, not being able to load entire dataset into memory (but can load fractions and/or aggregations of it!)
# MAGIC     - store data to optimize for column/row retrieval based on algorithm (parquet for DT & avro for LR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prospective Models for the Departure Delay Classification Task
# MAGIC To motivate our EDA in this section for scalability investigation, we will keep in mind the following models, which we will explore in more detail in section IV:
# MAGIC * Logistic Regression
# MAGIC * Decision Trees
# MAGIC * Naive Bayes
# MAGIC * Support Vector Machines

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #1: Exploring Scheduled Departure & Arrival Times
# MAGIC * There are a lot of unique departure & arrival times that are numerical features, but the time 2300 isn't much different from 2305 to 2310
# MAGIC * May be worthwhile to bin things
# MAGIC     - fewer splits for decision tree to consider
# MAGIC     - can estimate effects of unique time blocks departure delays (more meaningful/interpretable)
# MAGIC     - do have more coefficients to estimate in LR (1 for each bin value)
# MAGIC * Show barplots of delay/no delay distributions for numerical values and binned values (maybe even show ordered by probability of departure delay--Diana EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #2: Categorical Variables & Reducing # of Splits (for Decision Trees)
# MAGIC * Some categorical variables have few values (e.g. carrier)
# MAGIC      - good for Decision Trees, because it introduces fewer splits to have to consider
# MAGIC      - can still define distinct sets of delayed & not delayed airlines
# MAGIC      - show probability charts b/c clearly some are more delayed than others
# MAGIC * Other categorical variables have a lot of values (e.g. origin/dest airports) & have no implicit ordering
# MAGIC      - should incorporate things like Breiman's method to reduce number of splits for decision trees
# MAGIC      - effectivley want to give an ordering to categories
# MAGIC      - using Breiman's method makes things more scalable
# MAGIC      - Can rank, for example based on probability of delay and use this ranking in place of actual origin/dest categories

# COMMAND ----------

# Helper function for Group 3 graphs that plot the probability of outcome on the x axis, the number of flights on the x axis
# With entries for each distinct value of the feature as separate bars.

def MakeProbBarChart(full_data_dep, var, xtype, numDecimals):
  # Filter out just to rows with delays or no delays
  threshold = 30
  outcomeName = 'Dep_Del' + str(threshold)
  if (outcomeName not in full_data_dep.columns):
    full_data_dep = full_data_dep.withColumn(outcomeName, (full_data_dep['Dep_Delay'] >= threshold).cast('integer'))
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
# Airline Codes to Airlines: https://www.bts.gov/topics/airlines-and-airports/airline-codes
MakeProbBarChart(train, "Op_Unique_Carrier", xtype='category', numDecimals=4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA Task #3: Unbalanced Dataset
# MAGIC * Show how dataset is drastically unbalanced (especially as you increase the dep_delay threshold)
# MAGIC * Discuss stacking and what that tries to do (how we'll eventually want an ensemble approach to support this)
# MAGIC * If can get SMOTE working to be scalable, also discuss SMOTE and what it effectively does
# MAGIC    - do discuss scalability concerns, b/c do need to apply knn algo & predict on each datapoint
# MAGIC * Discuss how balancing the dataset with stacking or SMOTE allows us to ensure that our models don't become biased towards the no-delay class, but it does introduce some more variance (bias-variance tradeoff)

# COMMAND ----------

# Look at balance for outcome in training
display(train.groupBy('Dep_Del30').count().sort('Dep_Del30'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Challenges when send algorithms to scale
# MAGIC * For training Decision Tree, will want to rely on parquet format (since we do feature eval independently for each row)
# MAGIC * for Decision Tree Prediction, we'll likely want avro format (since we do inference on unique rows)
# MAGIC * for Logistic regression training & prediction, we'll want to train & predict on avro
# MAGIC * SVM challenges with categorical variables with large numbers of cateogries (origin & dest--will have very long 1-hot encoded vectors)
# MAGIC * Naive Bayes ____________________
# MAGIC * For Ensemble methods, training models in parallel

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
# MAGIC ### Creating our Outcome Variable: `Dep_Del30`

# COMMAND ----------

# Generate other Departure Delay outcome indicators for n minutes
def CreateNewDepDelayOutcome(data, thresholds):
  for threshold in thresholds:
    data = data.withColumn('Dep_Del' + str(threshold), (data['Dep_Delay'] >= threshold).cast('integer'))
  return data  
  
airlines = CreateNewDepDelayOutcome(airlines, [30])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter to Columns Available at Inference Time

# COMMAND ----------

outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
joiningFeatures = ['FL_Date'] # Features needed to join with the holidays dataset--not needed for training

airlines = airlines.select([outcomeName] + numFeatureNames + catFeatureNames + joiningFeatures)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split Dataset into train/val/test & save to disk as parquet & avro

# COMMAND ----------

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

mini_train, train, val, test = SplitDataset(airlines)

# COMMAND ----------

# Write train & val data to parquet for easier EDA
def WriteAndRefDataToParquet(data, dataName):
  # Write data to parquet format (for easier EDA)
  data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

train_and_val = WriteAndRefDataToParquet(train.union(val), 'train_and_val')

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
# MAGIC * Ordering of Categorical Variables (Brieman's Theorem)
# MAGIC * only do modifications to training split
# MAGIC * save result to cluster

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bin Numerical Features (reducing potential splits)

# COMMAND ----------

from pyspark.ml.feature import Bucketizer

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
airlines = BinFeature(airlines, 'CRS_Dep_Time', splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]) # 2-hour blocks
airlines = BinFeature(airlines, 'CRS_Arr_Time', splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]) # 2-hour blocks
airlines = BinFeature(airlines, 'CRS_Elapsed_Time', splits = [float("-inf"), 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, float("inf")]) # 1-hour blocks
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
  df = df.withColumn("flightbucket", F.concat_ws("-", F.col("Month"), F.col("Day_Of_Month"), F.col("CRS_Dep_Time_bin"), F.col("Origin")))
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
# MAGIC ### Apply Brieman's Theorem to Categorical Features

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

# Applies Breiman's Method to the categorical feature
# Generates the ranking of the categories in the provided categorical feature
# Orders the categories by the average outcome ascending, from integer 1 to n
# Note that this should only be run on the training data
def GenerateBriemanRanks(df, catFeatureName, outcomeName):
  window = Window.orderBy('avg(' + outcomeName + ')')
  briemanRanks = df.groupBy(catFeatureName).avg(outcomeName) \
                   .sort(F.asc('avg(' + outcomeName + ')')) \
                   .withColumn(catFeatureName + "_brieman", F.row_number().over(window))
  return briemanRanks

# Using the provided Brieman's Ranks, applies Brieman's Method to the categorical feature
# and creates a column in the original table using the mapping in briemanRanks variable
# Note that this effectively transforms the categorical feature to a numerical feature
# The new column will be the original categorical feature name, suffixed with '_brieman'
def ApplyBriemansMethod(df, briemanRanks, catFeatureName, outcomeName):
  if (catFeatureName + "_brieman" in df.columns):
    print("Variable '" + catFeatureName + "_brieman" + "' already exists")
    return df
  
  res = df.join(F.broadcast(briemanRanks), df[catFeatureName] == briemanRanks[catFeatureName], how='left') \
          .drop(briemanRanks[catFeatureName]) \
          .drop(briemanRanks['avg(' + outcomeName + ')']) \
          .fillna(-1, [catFeatureName + "_brieman"])
  return res

# COMMAND ----------

# Apply brieman ranking to all datasets, based on ranking developed from training data
briemanRanksDict = {} # save to apply later as needed
featuresToApplyBriemanRanks = catFeatureNames + intFeatureNames + holFeatureNames
for feature in featuresToApplyBriemanRanks:
  # Get ranks for feature, based on training data only
  ranksDict = GenerateBriemanRanks(train, feature, outcomeName)
  briemanRanksDict[feature] = ranksDict
  
  # Apply Brieman's method & do feature transformation for all datasets
  mini_train = ApplyBriemansMethod(mini_train, ranksDict, feature, outcomeName)
  train = ApplyBriemansMethod(train, ranksDict, feature, outcomeName)
  val = ApplyBriemansMethod(val, ranksDict, feature, outcomeName)
  test = ApplyBriemansMethod(test, ranksDict, feature, outcomeName)
  airlines = ApplyBriemansMethod(airlines, ranksDict, feature, outcomeName)
  
briFeatureNames = [entry + "_brieman" for entry in briemanRanksDict]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write and Reference augmented airlines

# COMMAND ----------

airlines = WriteAndRefDataToParquet(airlines, 'augmented')

# COMMAND ----------

# Regenerate splits & save & reference
mini_train, train, val, test = SplitDataset(airlines)
mini_train = WriteAndRefDataToParquet(mini_train, 'augmented_mini_train')
train = WriteAndRefDataToParquet(train, 'augmented_train')
val = WriteAndRefDataToParquet(val, 'augmented_val')
test = WriteAndRefDataToParquet(test, 'augmented_test')

# COMMAND ----------

# Write data to avro for easier row-wise training
def WriteAndRefDataToAvro(data, dataName):
  # Write data to avro format
  data.write.mode('overwrite').format("avro").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".avro")
  
  # Read data back directly from disk 
  return spark.read.format("avro").load(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".avro")

mini_train_avro = WriteAndRefDataToAvro(mini_train, 'augmented_mini_train')
train_avro = WriteAndRefDataToAvro(train, 'augmented_train')
val_avro = WriteAndRefDataToAvro(val, 'augmented_val')
test_avro = WriteAndRefDataToAvro(test, 'augmented_test')

# COMMAND ----------

print("All Available Features:")
print("------------------------")
print(" - Numerical Features: \t\t", numFeatureNames) # numerical features
print(" - Categorical Features: \t", catFeatureNames) # categorical features
print(" - Binned Features: \t\t", binFeatureNames) # numerical features
print(" - Interaction Features: \t", intFeatureNames) # categorical features
print(" - Holiday Feature: \t\t", holFeatureNames) # categorical features
print(" - Origin Activity Feature: \t", orgFeatureNames) # numerical features
print(" - Brieman Ranked Features: \t", briFeatureNames) # numerical features

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

# For the 4 models, can take train dataset & subset to the features in numFeatureNames & catFeatureNames
# Bring in code for 4 models, train, and do prediction & evaluation on validation set

# COMMAND ----------

# - Numerical Features: 		 ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
# - Categorical Features: 	 ['Op_Unique_Carrier', 'Origin', 'Dest']
# take the train dataset and subset to the features in numFeatureNames & catFeatureNames
# outcomeName + numFeatureNames + catFeatureNames
mini_train_algo = mini_train.select([outcomeName] + numFeatureNames + catFeatureNames)

train_algo = train.select([outcomeName] + numFeatureNames + catFeatureNames)
val_algo = val.select([outcomeName] + numFeatureNames + catFeatureNames)

# Define outcome & features to use in model development
outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest'] 


# COMMAND ----------

# take the train dataset and subset to the features in numFeatureNames & catFeatureNames
# outcomeName + numFeatureNames + catFeatureNames

def train_model(df,algorithm,categoricalCols,continuousCols,labelCol,svmflag):

    indexers = [ StringIndexer(inputCol=c, outputCol= c + "_indexed", handleInvalid="keep") for c in categoricalCols ]

    encoder = OneHotEncoderEstimator(inputCols=[indexer.getOutputCol() for indexer in indexers],
                                 outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers])
    
    # If it si svm do hot encoding
    if svmflag == True:
      assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")
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
      nb = NaiveBayes(labelCol = outcomeName, featuresCol = "features", smoothing = 1)
      pipeline = Pipeline(stages=indexers + [assembler,nb])
      
    elif algorithm == 'svm':
      svc = LinearSVC(labelCol = outcomeName, featuresCol = "features", maxIter=50, regParam=0.1)
      pipeline = Pipeline(stages=indexers + encoders + [assembler,svc])
      
    else:
      pass
    
    tr_model=pipeline.fit(df)

    return tr_model

# COMMAND ----------

def PredictAndEvaluate(model, data, dataName, outcomeName, algorithm):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName, algorithm)

# COMMAND ----------

def EvaluateModelPredictions(predictions, dataName, outcomeName,algoName):
  # Make predictions on test data to measure the performance of the model
  
  print(f'\nModel Evaluation - {dataName} for {algoName}', )
  print("------------------------------------------")

  # Accuracy
  evaluator = MulticlassClassificationEvaluator(labelCol=outcomeName, predictionCol="prediction", metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)
  print("Accuracy:\t\t", accuracy)

  # Recall
  evaluator = MulticlassClassificationEvaluator(labelCol=outcomeName, predictionCol="prediction", metricName="weightedRecall")
  recall = evaluator.evaluate(predictions)
  print("Recall:\t\t", recall)

  # Precision
  evaluator = MulticlassClassificationEvaluator(labelCol=outcomeName, predictionCol="prediction", metricName="weightedPrecision")
  precision = evaluator.evaluate(predictions)
  print("Precision:\t", precision)

  # F1
  evaluator = MulticlassClassificationEvaluator(labelCol=outcomeName, predictionCol="prediction",metricName="f1")
  f1 = evaluator.evaluate(predictions)
  print("F1:\t\t", f1)

  # Precision-Recall
  evaluator = BinaryClassificationEvaluator(labelCol = outcomeName, rawPredictionCol = "rawPrediction", metricName = "areaUnderPR")
  PR = evaluator.evaluate(predictions)
  print("Area-Under-PR:\t", PR)
  
  # Are under the curve
  evaluator = BinaryClassificationEvaluator(labelCol = outcomeName, rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
  AUC = evaluator.evaluate(predictions)
  print("Area-Under-ROC:\t", AUC)


# COMMAND ----------

dataName = 'train'
algorithms = ['lr','dt','nb','svm']
for algorithm in algorithms:
  if algorithm == 'svm':
    tr_model = train_model(train_algo,algorithm,catFeatureNames,numFeatureNames,outcomeName,svmflag=True)
    PredictAndEvaluate(tr_model, val_algo, dataName, outcomeName, algorithm)
  else:
    tr_model = train_model(train_algo,algorithm,catFeatureNames,numFeatureNames,outcomeName,svmflag=False)
    predictions = tr_model.transform(val_algo)
    PredictAndEvaluate(tr_model, val_algo, dataName, outcomeName, algorithm)

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
toy_dataset = mini_train.select(['Dep_Del30', 'Day_Of_Week', 'CRS_Dep_Time', 'Origin', 'Op_Unique_Carrier']) \
                        .filter(mini_train['Dep_Del30'] == 0) \
                        .sample(False, 0.0039, 8)

toy_dataset = toy_dataset.union(mini_train.select(['Dep_Del30', 'Day_Of_Week', 'CRS_Dep_Time', 'Origin', 'Op_Unique_Carrier']) \
                         .filter(mini_train['Dep_Del30'] == 1) \
                         .sample(False, 0.025, 8))

display(toy_dataset)

# COMMAND ----------

# Save the toy example dataset
toy_dataset = WriteAndRefDataToParquet(toy_dataset, 'toy_dataset')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Introduction to Decision Trees
# MAGIC Decision trees predict the label (or class) by evaluating a set of rules that follow an IF-THEN-ELSE pattern. The questions are the nodes, and the answers (true or false) are the branches in the tree to the child nodes. A decision tree model estimates the minimum number of true/false questions needed, to assess the probability of making a correct decision. 
# MAGIC 
# MAGIC We use CART decision tree algorithm (Classification and Regression Trees). Decision Tree is a greedy algorithm that considers all features to select the best feature for the split. Initially, we have a root node for the tree. The root node receives the entire training set as input and all subsequent nodes receive a subset of rows as input. Each node asks a true/false question about one of the features using a threshold and in response, the dataset is split into two subsets. The subsets become input to the child nodes added to the tree for the next level of splitting. The goal is to produce the purest distribution of labels at each node.
# MAGIC 
# MAGIC If a node contains examples of only a single type of label, it has 100% purity and becomes a leaf node. The subset doesn't need to be split any further. On the other hand, if a node still contains mixed labels, the decision tree chooses another question and threshold, based on which the dataset is split further. The trick to building an effective tree is to decide which questions to ask and when. To do this, we need to quantify how well a question can split the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Entropy
# MAGIC Entropy is a measure of disorder in the dataset. It characterizes the (im)purity of an arbitrary collection of examples. In decision trees, at each node, we split the data and try to group together samples that belong in the same class. The objective is to maximize the purity of the groups each time a new child node of the tree is created. The goal is to decrease the entropy at each split. If the entropy decreases, the split is validated and we can proceed to the next step, else, the decision tree picks another feature. 
# MAGIC 
# MAGIC Entropy ranges between 0 and 1. It is 0 for a pure set (i.e), group of observations containing only one label. 
# MAGIC 
# MAGIC ##### Gini impurity and Information gain
# MAGIC 
# MAGIC We quantify the amount of uncertainity at a single node by a metric called the gini impurity. We can quantify how much a split reduces the uncertainity by using a metric called the information gain. Information gain is the expected reduction in entropy caused by partitioning the examples according to a given attribute. These two metrics are used to select the best feature and threshold at each split point. The best feature reduces the uncertainity the most. Given the feature and threshold, the algorithm recursively buils the tree at each of the new child nodes (that are not leaf nodes). This process continues until all the nodes are pure or we reach a stopping criteria (such as a minimum number of examples).

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Mathematical definition of entropy
# MAGIC 
# MAGIC The general formula for entropy is:
# MAGIC $$ E = \sum_i -p_i {\log_2 p_i} $$
# MAGIC 
# MAGIC Since our dataset has a binary classification, all the observations fall into one of two classes. Suppose we have a set of N observations in the dataset. Let's assume that n observations have label 1 and m = N - n observations have label 0. p and q, the ratios of elements of each label in the dataset are given by:
# MAGIC 
# MAGIC $$p = \frac{n}{N}$$ $$q = \frac{m}{N} = 1-p $$
# MAGIC 
# MAGIC Entropy is given by the following equation:
# MAGIC $$E = -p {\log_2 (p)} -q {\log_2 (q)}$$

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Building the decision tree 
# MAGIC 
# MAGIC In our toy dataset, we have ten observations. Four of them have label 1 and six of them have label 0. Thus, entropy at the root node is given by:
# MAGIC 
# MAGIC $$ Entropy = -\frac{4}{10} {\log_2 (\frac{4}{10})} -\frac{6}{10} {\log_2 (\frac{6}{10})} = 0.966 $$
# MAGIC 
# MAGIC Entropy is close to 1 as we have a distribution close to 50/50 for the observations belonging to each class.

# COMMAND ----------

# MAGIC %md
# MAGIC The data set that goes down each branch of the tree has its own entropy value. We can calculate the expected entropy for each possible attribute. This is the degree to which the entropy would change if we branch on this attribute. We add the entropies of the two child nodes, weighted by the proportion of examples from the parent node that ended up at that child.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Insert picture of tree here.

# COMMAND ----------

# MAGIC %md
# MAGIC Information gain (or entropy / uncertainity reduction) at child node is calculated as 
# MAGIC $$ G = Entropy (parent) - Entropy (child) $$

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
# MAGIC Stacking is a general framework that involves training a learning algorithm to combine the predictions of several other learning algorithms to make a final prediction. Stacking can be used to handle an imbalanced dataset. The steps can be outlined as below.  
# MAGIC (a) Group the **training** data into majority and minority class.   
# MAGIC (b) Split the  majority class into \\(N + 1\\) groups, each group containing same number of data points as that in minority class.    
# MAGIC (c) Create \\(N + 1\\) datasets for training by combining the each group from (b) with minority class from (a). Each of these groups will be balanced.  
# MAGIC (d) Use \\(N\\) datasets from (c) to train the first level classifier. Once the models are generated, use the remaing one dataset from (c) to generate predictions for each of these models. These \\(N\\) predictions are the features for the second level classifier. The target/label value for the second level classifier is the target/label value of this remaining dataset.  
# MAGIC (e) Train the second level classifier. A final pipeline can be created by combining the models.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform data for Ensemble
# MAGIC Do assembler transformation before train test split for `VectorIndexer` to work 

# COMMAND ----------

def TransformDataForEnsemble(full_dataset):
  str_features = ["Op_Unique_Carrier",	"Origin", "Dest", "Holiday"] # String features

  # Convert strings to index
  str_indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in str_features]
  cat_features = ["Month", "Day_Of_Month", "Day_Of_Week", "Distance_Group", 'CRS_Dep_Time_bin', 'CRS_Arr_Time_bin', 'CRS_Elapsed_Time_bin']
  target       = ["Dep_Del30"]
  all_features = cat_features + ["Origin_Activity"]
  
  assembler = VectorAssembler(inputCols=all_features, outputCol="features")
  ensemble_pipeline = Pipeline(stages=str_indexers + [assembler])
  
  transformed_data =  (ensemble_pipeline
          .fit(full_dataset)
          .transform(full_dataset)
          .select(["Year", "features"] + target)
          .withColumnRenamed("Dep_Del30", "label"))
  
  # Here, create automatic Categories. All variables > 400 Categories will be treated as Continuous.
  # else will be treated as Categories by random forest.
  featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=400).fit(transformed_data)
  
  return all_features, featureIndexer, transformed_data

all_ensemble_features, ensamble_featureIndexer, ensemble_transformed_data = TransformDataForEnsemble(airlines)

# COMMAND ----------

# Debug - to remove
ensemble_transformed_data.show(4, truncate=False)

# COMMAND ----------

ensemble_mini_train, ensemble_train, ensemble_val, ensemble_test = SplitDataset(ensemble_transformed_data)

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

# Debug - to remove
[[d.groupBy('label').count().toPandas()["count"].to_list()] for d in train_group]

# COMMAND ----------

# Debug - to remove
train_combiner.groupBy('label').count().toPandas()["count"].to_list()

# COMMAND ----------

# Debug - to remove
train_group[6].show(4, truncate=False)

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

# COMMAND ----------

def TrainEnsembleModels(en_train, featureIndexer, classifier) :
  job = []
  for num, _ in enumerate(en_train):
      print("Create ensemble model : " + str(num))      
      # Chain indexer and classifier in a Pipeline 
      job.append(Pipeline(stages=[featureIndexer, classifier]))
      
  return pool.map(lambda x: x[0].fit(x[1]), zip(job, en_train))
      
ensemble_model = TrainEnsembleModels(train_group, ensamble_featureIndexer, 
                    # Type of model we can use.
                    RandomForestClassifier(featuresCol="indexedFeatures", maxBins=369, maxDepth=5, numTrees=5, impurity='gini')
                   )
print("Training done")

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
      marker_color=list(map(lambda x: px.colors.sequential.Plasma[x], range(0,len(plt[key]['features'])))),
      name = '',
      showlegend = False,
    ), row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(categoryorder='total descending', row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    if plt[key]['y_pos'] == 1: fig.update_yaxes(title_text="Feature importance", row=plt[key]['x_pos'], col=plt[key]['y_pos'])
    fig.update_xaxes(tickangle=-45)
    
fig.update_layout(height=1000, width=1000, title_text="Feature importance for individual ensembles")
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Construct a new data set based on the output of base classifiers

# COMMAND ----------

def do_ensemble_prediction(em, train_en) :
    prediction_array = []
    for num, m in enumerate(em) :
        print("Run prediction on ensemble model : " + str(num))
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
reduced_df.show(2)
ensemble_transformed = do_transform_final(reduced_df)
ensemble_transformed.show(2) 

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

# Final ensemble 
model_trained_ensemble = TrainCombiner(ensemble_transformed, ensemble_featureIndexer, 
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
# MAGIC #### Model evaluation

# COMMAND ----------

ensemble_test_prediction = FinalEnsmblePipeline(model_trained_ensemble, ensemble_model, ensemble_test)
EvaluateModelPredictions(ensemble_test_prediction, dataName='test')

# COMMAND ----------

ensemble_val_prediction = FinalEnsmblePipeline(model_trained_ensemble, ensemble_model, ensemble_val)
EvaluateModelPredictions(ensemble_val_prediction, dataName='validation')

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

