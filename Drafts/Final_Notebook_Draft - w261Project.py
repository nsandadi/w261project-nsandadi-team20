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

from pyspark.sql import functions as F

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
# MAGIC ### Explanation for why we chose the variable we chose (rationale for why not all variables are available at inference time)

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

# MAGIC %md
# MAGIC ### EDA Task #3: Unbalanced Dataset
# MAGIC * Show how dataset is drastically unbalanced (especially as you increase the dep_delay threshold)
# MAGIC * Discuss stacking and what that tries to do (how we'll eventually want an ensemble approach to support this)
# MAGIC * If can get SMOTE working to be scalable, also discuss SMOTE and what it effectively does
# MAGIC    - do discuss scalability concerns, b/c do need to apply knn algo & predict on each datapoint
# MAGIC * Discuss how balancing the dataset with stacking or SMOTE allows us to ensure that our models don't become biased towards the no-delay class, but it does introduce some more variance (bias-variance tradeoff)

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
print(" - Numerical Features: \t\t", numFeatureNames)
print(" - Categorical Features: \t", catFeatureNames)
print(" - Binned Features: \t\t", binFeatureNames)
print(" - Interaction Features: \t", intFeatureNames)
print(" - Holiday Feature: \t\t", holFeatureNames)
print(" - Origin Activity Feature: \t", orgFeatureNames)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Breiman's Theorem to Categorical & Interaction Features!!!

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
outcomeName = 'Dep_Del30'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Toy Example: Decision Trees

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Dataset

# COMMAND ----------

# Build the toy example dataset from the cleaned and transformed mini training set
toy_dataset = mini_train.select(['Dep_Del30', 'Year', 'Month', 'Day_Of_Week', 'CRS_Dep_Time', 'Origin_Dest', 'Op_Unique_Carrier']) \
                        .filter(mini_train['Dep_Del30'] == 0) \
                        .sample(False, 0.00252, 8)

toy_dataset = toy_dataset.union(mini_train.select(['Dep_Del30', 'Year', 'Month', 'Day_Of_Week', 'CRS_Dep_Time', 'Origin_Dest', 'Op_Unique_Carrier']) \
                         .filter(mini_train['Dep_Del30'] == 1) \
                         .sample(False, 0.005, 8))

display(toy_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Introduction
# MAGIC - The first column 'Dep_Del30' is the label or outcome variable, and the rest of the columns are features. 
# MAGIC - The goal of the decision tree is to predict the departure delay.
# MAGIC - The dataset has both numeric and categorical features.
# MAGIC - We use CART decision tree algorithm (Classification and Regression Trees). 
# MAGIC - Decision trees use a process to ask certain questions at a certain point to make decisions about splitting the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Decision tree learning
# MAGIC - Initially, we have a root node for the tree.
# MAGIC - The root node receives the entire training set as input and all subsequent nodes receive a subset of rows as input.
# MAGIC - Each node asks a true/false question about one of the features using a threshold and in response, the dataset is split into two subsets.
# MAGIC - The subsets become input to the child nodes added to the tree for the next level of splitting.
# MAGIC - The ultimate goal is to produce the purest distribution of labels at each node.
# MAGIC - If a node contains examples of only a single type of label, it has 100% purity and becomes a leaf node. The subset doesn't need to be split any further.
# MAGIC - On the other hand, if a node still contains mixed labels, the decision tree chooses another question and threshold, based on which the dataset is split further.
# MAGIC - The trick to building an effective tree is to decide which questions to ask and when.
# MAGIC - To do this, we need to quantify how well a question can split the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Gini impurity and Information gain
# MAGIC - We quantify the amount of uncertainity at a single node by a metric called the gini impurity. 
# MAGIC - We can quantify how much a question reduces the uncertainity by using a metric called the information gain.
# MAGIC - These two metrics are used to select the best question at each split point. 
# MAGIC - The best question reduces the uncertainity the most.
# MAGIC - Given the question, the algorithm recursively buils the tree at each of the new child nodes (that are not leaf nodes).
# MAGIC - This process continues until all the nodes are pure or we reach a stopping criteria (such as a certain number of examples).

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### View of tree building / completed tree

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling Helpers
# MAGIC * Evaluation functions
# MAGIC * Decision Tree PrintModel Function

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

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
from pyspark.ml import Pipeline

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

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluates model predictions for the provided predictions
# Predictions must have two columns: prediction & label
def EvaluateModelPredictions(predictions, dataName=None, outcomeName='label'):   
  print("\nModel Evaluation - ", dataName)
  print("------------------------------------------")

  # Accuracy
  evaluator = MulticlassClassificationEvaluator(labelCol=outcomeName, predictionCol="prediction", metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)
  print("Accuracy:\t", accuracy)

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

# COMMAND ----------

def PredictAndEvaluate(model, data, dataName, outcomeName):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName)

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
# MAGIC ### Discussion on Dataset Balancing

# COMMAND ----------

def PrepareDatasetForStacking(train, outcomeName, majClass = 0, minClass = 1):
  # Determine distribution of dataset for each outcome value (zero & one)
  ones, zeros = train.groupBy(outcomeName).count().sort(train[outcomeName].desc()).toPandas()["count"].to_list()

  # Set number of models & number of datasets (2 more than ratio majority to minority class)
  num_models = int(zeros/ones) + 2
  print("Number of models : " + str(num_models))
  
  # Split dataset for training individual modesl and for training the voting (ensemble) model
  zero_df, zero_df_train_ensemble = train.filter(outcomeName + ' == ' + str(majClass)).randomSplit([0.5, 0.5], 1)
  one_df, one_df_train_ensemble  = train.filter(outcomeName + ' == ' + str(minClass)).randomSplit([0.5, 0.5], 1)

  # Construct dataset for voting (ensemble) model
  train_ensemble = zero_df_train_ensemble.union(one_df_train_ensemble).sample(False, 0.999999999999, 1)

  # get number of values in minority class
  one_df_count = one_df.count()
  print("Minority Class Size: ", one_df_count)
  
  zeros_array = zero_df.randomSplit([1.0] * num_models, 1)
  zeros_array_count = [s.count() for s in zeros_array]
  ones_array = [one_df.sample(False, min(0.999999999999, r/one_df_count), 1) for r in zeros_array_count]
  ones_array_count = [s.count() for s in ones_array]

  # Array of `num_models` datasets
  # below resampling may not be necessary for random forest.
  # Need to remove it in case of performance issues
  train_models = [a.union(b).sample(False, 0.999999999999, 1) for a, b in zip(zeros_array, ones_array)]
  
  return (train_models, train_ensemble)

# Prepare datasets for stacking
train_datasets, train_ensemble_dataset = PrepareDatasetForStacking(train, outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest of Decision Trees with Dataset Stacking

# COMMAND ----------

def TrainEnsembleDecisionTrees(train_datasets, train_ensemble_dataset, stages, outcomeName, maxDepth, maxBins):
  ensemble_model = []
  for num, df in enumerate(train_datasets):
    print("Training DecisionTreeClassifier model : " + str(num))
    TrainDecisionTreeModel(df, stages, outcomeName, maxDepth=maxDepth, maxBins=maxBins)
    ensemble_model.append(model_dt)

  return ensemble_model

dt_ensemble = 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest of Random Forests with Dataset Stacking

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

