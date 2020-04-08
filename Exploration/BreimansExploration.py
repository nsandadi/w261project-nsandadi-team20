# Databricks notebook source
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

# Generate other Departure Delay outcome indicators for n minutes
def CreateNewDepDelayOutcome(data, thresholds):
  for threshold in thresholds:
    data = data.withColumn('Dep_Del' + str(threshold), (data['Dep_Delay'] >= threshold).cast('integer'))
  return data  
  
airlines = CreateNewDepDelayOutcome(airlines, [30])

# COMMAND ----------

outcomeName = 'Dep_Del30'
numFeatureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']
catFeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
joiningFeatures = ['FL_Date'] # Features needed to join with the holidays dataset--not needed for training

# COMMAND ----------

airlines = airlines.select([outcomeName] + numFeatureNames + catFeatureNames + joiningFeatures)

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

display(test.filter('Year != 2019'))

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

briemanRanksDict = {}
for catFeatureName in catFeatureNames:
  # Get ranks for feature
  briemanRanksDict[catFeatureName] = GenerateBriemanRanks(train, catFeatureName, outcomeName)
  
  # Apply Brieman method & do feature transformation
  mini_train = ApplyBriemansMethod(mini_train, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  train = ApplyBriemansMethod(train, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  val = ApplyBriemansMethod(val, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  test = ApplyBriemansMethod(test, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  airlines = ApplyBriemansMethod(airlines, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  
briFeatureNames = [entry for entry in briemanRanksDict]

# COMMAND ----------

display(test.filter("Origin == 'ATY' or Origin == 'PIR' or Origin == 'PAE' or Origin == 'BFM' or Origin == 'XWA'"))

# COMMAND ----------

display(mini_train.take(10))

# COMMAND ----------

display(airlines.sample(fraction=0.00001, seed=6))

# COMMAND ----------

display(briemanRanks['Op_Unique_Carrier'])

# COMMAND ----------

display(airlines.filter("Dest == 'FNL'"))

# COMMAND ----------

airlines.createOrReplaceTempView("airlines")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify if there are any entries where thre are only unique values for 1 year
# MAGIC select Op_Unique_Carrier, min(Year), min(Op_Unique_Carrier_brieman) from airlines group by Op_Unique_Carrier having count(distinct Year) == 1 and min(Year) == 2019

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify if there are any entries where thre are only unique values for 1 year
# MAGIC select Dest, min(Year), min(Dest_brieman) from airlines group by Dest having count(distinct Year) == 1 and min(Year) == 2019

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify if there are any entries where thre are only unique values for 1 year
# MAGIC select Origin, min(Year), min(Origin_brieman) from airlines group by Origin having count(distinct Year) == 1 and min(Year) == 2019

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select Op_Unique_Carrier, Dest, Origin, Year from airlines where Op_Unique_Carrier_brieman == -1 or Dest_brieman == -1 or Origin_brieman == -1

# COMMAND ----------

