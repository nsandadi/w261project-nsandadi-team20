# Databricks notebook source
# MAGIC %md
# MAGIC ## Regular Dataset Prep

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

# MAGIC %md
# MAGIC ##Model Prep

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

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
  
def PredictAndEvaluate(model, data, dataName, outcomeName):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SMOTE the training dataset

# COMMAND ----------

dataName = 'smoted_train_data'
train_smoted =  spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

# COMMAND ----------

display(train_smoted.take(10))

# COMMAND ----------

display(train.groupBy(outcomeName).count())

# COMMAND ----------

display(train_smoted.groupBy(outcomeName).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model (no smote)

# COMMAND ----------

si_base = PrepStringIndexer(catFeatureNames)
va_base = PrepVectorAssembler(numericalFeatureNames = numFeatureNames, stringFeatureNames = catFeatureNames)

# COMMAND ----------

model_nosmote = TrainDecisionTreeModel(train, si_base + [va_base], outcomeName, maxDepth=8, maxBins=6647)
PrintDecisionTreeModel(model_nosmote.stages[-1], numFeatureNames + catFeatureNames)

# COMMAND ----------

PredictAndEvaluate(model_nosmote, train, "train", outcomeName)

# COMMAND ----------

PredictAndEvaluate(model_nosmote, val, "val", outcomeName)

# COMMAND ----------

PredictAndEvaluate(model_nosmote, test, "test", outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model (smote)

# COMMAND ----------

train_smoted = train_smoted.withColumnRenamed("DEP_DEL30", "Dep_Del30") \
                           .withColumnRenamed("YEAR", "Year") \
                           .withColumnRenamed("MONTH", "Month").withColumnRenamed("DAY_OF_MONTH", "Day_Of_Month") \
                           .withColumnRenamed("DAY_OF_WEEK", "Day_Of_Week") \
                           .withColumnRenamed("CRS_DEP_TIME", "CRS_Dep_Time").withColumnRenamed("CRS_ARR_TIME", "CRS_Arr_Time") \
                           .withColumnRenamed("CRS_ELAPSED_TIME", "CRS_Elapsed_Time") \
                           .withColumnRenamed("DISTANCE", "Distance") \
                           .withColumnRenamed("DISTANCE_GROUP", "Distance_Group") \
                           .withColumnRenamed("OP_UNIQUE_CARRIER", "Op_Unique_Carrier") \
                           .withColumnRenamed("ORIGIN", "Origin").withColumnRenamed("DEST", "Dest")

# COMMAND ----------

model_smote = TrainDecisionTreeModel(train_smoted, si_base + [va_base], outcomeName, maxDepth=8, maxBins=6647)
PrintDecisionTreeModel(model_smote.stages[-1], numFeatureNames + catFeatureNames)

# COMMAND ----------

PredictAndEvaluate(model_smote, train_moted, "train_smoted", outcomeName)
PredictAndEvaluate(model_smote, train, "train", outcomeName)
PredictAndEvaluate(model_smote, val, "val", outcomeName)
PredictAndEvaluate(model_smote, test, "test", outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Brieman

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

# MAGIC %md
# MAGIC ## Apply Brieman to unsmoted & train

# COMMAND ----------

# no smote data
briemanRanksDict = {}
for catFeatureName in catFeatureNames:
  # Get ranks for feature
  briemanRanksDict[catFeatureName] = GenerateBriemanRanks(train, catFeatureName, outcomeName)
  
  # Apply Brieman method & do feature transformation
  train_brieman = ApplyBriemansMethod(train, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  val_brieman = ApplyBriemansMethod(val, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  test_brieman = ApplyBriemansMethod(test, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  
briFeatureNames = [entry + "_brieman" for entry in briemanRanksDict]

# COMMAND ----------

si_brieman = PrepStringIndexer([])
va_brieman = PrepVectorAssembler(numericalFeatureNames = numFeatureNames + briFeatureNames, stringFeatureNames = [])

# COMMAND ----------

model_nosmote_brieman = TrainDecisionTreeModel(train_brieman, si_brieman + [va_brieman], outcomeName, maxDepth=8, maxBins=6647)
PrintDecisionTreeModel(model_nosmote_brieman.stages[-1], numFeatureNames + briFeatureNames)

# COMMAND ----------

PredictAndEvaluate(model_nosmote_brieman, train_brieman, "train_brieman", outcomeName)
PredictAndEvaluate(model_nosmote_brieman, val_brieman, "val_brieman", outcomeName)
PredictAndEvaluate(model_nosmote_brieman, test_brieman, "test_brieman", outcomeName)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Brieman to smoted & train

# COMMAND ----------

# smote data
briemanRanksDict = {}
for catFeatureName in catFeatureNames:
  # Get ranks for feature
  briemanRanksDict[catFeatureName] = GenerateBriemanRanks(train_smoted, catFeatureName, outcomeName)
  
  # Apply Brieman method & do feature transformation
  train_smoted_brieman = ApplyBriemansMethod(train_smoted, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  train_orig_brieman = ApplyBriemansMethod(train, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  val_smoted_brieman = ApplyBriemansMethod(val, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  test_smoted_brieman = ApplyBriemansMethod(test, briemanRanksDict[catFeatureName], catFeatureName, outcomeName)
  
briFeatureNames = [entry + "_brieman" for entry in briemanRanksDict]

# COMMAND ----------

model_smote_brieman = TrainDecisionTreeModel(train_smoted_brieman, si_brieman + [va_brieman], outcomeName, maxDepth=8, maxBins=6647)
PrintDecisionTreeModel(model_smote_brieman.stages[-1], numFeatureNames + briFeatureNames)

# COMMAND ----------

PredictAndEvaluate(model_smote_brieman, train_smoted_brieman, "train_smoted_brieman", outcomeName)
PredictAndEvaluate(model_smote_brieman, train_orig_brieman, "train_orig_brieman", outcomeName)
PredictAndEvaluate(model_smote_brieman, val_smoted_brieman, "val_smoted_brieman", outcomeName)
PredictAndEvaluate(model_smote_brieman, test_smoted_brieman, "test_smoted_brieman", outcomeName)

# COMMAND ----------

