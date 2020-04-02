# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Tree Exploration - 2 - dianai
# MAGIC ## Load Data from Parquet (in split form)

# COMMAND ----------

airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# Filter to datset with entries where diverted != 1, cancelled != 1, dep_delay != Null, and arr_delay != Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print("Number of records in full dataset:", airlines.count())

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

# Generate other Departure Delay outcome indicators (30, 45, 60)
def CreateDepDelayFeatures(data, thresholds):
  for threshold in thresholds:
    data = data.withColumn('Dep_Del' + str(threshold), (data['Dep_Delay'] >= threshold).cast('integer'))
  return data  
  
mini_train = CreateDepDelayFeatures(mini_train, [30, 45, 60])
train = CreateDepDelayFeatures(train, [30, 45, 60])
val = CreateDepDelayFeatures(val, [30, 45, 60])
test = CreateDepDelayFeatures(test, [30, 45, 60])

# COMMAND ----------

print("train size = " + str(train.count()))
print("mini_train size = " + str(mini_train.count()))
print("val size = " + str(val.count()))
print("test size = " + str(test.count()))

# COMMAND ----------

# save data as parquet
mini_train.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_mini_train.parquet")
train.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_train.parquet")
val.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_val.parquet")
test.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_test.parquet")

display(dbutils.fs.ls("dbfs/user/team20"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### >>> START HERE <<<

# COMMAND ----------

# read data back in directly from parquet for easier analysis
mini_train = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_mini_train.parquet")
train = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_train.parquet")
val = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_val.parquet")
test = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_test.parquet")

# COMMAND ----------

print("Mini_Train - \tActual: " + str(mini_train.count()), "\t\tExpected: 2094")
print("Train - \tActual: " + str(train.count()), "\tExpected: 20916420")
print("Val - \t\tActual: " + str(val.count()), "\tExpected: 2986961")
print("Test - \t\tActual: " + str(test.count()), "\tExpected: 7268232")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Initial Specification for Departure Delay Model

# COMMAND ----------

# Outcome Variable
outcomeName = 'Dep_Del30'

# Numerical features
nfeatureNames = [
  # 0        1            2              3              4                5                6               7               8 
  'Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group'
]

# Categorical features
#                         9              10       11
cfeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']

# Filter full data to just relevant columns
mini_train_dep = mini_train.select([outcomeName] + nfeatureNames + cfeatureNames)
train_dep = train.select([outcomeName] + nfeatureNames + cfeatureNames)
val_dep = val.select([outcomeName] + nfeatureNames + cfeatureNames)
test_dep = test.select([outcomeName] + nfeatureNames + cfeatureNames)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling Helper Functions

# COMMAND ----------

import ast
import random

# Visualize the decision tree model that was trained in text form
def printModel(model, featureNames):
  lines = model.toDebugString.split("\n")
  featuresUsed = set()
  
  for line in lines:
    parts = line.split(" ")

    # Replace "feature #" with feature name
    if ("feature" in line):
      featureNumIdx = parts.index("(feature") + 1
      featureNum = int(parts[featureNumIdx])
      parts[featureNumIdx] = featureNames[featureNum] # replace feature number with actual feature name
      parts[featureNumIdx - 1] = "" # remove word "feature"
      featuresUsed.add(featureNames[featureNum])
      
    # Summarize sets of values for easier reading
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
    
  print("\n", "All Features:  ", featureNames)
  print("\n", "Features Used: ", featuresUsed)

# COMMAND ----------

# Model Evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def EvaluateModelPredictions(model, data, dataName, outcomeName):
  # Make predictions on test data to measure the performance of the model
  predictions = model.transform(data)
  
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

# MAGIC %md
# MAGIC ## Make Decision Tree Pipeline
# MAGIC Resources: 
# MAGIC * Documentation on Decision Trees & Pipelines Api from Pyspark ML - https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
# MAGIC * DataFrames: https://spark.apache.org/docs/1.5.2/mllib-decision-tree.html

# COMMAND ----------

