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
  
  return (accuracy, recall, precision, f1)

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
# MAGIC ## Feature Engineering and Application to Dataset
# MAGIC Includes binning terms & generating interaction terms

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Bucketizer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import functions as f

# COMMAND ----------

# Augments the provided dataset for the given variable with binned/bucketized
# versions of that variable, as defined by splits parameter
# Column name suffixed with '_bin' will be the bucketized column
# Column name suffixed with '_binlabel' will be the nicely-named version of the bucketized column
def BinValues(df, var, splits, labels):
  bucketizer = Bucketizer(splits=splits, inputCol=var, outputCol=var + "_bin")
  df_buck = bucketizer.setHandleInvalid("keep").transform(df)
  
  bucketMaps = {}
  bucketNum = 0
  for l in labels:
    bucketMaps[bucketNum] = l
    bucketNum = bucketNum + 1
    
  def newCols(x):
    return bucketMaps[x]
  
  callnewColsUdf = udf(newCols, StringType())
    
  return df_buck.withColumn(var + "_binlabel", callnewColsUdf(f.col(var + "_bin")))

def Bin_CRS_Times(data):
  data = BinValues(data, 'CRS_Dep_Time', 
                   splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
                   labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm',
                             '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])
  data = BinValues(data, 'CRS_Arr_Time', 
                   splits = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
                   labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm',
                             '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am'])
  return BinValues(data, 'CRS_Elapsed_Time', 
                   splits = [Double.NegativeInfinity, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, Double.PositiveInfinity], 
                   labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', 
                             '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12+ hours'])

bfeatureNames = ['CRS_Dep_Time_binlabel', 'CRS_Arr_Time_binlabel', 'CRS_Elapsed_Time_binlabel']
mini_train_dep = Bin_CRS_Times(mini_train_dep)
train_dep = Bin_CRS_Times(train_dep)
val_dep = Bin_CRS_Times(val_dep)
test_dep = Bin_CRS_Times(test_dep)

# COMMAND ----------

def AddInteractions(data, varPairsAndNames):
  for (var1_prefix, var1, var2, newName) in varPairsAndNames:
    data = data.withColumn(newName, f.concat(f.lit(var1_prefix), f.col(var1), f.lit('-'), f.col(var2)))
  return data

# Make interaction variables: Day_of_Year, OriginxDest
interactions = [('', 'Month', 'Day_Of_Month', 'Day_Of_Year'), 
                ('', 'Origin', 'Dest', 'Origin-Dest'),
                ('Day_', 'Day_Of_Week', 'CRS_Dep_Time_binlabel', 'Dep_Time_Of_Week'),
                ('Day_', 'Day_Of_Week', 'CRS_Arr_Time_binlabel', 'Arr_Time_Of_Week')]
ifeatureNames = [i[3] for i in interactions]
mini_train_dep = AddInteractions(mini_train_dep, interactions)
train_dep = AddInteractions(train_dep, interactions)
val_dep = AddInteractions(val_dep, interactions)
test_dep = AddInteractions(test_dep, interactions)

# COMMAND ----------

# save updated departure delay data
mini_train_dep.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_mini_train_dep.parquet")
train_dep.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_train_dep.parquet")
val_dep.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_val_dep.parquet")
test_dep.write.mode('overwrite').format("parquet").save("dbfs/user/team20/airlines_test_dep.parquet")

# COMMAND ----------

display(dbutils.fs.ls("dbfs/user/team20"))

# COMMAND ----------

# Read prepared dataset
mini_train_dep = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_mini_train_dep.parquet")
train_dep = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_train_dep.parquet")
val_dep = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_val_dep.parquet")
test_dep = spark.read.option("header", "true").parquet(f"dbfs/user/team20/airlines_test_dep.parquet")

print("Mini_Train - \tActual: " + str(mini_train_dep.count()), "\t\tExpected: 2094")
print("Train - \tActual: " + str(train_dep.count()), "\tExpected: 20916420")
print("Val - \t\tActual: " + str(val_dep.count()), "\tExpected: 2986961")
print("Test - \t\tActual: " + str(test_dep.count()), "\tExpected: 7268232")

# COMMAND ----------

# Outcome Variable
outcomeName = 'Dep_Del30'

# Numerical features
nfeatureNames = [
  'Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group'
]

# Categorical features
cfeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest']
bfeatureNames = ['CRS_Dep_Time_binlabel', 'CRS_Arr_Time_binlabel', 'CRS_Elapsed_Time_binlabel']
interactions = [('', 'Month', 'Day_Of_Month', 'Day_Of_Year'), 
                ('', 'Origin', 'Dest', 'Origin-Dest'),
                ('Day_', 'Day_Of_Week', 'CRS_Dep_Time_binlabel', 'Dep_Time_Of_Week'),
                ('Day_', 'Day_Of_Week', 'CRS_Arr_Time_binlabel', 'Arr_Time_Of_Week')]
ifeatureNames = [i[3] for i in interactions]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Decision Tree Pipeline
# MAGIC Resources: 
# MAGIC * Documentation on Decision Trees & Pipelines Api from Pyspark ML - https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
# MAGIC * DataFrames: https://spark.apache.org/docs/1.5.2/mllib-decision-tree.html

# COMMAND ----------

# Encodes a string column of labels to a column of label indices
# Set HandleInvalid to "keep" so that the indexer adds new indexes when it sees new labels (could also do "error" or "skip")
# Apply string indexer to categorical, binned, and interaction features (all string formatted), as applicable
# Docs: https://spark.apache.org/docs/latest/ml-features#stringindexer
def PrepStringIndexer(sfeatureNames):
  return [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in sfeatureNames]

# Use VectorAssembler() to merge our feature columns into a single vector column, which will be passed into the model. 
# We will not transform the dataset just yet as we will be passing the VectorAssembler into our ML Pipeline.
def PrepVectorAssembler(numericalfeatureNames, stringfeatureNames):
  return VectorAssembler(inputCols = numericalfeatureNames + [f + "_idx" for f in stringfeatureNames], outputCol = "features")

# COMMAND ----------

def TrainAndEvaluate(trainingData, varStages, outcomeName, maxDepth, maxBins, evalTrainingData = False):
  # Train Model
  dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = maxDepth, maxBins=maxBins) 
  pipeline = Pipeline(stages = varStages + [dt])
  dt_model = pipeline.fit(trainingData)
  
  # Evaluate Model
  if (evalTrainingData):
    training_data_res = EvaluateModelPredictions(dt_model, trainingData, "training data", outcomeName)
  mini_train_res = EvaluateModelPredictions(dt_model, mini_train_dep, "mini-training", outcomeName)
  train_res = EvaluateModelPredictions(dt_model, train_dep, "training", outcomeName)
  val_res = EvaluateModelPredictions(dt_model, val_dep, "validation", outcomeName)
  
  if (evalTrainingData):
    return (dt_model, training_data_res, mini_train_res, train_res, val_res)
  else:
    return (dt_model, mini_train_res, train_res, val_res)

# COMMAND ----------

# Model 1: Use all prepared variables on mini_training
si1 = PrepStringIndex(cfeatureNames + bfeatureNames + ifeatureNames)
va1 = PrepVectorAssembler(nfeatureNames, cfeatureNames + bfeatureNames + ifeatureNames)
modelResults1 = TrainAndEvaluate(mini_train_dep, si1 + [va1], outcomeName, maxDepth=5, maxBins=1423)
display(modelResults1[0].stages[-1])

# COMMAND ----------

modelResults1[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults1_dtmodel.txt")

# COMMAND ----------

#display(modelResults1[0].stages[-1])
printModel(modelResults1[0].stages[-1], nfeatureNames + cfeatureNames + bfeatureNames + ifeatureNames)

# COMMAND ----------

# Model 2: Use all prepared variables on training (no features = baseline :P)
si2 = PrepStringIndexer(cfeatureNames + bfeatureNames + ifeatureNames)
va2 = PrepVectorAssembler(nfeatureNames, cfeatureNames + bfeatureNames + ifeatureNames)
modelResults2 = TrainAndEvaluate(train_dep, si2 + [va2], outcomeName, maxDepth=5, maxBins=6647)
printModel(modelResults2[0].stages[-1], nfeatureNames + cfeatureNames + bfeatureNames + ifeatureNames)

# COMMAND ----------

modelResults2[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults2_dtmodel.txt")

# COMMAND ----------

# Model 3: Use all prepared variables on training, but with greather depth (8)
si3 = PrepStringIndexer(cfeatureNames + bfeatureNames + ifeatureNames)
va3 = PrepVectorAssembler(nfeatureNames, cfeatureNames + bfeatureNames + ifeatureNames)
modelResults3 = TrainAndEvaluate(train_dep, si3 + [va3], outcomeName, maxDepth=8, maxBins=6647)
printModel(modelResults3[0].stages[-1], nfeatureNames + cfeatureNames + bfeatureNames + ifeatureNames)

# COMMAND ----------

modelResults3[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults3_dtmodel.txt")

# COMMAND ----------

# Model 4: Use all prepared variables on training, removing variables that have been binned already
                  # 0        1            2              3              4 
n4featureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group']
si4 = PrepStringIndexer(cfeatureNames + bfeatureNames + ifeatureNames)
va4 = PrepVectorAssembler(n4featureNames, cfeatureNames + bfeatureNames + ifeatureNames)
modelResults4 = TrainAndEvaluate(train_dep, si4 + [va4], outcomeName, maxDepth=8, maxBins=6647)
printModel(modelResults4[0].stages[-1], n4featureNames + cfeatureNames + bfeatureNames + ifeatureNames)

# COMMAND ----------

modelResults4[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults4_dtmodel.txt")

# COMMAND ----------

# Model 5: Use all variables, but treat binned variables as numerical instead of categorical (to give order)
b5featureNames = [f[0:-5] for f in bfeatureNames]
n5featureNames = nfeatureNames + b5featureNames
s5featureNames = cfeatureNames + ifeatureNames

si5 = PrepStringIndexer(s5featureNames)
va5 = PrepVectorAssembler(n5featureNames, s5featureNames)
modelResults5 = TrainAndEvaluate(train_dep, si5 + [va5], outcomeName, maxDepth=8, maxBins=6647)
printModel(modelResults5[0].stages[-1], n5featureNames + s5featureNames)

# COMMAND ----------

modelResults5[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults5_dtmodel.txt")

# COMMAND ----------

# Model 6: Same mode as in 4, except using binned variable as numerical instead of categorical
b6featureNames = [f[0:-5] for f in bfeatureNames]
n6featureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group'] + b6featureNames
s6featureNames = cfeatureNames + ifeatureNames

si6 = PrepStringIndexer(s6featureNames)
va6 = PrepVectorAssembler(n6featureNames, s6featureNames)
modelResults6 = TrainAndEvaluate(train_dep, si6 + [va6], outcomeName, maxDepth=8, maxBins=6647)
printModel(modelResults6[0].stages[-1], n6featureNames + s6featureNames)

# COMMAND ----------

modelResults6[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults6_dtmodel.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Balance Majority & Minority Classes

# COMMAND ----------

display(train_dep.groupBy(outcomeName).count())

# COMMAND ----------

# MAGIC %md
# MAGIC Because when the departure delay threshold is 30, about 90% of the data is no delay and 10% is delay, we'll need to balance the dataset by splitting it into subsets of 20%, where half is the 10% comprising the minority class and the other half is 10% from the majority class. 

# COMMAND ----------

def SplitAndStackUnbalancedDataset(data, outcomeName, min_label = 1, maj_label = 0, seed=6):
  # Split dataset into majority & minority class
  data_min_class = data.filter(data[outcomeName] == min_label)
  data_maj_class = data.filter(data[outcomeName] == maj_label)
  
  # Split the majority class to chunks about the same size as the minority class
  min_count = train_dep_min_class.count()
  maj_count = train_dep_maj_class.count()
  num_datasets = round(maj_count / min_count)
  splitWeights = [1/num_datasets for i in range(num_datasets)]
  maj_class_splits = data_maj_class.randomSplit(weights=splitWeights, seed=seed)
  
  # Stack minority class with each majority class split individually
  datasets = []
  for maj_class in maj_class_splits:
    datasets.append(maj_class.union(data_min_class))
  return datasets

# COMMAND ----------

mini_train_dep_stacks = SplitAndStackUnbalancedDataset(mini_train_dep, outcomeName, 1, 0, 6)
for stack in mini_train_dep_stacks:
  print(stack.count())

# COMMAND ----------

# Split & Stack training dataset
train_dep_stacks = SplitAndStackUnbalancedDataset(train_dep, outcomeName, 1, 0, 6)
display(train_dep_stacks[0].groupBy(outcomeName).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training on subset of train with stacking

# COMMAND ----------

# Retrain Model 2 as Model 7: Use all prepared variables on training, but only first stack
si7 = PrepStringIndexer(cfeatureNames + bfeatureNames + ifeatureNames)
va7 = PrepVectorAssembler(nfeatureNames, cfeatureNames + bfeatureNames + ifeatureNames)
modelResults7 = TrainAndEvaluate(train_dep_stacks[0], si7 + [va7], outcomeName, maxDepth=5, maxBins=6647, evalTrainingData=True)
printModel(modelResults7[0].stages[-1], nfeatureNames + cfeatureNames + bfeatureNames + ifeatureNames)

# COMMAND ----------

modelResults7[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults7_dtmodel.txt")

# COMMAND ----------

# Try saving first split to cluster for faster retrieval?
train_dep_stacks[0].write.mode('overwrite').format("parquet").save("dbfs/user/team20/train_dep_stacks/stack0.parquet")
train_dep_stacks[0] = spark.read.option("header", "true").parquet(f"dbfs/user/team20/train_dep_stacks/stack0.parquet")

# COMMAND ----------

# Retried Model 6: Same model 6, exception only using the first stack of data
b8featureNames = [f[0:-5] for f in bfeatureNames]
n8featureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Distance_Group'] + b8featureNames
s8featureNames = cfeatureNames + ifeatureNames

si8 = PrepStringIndexer(s8featureNames)
va8 = PrepVectorAssembler(n8featureNames, s8featureNames)
modelResults8 = TrainAndEvaluate(train_dep_stacks[0], si8 + [va8], outcomeName, maxDepth=8, maxBins=6647, evalTrainingData=True)
printModel(modelResults8[0].stages[-1], n8featureNames + s8featureNames)

# COMMAND ----------

modelResults8[0].save(f"dbfs/user/team20/DecisionTree-2-dianai-models/modelResults8_dtmodel.txt")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# bucketize variables: CRS_Dep_Time, CRS_Arr_Time, CRS_Elapsed_Time
# Use "keep" so that the indexer adds new indexes when it sees new labels (could also do "error" or "skip")

# CRS_Dep_Time & CRS_Arr_Time
bi_crs_dep_time = Bucketizer(splits=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
                             inputCol='CRS_Dep_Time', outputCol='CRS_Dep_Time_bin', handleInvalid="keep")
bi_crs_arr_time = Bucketizer(splits=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
                             inputCol='CRS_Arr_Time', outputCol='CRS_Arr_Time_bin', handleInvalid="keep")
bi_crs_time_labels = ['12am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm',
                      '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am']
  
# CRS_Elapsed_Time
bi_crs_dep_time = Bucketizer(splits=[Double.NegativeInfinity, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, Double.PositiveInfinity], 
                             inputCol='CRS_Elapsed_Time', outputCol='CRS_Elapsed_Time_bin', handleInvalid="keep")
bi_crs_elapsed_labels = ['1 hour', '2 hours', '3 hours', '4 hours', '5 hours', '6 hours', '7 hours', '8 hours', '9 hours', '10 hours', '11 hours', '12+ hours']

# Make a set of bucketizers
bi = [bi_crs_dep_time, bi_crs_arr_time, bi_crs_dep_time]