# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Tree Model Exloration
# MAGIC 
# MAGIC ### Initial Data Prep Work
# MAGIC Read Data, Filter to well-formed data (no nulls on oucomes), split dataset, 

# COMMAND ----------

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

# MAGIC %md
# MAGIC ### Model for Prediction Departure Delay 
# MAGIC - Variables to predict Departure Delay (1/0) - `Dep_Del15`
# MAGIC - Inference Time: 6 hours before CRS_Dep_Time
# MAGIC 
# MAGIC ##### Year, Month, Day of week, Day of Month
# MAGIC - Day of Month -- include when we join on holidays
# MAGIC - Year by itself -- continuous variable
# MAGIC - Month by itself -- categorical 
# MAGIC - Day of week -- categorical
# MAGIC 
# MAGIC ##### Unique_Carrer
# MAGIC - categorical
# MAGIC 
# MAGIC ##### Origin-attribute
# MAGIC - categorical
# MAGIC 
# MAGIC ##### Destination-attribute
# MAGIC - categorical
# MAGIC 
# MAGIC ##### CRS_Dep_Time, CRS_Arr_Time
# MAGIC - If continuous: minutes after midnight
# MAGIC - If categorical: groups of 15 minutes, 30 minutes, or 1 hr (binning)
# MAGIC - can use continuous and/or categorical
# MAGIC - Interaction of Day of week with CRS_Dep_Time (by hr)
# MAGIC - Interaction of Day of week with CRS_Arr_Time (by hr) -- might not be useful, but can eval with L1 Norm
# MAGIC 
# MAGIC ##### CRS_Elapsed_Time
# MAGIC - If continuous: minutes after midnight
# MAGIC - If categorical: groups of 15 minutes, 30 minutes, or 1 hr (binning)
# MAGIC - can use continuous and/or categorical
# MAGIC 
# MAGIC ##### Distance & Distance_Group
# MAGIC - experiment with using either or
# MAGIC - have both categorical & continuous depending on which we want to use
# MAGIC 
# MAGIC ##### Outcome: Boolean(`Dep_Delay > 15` === `Dep_Del15 = 1`)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Resources
# MAGIC - Documentation on Pipelines Api from MLlib - https://spark.apache.org/docs/1.5.2/ml-decision-tree.html
# MAGIC - Documentation on Decision Trees from MLlib - https://spark.apache.org/docs/1.5.2/mllib-decision-tree.html

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### EDA Space (in Dataframes & SQL!)

# COMMAND ----------

# Register table so it is accessible via SQL Context
mini_train_dep.createOrReplaceTempView("mini_train_dep")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Looks like we can run SQL directly for EDA -- want to return back to this
# MAGIC select Dep_Del15 from mini_train_dep

# COMMAND ----------

# Get number of distinct values for each column in full training dataset
from pyspark.sql.functions import col, countDistinct
display(train.agg(*(countDistinct(col(c)).alias(c) for c in mini_train_dep.columns)))

# COMMAND ----------

# Get number of distinct values for each column in mini training dataset
display(mini_train_dep.agg(*(countDistinct(col(c)).alias(c) for c in mini_train_dep.columns)))

# COMMAND ----------

display(mini_train_dep.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Prep the Dataset for Decision Tree Training
# MAGIC ##### Plan as reading documentation
# MAGIC - Source: 
# MAGIC      - DataFrames: https://spark.apache.org/docs/1.5.2/mllib-decision-tree.html
# MAGIC      - Python RDDs: https://spark.apache.org/docs/2.2.0/mllib-decision-tree.html
# MAGIC - Using classification decision tree
# MAGIC - want to use entropy impurity measure (or Gini impurity measure, but not covered in async)
# MAGIC - should try to bin continuous variables (either in advance or tell spark to do so) -- discretizing continuous variables
# MAGIC - for ordered categorical vars, treat them as categorical
# MAGIC - for unordered cateogrical vars, use Breiman's method (Async videos week 12) to reduce number of candidate splits
# MAGIC 
# MAGIC #####Features:
# MAGIC - Year:integer
# MAGIC - Month:integer
# MAGIC - Day_Of_Month:integer
# MAGIC - Day_Of_Week:integer
# MAGIC - Op_Unique_Carrier:string
# MAGIC - Origin:string
# MAGIC - Dest:string
# MAGIC - CRS_Dep_Time:integer
# MAGIC - CRS_Arr_Time:integer
# MAGIC - CRS_Elapsed_Time:double
# MAGIC - Distance:double
# MAGIC - Distance_Group:integer

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint

# Define outcome & features to use in model development
outcomeName = ['Dep_Del15']
featureNames = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 
               #    'Op_Unique_Carrier', 'Origin', 'Dest', # need to figure out how to bring in categorical vars
                   'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group']

# Prep Data to be used to train and do predictions with decision tree
def PrepDataForDecisionTreeTraining(df, outcome, featureNames):
  # Extract only columns of interest
  df = df.select(outcomeVar + featureNames)

  # Convert dataframe to RDD of LabelPoints (for easier training)
  rdd = df.rdd.map(lambda x: LabeledPoint(x[0], [x[1:]])).cache()
  
  return rdd
  
train_dep_rdd = PrepDataForDecisionTreeTraining(train, outcomeName, featureNames)
mini_train_dep_rdd = PrepDataForDecisionTreeTraining(mini_train, outcomeName, featureNames)
val_dep_rdd = PrepDataForDecisionTreeTraining(val, outcomeName, featureNames)
test_dep_rdd = PrepDataForDecisionTreeTraining(test, outcomeName, featureNames)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Decision Tree Model

# COMMAND ----------

# Print out model so it's more human-readable 
def printModel(modelString, featureNames):
  lines = modelString.split("\n")
  
  for line in lines:
    # figure out feature name (to not write "feature #" in line)
    if ("feature" in line):
      parts = line.split(" ")
      featureNumIdx = parts.index("(feature") + 1
      featureNum = int(parts[featureNumIdx])
      parts[featureNumIdx] = featureNames[featureNum]
      line = " ".join(parts)
    
    print(line)

# Train a DecisionTree model.
# Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(train_dep_rdd,
                                     numClasses=2,
                                     categoricalFeaturesInfo={},
                                     impurity='gini', 
                                     maxDepth=10, 
                                     maxBins=32)

printModel(model.toDebugString(), featureNames)

# COMMAND ----------

import numpy as np

# Do prediction on validation set & capture error
def PredictAndPrintError(model, data, datasetTypeName):
  predictions = model.predict(data.map(lambda x: x.features))
  labelsAndPredictions = data.map(lambda lp: lp.label).zip(predictions)
  
  # figure out perf (first value is label, second value is prediction)
  #accuracy = labelsAndPredictions.map(lambda lp: lp[0] == lp[1]).mean()
  
  tp = labelsAndPredictions.filter(lambda lp: lp[0] == 1.0 and lp[1] == 1.0).count()
  tn = labelsAndPredictions.filter(lambda lp: lp[0] == 0.0 and lp[1] == 0.0).count()
  fp = labelsAndPredictions.filter(lambda lp: lp[0] == 0.0 and lp[1] == 1.0).count()
  fn = labelsAndPredictions.filter(lambda lp: lp[0] == 1.0 and lp[1] == 0.0).count()
  
  accuracy = (tp + tn) / (tp + tn + fp + fn) if ((tp + tn + fp + fn) != 0) else 0.0
  precision = tp / float(tp + fp) if ((tp + fp) != 0) else 0.0
  recall = tp / float(tp + fn) if ((tp + fn) != 0) else 0.0
  f1 = 2 * ((precision * recall) / float(precision + recall)) if ((precision + recall) != 0) else 0.0
  
  res = [datasetTypeName, str(np.round(accuracy, 6)), str(np.round(precision, 6)), str(np.round(recall, 6)), str(np.round(f1, 6))]
  print("\t".join(res))
  
print("Dataset\t\tAccuracy\tPrecision\tRecall\tF1-Score")
PredictAndPrintError(model, mini_train_dep_rdd, "Mini Training")
PredictAndPrintError(model, train_dep_rdd, "Training")
PredictAndPrintError(model, val_dep_rdd, "Validation")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Running Decision Tree with Pipelines

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Define outcome & features to use in model development
outcomeName = 'Dep_Del15'
nfeatureNames = [
  # 1        2            3              4              5                6                7               8               9 
  'Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group'
]
#                         10             11       12
cfeatureNames = ['Op_Unique_Carrier', 'Origin', 'Dest'] # need to figure out how to bring in categorical vars

# Prep data to relevant rows
mini_train_dep = mini_train.select([outcomeName] + nfeatureNames + cfeatureNames)
train_dep = train.select([outcomeName] + nfeatureNames + cfeatureNames)
val_dep = val.select([outcomeName] + nfeatureNames + cfeatureNames)

# Encodes a string column of labels to a column of label indices
si = [StringIndexer(inputCol=column, outputCol=column+"_idx") for column in cfeatureNames]

# Use VectorAssembler() to merge our feature columns into a single vector column, which will be passed into the model. 
# We will not transform the dataset just yet as we will be passing the VectorAssembler into our ML Pipeline.
va = VectorAssembler(inputCols = nfeatureNames + [cat + "_idx" for cat in cfeatureNames], outputCol = "features")

# Define dthe decision tree model
dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = 5, maxBins=205)

# Chain assembler and model in a pipeline
#pipeline = Pipeline(stages= si + [va, nb])
pipeline = Pipeline(stages = si + [va, dt])

# Run stages in pipeline and train the model
dt_model = pipeline.fit(mini_train_dep)

# COMMAND ----------

# Visualize the decision tree model that was trained
display(dt_model.stages[-1])

# COMMAND ----------

# Model Evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def EvaluateModelPredictions(model, modelDescription, data, dataName, outcomeName):
  # Make predictions on test data to measure the performance of the model
  predictions = model.transform(data)
  
  print("\nModel Evaluation - ", dataName)
  print(modelDescription)
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

modelDesc = "First model training with numerical features only, training on mini-training"
EvaluateModelPredictions(dt_model, modelDesc, mini_train_dep, "mini-training", outcomeName)
EvaluateModelPredictions(dt_model, modelDesc, train_dep, "training", outcomeName)
EvaluateModelPredictions(dt_model, modelDesc, val_dep, "validation", outcomeName)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Before -- trying things out with Dataframes (no good...)

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Prep training data to be in format
# TODO: FIGURE OUT HOW TO LOAD DATA AS RDD OF LABEL POINTS!

#indexers = [StringIndexer(inputCol=column, outputCol=column+"_idx") for column in ['Op_Unique_Carrier', 'Origin', 'Dest']]

assembler = VectorAssembler(inputCols = ['Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 
                              #    'Op_Unique_Carrier', 'Origin', 'Dest', 
                                  'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group'], 
                     outputCol = "features")
transformed = assembler.transform(mini_train_dep)

res = (transformed.select(col("Dep_Del15").alias("label"), col("features"))
  .rdd
  .map(lambda row: LabeledPoint(row.label, row.features)))


# COMMAND ----------

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils


# Train a DecisionTree model.
# Variables not included in categoricalFeaturesInfo indicates these features are continuous.
model = DecisionTree.trainClassifier(
  res, 
  numClasses=2, 
  categoricalFeaturesInfo={}, # 4 -> 19, # Op_Unique_Carrier
                              # 5 -> 195, # Origin
                              # 6 -> 204},# Dest
  impurity='gini', 
  maxDepth=5, 
  maxBins=32)


# COMMAND ----------

