# Databricks notebook source
# MAGIC %md
# MAGIC # Airline Delay Prediction
# MAGIC ## W261 Final Project

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

# Load the data into dataframe
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# COMMAND ----------

print(airlines.count())

# COMMAND ----------

airlines.printSchema()

# COMMAND ----------

display(airlines.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean the data

# COMMAND ----------

# Remove entries where diverted = 1, cancelled = 1, dep_delay = Null, and arr_delay = Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print(airlines.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Transformations

# COMMAND ----------

# Create the outcome variable DEP_DEL30


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Split the data into train and test set

# COMMAND ----------

# Helper function to split the dataset into train and test (by Diana)
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
# MAGIC ### EDA

# COMMAND ----------

# Count of flights delayed vs. not delayed in entire dataset
display(airlines.groupby('DEP_DEL15').count())

# COMMAND ----------

# Count of flights delayed vs. not delayed in training dataset
display(train.groupby('DEP_DEL15').count())

# COMMAND ----------

# Count of flights delayed vs. not delayed in mini training dataset
display(mini_train.groupby('DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting interesting features from dataset

# COMMAND ----------

# Creating a new mini train dataframe with features of interest 
mini_train_depdelay = mini_train.select('DEP_DEL15', 'YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST')

display(mini_train_depdelay.take(5))

# COMMAND ----------

# Count of flights delayed vs. not delayed in mini training dataset
display(mini_train_depdelay.groupby('DEP_DEL15').count())

# COMMAND ----------

# Creating a new training dataframe with features of interest
train_depdelay = train.select('DEP_DEL15', 'YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST')
# 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST'

display(train_depdelay.take(5))

# COMMAND ----------

# Count of flights delayed vs. not delayed in mini training dataset
display(train_depdelay.groupby('DEP_DEL15').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### SMOTE for handling unbalanced data

# COMMAND ----------

# Homegrown SMOTE approach (without sklearn)
import random
import math
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import spark.sparkcontext

def vectorizerFunction(dataInput, outcomeName, strVars, numVars):
    # Prep Vector assembler
    si = [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in strVars]
    va = VectorAssembler(inputCols = numVars + [f + "_idx" for f in strVars], outputCol = "features")
    
    # Build a pipeline
    pipeline = Pipeline(stages= si + [va])
    pipelineModel = pipeline.fit(dataInput)
    
    # Vectorize
    pos_vectorized = pipelineModel.transform(dataInput)
    vectorized = pos_vectorized.select('features', outcomeName).withColumn('label',pos_vectorized[outcomeName]).drop(outcomeName)

    return vectorized
  

# Calculate the Euclidean distance between two feature vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Generate synthetic records
def synthetic(list1, list2):
    synthetic_records = []
    for i in range(len(list1)):
      synthetic_records.append(round(list1[i] + ((list2[i]-list1[i])*random.uniform(0, 1))))
    return synthetic_records


def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0):
        
    # Partition Vectorized Data to minority & majority classes
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    
    # Extracting feature vectors of minority group
    featureVect = dataInput_min.select('features')
    
    # Convert features dataframe into a list
    feature_list_rdd = featureVect.rdd.map(lambda x: list(x[0]))
    
    # Broadcast features
    sc = spark.sparkContext
    feat_broadcast = sc.broadcast(feature_list_rdd) 
    
    # Generate the Augmented Dataset with k synthetic data points for a 
    # given feature entry in the original dataset
    # Emit the original dataset entries and the synthetic dataset entries
    def generateAugmentedDataset(feature, allPossibleNeighbors, k):
      # Get all nearest neighbors with euclidean distances
      nearestNeighbors = allPossibleNeighbors.map(lambda n: (euclidean_distance(feature, n), n)) \
                                             .takeOrdered(k+1, key=lambda x: x[0])
      
      # For each neighbor, compute the difference, & generate the synthetic data point
      syntheticPoints = nearestNeighbors.map(lambda n: synthetic(feature, n)).collect()
      
      # Emit synthetic data points & original data points
      for i in range(1, len(syntheticPoints)):
        yield(syntheticPoints[i])
      yield(feature)
    
    # For each feature example, get the k nearest neighbors 
    # of the feature & generate a synthetic datapoint
    augmentedData = feature_list_rdd.flatMap(lambda x: generateAugmentedDataset(x, feat_broadcast.value, k))
        
    # Convert the synthetic data into a dataframe
    augmentedData = augmentedData.map(lambda x: Row(features = DenseVector(x), label = 1))
    augmentedData_DF = augmentedData.toDF()
    
    return dataInput_maj.unionAll(augmentedData_DF)


# Target Variable
outcomeName = 'Dep_Del15'
  
# Numerical Features
nfeatureNames = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP']

# Categorical features
cfeatureNames = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
  
vect = vectorizerFunction(mini_train_depdelay, outcomeName, cfeatureNames, nfeatureNames)

# COMMAND ----------

vect.take(10)

# COMMAND ----------

# Applying SMOTE on mini_train dataset with selected features
balanced_mini_train_depdelay_df = SmoteSampling(
  vectorizerFunction(mini_train_depdelay, outcomeName, cfeatureNames, nfeatureNames), 
  k = 3, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 90)


# COMMAND ----------

# Count of flights delayed vs. not delayed in mini training dataset w/ categorical features after applying SMOTE (label = DEP_DEL15)
display(balanced_mini_train_depdelay_df.groupby('label').count())

# COMMAND ----------

# Applying SMOTE on train dataset with selected features
balanced_train_depdelay_df = SmoteSampling(
  vectorizerFunction(train_depdelay, outcomeName, cfeatureNames, nfeatureNames), 
  k = 3, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 90)


# COMMAND ----------

# Code to undersample majority class
# new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree Baseline Model on mini train set

# COMMAND ----------

# Decision Tree on Mini Train
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Target Variable
outcomeName = 'DEP_DEL15'
  
# Numerical Features
numFeatures = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP']

# Categorical features
catFeatures = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']

# StringIndexer and VectorAssembler
labelIndexer = [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in catFeatures]
featureIndexer = VectorAssembler(inputCols = numFeatures + [f + "_idx" for f in catFeatures], outputCol = "features")

# Train a DecisionTree model
dt = DecisionTreeClassifier(labelCol= "DEP_DEL15", featuresCol = "features", maxDepth = 6, maxBins=366)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=labelIndexer + [featureIndexer, dt])

# Train the model
dt_model = pipeline.fit(mini_train)
# mini_train_depdelay_str

# Make predictions
predictions = dt_model.transform(val)



# COMMAND ----------

# Display a few predictions
display(predictions.select("DEP_DEL15", "prediction", "features").show(5))

# COMMAND ----------

# Display tree
treeModel = dt_model.stages[-1]

# summary only
print(treeModel)
display(treeModel)

# COMMAND ----------

# Performance Evaluation
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

# Test Error
print("Test Error: %g " % (1.0 - accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree Baseline Model on Training Set

# COMMAND ----------

# Decision Tree on Training Data
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Target Variable
outcomeName = 'DEP_DEL15'
  
# Numerical Features
numFeatures = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP']

# Categorical features
catFeatures = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']

# StringIndexer and VectorAssembler
labelIndexer = [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in catFeatures]
featureIndexer = VectorAssembler(inputCols = numFeatures + [f + "_idx" for f in catFeatures], outputCol = "features")

# Train a DecisionTree model
dt = DecisionTreeClassifier(labelCol= "DEP_DEL15", featuresCol = "features", maxDepth = 8, maxBins=366)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=labelIndexer + [featureIndexer, dt])

# Train the model
dt_model = pipeline.fit(train)
# mini_train_depdelay_str

# Make predictions
predictions = dt_model.transform(val)



# COMMAND ----------

# Display tree
treeModel = dt_model.stages[-1]

# summary only
print(treeModel)
display(treeModel)

# COMMAND ----------

# Performance Evaluation
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

# Test Error
print("Test Error: %g " % (1.0 - accuracy))