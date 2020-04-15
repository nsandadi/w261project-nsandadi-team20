# Databricks notebook source
# MAGIC %md
# MAGIC ### Load the data

# COMMAND ----------

# Load the data into dataframe
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# COMMAND ----------

print("Number of rows in original dataset:", airlines.count())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Clean Data

# COMMAND ----------

# Remove entries where diverted = 1, cancelled = 1, dep_delay = Null, and arr_delay = Null
airlines = airlines.where('DIVERTED != 1') \
                     .where('CANCELLED != 1') \
                     .filter(airlines['DEP_DELAY'].isNotNull()) 

print("Number of rows in cleaned dataset:", airlines.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Dependencies

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import broadcast
from pyspark.ml.linalg import DenseVector
from pyspark.ml import Pipeline
from pyspark.sql import Row


import math
import random

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create Outcome Variable 
# MAGIC ##### DEP_DEL30

# COMMAND ----------

# Generate outcome variable
def CreateNewDepDelayOutcome(data, thresholds):
  for threshold in thresholds:
    data = data.withColumn('DEP_DEL' + str(threshold), (data['DEP_DELAY'] >= threshold).cast('integer'))
  return data  
  
airlines = CreateNewDepDelayOutcome(airlines, [30])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection

# COMMAND ----------

outcomeName = 'DEP_DEL30'
numFeatureNames = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP']
catFeatureNames = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
joiningFeatures = ['FL_DATE'] # Features needed to join with the holidays dataset--not needed for training

airlines = airlines.select([outcomeName] + numFeatureNames + catFeatureNames + joiningFeatures)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Split the data into train and test set

# COMMAND ----------

# Helper function to split the dataset into train, val, test
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
# MAGIC ### EDA for data imbalance

# COMMAND ----------

# Count of flights delayed vs. not delayed in training dataset
display(train.groupby('DEP_DEL30').count())

# COMMAND ----------

# # Majority Class Undersampling using imblearn OneSidedSelection
# import sklearn
# from collections import Counter
# from sklearn.datasets import make_classification
# from imblearn import under_sampling, over_sampling
# from imblearn.under_sampling import OneSidedSelection
# from matplotlib import pyplot
# from numpy import where
# # define dataset
# X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
# 	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# # summarize class distribution
# counter = Counter(y)
# print(counter)
# # define the undersampling method
# undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
# # transform the dataset
# X, y = undersample.fit_resample(X, y)
# # summarize the new class distribution
# counter = Counter(y)
# print(counter)
# # scatter plot of examples by class label
# for label, _ in counter.items():
# 	row_ix = where(y == label)[0]
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
# pyplot.legend()
# pyplot.show()

# COMMAND ----------

# # Random undersampling of the majority class
# def resample(base_features,ratio,class_field,base_class):
#     min_class = base_features.filter(col(DEP_DEL30)==1)
#     maj_class = base_features.filter(col(DEP_DEL30)==0)
#     total_min = min_class.count()
#     total_maj = maj_class.count()
#     fraction=float(total_maj*ratio)/float(total_maj)
#     sampled = maj_class.sample(False,fraction)
#     return sampled.union(min_class)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vectorize the training data

# COMMAND ----------

# String Indexer for categorical variables
indexers = [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in catFeatureNames]
pipeline = Pipeline(stages=indexers)
indexed = pipeline.fit(train).transform(train)

# COMMAND ----------

display(indexed.take(5))

# COMMAND ----------

# Prep Vector assembler
va = VectorAssembler(inputCols = numFeatureNames + [f + "_idx" for f in catFeatureNames], outputCol = "features")

# Build a pipeline
pipeline = Pipeline(stages= indexers + [va])
pipelineModel = pipeline.fit(train)

# Vectorize
pos_vectorized = pipelineModel.transform(train)
vectorized = pos_vectorized.select('features', outcomeName).withColumn('label',pos_vectorized[outcomeName]).drop(outcomeName)

# COMMAND ----------

display(pos_vectorized.take(1))

# COMMAND ----------

vectorized.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter minority and majority data

# COMMAND ----------

# Filter the minority and majority data
majority_data = vectorized[vectorized.label == 0]
minority_data = vectorized[vectorized.label == 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Undersample the majority data

# COMMAND ----------

# Randomly undersample the majority data
maj_sampled = majority_data.sample(False,0.5)

# COMMAND ----------

# Join the undersampled majority data with minority data
vectorized_maj_undersampled = maj_sampled.union(minority_data)

# COMMAND ----------

# Distribution of data after undersampling
display(vectorized_maj_undersampled.groupby('label').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert minority data into feature vectors

# COMMAND ----------

# Convert minority data into feature vectors
featureVect = minority_data.select('features')
featureVect.take(5)

# COMMAND ----------

featureVect.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### K-means 
# MAGIC ##### Splits minority data into clusters

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Trains a k-means model on the feature vectors of minority data
kmeans = KMeans().setK(1000).setSeed(1)
model = kmeans.fit(featureVect)
predict = model.transform(featureVect)

# COMMAND ----------

# Visualize the distribution of data in each cluster
display(predict.groupBy('prediction').count().orderBy('prediction'))

# COMMAND ----------

# Order the columns
predict = predict.select(['prediction','features'])
predict.show(5)

# COMMAND ----------

predict.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SMOTE

# COMMAND ----------

# HELPER FUNCTIONS

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

smote_rdd = SmoteSampling(predict, 3)

# COMMAND ----------

# Convert the synthetic data into a dataframe
augmentedData_DF = smote_rdd.toDF()     


# COMMAND ----------

# Combine the original dataset with the synthetic data
smote_data = vectorized_maj_undersampled.unionAll(augmentedData_DF)

# COMMAND ----------

# EDA of data balance after applying SMOTE
display(smote_data.groupBy('label').count())

# COMMAND ----------

smote_data.take(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save balanced dataset as columns & reverse string indexing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reverse Vector Assembler

# COMMAND ----------

# Reverse Vector Assembler
from pyspark.ml.linalg import Vectors

def vectorToDF(df):

  def extract(row):
      return (row.label, ) + tuple(row.features.toArray().tolist())
    
  extracted_df = df.rdd.map(extract).toDF(['label'])

  # Rename Columns
  extracted_df = extracted_df.withColumnRenamed("label","DEP_DEL30") \
                             .withColumnRenamed("_2","YEAR") \
                             .withColumnRenamed("_3","MONTH") \
                             .withColumnRenamed("_4","DAY_OF_MONTH") \
                             .withColumnRenamed("_5","DAY_OF_WEEK") \
                             .withColumnRenamed("_6","CRS_DEP_TIME") \
                             .withColumnRenamed("_7","CRS_ARR_TIME") \
                             .withColumnRenamed("_8","CRS_ELAPSED_TIME") \
                             .withColumnRenamed("_9","DISTANCE") \
                             .withColumnRenamed("_10","DISTANCE_GROUP") \
                             .withColumnRenamed("_11","OP_UNIQUE_CARRIER_idx") \
                             .withColumnRenamed("_12","ORIGIN_idx") \
                             .withColumnRenamed("_13","DEST_idx") \

  return extracted_df

# COMMAND ----------

smoted_train_cols = vectorToDF(smote_data)
smoted_train_cols.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reverse StringIndexer

# COMMAND ----------

# Create lookup table for OP_UNIQUE_CARRIER
carrier_index_lookup = indexed['OP_UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER_idx'].distinct()
display(carrier_index_lookup.orderBy('OP_UNIQUE_CARRIER_idx'))

# COMMAND ----------

# Create lookup table for ORIGIN
origin_index_lookup = indexed['ORIGIN', 'ORIGIN_idx'].distinct()
display(origin_index_lookup.orderBy('ORIGIN_idx'))

# COMMAND ----------

# Create lookup table for DEST
dest_index_lookup = indexed['DEST', 'DEST_idx'].distinct().orderBy('DEST_idx')
display(dest_index_lookup.orderBy('DEST_idx'))

# COMMAND ----------

# Map OP_UNIQUE_CARRIER to OP_UNIQUE_CARRIER_idx
smoted_train_cols_carrier = smoted_train_cols.join(broadcast(carrier_index_lookup), 
                                                       (smoted_train_cols.OP_UNIQUE_CARRIER_idx == carrier_index_lookup.OP_UNIQUE_CARRIER_idx))

smoted_train_cols_carrier = smoted_train_cols_carrier.drop('OP_UNIQUE_CARRIER_idx')

# COMMAND ----------

# Map ORIGIN to ORIGIN_idx
smoted_train_cols_origin = smoted_train_cols_carrier.join(broadcast(origin_index_lookup), 
                                                       (smoted_train_cols_carrier.ORIGIN_idx == origin_index_lookup.ORIGIN_idx))

smoted_train_cols_origin = smoted_train_cols_origin.drop('ORIGIN_idx')

# COMMAND ----------

# Map DEST to DEST_idx
smoted_train_cols_dest = smoted_train_cols_origin.join(broadcast(dest_index_lookup), 
                                                       (smoted_train_cols_origin.DEST_idx == dest_index_lookup.DEST_idx))

smoted_train_cols_dest = smoted_train_cols_dest.drop('DEST_idx')

# COMMAND ----------

# Perform an action as the transformations are lazily evaluated 
smoted_train_cols_dest.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the dataset to parquet

# COMMAND ----------

# Write train & val data to parquet for easier EDA
def WriteAndRefDataToParquet(data, dataName):
  # Write data to parquet format (for easier EDA)
  data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

# COMMAND ----------

smoted_train_kmeans_maj_undsamp = WriteAndRefDataToParquet(smoted_train_cols_dest, 'smoted_train_kmeans_maj_undsamp')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the smoted dataset to dataframe

# COMMAND ----------

# Load the data into dataframe
smoted_train_kmeans_maj_undsamp = spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_smoted_train_kmeans_maj_undsamp.parquet")

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp.take(10))

# COMMAND ----------

smoted_train_kmeans_maj_undsamp.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA of balanced vs. unbalanced train dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delayed vs. Not Delayed

# COMMAND ----------

display(train.groupby('DEP_DEL30').count())

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp.groupby('DEP_DEL30').count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filtering out delayed data within train and smoted_train to observe the distribution of data

# COMMAND ----------

train_delay = train.filter(train.DEP_DEL30 == 1)
smoted_train_kmeans_maj_undsamp_delay = smoted_train_kmeans_maj_undsamp.filter(smoted_train_kmeans_maj_undsamp.DEP_DEL30 == 1)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. OP_UNIQUE_CARRIER

# COMMAND ----------

display(train_delay.groupby('OP_UNIQUE_CARRIER').count().orderBy('OP_UNIQUE_CARRIER'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('OP_UNIQUE_CARRIER').count().orderBy('OP_UNIQUE_CARRIER'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. ORIGIN

# COMMAND ----------

display(train_delay.groupby('ORIGIN').count().orderBy('ORIGIN'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('ORIGIN').count().orderBy('ORIGIN'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. DEST

# COMMAND ----------

display(train_delay.groupby('DEST').count().orderBy('DEST'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('DEST').count().orderBy('DEST'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. DISTANCE_GROUP

# COMMAND ----------

display(train_delay.groupby('DISTANCE_GROUP').count().orderBy('DISTANCE_GROUP'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('DISTANCE_GROUP').count().orderBy('DISTANCE_GROUP'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5. DISTANCE

# COMMAND ----------

display(train_delay.groupby('DISTANCE').count().orderBy('DISTANCE'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('DISTANCE').count().orderBy('DISTANCE'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6. YEAR

# COMMAND ----------

display(train_delay.groupby('YEAR').count().orderBy('YEAR'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('YEAR').count().orderBy('YEAR'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7. MONTH

# COMMAND ----------

display(train_delay.groupby('MONTH').count().orderBy('MONTH'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('MONTH').count().orderBy('MONTH'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 8. DAY_OF_MONTH

# COMMAND ----------

display(train_delay.groupby('DAY_OF_MONTH').count().orderBy('DAY_OF_MONTH'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('DAY_OF_MONTH').count().orderBy('DAY_OF_MONTH'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 9. DAY_OF_WEEK

# COMMAND ----------

display(train_delay.groupby('DAY_OF_WEEK').count().orderBy('DAY_OF_WEEK'))

# COMMAND ----------

display(smoted_train_kmeans_maj_undsamp_delay.groupby('DAY_OF_WEEK').count().orderBy('DAY_OF_WEEK'))