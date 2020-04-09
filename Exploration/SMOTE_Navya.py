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

print("Number of rows in original dataset:", airlines.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean the data

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
# MAGIC ### Feature Selection and Transformations

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

# Write train & val data to parquet for easier EDA
def WriteAndRefDataToParquet(data, dataName):
  # Write data to parquet format (for easier EDA)
  data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA for data imbalance

# COMMAND ----------

# Count of flights delayed vs. not delayed in training dataset
display(train.groupby('DEP_DEL30').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vectorize the train dataset

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
# MAGIC ### SMOTE implementation

# COMMAND ----------

# Helper Functions

# Calculate the Euclidean distance between two feature vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)
  
  
# Locate the nearest neighbors
def get_neighbors(train, test_row, num_neighbors):
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
    synthetic_records = []
    for i in range(len(list1)):
      synthetic_records.append(round(list1[i] + ((list2[i]-list1[i])*random.uniform(0, 1))))
    return synthetic_records

# COMMAND ----------

# SMOTE function
def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, sample_size = 1):
        
    # Partition Vectorized Data to minority & majority classes
    dataInput_min = vectorized[vectorized['label'] == minorityClass]
    dataInput_maj = vectorized[vectorized['label'] == majorityClass]
    
    # Extracting feature vectors of minority group
    featureVect = dataInput_min.select('features')
    
    # Convert features dataframe into RDD
    feature_list_rdd = featureVect.rdd.map(lambda x: list(x[0])).cache()
    
    # Store a sample of the feature list as a broadcast variable
    feature_list = feature_list_rdd.sample(False, sample_size, 6).collect()
    feature_list_broad = sc.broadcast(feature_list)
        
    # For each feature example, get the k nearest neighbors of the feature & generate a synthetic datapoint
    augmentedData = feature_list_rdd.map(lambda x: (x, get_neighbors(feature_list_broad.value, x, k))) \
                                    .flatMap(lambda x: [synthetic(x[0], n) for n in x[1]]) \
                                    .map(lambda x: Row(features = DenseVector(x), label = 1)) \
                                    .cache()
                                   
    # Convert the synthetic data into a dataframe
    augmentedData_DF = augmentedData.toDF()     
    

    return vectorized.unionAll(augmentedData_DF)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Apply SMOTE to train dataset

# COMMAND ----------

# Applying SMOTE on train dataset 
balanced_train = SmoteSampling(vectorized, k = 7, minorityClass = 1, majorityClass = 0, sample_size = 0.0005)


# COMMAND ----------

# Check if dataset is balanced
display(balanced_train.groupby('label').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ###### SmoteSampling for k=6 gives 47% delay - 53% no delay distribution in 29 + 42 mins

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save balanced dataset as columns & reverse indexing

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

balanced_train_cols = vectorToDF(balanced_train)
balanced_train_cols.show(5)

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
balanced_train_cols_carrier = balanced_train_cols.join(carrier_index_lookup, 
                                                       (balanced_train_cols.OP_UNIQUE_CARRIER_idx == carrier_index_lookup.OP_UNIQUE_CARRIER_idx))

balanced_train_cols_carrier = balanced_train_cols_carrier.drop('OP_UNIQUE_CARRIER_idx')

# COMMAND ----------

# balanced_train_cols_carrier = balanced_train_cols_carrier.drop('OP_UNIQUE_CARRIER_idx')

# COMMAND ----------

# Map ORIGIN to ORIGIN_idx
balanced_train_cols_origin = balanced_train_cols_carrier.join(origin_index_lookup, 
                                                       (balanced_train_cols_carrier.ORIGIN_idx == origin_index_lookup.ORIGIN_idx))

balanced_train_cols_origin = balanced_train_cols_origin.drop('ORIGIN_idx')


# COMMAND ----------

# Map DEST to DEST_idx
balanced_train_cols_dest = balanced_train_cols_origin.join(dest_index_lookup, 
                                                       (balanced_train_cols_origin.DEST_idx == dest_index_lookup.DEST_idx))

balanced_train_cols_dest = balanced_train_cols_dest.drop('DEST_idx')

# COMMAND ----------

balanced_train_cols_dest.count()

# COMMAND ----------

# IndexToString - Reverse of StringIndexer - Can use only on the same dataset
def reverseStringIndexer(df):
  
  indexers = [StringIndexer(inputCol=f, outputCol=f+"_idx", handleInvalid="keep") for f in catFeatureNames]
  pipeline = Pipeline(stages=indexers)
  indexed = pipeline.fit(df).transform(df)
  
  converters = [IndexToString(inputCol=f+"_idx", outputCol=f+"_rev_idx") for f in catFeatureNames]
  pipeline = Pipeline(stages=converters)
  converted_all = pipeline.fit(indexed_multiple).transform(indexed_multiple)
  
  return converted_all

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

smoted_train_data = WriteAndRefDataToParquet(balanced_train_cols_dest, 'smoted_train_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the dataset to dataframe

# COMMAND ----------

# Load the data into dataframe
smoted_train = spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_smoted_train_data.parquet")

# COMMAND ----------

display(smoted_train.take(10))

# COMMAND ----------

smoted_train.count()

# COMMAND ----------

display(smoted_train.groupby('DEP_DEL30').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA of Smoted vs. Unsmoted training set 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delayed vs. Not Delayed

# COMMAND ----------

display(train.groupby('DEP_DEL30').count())

# COMMAND ----------

display(smoted_train.groupby('DEP_DEL30').count())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filtering out delayed data within train and smoted_train

# COMMAND ----------

train_delay = train.filter(train.DEP_DEL30 == 1)
smoted_train_delay = smoted_train.filter(smoted_train.DEP_DEL30 == 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1. OP_UNIQUE_CARRIER

# COMMAND ----------

display(train_delay.groupby('OP_UNIQUE_CARRIER').count())

# COMMAND ----------

display(smoted_train_delay.groupby('OP_UNIQUE_CARRIER').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2. ORIGIN

# COMMAND ----------

display(train_delay.groupby('ORIGIN').count())

# COMMAND ----------

display(smoted_train_delay.groupby('ORIGIN').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3. DEST

# COMMAND ----------

display(train_delay.groupby('DEST').count())

# COMMAND ----------

display(smoted_train_delay.groupby('DEST').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4. DISTANCE_GROUP

# COMMAND ----------

display(train_delay.groupby('DISTANCE_GROUP').count())

# COMMAND ----------

display(smoted_train_delay.groupby('DISTANCE_GROUP').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5. DISTANCE

# COMMAND ----------

display(train_delay.groupby('DISTANCE').count())

# COMMAND ----------

display(smoted_train_delay.groupby('DISTANCE').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 6. CRS_DEP_TIME

# COMMAND ----------

display(train_delay.groupby('CRS_DEP_TIME').count())

# COMMAND ----------

display(smoted_train_delay.groupby('CRS_DEP_TIME').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7. CRS_ARR_TIME

# COMMAND ----------

display(train_delay.groupby('CRS_ARR_TIME').count())

# COMMAND ----------

display(smoted_train_delay.groupby('CRS_ARR_TIME').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 8. CRS_ELAPSED_TIME

# COMMAND ----------

display(train_delay.groupby('CRS_ELAPSED_TIME').count())

# COMMAND ----------

display(smoted_train_delay.groupby('CRS_ELAPSED_TIME').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 9. YEAR

# COMMAND ----------

display(train_delay.groupby('YEAR').count())

# COMMAND ----------

display(smoted_train_delay.groupby('YEAR').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 10. MONTH

# COMMAND ----------

display(train_delay.groupby('MONTH').count())

# COMMAND ----------

display(smoted_train_delay.groupby('MONTH').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 11. DAY_OF_MONTH

# COMMAND ----------

display(train_delay.groupby('DAY_OF_MONTH').count())

# COMMAND ----------

display(smoted_train_delay.groupby('DAY_OF_MONTH').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 12. DAY_OF_WEEK

# COMMAND ----------

display(train_delay.groupby('DAY_OF_WEEK').count())

# COMMAND ----------

display(smoted_train_delay.groupby('DAY_OF_WEEK').count())