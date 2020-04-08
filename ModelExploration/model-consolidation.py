# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

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

# Write train & val data to parquet for easier EDA
def WriteAndRefDataToParquet(data, dataName):
  # Write data to parquet format (for easier EDA)
  data.write.mode('overwrite').format("parquet").save("dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")
  
  # Read data back directly from disk 
  return spark.read.option("header", "true").parquet(f"dbfs/user/team20/finalnotebook/airlines_" + dataName + ".parquet")

train_and_val = WriteAndRefDataToParquet(train.union(val), 'train_and_val')

# COMMAND ----------

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

def train_model(df,categoricalCols,continuousCols,labelCol,model,svmflag):

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col

    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in categoricalCols ]

    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in indexers ]
    # If it si svm do hot encoding
    if svmflag == True:
      assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")
    # else skip it
    else:
      assembler = VectorAssembler(inputCols = continuousCols + [cat + "_indexed" for cat in categoricalCols], outputCol = "features")
      pipeline = Pipeline(stages=indexers + [assembler])
      
    # choose the appropriate model regression  
    if model == 'lr':
      lr = LogisticRegression(labelCol = outcomeName, featuresCol="features", maxIter=100, regParam=0.1, elasticNetParam=0)
      pipeline = Pipeline(stages=indexers + encoders + [assembler,lr])

    elif model == 'dt':
      dt = DecisionTreeClassifier(labelCol = outcomeName, featuresCol = "features", seed = 6, maxDepth = 8, maxBins=366)
      pipeline = Pipeline(stages=indexers + encoders + [assembler,dt])
      
    elif model == 'nb':
      nb = NaiveBayes(labelCol = outcomeName, featuresCol = "features", smoothing = 1)
      pipeline = Pipeline(stages=indexers + encoders + [assembler,nb])
      
    elif model == 'svm':
      svc = LinearSVC(labelCol = outcomeName, featuresCol = "features", maxIter=50, regParam=0.1)
      pipeline = Pipeline(stages=indexers + encoders + [assembler,svc])
      
    else:
      pass
    
    model=pipeline.fit(df)
#     data = model.transform(df)

#     data = data.withColumn('label',col(labelCol))

#     return data.select('features','label')
    return model

# COMMAND ----------

model = train_model(mini_train_algo,catFeatureNames,numFeatureNames,outcomeName,svmflag=False)
data = model.transform(mini_train_algo)
display(data.take(10))

# COMMAND ----------

def PredictAndEvaluate(model, data, dataName, outcomeName):
  predictions = model.transform(data)
  EvaluateModelPredictions(predictions, dataName, outcomeName)