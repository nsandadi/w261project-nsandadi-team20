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

# Extract only columns of interest
def ExtractColumnsForDepDelayModel(df):
  return df.select('Dep_Del15', 'Year', 'Month', 'Day_Of_Month', 'Day_Of_Week', 'Op_Unique_Carrier', 'Origin', 'Dest', 'CRS_Dep_Time', 'CRS_Arr_Time', 'CRS_Elapsed_Time', 'Distance', 'Distance_Group')

mini_train_dep = ExtractColumnsForDepDelayModel(mini_train)
val_dep = ExtractColumnsForDepDelayModel(val)

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

# MAGIC %md
# MAGIC #### Plan as reading documentation
# MAGIC - Source: https://spark.apache.org/docs/1.5.2/mllib-decision-tree.html
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
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(
  mini_train, 
  numClasses=2, 
  categoricalFeaturesInfo={ 4 -> 19, # Op_Unique_Carrier
                           5 -> 195, # Origin
                           6 -> 204},# Dest
  impurity='gini', 
  maxDepth=5, 
  maxBins=205)
