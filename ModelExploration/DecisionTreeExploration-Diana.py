# Databricks notebook source
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