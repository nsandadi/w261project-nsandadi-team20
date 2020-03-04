# Databricks notebook source
DATAPATH = '/databricks-datasets/airlines'
all_files = dbutils.fs.ls(DATAPATH)
chunks = [c.path for c in all_files if "part" in c.path]
print("Number of parquet = {}".format(len(chunks)))
airlines0 = spark.read.option("header", "true").csv(chunks[0])
airlines_rest = spark.read.option("header", "false").csv(chunks[1:])
airlines = airlines0.union(airlines_rest)

# COMMAND ----------

print(airlines.columns)
airlines.printSchema()

# COMMAND ----------

# MAGIC %timeit
# MAGIC airlines.select('Year').distinct().collect()

# COMMAND ----------



# COMMAND ----------

# Load data
data_csv = spark.read.option("header", "true").csv(DATA_PATH+"part-00***")

# COMMAND ----------

data_csv.write.format("parquet").save(AIRLINES_path+"airline_delays_team20_quick.parquet")

# COMMAND ----------

airlines_quick_parquet = "airline_delays_team20_quick.parquet"

sum = 0
for item in dbutils.fs.ls(AIRLINES_path+airlines_quick_parquet+"/"):
  sum = sum+item.size
sum

# COMMAND ----------

data_parquet = spark.read.parquet(AIRLINES_path+airlines_quick_parquet)
data_parquet.count()

# COMMAND ----------

display(data_parquet.take(10))

# COMMAND ----------

