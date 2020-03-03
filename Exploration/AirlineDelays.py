# Databricks notebook source
# MAGIC %md # Airline delays 
# MAGIC ## Bureau of Transportation Statistics
# MAGIC https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp   
# MAGIC https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations
# MAGIC 
# MAGIC ~140GB

# COMMAND ----------

dbutils.fs.ls('/databricks-datasets/airlines')

# COMMAND ----------

sum = 0
DATA_PATH = 'dbfs:/databricks-datasets/airlines/'
for item in dbutils.fs.ls(DATA_PATH):
  sum = sum+item.size
sum


# COMMAND ----------

with open("/dbfs/databricks-datasets/airlines/README.md") as f:
    x = ''.join(f.readlines())

print(x)

# COMMAND ----------

airlines = spark.read.option("header", "true").csv("dbfs:/databricks-datasets/airlines/part-00000")

# COMMAND ----------

display(airlines.take(10))

# COMMAND ----------

