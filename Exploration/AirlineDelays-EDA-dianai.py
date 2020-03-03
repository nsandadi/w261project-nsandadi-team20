# Databricks notebook source
# Import data
ROOT = '/dbfs/'
DATA_PATH = '/databricks-datasets/airlines'
dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# Print the top 5 lines of first & second files
dbutils.fs.head(DATA_PATH + '/part-00000', 500)

def printNLines(filepath, n):
  with open(filepath, "r") as f_read:
    num_lines = n
    for line in f_read:
      if (num_lines == 0):
        break
      print(line)
      num_lines -= 1

printNLines('/dbfs/databricks-datasets/airlines/part-00000', 5)
print("\n*************************************************\n")
printNLines('/dbfs/databricks-datasets/airlines/part-00001', 5)

# Note that only the first file has headers

# COMMAND ----------

# Load the first file with headers
airlines = spark.read.option("header", "true").csv("dbfs:/databricks-datasets/airlines/part-00000")
display(airlines.take(5))

# COMMAND ----------

# Load all subsequent files without headers and join with dataframe that has headers
partial_airlines = None
for file in dbutils.fs.ls(DATA_PATH):
  if ('part' in file.name and file.name != 'part-00000'):
    partial_airlines = spark.read.option("header", "false").csv(file.path)
    break
display(airlines.union(partial_airlines).take(5))

# COMMAND ----------



# COMMAND ----------

