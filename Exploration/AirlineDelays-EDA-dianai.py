# Databricks notebook source
# Import data
ROOT = '/dbfs/'
DATA_PATH = '/databricks-datasets/airlines'
dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# Print the top 5 lines of first & second files
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

airlines.count()

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
AIRLINES_path = userhome + "/AIRLINES/" 
AIRLINES_path_open = '/dbfs' + AIRLINES_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(AIRLINES_path)

# COMMAND ----------

# Save initial parquet file
parquet_file_name = AIRLINES_path + "airline_delays_team20.parquet"
airlines.write.mode('overwrite').parquet(parquet_file_name)

# COMMAND ----------

# Load all subsequent files without headers and join with dataframe that has headers
i = 0
for file in dbutils.fs.ls(DATA_PATH):
  if ('part' in file.name and file.name != 'part-00000'):
    partial_airlines = spark.read.option("header", "false").csv(file.path)
    partial_airlines.write.mode('append').parquet(parquet_file_name)
    
    i = i + 1    
    if (i % 10 == 0):
      print(str(i) + " Files Processed (Last processed '" + file.path + "')")

# COMMAND ----------

# read from parquet file
airlines_parquet = spark.read.parquet(parquet_file_name)

airlines_parquet.count()

# COMMAND ----------



# COMMAND ----------

