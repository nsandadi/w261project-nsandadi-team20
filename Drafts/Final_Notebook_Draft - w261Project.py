# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Airline Delays - Final Notebook
# MAGIC ###### W261 Spring 2020 
# MAGIC ###### Presentation Date: April 16th, 2020
# MAGIC ###### Team 20: Diana Iftimie, Shaji K Kunjumohamed, Navya Sandadi, & Shobha Sankar

# COMMAND ----------

# MAGIC %md 
# MAGIC ## I. Question Formulation & Introduction
# MAGIC 
# MAGIC As we've all probably experienced at some point in our lives, air travel is never easy. Whether you're the person getting on a flight traveling around the world, the folks in the air traffic control towers orchestrating incoming and outgoing flights, or the airports and airlines trying their best to effectively coordinate flights at every hour of every day of every year, so much can go wrong. The delays alone are enough to completely derail anyone's plans and trigger a cascading effect of consequences down the line as delays continue to stack up on top of each other over the course of time. And the biggest problem is that these delays often occur when we least expect them and at the worst possible times.
# MAGIC 
# MAGIC To attempt to solve this problem, we introduce the *Airline Delays* dataset, a dataset of US domestic flights from 2015 to 2019 collected by the Bureau of Transportation Statistics for the purpose of studying airline delays. For this analysis, we will primarily use this dataset to study the nature of airline delays in the united states over the last few years, with the ultimate goal of developing models for predicting signitifact flight departure delays (30 minutes or more) in the United States. 
# MAGIC 
# MAGIC In developing such models, we seek to answer the core question, **"Given known information prior to a flight's departure, can we predict departure delays and identify the likely causes of such delays?"**. In the last few years, about 11% of all US domestic flights resulted in significant delays, and answering these questions can truly help us to understand why such delays happen. In doing so, not only can airlines and airports start to identify likely causes and find ways to mitigate them and save both time and money, but air travelers also have the potential to better prepare for likely delays and possibly even plan for different flights in order to reduce their chance of significant delay. 
# MAGIC 
# MAGIC To effectively investigate this question and produce a practically useful model, we will aim to develop a model that performs better than a baseline model that predicts the majority class of 'no delay' 89% of the time (the equivalent of random guessing, which would have an accuracy of 89%). Given the classificatio nature of this problem, we will concentrate on improving metrics such precision, recall and F1 over our baseline model. We will also concentrate on producing models that can explain what features of flights known prior to departure time can best predict departure delays and from these, attempt to best infer possible causes of departure delays. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. EDA & Discussion of Challenges
# MAGIC - 2-3 EDA Tasks to help make decisions on how we implement the algorithm to be scalable
# MAGIC     - Binning could be a way to scale decision tree algorithms (when defining splits, e.g. CRS_Dep_Time)
# MAGIC     - More customized binning for the origin-destination to fit some custom criterion (to make new features)
# MAGIC     - Brieman's Theorem can help with binning on categorical variables -- we could do an EDA task to rank some categorical variables (reduce number of possible splits -- bring order to unordered cateogrical variables)
# MAGIC - Challenges anticipated based on EDA
# MAGIC     - Cleaning/joining datasets?
# MAGIC     - Feature Selection (or feature augmentation)
# MAGIC     - gradient descent for logistic regression?
# MAGIC     - sharing clusters, not being able to load entire dataset into memory (but can load fractions and/or aggregations of it!)
# MAGIC     - store data to optimize for column/row retrieval based on algorithm (parquet for DT & avro for LR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. EDA & Discussion of Challenges
# MAGIC 
# MAGIC ### Dataset Introduction
# MAGIC The Bureau of Transporation Statistics provides us with a wide variety of features relating to each flight, ranging from features about the scheduled flight such as the planned departure, arrival, and elapsed times, the planned distance, the carrier and airport information, as well as information regarding the causes of certain delays for the entire flight, as well as the amounts of delay (for both flight departure and arrival), among many other features. 
# MAGIC 
# MAGIC Given that for this analysis, we will be concentrating on predicting and identify the likely causes of departure delays before any such delay happens, we will primarily concentrate our EDA and model development using features of flights that would be known at inference time. We will choose the inference time to be 6 hours prior to the scheduled departure time of a flight. Realistically speaking, providing someone with a notice that a flight will likely be delayed 6 hours in advance is likely a sufficient amount of time to let people prepare for such a delay to reduce the cost of the departure delay, if it occurs. Such features that fit this criterion include those that are related to:
# MAGIC 
# MAGIC * **Time of year** (e.g. `Year`, `Month`, `Day_Of_Month`, `Day_Of_Week`)
# MAGIC * **Airline Carrier** (e.g. `Op_Unique_Carrier`)
# MAGIC * **Origin & Destination Airports** (e.g. `Origin`, `Dest`)
# MAGIC * **Scheduled Departure & Arrival Times** (e.g. `CRS_Dep_Time`, `CRS_Arr_Time`)
# MAGIC * **Planned Elapsed Times & Distances** (e.g. `CRS_Elapsed_Time`, `Distance`, `Distance_Group`)
# MAGIC 
# MAGIC Additionally, we will use the variable `Dep_Delay` to define our outcome variable for "significiant" departure delays (i.e. delays of 30 minutes or more). Finally , we will focus our analysis to the subset of flights that are not diverted, are not cancelled and have departure delays defined to ensure that we can accurately predict departure delays for flights. Below are a few example flights taken from the *Airline Delays* dataset that we will use for our analysis.

# COMMAND ----------

# Read in original dataset
airlines = spark.read.option("header", "true").parquet(f"dbfs:/mnt/mids-w261/data/datasets_final_project/parquet_airlines_data/201*.parquet")

# Filter to datset with entries where diverted != 1, cancelled != 1, and dep_delay != Null
airlines = airlines.where('DIVERTED != 1') \
                   .where('CANCELLED != 1') \
                   .filter(airlines['DEP_DEL15'].isNotNull()) \
                   .filter(airlines['ARR_DEL15'].isNotNull())

print("Number of records in full dataset:", airlines.count())

# Print examples of flights
display(airlines.take(6))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that because we are interested in predicting departure delays for future flights, we will define our test set to be the entirty of flights from the year 2019 and use the years 2015-2018 for training. This way, we will simulate the conditions for training a model that will predict departure delays for flights that will occur in the future. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## III. Feature Engineering
# MAGIC - Cleaning
# MAGIC     - Remove cancelled/diverted flights (don't have full data)
# MAGIC     - Remove flights with null outcomes (can't estimate)
# MAGIC - Transformations
# MAGIC     - Binning (elapsed time, departure/arrival time, )
# MAGIC     - Forming Indiciator Variables (e.g. departure/arrival delay > 30 min (true/false))
# MAGIC - Interaction Terms
# MAGIC     - Interacting variables to make new features ("Day of Year", "Time of week")
# MAGIC     - Binning interaction terms
# MAGIC     - Joining interaction terms to make new features
# MAGIC          - Day of month with month ("Day of Year") -> Holidays indiciator (or near holiday indicator)
# MAGIC          - Departure/Arrival time day of week ("Time of week")
# MAGIC - Treatment of Categorical Variables
# MAGIC     - Categorical variables: Time of Year Variables, Distance Group, Carrier, origin/destination airports
# MAGIC     - Order  Breiman's Theorem (rank by volume of flights, probability of departure delay, etc)
# MAGIC     - Make broader categories for categorical variables (not something we've tried yet, but we can do this)
# MAGIC     - one-hot encoding (svms)
# MAGIC - Treatment for categorical / binned variables
# MAGIC     - computing # of flights per category
# MAGIC     - computing probability of delay per category
# MAGIC     - Try a model that includes these to see if the model selects them (Decision tree)

# COMMAND ----------

# MAGIC %md
# MAGIC ## IV. Algorithm Exploration
# MAGIC - Apply 2-3 Algorithms
# MAGIC     - Logistic Regression
# MAGIC     - SVM
# MAGIC     - Naive Bayes
# MAGIC     - Decision Tree
# MAGIC - Expectations / Tradeoffs
# MAGIC     - All able to do classificaiton (Delay/No Delay)
# MAGIC     - LR: Interpretable (good), get estimate for effect of each variable, manual feature selection required, need to deal with multi-collinearity, among other things
# MAGIC     - SVM: needed to transform & one-hot encode categorical variables
# MAGIC     - DT: Not a lot of hyper parameter tuning/feature selection (most automated), 
# MAGIC - Results
# MAGIC 
# MAGIC -- See Task 3 Submission :) --
# MAGIC https://docs.google.com/document/d/1IaGOgYWSRCH-WgDzJ7N2Lw6y5h8zxpTQ8HHg9AqpML8/edit?usp=sharing

# COMMAND ----------

# MAGIC %md
# MAGIC ## V. Algorithm Implementation
# MAGIC - Toy example likely with a decision tree on a mini_mini_train dataset (like 10 rows)
# MAGIC - Walk through training the model and doing inference

# COMMAND ----------

# MAGIC %md
# MAGIC ## VI. Conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC ## V. Applications of Course Concepts

# COMMAND ----------

