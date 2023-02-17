# Databricks notebook source
# MAGIC %md
# MAGIC # Featurization
# MAGIC 
# MAGIC Cleaning data and adding features creates the inputs for machine learning models, which are only as strong as the data they are fed.  This notebook examines the process of featurization including common tasks such as:
# MAGIC 
# MAGIC - Exercise 1: Handling missing data
# MAGIC - Exercise 2: Feature Engineering
# MAGIC - Exercise 3: Scaling Numeric features
# MAGIC - Exercise 4: Encoding Categorical Features

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to load common libraries.

# COMMAND ----------

import urllib.request
import os
import numpy as np
from pyspark.sql.types import * 
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import udf
print("Imported common libraries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training data
# MAGIC 
# MAGIC In this notebook, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from [Azure Open Datasets]( https://azure.microsoft.com/en-us/services/open-datasets/). The data is enriched with holiday and weather data. Each row of the table represents a taxi ride that includes columns such as number of passengers, trip distance, datetime information, holiday and weather information, and the taxi fare for the trip.
# MAGIC 
# MAGIC Run the following cell to load the table into a Spark dataframe and reivew the dataframe.

# COMMAND ----------

dataset = spark.sql("select * from nyc_taxi")
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 1: Handling missing data
# MAGIC 
# MAGIC Null values refer to unknown or missing data as well as irrelevant responses. Strategies for dealing with this scenario include:
# MAGIC * **Dropping these records:** Works when you do not need to use the information for downstream workloads
# MAGIC * **Adding a placeholder (e.g. `-1`):** Allows you to see missing data later on without violating a schema
# MAGIC * **Basic imputing:** Allows you to have a "best guess" of what the data could have been, often by using the mean of non-missing data
# MAGIC * **Advanced imputing:** Determines the "best guess" of what data should be using more advanced strategies such as clustering machine learning algorithms or oversampling techniques <a href="https://jair.org/index.php/jair/article/view/10302" target="_blank">such as SMOTE.</a>

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to review summary statistics of each column in the data frame. Observe that based on the **count** metric the two columns `passenger count` and `totalAmount` have some null or missing values.

# COMMAND ----------

display(dataset.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC A common option for working with missing data is to impute the missing values with a best guess for their value. We will try imputing the `passenger count` column with its median value. Run the following cell to create the **Imputer** with **strategy="median"** and impute the `passenger count` column.

# COMMAND ----------

from pyspark.ml.feature import Imputer

inputCols = ["passengerCount"]
outputCols = ["passengerCount"]

imputer = Imputer(strategy="median", inputCols=inputCols, outputCols=outputCols)
imputerModel = imputer.fit(dataset)
imputedDF = imputerModel.transform(dataset)
display(imputedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In the next, lesson we will train a machine learning model to predict the taxi fares, thus the `totalAmount` column will be the target column for training the machine learning model. Given the importance of this column, the strategy to deal with `totalAmount` column will be to drop the rows with null values in that column. Run the following cell to drop the null rows and review the final imputed dataset.

# COMMAND ----------

imputedDF = imputedDF.na.drop(subset=["totalAmount"])

display(imputedDF.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 2: Feature Engineering
# MAGIC 
# MAGIC In some situations, it is beneficial to engineer new features or columns from existing data. In this case, the `hour_of_day` column represents hours from 0 – 23. Given that time is cyclical in nature, for example hour 23 is very close hour 0. Thus, it can be useful to transform the ` hour_of_day ` column as **sine** and **cosine** functions that are inherently cyclical in nature. Run the following cell to setup an user defined function (UDF) that will take in the ` hour_of_day ` column and transforms the column to its sine and cosine representation.

# COMMAND ----------

def get_sin_cosine(value, max_value):
  sine =  np.sin(value * (2.*np.pi/max_value))
  cosine = np.cos(value * (2.*np.pi/max_value))
  return (sine.tolist(), cosine.tolist())

schema = StructType([
    StructField("sine", DoubleType(), False),
    StructField("cosine", DoubleType(), False)
])

get_sin_cosineUDF = udf(get_sin_cosine, schema)

print("UDF get_sin_cosineUDF defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Run the following cell to do the ` hour_of_day `  column transformation and name the two new columns as `hour_sine` and `hour_cosine` and drop the original column. To review the resulting dataframe, scroll to the right to observe the two new columns.

# COMMAND ----------

engineeredDF = imputedDF.withColumn("udfResult", get_sin_cosineUDF(col("hour_of_day"), lit(24))).withColumn("hour_sine", col("udfResult.sine")).withColumn("hour_cosine", col("udfResult.cosine")).drop("udfResult").drop("hour_of_day")
display(engineeredDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 3: Scaling Numeric features
# MAGIC 
# MAGIC Common types of data in machine learning include:
# MAGIC - Numerical
# MAGIC   - Numerical values, either integers or floats
# MAGIC   - Example, predict house prices
# MAGIC - Categorical
# MAGIC   - Discrete and limited set of values
# MAGIC   - The values typically do not make sense unless there is a meaning or a category attached to the values
# MAGIC   - Example, persons gender or ethnicity
# MAGIC - Time-Series
# MAGIC   - Data series over time
# MAGIC   - Typically, data collected over equally spaced points in time
# MAGIC   - Example, real-time stock performance
# MAGIC - Text
# MAGIC   - Words or sentences
# MAGIC   - Example, news articles
# MAGIC   
# MAGIC In the example we are working with, we have **numerical** and **categorical** features. Run the following cell to create list of numerical and categorical features in the dataset. In this exercise, we will look at how to work with numerical features and in the next exercise we will look at encoding categorical features.

# COMMAND ----------

numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", "hour_sine", "hour_cosine"]
categorical_cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
label_column = "totalAmount"
print("Numerical and categorical features list defined. Label column identified.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For numerical features the ranges of values can vary widely and thus it is common practice in machine learning to scale the numerical features. The two common approaches for data scaling are:
# MAGIC - **Normalization**: Rescales the data into the range [0, 1].
# MAGIC - **Standardization**: Rescales the data to have Mean = 0 and Variance = 1.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Run the following cell to see how to use the **VectorAssembler**  and **MinMaxScaler** to scale the numerical features into the range of [0,1]. Observe how we combine the two-step transformation into a single pipeline. Finally, review the resulting dataframe by scrolling to right to observe the new assembled and scaled column: **scaled_numerical_features**.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline


assembler = VectorAssembler().setInputCols(numerical_cols).setOutputCol('numerical_features')
scaler = MinMaxScaler(inputCol=assembler.getOutputCol(), outputCol="scaled_numerical_features")

partialPipeline = Pipeline().setStages([assembler, scaler])
pipelineModel = partialPipeline.fit(engineeredDF)
scaledDF = pipelineModel.transform(engineeredDF)

display(scaledDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 4: Encoding Categorical Features
# MAGIC 
# MAGIC It is important to note that in machine learning, we ultimately always work with numbers or specifically, vectors. In this context, a vector is either an array of numbers or nested arrays of arrays of numbers. All non-numeric data types, such as categories, like `normalizeHolidayName`, `isPaidTimeOff` in the dataframe are eventually represented as numbers. Also, for numerical categories, such as `day_of_week` and `month_num`, it is important to encode them. Otherwise, machine learning model might assume that month 6 (June) is six times as much as the month 1 (January).
# MAGIC 
# MAGIC **One Hot Encoding** is often the recommended approach to encode categorical features. In this approach, for each categorical column, a number of N new columns are added to the data set, where N is the cardinality (the number of distinct values) of the column. Each column corresponds to one of the categories and will have a value of 0 if the row has that category or 1 if it hasn’t.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Run the following cell to encode the categorical features in the dataset using One Hot encoding. Since, ** OneHotEncoder** only operates on numerical values, we will first use **StringIndexer** to index the categorical columns to a numerical value and the then encode using the **OneHotEncoder**.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

stages = [] # stages in our Pipeline
for categorical_col in categorical_cols:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index", handleInvalid="skip")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categorical_col + "_classVector"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

encodedDF = scaledDF.withColumn("isPaidTimeOff", col("isPaidTimeOff").cast("integer"))
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(encodedDF)
encodedDF = pipelineModel.transform(encodedDF)

display(encodedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In the resulting dataframe, observe the new column **isPaidTimeOff_classVector** is a vector. The difference between a sparse and dense vector is whether Spark records all of the empty values. In a sparse vector, like we see here, Spark saves space by only recording the places where the vector has a non-zero value. The value of 0 in the first position indicates that it's a sparse vector. The second value indicates the length of the vector.
# MAGIC 
# MAGIC Example interpretation of the following vector: **[0, 1, [0], [1]]**
# MAGIC - 0 - it’s a sparse vector
# MAGIC - 1 – length of the vector is 1
# MAGIC - [0] – in this case the values only present in the 0th position of the vector
# MAGIC - [1] – values in the corresponding positions