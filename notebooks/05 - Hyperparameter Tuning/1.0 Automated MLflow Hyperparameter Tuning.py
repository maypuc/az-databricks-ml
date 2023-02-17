# Databricks notebook source
# MAGIC %md
# MAGIC # Automated MLflow Hyperparameter Tuning
# MAGIC 
# MAGIC In this lab, you will learn to tune hyperparameters in Azure Databricks. This lab will cover the following exercise:
# MAGIC - Exercise 1: Using Automated MLflow for hyperparameter tuning.
# MAGIC 
# MAGIC To upload the necessary data, please follow the instructions in the lab guide.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Attach notebook to your cluster
# MAGIC Before executing any cells in the notebook, you need to attach it to your cluster. Make sure that the cluster is running.
# MAGIC 
# MAGIC In the notebook's toolbar, select the drop down arrow next to Detached, and then select your cluster under Attach to.
# MAGIC 
# MAGIC Make sure you run each cells in order.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 1: Using Automated MLflow for hyperparameter tuning
# MAGIC In this exercise, you will perform hyperparameter tuning by using the automated MLflow libary. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load the data
# MAGIC In this exercise, you will be using a dataset of real estate sales transactions to predict the price-per-unit of a property based on its features. The price-per-unit in this data is based on a unit measurement of 3.3 square meters
# MAGIC 
# MAGIC The data consists of the following variables:
# MAGIC - **transaction_date** - the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
# MAGIC - **house_age** - the house age (in years)
# MAGIC - **transit_distance** - the distance to the nearest light rail station (in meters)
# MAGIC - **local_convenience_stores** - the number of convenience stores within walking distance
# MAGIC - **latitude** - the geographic coordinate, latitude
# MAGIC - **longitude** - the geographic coordinate, longitude
# MAGIC - **price_per_unit** - house price of unit area (3.3 square meters) 
# MAGIC 
# MAGIC 
# MAGIC Run the following cell to load the table into a Spark dataframe and review the dataframe.

# COMMAND ----------

dataset = spark.sql("select * from real_estate")
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a linear regression model
# MAGIC Start by performing a train/test split on the housing dataset and building a pipeline for linear regression.
# MAGIC 
# MAGIC In the cell below, a dataframe `housingDF` is created from the table you created before. The dataframe is then randomnly split into a training set that contains 80% of the data, and a test set that contains 20% of the original dataset. All columns except for the last one are then marked as features so that a Linear Regression model can be trained on the data. 

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

housingDF = dataset.dropna(subset=['price_per_unit'])

trainDF, testDF = housingDF.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=housingDF.columns[:-1], outputCol="features")

lr = (LinearRegression()
  .setLabelCol("price_per_unit")
  .setFeaturesCol("features")
)

pipeline = Pipeline(stages = [assembler, lr])

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the model parameters using the `.explainParams()` method.

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC `ParamGridBuilder()` allows us to string together all of the different possible hyperparameters we would like to test.  In this case, we can test the maximum number of iterations, whether we want to use an intercept with the y axis, and whether we want to standardize our features.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
  .addGrid(lr.maxIter, [1, 10, 100])
  .addGrid(lr.fitIntercept, [True, False])
  .addGrid(lr.standardization, [True, False])
  .build()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now `paramGrid` contains all of the combinations we will test in the next step.  Take a look at what it contains.

# COMMAND ----------

paramGrid

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Cross-Validation
# MAGIC 
# MAGIC There are a number of different ways of conducting cross-validation, allowing us to trade off between computational expense and model performance.  An exhaustive approach to cross-validation would include every possible split of the training set.  More commonly, _k_-fold cross-validation is used where the training dataset is divided into _k_ smaller sets, or folds.  A model is then trained on _k_-1 folds of the training data and the last fold is used to evaluate its performance.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a `RegressionEvaluator()` to evaluate our grid search experiments and a `CrossValidator()` to build our models.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(
  labelCol = "price_per_unit", 
  predictionCol = "prediction"
)

cv = CrossValidator(
  estimator = pipeline,             # Estimator (individual model or pipeline)
  estimatorParamMaps = paramGrid,   # Grid of parameters to try (grid search)
  evaluator=evaluator,              # Evaluator
  numFolds = 3,                     # Set k to 3
  seed = 42                         # Seed to make sure our results are the same if ran again
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fit the `CrossValidator()`
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This will train a large number of models.  If your cluster size is too small, it could take a while.

# COMMAND ----------

cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the scores from the different experiments.

# COMMAND ----------

for params, score in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics):
  print("".join([param.name+"\t"+str(params[param])+"\t" for param in params]))
  print("\tScore: {}".format(score))

# COMMAND ----------

# MAGIC %md
# MAGIC You can then access the best model using the `.bestModel` attribute. 

# COMMAND ----------

bestModel = cvModel.bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC To see the predictions of the best model on the test dataset, execute the code below:

# COMMAND ----------

predictions = cvModel.bestModel.transform(testDF)
display(predictions)