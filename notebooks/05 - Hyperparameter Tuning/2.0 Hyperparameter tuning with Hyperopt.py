# Databricks notebook source
# MAGIC %md
# MAGIC # Hyperparameter tuning with Hyperopt
# MAGIC 
# MAGIC In this lab, you will learn to tune hyperparameters in Azure Databricks. This lab will cover the following exercises:
# MAGIC - Exercise 2: Using Hyperopt for hyperparameter tuning.
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
# MAGIC ## Exercise 2: Using Hyporopt for hyperparameter tuning
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for hyperparameter tuning. Databricks Runtime for Machine Learning includes an optimized and enhanced version of Hyperopt, including automated MLflow tracking and the `SparkTrials` class for distributed tuning.  
# MAGIC 
# MAGIC This exercise illustrates how to scale up hyperparameter tuning for a single-machine Python ML algorithm and track the results using MLflow. Run the cell below to import the required packages.

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

# If you are running Databricks Runtime for Machine Learning, `mlflow` is already installed and you can skip the following line. 
import mlflow

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load the data
# MAGIC In this exercise, you will use a dataset that includes chemical and visual features of wine samples to classify them based on their cultivar (grape variety).
# MAGIC 
# MAGIC The dataset consists of 12 numeric features and a classification label with the following classes:
# MAGIC - **0** (variety A)
# MAGIC - **1** (variety B)
# MAGIC - **2** (variety C)
# MAGIC 
# MAGIC Run the following cell to load the table into a Spark dataframe and reivew the dataframe.

# COMMAND ----------

df = spark.sql("select * from wine")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Separate the features from the label (WineVariety):

# COMMAND ----------

import numpy as np
df_features = df.select('Alcohol','Malic_acid','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color_intensity','Hue','OD280_315_of_diluted_wines','Proline').collect()
X = np.array(df_features)

df_label = df.select('WineVariety').collect()
y = np.array(df_label)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the first four wines to see if the data is loaded in correctly:

# COMMAND ----------

for n in range(0,4):
    print("Wine", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])

# COMMAND ----------

# MAGIC %md ## Part 1. Single-machine Hyperopt workflow
# MAGIC 
# MAGIC Here are the steps in a Hyperopt workflow:  
# MAGIC 1. Define a function to minimize.  
# MAGIC 2. Define a search space over hyperparameters.  
# MAGIC 3. Select a search algorithm.  
# MAGIC 4. Run the tuning algorithm with Hyperopt `fmin()`.
# MAGIC 
# MAGIC For more information, see the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin).

# COMMAND ----------

# MAGIC %md ### Define a function to minimize
# MAGIC In this example, we use a support vector machine classifier. The objective is to find the best value for the regularization parameter `C`.  
# MAGIC 
# MAGIC Most of the code for a Hyperopt workflow is in the objective function. This example uses the [support vector classifier from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def objective(C):
    # Create a support vector classifier model
    clf = SVC(C=C)
    
    # Use the cross-validation accuracy to compare the models' performance
    accuracy = cross_val_score(clf, X, y).mean()
    
    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ### Define the search space over hyperparameters
# MAGIC 
# MAGIC See the [Hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions) for details on defining a search space and parameter expressions.

# COMMAND ----------

search_space = hp.lognormal('C', 0, 1.0)

# COMMAND ----------

# MAGIC %md ### Select a search algorithm
# MAGIC 
# MAGIC The two main choices are:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach which iteratively and adaptively selects new hyperparameter settings to explore based on past results
# MAGIC * `hyperopt.rand.suggest`: Random search, a non-adaptive approach that samples over the search space

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

# MAGIC %md Run the tuning algorithm with Hyperopt `fmin()`
# MAGIC 
# MAGIC Set `max_evals` to the maximum number of points in hyperparameter space to test, that is, the maximum number of models to fit and evaluate.

# COMMAND ----------

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=16)

# COMMAND ----------

# Print the best value found for C
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md ## Part 2. Distributed tuning using Apache Spark and MLflow
# MAGIC 
# MAGIC To distribute tuning, add one more argument to `fmin()`: a `Trials` class called `SparkTrials`. 
# MAGIC 
# MAGIC `SparkTrials` takes 2 optional arguments:  
# MAGIC * `parallelism`: Number of models to fit and evaluate concurrently. The default is the number of available Spark task slots.
# MAGIC * `timeout`: Maximum time (in seconds) that `fmin()` can run. The default is no maximum time limit.
# MAGIC 
# MAGIC This example uses the very simple objective function defined in Cmd 12. In this case, the function runs quickly and the overhead of starting the Spark jobs dominates the calculation time, so the calculations for the distributed case take more time. For typical real-world problems, the objective function is more complex, and using `SparkTrails` to distribute the calculations will be faster than single-machine tuning.
# MAGIC 
# MAGIC Automated MLflow tracking is enabled by default. To use it, call `mlflow.start_run()` before calling `fmin()` as shown in the example.

# COMMAND ----------

from hyperopt import SparkTrials

# To display the API documentation for the SparkTrials class, uncomment the following line.
# help(SparkTrials)

# COMMAND ----------

spark_trials = SparkTrials()

with mlflow.start_run():
  argmin = fmin(
    fn=objective,
    space=search_space,
    algo=algo,
    max_evals=16,
    trials=spark_trials)

# COMMAND ----------

# Print the best value found for C
print("Best value found: ", argmin)

# COMMAND ----------

# MAGIC %md To view the MLflow experiment associated with the notebook, click the **Experiment** icon in the notebook context bar on the upper right.  There, you can view all runs. To view runs in the MLflow UI, click the icon at the far right next to **Experiment Runs**. 
# MAGIC 
# MAGIC To examine the effect of tuning `C`:
# MAGIC 
# MAGIC 1. Select the resulting runs and click **Compare**.
# MAGIC 1. In the Scatter Plot, select **C** for X-axis and **loss** for Y-axis.