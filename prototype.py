# Importing required libraries
!pip install h2o
from h2o.automl import H2OAutoML
import h2o
import pandas as pd
import numpy as np

# Initialize the H2O cluster
h2o.init()

# Load the dataset
data = pd.read_csv("/path/to/training.csv")  # Replace with your actual path
data.drop(columns=["lineID", "pid"], inplace=True)
data.to_csv("dataset.csv")
data_h2o = h2o.import_file(path="dataset.csv")

# Summary statistics
data_h2o.describe(chunk_summary=True)

# Train-Test Split
train, test = data_h2o.split_frame(ratios=[0.85], seed=1)

# Identify predictors and response variable
x = data_h2o.columns
y = "revenue"
x.remove(y)

# Initialize AutoML
AutoML = H2OAutoML(max_models=100, seed=10, max_runtime_secs=25000, project_name="Revenue_Forecast_for_Dynamic_Pricing")

# Train the model
AutoML.train(x=x, y=y, training_frame=train)

# View the leaderboard
automl_leaderboard = AutoML.leaderboard
print(automl_leaderboard.head(rows=automl_leaderboard.nrows))

# Get the best model
best_model = AutoML.get_best_model()
print("Best Model:", best_model)

# Evaluate the model
train_evaluation = best_model.model_performance(train)
test_evaluation = best_model.model_performance(test)

print("\nR2 Score on Training Set:", train_evaluation.r2())
print("R2 Score on Testing Set:", test_evaluation.r2())
print("\nRMSE Score on Training Set:", train_evaluation.rmse())
print("RMSE Score on Testing Set:", test_evaluation.rmse())

# Save the model
model_path = h2o.save_model(model=best_model, path='/path/to/save', force=True)  # Replace with your actual path

# Model Explainability
explain_model = AutoML.explain(frame=test)
AutoML.explain_row(frame=test, row_index=15)
