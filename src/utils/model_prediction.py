# Import packages
import pandas as pd
import os
import pickle


# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read train data
X_test = pd.read_csv("data/final/test/X_test.csv")
y_test = pd.read_csv("data/final/test/y_test.csv")

# Load the saved model
with open("model/model.pkl", "rb") as f:
    model_obj = pickle.load(f)


# Function to predict on the test data using the saved model
def model_predict(model_obj, X_test, y_test):
    y_test["pred_prob"] = model_obj.predict(X_test)
    return y_test


# Predict on the test data using the saved model
model_obj = model_predict(model_obj, X_test, y_test)


# Write the files in the respective folder
y_test.to_csv("results/y_test.csv", index=False)
