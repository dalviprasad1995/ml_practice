# Import packages
import pandas as pd
import os
import pickle


# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read train data
X_test = pd.read_csv("data/final/test/X_test.csv")
y_test = pd.read_csv("data/final/test/y_test.csv")


# Function to load the saved model
def model_predict(X_test, y_test):
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# Load model on the train data
model_obj = model_predict(X_test, y_test)

# Predict on the test data using the saved model
y_test["pred_prob"] = model_obj.predict(X_test)

# Write the files in the respective folder
y_test.to_csv("results/y_test.csv", index=False)
