# Import packages
import pandas as pd
import os
import sys
import pickle


# Adding Folder_2 to the system path
sys.path.insert(0, "E:/Prasad/Projects/ml_practice")

# Import customer packages
from src.utils.data_preparation import data_split
from src.utils.model_building import model_fit
from src.utils.model_prediction import model_predict


# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read data
iris = pd.read_csv("data/raw/iris_data.csv")

# Separate the features and labels
X = iris.drop(["Id", "Species"], axis=1)
y = iris[["Species"]]


# Divide the data in train test
X_train, X_test, y_train, y_test = data_split(X, y)

# Write the files in the respective folder
X_train.to_csv("data/final/train/X_train.csv", index=False)
X_test.to_csv("data/final/test/X_test.csv", index=False)
y_train.to_csv("data/final/train/y_train.csv", index=False)
y_test.to_csv("data/final/test/y_test.csv", index=False)


# Read train data
X_train = pd.read_csv("data/final/train/X_train.csv")
y_train = pd.read_csv("data/final/train/y_train.csv")


# Build model on the train data
model_obj = model_fit(X_train, y_train.values.ravel())

# Save model object for further use
with open("model/model.pkl", "wb") as f:
    pickle.dump(model_obj, f)


# Read train data
X_test = pd.read_csv("data/final/test/X_test.csv")
y_test = pd.read_csv("data/final/test/y_test.csv")


# Load model on the train data
model_obj = model_predict(X_test, y_test)

# Predict on the test data using the saved model
y_test["pred_prob"] = model_obj.predict(X_test)

# Write the files in the respective folder
y_test.to_csv("results/y_test.csv", index=False)
