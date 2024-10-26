# Import packages
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import pickle


# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read train data
X_train = pd.read_csv("data/final/train/X_train.csv")
y_train = pd.read_csv("data/final/train/y_train.csv")


# Function to fit the model
def model_fit(X_train, y_train):
    """
    This function takes the features and lables as input builds a logistic regression model

    Input :
    X_train : Features of the training data
    y_train : Labels of the training data

    Output :
    The output is a model object

    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# Build model on the train data
model_obj = model_fit(X_train, y_train.values.ravel())

# Save model object for further use
with open("model/model.pkl", "wb") as f:
    pickle.dump(model_obj, f)
