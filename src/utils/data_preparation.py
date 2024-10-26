# Import packages
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read data
iris = pd.read_csv("data/raw/iris_data.csv")

# Separate the features and labels
X = iris.drop(["Id", "Species"], axis=1)
y = iris[["Species"]]


# Function to divide the data in train and test
def data_split(X, y):
    """
    This function takes the entire features and lables data as input and divide the into train an test data

    Input :
    X : Features of the data
    y : Labels of the data

    Output :
    The entire features and labels data is divided into training data and testing data

    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=5
    )

    return (X_train, X_test, y_train, y_test)


# Divide the data in train test
X_train, X_test, y_train, y_test = data_split(X, y)

# Write the files in the respective folder
X_train.to_csv("data/final/train/X_train.csv", index=False)
X_test.to_csv("data/final/test/X_test.csv", index=False)
y_train.to_csv("data/final/train/y_train.csv", index=False)
y_test.to_csv("data/final/test/y_test.csv", index=False)
