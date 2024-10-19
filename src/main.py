# Import packages
import pandas as pd
import os
import sys
import pickle
import logging
import datetime

# Get current time
timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

# Adding Folder_2 to the system path to add directories in the setup file
sys.path.insert(0, "E:/Prasad/Projects/ml_practice/")

# Logging files
logging.basicConfig(
    filename="E:/Prasad/Projects/ml_practice/logs/model_logs_{}.log".format(timestamp)
)

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


# Import customer packages
from src.utils.data_preparation import data_split
from src.utils.model_building import model_fit
from src.utils.model_prediction import model_predict


# Set working directory
os.chdir("E:/Prasad/Projects/ml_practice/")

# Read data
iris = pd.read_csv("data/raw/iris_data.csv")
logging.info("Iris Data is loaded completely")

# Separate the features and labels
X = iris.drop(["Id", "Species"], axis=1)
y = iris[["Species"]]


# Divide the data in train test
try:
    X_train, X_test, y_train, y_test = data_split(X, y)
    logging.debug("Data is splitted in training and testing")
except:
    logging.error("Error in splitting the error in training and testing")

# Write the files in the respective folder
X_train.to_csv("data/final/train/X_train.csv", index=False)
X_test.to_csv("data/final/test/X_test.csv", index=False)
y_train.to_csv("data/final/train/y_train.csv", index=False)
y_test.to_csv("data/final/test/y_test.csv", index=False)
logging.info("Train and test data exported")


# Read train data
X_train = pd.read_csv("data/final/train/X_train.csv")
y_train = pd.read_csv("data/final/train/y_train.csv")
logging.info("Train data is loaded completely")

# Build model on the train data
try:
    model_obj = model_fit(X_train, y_train.values.ravel())
    logging.debug("Model building is completed")
except:
    logging.error("Error in building the model")

# Save model object for further use
with open("model/model.pkl", "wb") as f:
    pickle.dump(model_obj, f)


# Read train data
X_test = pd.read_csv("data/final/test/X_test.csv")
y_test = pd.read_csv("data/final/test/y_test.csv")

# Load the saved model
with open("model/model.pkl", "rb") as f:
    model_obj = pickle.load(f)

# Predict on the test data using the saved model
try:
    y_test = model_predict(model_obj, X_test, y_test)
    logging.debug("Prediction complete on new data")
except:
    logging.error("Error in prediction on the new data")

# Write the files in the respective folder
y_test.to_csv("results/y_test.csv", index=False)
logging.info("Code run complete !!")
