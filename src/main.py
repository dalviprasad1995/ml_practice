# Import packages
import pandas as pd
import os
import sys
import pickle
import logging
import datetime
import argparse
import configparser


# Command line argument
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root_dir", type=str, required=False)
parser.add_argument("--raw_data", type=str, required=False)
parser.add_argument("--train_data", type=str, required=False)
parser.add_argument("--test_data", type=str, required=False)
parser.add_argument("--save_model", type=str, required=False)
parser.add_argument("--save_prediction", type=str, required=False)
parser.add_argument("--log_path", type=str, required=False)
parser.add_argument("--log_level", type=str, required=False)
args = parser.parse_args()


# Get path
config = configparser.ConfigParser()
config.read("E:/Prasad/Projects/ml_practice/config.ini")

if args.root_dir is None:
    root_dir_path = config["Defaults"]["root_dir"]
else:
    root_dir_path = args.root_dir

if args.raw_data is None:
    raw_data_path = config["Defaults"]["raw_data"]
else:
    raw_data_path = args.raw_data

if args.train_data is None:
    train_data_path = config["Defaults"]["train_data"]
else:
    train_data_path = args.train_data

if args.test_data is None:
    test_data_path = config["Defaults"]["test_data"]
else:
    test_data_path = args.test_data

if args.save_model is None:
    save_model_path = config["Defaults"]["save_model"]
else:
    save_model_path = args.save_model

if args.save_prediction is None:
    save_prediction_path = config["Defaults"]["save_prediction"]
else:
    save_prediction_path = args.save_prediction

if args.log_path is None:
    log_path = config["Defaults"]["log_path"]
else:
    log_path = args.log_path

if args.log_level is None:
    log_level = config["Defaults"]["log_level"]
else:
    log_level = args.log_level


# Set working directory
os.chdir(root_dir_path)


# Get current time
timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))


# Logging files
logging.basicConfig(filename=log_path + "model_logs_{}.log".format(timestamp))

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger
if log_level == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif log_level == "INFO":
    logger.setLevel(logging.INFO)
elif log_level == "WARNING":
    logger.setLevel(logging.WARNING)
elif log_level == "ERROR":
    logger.setLevel(logging.ERROR)
elif log_level == "CRITICAL":
    logger.setLevel(logging.CRITICAL)


# Adding Folder_2 to the system path to add directories in the setup file
sys.path.insert(0, root_dir_path)


# Import customer packages
from src.utils.data_preparation import data_split
from src.utils.model_building import model_fit
from src.utils.model_prediction import model_predict


# Read data
iris = pd.read_csv(raw_data_path + "iris_data.csv")
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
X_train.to_csv(train_data_path + "X_train.csv", index=False)
X_test.to_csv(test_data_path + "X_test.csv", index=False)
y_train.to_csv(train_data_path + "y_train.csv", index=False)
y_test.to_csv(test_data_path + "y_test.csv", index=False)
logging.info("Train and test data exported")


# Read train data
X_train = pd.read_csv(train_data_path + "X_train.csv")
y_train = pd.read_csv(train_data_path + "y_train.csv")
logging.info("Train data is loaded completely")


# Build model on the train data
try:
    model_obj = model_fit(X_train, y_train.values.ravel())
    logging.debug("Model building is completed")
except:
    logging.error("Error in building the model")

# Save model object for further use
with open(save_model_path + "model.pkl", "wb") as f:
    pickle.dump(model_obj, f)


# Read train data
X_test = pd.read_csv(test_data_path + "X_test.csv")
y_test = pd.read_csv(test_data_path + "y_test.csv")


# Load the saved model
with open(save_model_path + "model.pkl", "rb") as f:
    model_obj = pickle.load(f)


# Predict on the test data using the saved model
try:
    y_test = model_predict(model_obj, X_test, y_test)
    logging.debug("Prediction complete on new data")
except:
    logging.error("Error in prediction on the new data")


# Write the files in the respective folder
y_test.to_csv(save_prediction_path + "y_test.csv", index=False)
logging.info("Code run complete !!")
