from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Define column names, features, and label
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

# Build the input function. Input data is passed into input_fn in the data_set argument.
# This allows the funtion to process any of the imported DataFrames
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x = pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set[LABEL].values),
        num_epochs = num_epochs,
        shuffle=shuffle)

# Load datasets into pandas
training_set = pd.read_csv("boston_train.csv", skipinitialspace = True,
                            skiprows = 1, names = COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace = True,
                        skiprows = 1, names = COLUMNS)
# Set of 6 examples for which to predict median house values
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace = True,
                          skiprows = 1, names = COLUMNS)

# Define feature columns
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols,
                                      hidden_units = [10, 10],
                                      model_dir = "/tmp/boston_model")

# Training the Regressor
regressor.train(input_fn=get_input_fn(training_set), steps=5000)

# Evaluate the model to see how trained data performs against the test set
ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

# Print out predictions over a slice of prediction_set.
# .predict() returns an iterator of dicts; convert to a list and print predictions
y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
