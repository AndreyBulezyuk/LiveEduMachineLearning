#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import logging
from sklearn import metrics, preprocessing, linear_model


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)


    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    prediction_data = prediction_data[prediction_data.data_type == 'live']

    eras = prediction_data.era
    eras.drop_duplicates


    print(eras.describe())
    print(type(eras))
    print(eras)
    print(len(eras))

    #27693 test
    #16686 validation
    #1280 live



    # This is your model that will learn to predict
    model = linear_model.LogisticRegression(n_jobs=-1)

    print("Training...")
    # Your model is trained on the training_data
    model.fit(X, Y)

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions4.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
