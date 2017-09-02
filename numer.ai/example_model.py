#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from sklearn.neural_network import MLPClassifier

def main():
    # Set seed for reproducibility
    np.random.seed(0)
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
    prediction_data = prediction_data[prediction_data.data_type == 'live']

    print(len(prediction_data))

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]


    model = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(5, 2),
                        random_state=1,
                        max_iter=3,
                        verbose=True)

    model.fit(X, Y)

    print("Predicting...")
    # Your trained model is now used to make predictions on the numerai_tournament_data
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    print("#"*50)
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    print(results_df)
    joined = pd.DataFrame(ids).join(results_df)
    print(joined)
    print("Writing predictions to predictions.csv")
    # Save the predictions out to a CSV file
    joined.to_csv("predictions_3.csv", index=False)
    # Now you can upload these predictions on numer.ai


if __name__ == '__main__':
    main()
