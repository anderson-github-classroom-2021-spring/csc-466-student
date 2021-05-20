import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab6.joblib")

# Import the student solutions
import Lab6_helper

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
df = pd.read_csv(f"{DIR}/../data/housing/boston_fixed.csv")

def test_exercise_1():
    X = Lab6_helper.scale(df)
    assert np.all(answers['exercise_1'].values == X.values)

def test_exercise_2():
    X = Lab6_helper.scale(df)
    X_pca = Lab6_helper.pca(X)
    assert np.all(answers['exercise_2'].values == X_pca.values)
    
def test_exercise_3():
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    assert set(answers['exercise_3'].keys()) == set(kmeans_models.keys())

def test_exercise_4():
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    assert np.all(answers['exercise_4'].values == cluster_labels.values)
    
def test_exercise_5():
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    n_clusters = 2
    scores = Lab6_helper.silhouette_scores(X,cluster_labels[n_clusters])
    assert np.all(answers['exercise_5'] == scores)
    
def test_exercise_6():
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    n_clusters = 2
    scores = Lab6_helper.silhouette_scores(X,cluster_labels[n_clusters])
    clusterer = Lab6_helper.bin_x(df[["MEDV"]])
    labels = clusterer.predict(df[["MEDV"]])
    assert np.all(answers['exercise_6'] == labels)
