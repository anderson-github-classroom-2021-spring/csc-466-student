import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab2.joblib")

# Import the student solutions
import Lab2_helper

import pandas as pd
import numpy as np

titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)

features = ['Pclass','Survived','Sex','Age']
titanic_df = titanic_df.loc[:,features]
titanic_df.loc[:,'Pclass']=titanic_df['Pclass'].fillna(titanic_df['Pclass'].mode()).astype(int)
titanic_df.loc[:,'Age']=titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df.loc[:,'Age']=(titanic_df['Age']/10).astype(str).str[0].astype(int)*10
titranic_df = titanic_df.dropna()

X = titanic_df.drop("Survived",axis=1)
y = titanic_df["Survived"]

def test_exercise_1():
    survived_priors = Lab2_helper.compute_priors(titanic_df['Survived'])
    assert answers['exercise_1'] == survived_priors

def test_exercise_2():
    prob = Lab2_helper.specific_class_conditional(titanic_df['Sex'],'female',titanic_df['Survived'],0)
    assert answers['exercise_2'] == prob

def test_exercise_3():
    probs = Lab2_helper.class_conditional(X,y)
    assert answers['exercise_3'] == probs

def test_exercise_4():
    probs = Lab2_helper.class_conditional(X,y)
    priors = Lab2_helper.compute_priors(y)
    x = titanic_df.drop("Survived",axis=1).loc[2]
    post_probs = Lab2_helper.posteriors(probs,priors,x)
    assert answers['exercise_4'] == post_probs

def test_exercise_5():
    np.random.seed(2)
    Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
    assert answers['exercise_5'][0].equals(Xtrain)

def test_exercise_6():
    np.random.seed(0)
    Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
    accuracy = Lab2_helper.exercise_6(Xtrain,ytrain,Xtest,ytest)
    assert answers['exercise_6'] == accuracy

def test_exercise_6():
    np.random.seed(0)
    Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
    accuracy = Lab2_helper.exercise_6(Xtrain,ytrain,Xtest,ytest)
    assert answers['exercise_6'] == accuracy

def test_exercise_7():
    np.random.seed(0)
    Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
    importances = Lab2_helper.exercise_7(Xtrain,ytrain,Xtest,ytest)
    assert answers['exercise_7'] == importances

def test_exercise_8():
    np.random.seed(0)
    Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
    importances = Lab2_helper.exercise_8(Xtrain,ytrain,Xtest,ytest)
    assert answers['exercise_8'] == importances
