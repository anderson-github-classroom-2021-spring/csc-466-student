import sys
import os
sys.path.append(".")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab1.joblib")

# Import the student solutions
import Lab1_helper as helper

import numpy as np
np.random.seed(1)
c = np.random.rand(6,4)

import pandas as pd
titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)

def test_exercise_1():
    assert np.all(answers['exercise_1'] == helper.exercise_1())

def test_exercise_2():
    assert np.all(answers['exercise_2'] == helper.exercise_2())

def test_exercise_3():
    assert answers['exercise_3'] == helper.exercise_3(c)

def test_exercise_4():
    one,two = helper.exercise_4(c)
    assert np.all(answers['exercise_4'][0] == one) and np.all(answers['exercise_4'][1] == two)
    
def test_exercise_5():
    b = helper.exercise_2()
    one,two = helper.exercise_5(b)
    assert np.all(answers['exercise_5'][0] == one) and np.all(answers['exercise_5'][1] == two)
    
def test_exercise_6():
    assert np.all(answers['exercise_6'] == helper.exercise_6())

def test_exercise_7():
    assert np.all(answers['exercise_7'] == helper.exercise_7())