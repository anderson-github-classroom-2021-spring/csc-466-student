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

def test_exercise_1():
    assert np.all(answers['exercise_1'] == helper.exercise_1())

def test_exercise_2():
    assert np.all(answers['exercise_2'] == helper.exercise_2())

def test_exercise_3():
    one,two = helper.exercise_4(c)
    assert np.all(answers['exercise_3'][0] == one) and np.all(answers['exercise_3'][0] == two)
