import sys
import os
sys.path.append(".")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab1.joblib")

# Import the student solutions
import Lab1_helper as helper

def test_exercise_1():
    assert answers['exercise_1'] == helper.exercise_1()

def test_exercise_2():
    assert answers['exercise_2'] == helper.exercise_2()

def test_exercise_3():
    assert answers['exercise_3'] == helper.process_exercise_3(result)

def test_exercise_4():
    assert answers['exercise_4'] == helper.process_exercise_4(result)
