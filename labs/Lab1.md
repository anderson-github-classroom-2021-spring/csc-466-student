---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Ethics with an introduction to NumPy and Pandas on the side


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.

Please see the README for instructions on how to submit and obtain the lab.

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab1_helper
```

# Python, NumPy, and Pandas

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy and Pandas that we will rely upon routinely. 


## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)


## Exercises 1-5
For the following exercises please read the Python appendix in the Marsland textbook and answer problems below in the space provided and in Lab1_helper.py.


#### Exercise 1. Make an array a of size 6 × 4 where every element is a 2.

```python
a = Lab1_helper.exercise_1()
a
```

#### Exercise 2. Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1 everywhere else. (You can do this without loops.)

```python
b = Lab1_helper.exercise_2()
b
```

#### Stop and think: Why can you multiply these two matrices together? Why does a * b work, but not dot(a,b)?

```python
a*b
```

```python
import numpy as np

np.dot(a,b)
```

#### YOUR SOLUTION HERE


#### Stop and Think: Compute dot(a.transpose(),b) and dot(a,b.transpose()). Why are the results different shapes?

```python
np.dot(a.transpose(),b)
```

```python
np.dot(a,b.transpose())
```

#### YOUR SOLUTION HERE


#### Exercise 3. Find the overall mean of matrix ``c``.

```python
np.random.seed(1)
c = np.random.rand(6,4)
display(c)
m = Lab1_helper.exercise_3(c)
m
```

#### Exercise 4. Find the column and row means of matrix ``c``.

```python
row_means,col_means = Lab1_helper.exercise_4(c)
display(row_means)
display(col_means)
```

#### Exercise 5. Write a function that consists of a set of loops that run through an array and counts the number of ones in it. Do the same thing without using a for loop. For inspiration, check out the following. You don't need to use all of them, but pick one.

```python
b == 1
```

```python
np.where(b == 1)
```

```python
np.array(b.flat)
```

```python
np.where(b.flat == 1)
```

```python
c1,c2 = Lab1_helper.exercise_5(b)
c1,c2
```

## Excercises 6
We will use Pandas at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


#### Exercise 6. Repeat Exercise 1, but create a Pandas DataFrame instead of a NumPy array.

```python
a = Lab1_helper.exercise_6()
a
```

#### Exercise 7. Repeat exercise 2 using a DataFrame instead.

```python
b = Lab1_helper.exercise_7()
b
```

#### Stop and think: What if we want to go from a pandas dataframe to a numpy array?

```python
b.values
```

## Exercises 8-10
Now let's look at a real dataset, and talk about ``.iloc`` and ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "../csc-466-student/data/titanic.csv"
)
titanic_df
```

```python
titanic_df.index
```

```python
df = titanic_df.set_index('sex').loc['female']
df
```

```python
inxs = np.where(titanic_df.survived==1)
inxs
titanic_df.iloc[inxs]
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE
titanic_df.set_index('sex',inplace=True)
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
```

## Ethics

We are finally ready to think about KDD ethics! 

We have preprocessed a dataset on loan applications to make this example appropriate for linear regression. The independent variable data is real and has not been modified apart from being transformed (e.g., Married=Yes => Married=1.). In other words, this is a real dataset with minimal modifcations. 

Our client is a loan company, they would like you to look at this historical data of 296 loans which have been approved for varying amounts (LoanAmountApproved). They are interested in extracting which independent variables are the most influential/important when predicting the amount of the approved loan. Upon an ethical review, they have determined that ``Gender`` is a protected column and should not be considered in the analysis.

```python
import pandas as pd

# Read in the data into a pandas dataframe
credit = pd.read_csv("../data/credit.csv",index_col=0)
credit.head()
```

### Exercise 15. Construct a linear model model to predict LoanAmountApproved (y = mx+b) using all of the columns except Gender which after an ethical review was deemed inappropriate to consider when make a determination on the amount of loan approved for an applicant.

```python
X = credit.drop(['Gender','LoanAmountApproved'],axis=1)
y = credit['LoanAmountApproved']

model = Lab1_helper.problem_15(X,y)

# Here is code that takes the numpy array of coefficients stored in model.coef_ and formats it nicely for the screen
coef = pd.Series(model.coef_,index=X.columns)
coef.abs().sort_values(ascending=False)
```

Now let's write some code that calculates the mean absolute error of our model which is one measure of how good our model is performing. Looks like we are approximately $27K off in our model on average.

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,model.predict(X))
```

The company asks you for your interpretation of the model. You say that being married is a high indicator of receiving a high amount for your loan. This seems reasonable, and everyone seems happy. An experienced data scientist on your team suggests you run a correlation of your columns used in the regression and the column Gender since it is considered a protected column. You do so quickly to satisfy this request and get:

```python
Xgender = X.copy()
Xgender['Gender'] = credit['Gender']
Xgender.corr().loc['Gender'].abs().sort_values(ascending=False)
```

What do you think about the results? Specifically, is the fact that Married is correlated with Gender at a correlation of 0.36 concerning from an ethical standpoint? What do you as an individual think? Can you think of any suggestions about what to do?


Assuming your suggestion was to remove it. Do so and then compare the accuracy of the model

```python
from sklearn.metrics import r2_score
model2 = LinearRegression().fit(X.drop('Married',axis=1), y)
mean_absolute_error(y,model2.predict(X.drop('Married',axis=1)))
```

What do you think now, should you drop it? Your prediction is now off by more than $10,000. Is this ok?

```python

```
