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

# Final Lab

## CSC 466 Spring 2021

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import FinalLab_helper
```

<!-- #region slideshow={"slide_type": "subslide"} -->
#### Exercise 1

Convert your naive Bayesian classifier code from Lab 2 to fit the object oriented estimator pattern from sklearn. In other words, fill out the class in FinalLab_helper. Here is a demo of it working.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

titanic_df = pd.read_csv(f"{home}/csc-466-student/data/titanic.csv")
features = ['Pclass','Survived','Sex','Age']
titanic_df = titanic_df.loc[:,features]
titanic_df.loc[:,'Pclass']=titanic_df['Pclass'].fillna(titanic_df['Pclass'].mode()).astype(int)
titanic_df.loc[:,'Age']=titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df.loc[:,'Age']=(titanic_df['Age']/10).astype(str).str[0].astype(int)*10
titranic_df = titanic_df.dropna()
titanic_df.head()
```

```python
X = titanic_df.drop('Survived',axis=1)
y = titanic_df['Survived']
```

```python slideshow={"slide_type": "subslide"}
clf = FinalLab_helper.NBClassifier()
clf.fit(X,y)
predictions = clf.predict(X)
predictions[:20]
```

<!-- #region slideshow={"slide_type": "subslide"} -->
#### Exercise 2
Convert the code in Chapter 7 to a sklearn style PCA transformer. In other words, complete the code provided in FinalLab_helper such that you find the eigenvectors of the covariance matrix and use them appropriate when calling transform. A sample usage is below.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
df = pd.read_csv(f"{home}/csc-466-student/data/breast_cancer_three_gene.csv",index_col=0)
X = df[['ESR1','AURKA']]
X
```

```python
pca_transformer = FinalLab_helper.PCA()
pca_transformer.fit(X)
Xt = pca_transformer.transform(X)
Xt
```

```python
# Good job!
# Don't forget to push with ./submit.sh
```

<!-- #region -->
#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_FinalLab.joblib")
answers.keys()
```
<!-- #endregion -->

```python

```
