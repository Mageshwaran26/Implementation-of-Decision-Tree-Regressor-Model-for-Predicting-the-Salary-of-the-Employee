# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mageshwaran T.A
RegisterNumber: 212224230146
*/
```
```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:
## DATA HEAD
![image](https://github.com/user-attachments/assets/40a317c6-2e26-46f9-8550-857b371b1a2d)
## DATA INFO
![image](https://github.com/user-attachments/assets/4152e9c3-8d2b-40a2-8daa-1617e173932d)
## isnull() sum():
![image](https://github.com/user-attachments/assets/43979d88-06d4-49d9-9340-8da0212d8e61)
## DATA HEAD FOR SALARY 
![image](https://github.com/user-attachments/assets/9678fb4b-bab8-4f48-bbf8-d0510e0ea305)
## MEAN SQUARED ERROR
![image](https://github.com/user-attachments/assets/a692d6e9-1e77-4344-a2a5-50bd6abe2384)
## R2 VALUE
![image](https://github.com/user-attachments/assets/db1eb5c1-f30a-4864-90ad-559c20d3312d)
## DATA PREDICTION

![image](https://github.com/user-attachments/assets/c8c498e6-b0ba-4772-8436-a2ecd1ca03c5)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
