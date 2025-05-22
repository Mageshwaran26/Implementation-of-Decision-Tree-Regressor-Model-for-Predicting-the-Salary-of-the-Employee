# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load Data – Import the dataset containing employee details and their salaries.

2. Preprocess Data – Handle missing values, encode categorical variables, and split into training and test sets.

3. Initialize Model – Create a DecisionTreeRegressor with suitable parameters (e.g., max_depth=5).

4. Train Model – Fit the regressor using training data (model.fit(X_train, y_train)).

5. Predict & Evaluate – Predict salaries on test data and evaluate using metrics like MAE, MSE, and R² score.

6. Visualize & Interpret – Plot the tree and analyze feature importance for salary prediction. 
 
## Program and Output:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Mageshwaran T.A
RegisterNumber: 212224230146
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
```
![alt text](<Screenshot 2025-04-24 092514.png>)
```
data.info()
data.isnull().sum()
```
![alt text](<Screenshot 2025-04-24 092524.png>)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![alt text](<Screenshot 2025-04-24 092532.png>)
```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
![alt text](<Screenshot 2025-04-24 092538.png>)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
![alt text](<Screenshot 2025-04-24 092544.png>)
```
from sklearn import metrics
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2
```
![alt text](<Screenshot 2025-04-24 092549.png>)
```
dt.predict([[5,6]])
```
![alt text](<Screenshot 2025-04-24 092558.png>)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
