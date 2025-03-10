# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students

2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored

3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis

4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
  
5.For each data point calculate the difference between the actual and predicted marks

5.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error

6.Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: J.JANANI
RegisterNumber:  212223230085

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('/content/student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:
Head Values :

![420179982-f8383e2e-1bcc-4137-b1cd-fd706bd438fb](https://github.com/user-attachments/assets/21c8d22f-fdb9-4ba5-934d-6bc0fe7d06c5)

Tail Values :

![420180048-d3b67821-e803-4413-9f0f-415f6b04a21f](https://github.com/user-attachments/assets/4c2a9a86-eff9-4fbe-ba6f-ef7f41770d31)


X Values:

![420180149-a4a3710b-6d70-4286-a728-cc641ba21638](https://github.com/user-attachments/assets/a20853d2-3777-4fe1-bf84-5f9fd78ac419)

y Values :

Predicted Values

![420180178-eee55077-1f9e-4a60-9471-d5e2a131a6e8](https://github.com/user-attachments/assets/2ce266e1-4996-4548-9318-b97297b245f8)

Actual Values

![420180211-0b4bed37-2dea-4330-abcc-cfb278976f69](https://github.com/user-attachments/assets/1113439d-8305-4bfd-a686-d936ad6f462a)

Training Set :

![420180269-d8a0d3b8-d698-4bbf-a7d1-505f963a0734](https://github.com/user-attachments/assets/e4cfa84d-aedf-4d15-9ec5-3bc58e6e6eab)

Testing Set :

![420180321-83dc9c01-c442-4aaf-ad30-245eb1905a07](https://github.com/user-attachments/assets/88a41f37-235b-43e0-a4b7-fa6f3901ad60)

MSE, MAE and RMSE :

![420180926-c438fe10-89be-4e8c-9eda-0e436fbf83ca](https://github.com/user-attachments/assets/fe7e70da-fe97-4abe-a52a-11e8dd4aa53c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
