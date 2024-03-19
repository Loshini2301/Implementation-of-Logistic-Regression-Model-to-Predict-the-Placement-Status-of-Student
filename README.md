# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
### NAME:LOSHINI.G
### REFERENCE NUMBER:212223220051
### DEPARTMENT:IT

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOSHINI.G
RegisterNumber:  212223220051
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-03-19 085939](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/5853fef5-511d-43c9-b7cc-7637d17e5bd6)
![Screenshot 2024-03-19 090000](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/df606171-4d82-4c37-9961-6be4d7dc414e)
![Screenshot 2024-03-19 090020](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/78f9c39c-4b81-4f16-b448-37af5fb3c4b7)
![Screenshot 2024-03-19 090053](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/7c14cf3d-34c6-46ee-b84c-24b0377c6f13)
![Screenshot 2024-03-19 090113](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/67dde93c-0fc7-465b-a62d-1debb85c1f92)
![Screenshot 2024-03-19 090128](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/71ea15f6-cdea-4428-bdf6-227af97f1117)
![Screenshot 2024-03-19 090203](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/874eedc9-d071-4d85-8d0c-8687e4897cc2)
![Screenshot 2024-03-19 090220](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/f58b971e-5ac6-4a3e-8b72-c022c1d96cb4)
![Screenshot 2024-03-19 085901](https://github.com/Loshini2301/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150007305/08f51c38-26ee-46c6-bda7-9c549c0c8970)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
