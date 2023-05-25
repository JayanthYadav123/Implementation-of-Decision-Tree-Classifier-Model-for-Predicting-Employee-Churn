# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2. Read the data set.

3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4. Determine training and test data set.

5. Apply decision tree Classifier and get the values of accuracy and data prediction.


## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: G.Jayanth.
RegisterNumber: 212221230030. 
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
1.df.head()

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/8eeb10fa-42f2-4533-852c-3f6579f580ad)

2. df.info()

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/5e882f38-79a9-4b60-823a-cd78bffc3102)

3. Null values

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/b88b201f-20b7-40ae-93c5-7665a5e8d4c0)

4. value_count() for left data

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/569a4e56-c888-4098-9503-554ded8b7b33)

5. data.head() for salary

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/7a443190-24bf-45ae-9ff0-92a76c8d780a)

6. x.head()
 
![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/0a379a53-e0ca-40cf-8f37-5e80dbfcfc47)

7. Accuracy value
 
![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/cd37994d-8107-4e94-b28b-d4f873dfdf69)

8. prediction value

![image](https://github.com/JayanthYadav123/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94836154/97bff433-e107-4570-83fa-cb70ed7129a6)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
