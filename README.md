# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Load the Dataset
3. Preprocess the Data
4. Split the Data
5. Build and Train the Model
6. Evaluate the Model
7. Stop

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Visalan H
RegisterNumber:  212223240183
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
y = data['left']
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(x_train,y_train)
y_predict = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_predict)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
### Data.head():
![image](https://github.com/user-attachments/assets/03eb4e62-2da8-4d20-800b-d6e66d384299)

### Data.info() and Data.isnull().sum():
![image](https://github.com/user-attachments/assets/c0ffc608-4c24-4c8b-8879-20dfd9149bbd)
![image](https://github.com/user-attachments/assets/fbfa1f1c-2e54-4dc5-9f7d-c8a53c7dcd28)

![image](https://github.com/user-attachments/assets/c8c979e9-e519-4a75-84c4-a32db90b89d8)

### Label Encoder :
![image](https://github.com/user-attachments/assets/8cdaef34-74fb-4a62-870d-b337fa126e98)

### y.head():
![image](https://github.com/user-attachments/assets/cd54a54d-e44b-497f-bf61-91c5f19f6b22)
### Accuracy:
![image](https://github.com/user-attachments/assets/5d5d780c-feb7-467d-8e82-8f232b00bb9b)
### Prediction:
![image](https://github.com/user-attachments/assets/f71bbb7a-a7c6-4414-9937-b80db8e170db)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
