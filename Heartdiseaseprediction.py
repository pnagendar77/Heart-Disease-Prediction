import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading heart.csv file
heart_data = pd.read_csv('/content/heart.csv')

heart_data.shape

#Information of Dataset
heart_data.info()

#Head and tail of Dataset
heart_data.head()
heart_data.tail()

#Describe function 
heart_data.describe()

#Check missing values in dataset
heart_data.isnull().sum()

heart_data['target'].value_counts()

#Splitting the features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print(Y)

#Training and Testing 
-> #Splitting the data into Training data and Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Model training - Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation 
-> #Accuracy Score on Training Data:
X_train_prediction = model.predict(X_train)
training_data_score = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score on training data is : ',training_data_score)

#Accuracy score on Testing Data
X_test_prediction = model.predict(X_test)
testing_data_score = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score on test data is : ',testing_data_score)

#Testing Data
input_data = (47,1,2,138,257,0,0,156,0,0,2,0,2) 
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

#Print the info
print(prediction)
if (prediction[0]==0):
  print('This person is not affected with any heart disease!')
else:
  print('This person is affected with heart disease!')
print("0 Represents the person with no disease & 1 Represents the person with Heart disease !")






