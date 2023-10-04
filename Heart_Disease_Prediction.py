import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND PROCESSING

# Loading the csv data to a pandas Dataframe

heart_data = pd.read_csv('D:\Files\Bhushan\My_WorkSpace\Machine_Learning _Projects\Heart_Disease_Prediction\heart.csv')

# print 1st 4 rows of the dataset
print(heart_data.head())


heart_data.tail()

heart_data.shape

# getting info about the data
heart_data.info()

# statisticaal measures about the data
heart_data.describe()
print(heart_data.describe())

# checking the distribution of target variable
heart_data['target'].value_counts()

'''  1 --> Defective Heart  '''
'''  0 --> Healthy Heart    '''

# Splitting the Features and Targets
X = heart_data.drop(columns = 'target', axis = 1)
Y = heart_data['target']

print(X)
print(Y)

# Splitting data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


                       # Model Training
                       # Logistic Regression

model = LogisticRegression()           

# training the logistic regression model with Training data
model.fit(X_train, Y_train)

                   
                       # Model Evaluation
                       # Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data : ' , training_data_accuracy)         

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data : ' , test_data_accuracy)


'''                   Building A Predictive System                  '''

input_data = (70,1,2,160,269,0,1,112,1,2.9,1,1,3)

# change a input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are presicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print('The person does not have a Heart Disease')

else:
  print('The person has Heart Disease')
