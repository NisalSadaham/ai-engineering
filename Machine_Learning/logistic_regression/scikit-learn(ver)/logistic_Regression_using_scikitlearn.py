import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_score, recall_score

#2 features are used here x1 = study hours and x2 = pervious score
X = np.array([
    [1, 35],
    [2, 50],
    [3, 45],
    [4, 60],
    [5, 55],
    [2, 65],
    [6, 70],
    [4, 45],
    [7, 80],
    [5, 75],
    [3, 60],
    [6, 50],
    [8, 85],
    [2, 40],
    [7, 65]
])

#actual values
y = np.array([
    0, 0, 0, 1, 0,
    1, 1, 0, 1, 1,
    1, 0, 1, 0, 1
])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42) #splitting the data for training and testing the model
model = LogisticRegression() #creating an instance of the logistic regression model
model.fit(X=x_train,y=y_train) #fitting the training data into the model
predictions_class = model.predict(X=x_test) #making predictions on the test data
predictions_prob = model.predict_proba(X=x_test) #getting the probabilities of the predictions
accuracy = accuracy_score(y_test,predictions_class) #calculating the accuracy of the model
f1 = f1_score(y_test,predictions_class) #calculating the F1 score of the model
precision = precision_score(y_test,predictions_class) #calculating the precision of the model relability of positive predicitions
recall = recall_score(y_test,predictions_class) #calculating the recall of the model means how many are actually positive and how many are predicted positive ability to find actual positive cases
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions_class)
print(cm)


print("Cofficient:",model.coef_) #gives the value of the parameter w(weight)
print("Intercept:",model.intercept_)#gives the value of the parameter b(bias)
print("Model score:", model.score(X=x_test,y=y_test))
print("Predicted classes:", predictions_class)#predicted classes for the test data(1,0)
print("Predicted probabilities:", predictions_prob) #predicted probabilities for the test data(eg: 0.1, 0.9)
print("Accuracy:", accuracy) #printing the accuracy of the model
print("F1 Score:", f1) #printing the F1 score of the model
print("Precision:", precision) #printing the precision of the model
print("Recall:", recall) #printing the recall of the model

x_new = np.array([[5, 70]]) #new data for prediction
new_predictions = model.predict(X=x_new) #making prediction on the new data
new_probabilities = model.predict_proba(X=x_new) #getting the probabilities of the new predictions
print("Predictions for new data:", new_predictions)
print("Probabilities for new data:", new_probabilities)