

# Reference : code provided by Prof. Murphy 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics




df = pd.read_csv("Treasury_Squeeze_test_DS1.csv")                      # loading Treasury squeeze dataset using pandas

df = df.drop(["rowindex","contract"], axis = 1)

print ("information of the dataset \n\n" , df.info(),"\n\n")           # Information about the variables

print ("Top 5 observations of the dataset \n\n", df.head(),"\n\n")     # Top 5 observations of the dataset

X = df.drop(["squeeze"], axis = 1)                                     # Creating matrix of feature variables

y = df["squeeze"]                                                      # Series of target variable

print("Top 5 observations of X matrix \n\n", X.head(),"\n\n")

print("Dimension of X \n\n", X.shape,"\n\n")

print("Top 5 observations of y \n\n", y.head(),"\n\n")

print("Dimension of y \n\n", y.shape,"\n\n")


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 30) # Splitting Data set

print("Dimensions in the same order as above split \n\n", X_train.shape, X_test.shape, y_train.shape, y_test.shape,"\n\n") 

print("Head of X_train \n\n", X_train.head(),"\n\n")
print("Head of X_test \n\n", X_test.head(),"\n\n")
print("Head of y_train \n\n", y_train.head(),"\n\n")
print("Head of y_test \n\n", y_test.head(),"\n\n")

dtc = DecisionTreeClassifier(max_depth = 4)                                # Defining Model

dtc.fit(X_train,y_train)                                                   # Fitting the Model

y_pred_train_dtc = dtc.predict(X_train)                                    # Predicting on training set

y_pred_test_dtc  = dtc.predict(X_test)                                     # Predicting on test set

Accuracy_train_dtc = metrics.accuracy_score(y_train, y_pred_train_dtc)     # Accuracy
Accuracy_test_dtc = metrics.accuracy_score(y_test , y_pred_test_dtc)

print("Accuracy_train_dtc:",Accuracy_train_dtc,"\n")
print("Accuracy_test_dtc:",Accuracy_test_dtc,"\n")

print(metrics.classification_report(y_test, y_pred_test_dtc))              # Claasification Report

print(metrics.confusion_matrix(y_test, y_pred_test_dtc),"\n\n\n")          # Confusion Matrix


k_range = range(1,26)                                                      # KNN Classifier

scores_train = []
scores_test  = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(X_train,y_train)
    
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn  = knn.predict(X_test)
    
    scores_train.append(metrics.accuracy_score(y_train,y_pred_train_knn))
    scores_test.append(metrics.accuracy_score(y_test,y_pred_test_knn))

    
plt.plot(scores_train,"g",label = "training set accuracy KNN")
plt.plot(scores_test,"r", label = "test set accuracy KNN")
plt.legend()
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()


print("k value of 8 seems to be the best from the graph","\n\n")



print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




########### END ##############


