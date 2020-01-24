# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

predict_result={"GK" : "The player is suitable tp be a goal keeper",
                "ST" : "The player is suitable tp be a striker",
                "DEF" : "The player is suitable tp be a defender",
                "MID" : "The player is suitable tp be a mid-fielder"}

balance_data = pd.read_csv("fitindata.csv") 
# Printing the dataswet shape 
print ("Dataset Length: ", len(balance_data)) 
print ("Dataset Shape: ", balance_data.shape) 
	
# Printing the dataset obseravtions 
print ("Dataset: ",balance_data.head()) 
X = balance_data.values[:, 0:5] 
Y = balance_data.values[:, 5] 
print(X)
# Spliting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 

#Creating a gini based model
model = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

# Actual training process 
model.fit(X_train, y_train) 

# Predicton on testdata 
y_pred = model.predict(X_test) 
print("Predicted values:") 
print(y_pred)

print("Confusion Matrix:\n ", confusion_matrix(y_test, y_pred)) 
	
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
	
print("Report : ", classification_report(y_test, y_pred))

print("Enter the Following0 out of 100")

a=[]
a.append(int(input("Acceleration :")))
a.append(int(input("Ball Control :")))
a.append(int(input("Finishing :")))
a.append(int(input("GK Diving :")))
a.append(int(input("Aggression :")))
test_ip=[]
test_ip.append(a)

test_pred=model.predict(test_ip)

print(predict_result[test_pred[0]])