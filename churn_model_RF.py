import pandas as pd
#import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix, classification_report



data = pd.read_excel(r'C:\Users\harsh\Documents\BEPEC Notebooks\Datasets\Churn_data.xlsx')

data_r = data.drop(['CIF','CUS_DOB','CUS_Customer_Since'], axis = 1)

data_r['CUS_Gender'] = data_r['CUS_Gender'].ffill()
data_r['CUS_Month_Income'] = data_r['CUS_Month_Income'].ffill()

#Feature Engineerind technique
label = LabelEncoder()
data_r['CUS_Gender'] = label.fit_transform(data_r['CUS_Gender'])
data_r['CUS_Marital_Status'] = label.fit_transform(data_r['CUS_Marital_Status'])
data_r['TAR_Desc'] = label.fit_transform(data_r['TAR_Desc'])
data_r['Status'] = label.fit_transform(data_r['Status'])

#ACTIVE - 0 
#CHURN -1

X = data_r.iloc[:,:24]
y = data_r.iloc[:,24:]

#Feature Scaling technique
scale = MinMaxScaler()
X = scale.fit_transform(X)

#Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
model = RF(n_estimators = 10, max_depth = 3)
model.fit(X_train,Y_train)

print(model)

#Model prediction
predict = model.predict(X_test)


confusion_matrix(Y_test, predict)
classification_report(Y_test, predict)

import pickle

pickle.dump(model, open("churn.pkl","wb"))

model_churn = pickle.load(open('churn.pkl',"rb"))

print(model.predict([[16,200000,'FEMALE','SINGLE',14,3,0,0,8693.6,\
                      0,0,0,0,0,0,0,0,8693.6,3,0,0,3,2223,'LOW']]))

