from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

spark = SparkSession.builder.getOrCreate()

data_path1="C:/Users/appal/Downloads/archive (2)/accepted_2007_to_2018q4.csv/"
data_path2="C:/Users/appal/Downloads/archive (2)/rejected_2007_to_2018q4.csv/"

file_path1=data_path1+"FINAL_DATA_FRAME.csv"
dataframe1=spark.read.format("csv").option("header","true").load(file_path1)
file_path2=data_path2+"Final_Data_Frame.csv"
dataframe2=spark.read.format("csv").option("header","true").load(file_path2)

df_sql = dataframe1.union(dataframe2)

pandas_DF = df_sql.toPandas()

pandas_DF = pandas_DF.dropna()

category= ['loan_grade','Employee_Title','Employee_length','Home_Ownership_Status','Employee_Verification_Status','Loan_Purpose','Loan_Title','Province_Code','Loan_Application_Type','Loan_Status','Bank_Name','Gender','Married_Status'] 
encoder= LabelEncoder()
for i in category:   
    pandas_DF[i] = encoder.fit_transform(pandas_DF[i])
        
X = pandas_DF.drop(columns=['Loan_Status','loan_id'],axis=1)
Y = pandas_DF['Loan_Status']
        # if pandas_DF.empty == False:
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=42)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_test)



with open('loan.pkl', 'wb') as f:
    pickle.dump(svc, f)

with open('loan.pkl', 'rb') as f:
    k = pickle.load(f)

c

