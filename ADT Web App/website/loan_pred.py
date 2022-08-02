from flask import Blueprint, render_template, url_for, request, send_file, render_template_string
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from fpdf import FPDF
from datetime import datetime, timedelta
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

model = pickle.load(open('C:/Users/appal/OneDrive/Documents/ADT Web App/ML_Model1.pkl', 'rb'))

WIDTH = 210
HEIGHT = 297

def create_title(day, pdf):
  # Unicode is not yet supported in the py3k version; use windows-1252 standard font
  pdf.set_font('Arial', '', 24)  
  pdf.ln(60)
  pdf.write(5, f"Loan Prediction report")
  pdf.ln(10)
  pdf.set_font('Arial', '', 16)
  pdf.write(4, f'{day}')
  pdf.ln(5)

loan_pred = Blueprint('loan_pred',__name__)

@loan_pred.route('/loan_preds',methods=["POST","GET"])
def loan_preds():
    return render_template("loan_prediction.html")

@loan_pred.route('/download',methods=["POST","GET"])
def download():
    spark = SparkSession.builder.getOrCreate()

    data_path1="C:/Users/appal/Downloads/archive (2)/accepted_2007_to_2018q4.csv/"
    data_path2="C:/Users/appal/Downloads/archive (2)/rejected_2007_to_2018q4.csv/"

    file_path1=data_path1+"FINAL_DATA_FRAME12.csv"
    dataframe1=spark.read.format("csv").option("header","true").load(file_path1)
    file_path2=data_path2+"Final_Data_Frame12.csv"
    dataframe2=spark.read.format("csv").option("header","true").load(file_path2)

    df_sql = dataframe1.union(dataframe2)

    pandas_DF = df_sql.toPandas()

    # emp_pandas = pd.read_csv("C:/Users/appal/OneDrive/Documents/emp_details.csv")
    # emp_pandas = emp_pandas.to_dict()

    if request.method == "POST":

        pdf = FPDF()

        LoanAmount = request.form.get("LoanAmount", type=int)
        LoanT = request.form.get("LoanTerm",type=int)
        LoanG = request.form.get("LoanGrade",type=int)
        HOS = request.form.get("HOS",type=int)
        ApplicantIncome = request.form.get("ApplicantIncome", type=int)
        LoanP = request.form.get("LoanPurpose",type=int)
        emp = request.form.get("EmployeeTitle",type=int)
        csr = request.form.get("credit",type=int)
        Province = request.form.get("Province",type=int)
        Gender = request.form.get("gender",type=int) 
        married = request.form.get("married",type=int)
        Employee_length = request.form.get("length", type=int)
        Employee_Verification_Status = request.form.get("verify", type=int)
        Loan_Application_Type = request.form.get("type", type=int)
        Bank_Name = request.form.get("bank", type=int)
        
        #HOS,Province,Gender
        # loan_amnt	loan_term	loan_grade	Employee_Title	Employee_length	Home_Ownership_Status	Annual_Income	Employee_Verification_Status	Loan_Purpose	Loan_Title	Province_Code Loan_Application_Type	Bank_Name	Gender	Married_Status	Credit_Score
        features = [[LoanAmount, LoanT, LoanG, emp, Employee_length, HOS,ApplicantIncome,Employee_Verification_Status,LoanP,LoanP,Province,Loan_Application_Type,Bank_Name,Gender,married,csr]]
        result = model.predict(features)

        if result == 1 :
            op = "Congrats Your Loan is Approved!!!"
        else:
            op = "Sorry your loan is Rejected!!!"
        
        ''' First Page '''
        pdf.add_page()
        pdf.image("C:/Users/appal/OneDrive/Documents/ADT Web App/letterhead.png", 0, 0, WIDTH)
        create_title(op, pdf)

        svm = sns.stripplot(data=pandas_DF, x="Bank_Name", y="Loan_Purpose", hue="Loan_Status")
        svm.set_xticklabels(svm.get_xticklabels(), rotation=40, ha="right")
        figure = svm.get_figure()    
        figure.set_size_inches(14, 14)
        figure.savefig('stripplot.png', dpi = 100)

        # svm = sns.countplot(x='Employee_Verification_Status',hue='Loan_Status',data=pandas_DF)
        # figure = svm.get_figure()    
        # figure.savefig('countplot.png')


        pdf.image("stripplot.png", 15, 90, WIDTH-10)
        #pdf.image("countplot.png", 45, 200, WIDTH/2-10)

        pdf.output('C:/Users/appal/OneDrive/Documents/ADT Web App/Prediction_report.pdf', 'F')
        pdf.close()
        return send_file('C:/Users/appal/OneDrive/Documents/ADT Web App/Prediction_report.pdf',as_attachment=True)
    return render_template("loan_prediction.html")
    
