from flask import Blueprint,render_template,request, send_file
from fpdf import FPDF
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession

WIDTH = 210
HEIGHT = 297

def create_title(day, pdf):
  # Unicode is not yet supported in the py3k version; use windows-1252 standard font
  pdf.set_font('Arial', '', 24)  
  pdf.ln(60)
  pdf.write(5, f"Bank Details Report")
  pdf.ln(10)
  pdf.set_font('Arial', '', 16)
  pdf.write(4, f'{day}')
  pdf.ln(5)

bank_pred = Blueprint('bank_pred',__name__)

@bank_pred.route('/bank_preds',methods=["POST","GET"])
def bank_preds():
   return render_template("bank_prediction.html")

@bank_pred.route('/download',methods=["POST","GET"])
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

   if request.method == "POST":
      pdf = FPDF()
      bankname = request.form.get("bankname", type=str)

     #    pandas_DF = pandas_DF["Bank_Name" == bankname]
      print(bankname)
      pandas_DF = pandas_DF[pandas_DF.Bank_Name == 'CIBC']

      ''' First Page '''
      pdf.add_page()
      pdf.image("C:/Users/appal/OneDrive/Documents/ADT Web App/letterhead.png", 0, 0, WIDTH)
      create_title(bankname, pdf)

      svm = sns.countplot(x='Loan_Purpose',hue='Loan_Status',data=pandas_DF)
      svm.set_xticklabels(svm.get_xticklabels(), rotation=40, ha="right")
      figure = svm.get_figure()    
      figure.set_size_inches(14, 14)
      figure.savefig('countplotbank.png', dpi = 100)

      svm = sns.countplot(x='Employee_Verification_Status',hue='Loan_Status',data=pandas_DF)
      figure = svm.get_figure()    
      figure.savefig('countplot.png')


      pdf.image("countplotbank.png", 15, 90, WIDTH-10)

      pdf.output('C:/Users/appal/OneDrive/Documents/ADT Web App/bank_report.pdf', 'F')
      return send_file('C:/Users/appal/OneDrive/Documents/ADT Web App/bank_report.pdf',as_attachment=True)
   return render_template("bank_prediction.html")