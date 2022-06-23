from flask import Blueprint, render_template

loan_pred = Blueprint('loan_pred',__name__)

@loan_pred.route('/loan_preds')
def loan_preds():
    return render_template("loan_prediction.html")