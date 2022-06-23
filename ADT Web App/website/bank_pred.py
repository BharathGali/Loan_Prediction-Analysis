from flask import Blueprint,render_template

bank_pred = Blueprint('bank_pred',__name__)

@bank_pred.route('/bank_preds')
def bank_preds():
     return render_template("bank_prediction.html")