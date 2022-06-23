from flask import Blueprint, render_template

team = Blueprint('team',__name__)

@team.route('/teams')
def teams():
    return render_template("team.html")