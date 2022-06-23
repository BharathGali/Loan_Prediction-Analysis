from flask import Blueprint,render_template

about = Blueprint('about',__name__)

@about.route('/abouts')
def abouts():
    return render_template("about.html")