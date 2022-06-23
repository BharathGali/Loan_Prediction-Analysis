from flask import Flask

def create_app():
    app = Flask(__name__)

    from .views import views
    from .about import about
    from .team import team
    from .bank_pred import bank_pred
    from .loan_pred import loan_pred

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(bank_pred, url_prefix='/bank_prediction')
    app.register_blueprint(loan_pred, url_prefix='/loan_prediction')
    app.register_blueprint(about, url_prefix='/about')
    app.register_blueprint(team, url_prefix='/team')

    return app