from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from yourheartai_api.config import Config

from flask_graphql import GraphQLView
# from .api import Api
from flask_smorest import Api, Blueprint
# from schema import schemas
from flask_cors import CORS
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'



mail = Mail()
cors = CORS(resources={r'/api/yha/*': {'origins':'http://localhost:4401'}})
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    cors.init_app(app)
    jwt.init_app(app)

    from .models import User

    @login_manager.user_loader #user loader cookie & login
    def load_user(user_id):
        return User.query.get(int(user_id))

    from yourheartai_api.users.routes import users
    from yourheartai_api.posts.routes import posts
    from yourheartai_api.main.routes import main
    from yourheartai_api.errors.handlers import errors
    _api = Api(app)
    _api.register_blueprint(users)
    _api.register_blueprint(posts)
    _api.register_blueprint(main)
    _api.register_blueprint(errors)

    # app.register_blueprint(users)
    # app.register_blueprint(posts)
    # app.register_blueprint(main)
    # app.register_blueprint(errors)

    app.add_url_rule(
        "/graphql",
        # view_func=GraphQLView.as_view("graphql", schema=schema, graphiql=True),
        view_func=GraphQLView.as_view("graphql", graphiql=True),
    )

    return app
    