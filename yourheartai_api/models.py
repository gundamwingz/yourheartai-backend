from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer
from datetime import datetime
from yourheartai_api import db, login_manager
from flask_login import UserMixin
from flask import current_app

# from sqlalchemy import (
#     create_engine,
#     Column,
#     Integer,
#     String,
#     DateTime,
#     func,
#     ForeignKey,
# )
# from sqlalchemy.orm import scoped_session, sessionmaker, relationship, backref
# from sqlalchemy.ext.declarative import declarative_base

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    firstName = db.Column(db.String(60), nullable=True)
    lastName = db.Column(db.String(60), nullable=True)
    role = db.Column(db.String(60), nullable=True)
    company = db.Column(db.String(60), nullable=True)
    address = db.Column(db.String(60), nullable=True)
    city = db.Column(db.String(60), nullable=True)
    country = db.Column(db.String(60), nullable=True)
    postCode = db.Column(db.String(60), nullable=True)
    aboutMe = db.Column(db.Text, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)   
    
    def get_reset_token(self):
        s = Serializer(current_app.config['SECRET_KEY'])                
        return s.dumps({'user_id': self.id})

    @staticmethod
    def verify_reset_token(token, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY']) 
        try:
            user_id = s.loads(token, expires_sec)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}','{self.date_posted}')"


# engine = create_engine(
#     f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}&MARS_Connection={mars}&ConnectRetryCount=5",
#     convert_unicode=True,
#     echo=True,
#     pool_pre_ping=True,
# )
# db_session = scoped_session(
#     sessionmaker(autocommit=False, autoflush=False, bind=engine)
# )


# Base = declarative_base()
# # We will need this for querying
# Base.query = db_session.query_property()
   

# class Document(Base):
#     __tablename__ = "OAM_Content"
#     __table_args__ = {"schema": "pcp"}
#     id = Column(Integer, primary_key=True)
#     uuid = Column(String)
#     description = Column(String)
#     content = Column(String)
#     lockedBy = Column(String)
#     tags = Column(String, default="[]")
#     active = Column(Integer, default=1)
#     type = Column(String)
#     contains = Column(String)
#     state = Column(String, default="Draft")
#     version = Column(Integer, default=1)
#     revision = Column(Integer, default=0)
