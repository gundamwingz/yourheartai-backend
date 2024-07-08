# settings.py
from os import environ

from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    SECRET_KEY = environ.get('SECRET_KEY_YHA')
    JWT_SECRET_KEY = environ.get('JWT_SECRET_KEY_YHA')
    SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
    MAIL_SERVER = environ.get('MAIL_SERVER_YHA')
    MAIL_PORT = environ.get('MAIL_PORT_YHA')
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    MAIL_DEBUG = True
    MAIL_USERNAME = environ.get('EMAIL_USER_YHA')
    MAIL_PASSWORD = environ.get('EMAIL_PASS_YHA')
    MAIL_DEFAULT_SENDER = environ.get('EMAIL_USER_YHA')
    MAIL_MAX_EMAILS = 5
    MAIL_SUPPRESS_SEND = False
    MAIL_ASCII_ATTACHMENTS = False

    API_TITLE = "yourheartai_api"
    API_VERSION = "v1"
    OPENAPI_VERSION = "3.0.3"
    OPENAPI_JSON_PATH = "api-spec.json"
    OPENAPI_URL_PREFIX = "/"
    OPENAPI_RAPIDOC_PATH = "/api/rapidoc"
    OPENAPI_RAPIDOC_URL = "https://unpkg.com/rapidoc/dist/rapidoc-min.js"
    OPENAPI_RAPIDOC_CONFIG = {"show-header": "false", "theme": "light"}