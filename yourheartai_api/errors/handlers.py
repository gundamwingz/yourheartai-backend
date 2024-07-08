from flask_smorest import Blueprint

errors = Blueprint('errors', __name__, url_prefix="/api/yha")


@errors.app_errorhandler(404)
def error_404(error):
    return "404",404

@errors.app_errorhandler(403)
def error_403(error):
    return "403",403

@errors.app_errorhandler(500)
def error_500(error):
    return "500",500