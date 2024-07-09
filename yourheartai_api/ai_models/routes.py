import os
from flask import (jsonify, request)
from flask_login import current_user
from yourheartai_api import db
from yourheartai_api.ai_models.utils import getPrediction
from yourheartai_api.models import Post

from flask import url_for, current_app
from flask_smorest import Blueprint, abort
from marshmallow import fields
from flask.views import MethodView
from flask import request
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from werkzeug.utils import secure_filename
from yourheartai_api.ai_models.AISchema import (
    YHANewPostSchema, 
    )

ai_models = Blueprint('ai_models', __name__, url_prefix="/api/yha")

@ai_models.route("/prediction/cancer", methods=["POST"])
class GetPrediction(MethodView):
    @jwt_required()

    ## TODO: Update AI Schemas to match incoming data etc...
    # @ai_models.arguments(YHANewPostSchema, location="query")
    def post(self, args):
        ########## FIXME: Random Boiler Plate Code Begins ##########

        title = args["title"]
        content = args["content"]
        post = Post(title=title, content=content, author=current_user)
        db.session.add(post)
        # db.session.commit()   

        ########## Random Boiler Plate Code Ends ##########
        
        ## FIXME: this may not work for getting image from Angular!
        if request.method == 'POST': 
            if 'file' not in request.files:
                response = jsonify('No file part')
                return response, 403
            file = request.files['file']
            if file.filename == '':
                response = jsonify('No file selected for uploading')
                return response, 403
            if file:
                filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
                ## TODO: sort out path name for saving
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                file.save(os.path.join(current_app.root_path, 'static/ai_images/cancer', filename))

                #getPrediction(filename)
                label = getPrediction(filename)
                response = jsonify(label)

                ## TODO: sort out path name for saving
                full_filename = os.path.join(current_app.root_path, 'static/ai_images/cancer', filename)
                response = jsonify(full_filename)
                return response, 200

           
