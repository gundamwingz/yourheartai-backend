import os
from flask import (jsonify, request)
from flask_login import current_user
from yourheartai_api import db
from yourheartai_api.ai_models.utils import getCancerPrediction, getStenosisPrediction, getCHDPrediction
from yourheartai_api.models import Post, User
from datetime import datetime

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

@ai_models.route("/ai-prediction/cvd-mrcnn", methods=["POST"])
class GetCvdCnnPrediction(MethodView):
    @jwt_required()
    ## TODO: Update AI Schemas to match incoming data etc...
    # @ai_models.arguments(YHANewPostSchema, location="query")    
    def post(self):
        print("Routing working")
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(id=current_user_id).first() 
        current_user_jwt = user 
        now = datetime.now() # current date and time

        ########## FIXME: Random Boiler Plate Code Begins ##########

        #FIXME: need to commit result to db

        # title = args["title"]
        # content = args["content"]
        # post = Post(title=title, content=content, author=current_user)
        # db.session.add(post)
        # db.session.commit()   

        ########## Random Boiler Plate Code Ends ##########
   
        try:            
            if 'file' not in request.files:
                response = jsonify('No file part')
                return response, 403
            file = request.files['file']
            if file.filename == '':
                response = jsonify('No file selected for uploading')
                return response, 403
            if file:
                aiFilename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
                path = current_app.root_path+'\\static\\ai_images\\chd-mrcnn\\raw'


                date_time = now.strftime("%Y%m%d_%H%M%S")
                print("date and time:",date_time)
            
                f_name, f_ext = os.path.splitext(aiFilename) #underscore used to discard variable
                aiFilename = f_name +"-"+ date_time + f_ext

                # aiFilename = aiFilename + date_time

                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                elif isExist:                
                    file.save(os.path.join(current_app.root_path, 'static/ai_images/chd-mrcnn/raw', aiFilename))

                #getCancerPrediction(filename)
                # label = getCancerPrediction(aiFilename)
                stenPred =  getStenosisPrediction(aiFilename)

                image_file = url_for('static', filename='/ai_images/chd-mrcnn/raw/' + aiFilename)
                
                resultsJson = {
                    # "prediction": label,
                    "image_url": image_file
                }
                                
                response = jsonify(resultsJson)
                return response, 200
        except:
            return jsonify("Failed"), 500
@ai_models.route("/prediction/cancer", methods=["POST"])
class GetCancerPrediction(MethodView):
    @jwt_required()
    def post(self):
        now = datetime.now() # current date and time
        try:            
            if 'file' not in request.files:
                response = jsonify('No file part')
                return response, 403
            file = request.files['file']
            if file.filename == '':
                response = jsonify('No file selected for uploading')
                return response, 403
            if file:
                aiFilename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
                path = current_app.root_path+'\\static\\ai_images\\cancer'


                date_time = now.strftime("%Y%m%d_%H%M%S")
                print("date and time:",date_time)
            
                f_name, f_ext = os.path.splitext(aiFilename) #underscore used to discard variable
                aiFilename = f_name +"-"+ date_time + f_ext

                # aiFilename = aiFilename + date_time

                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                elif isExist:                
                    file.save(os.path.join(current_app.root_path, 'static/ai_images/cancer', aiFilename))

                #getCancerPrediction(filename)
                label = getCancerPrediction(aiFilename)

                image_file = url_for('static', filename='ai_images/cancer/' + aiFilename)
                
                resultsJson = {
                    "prediction": label,
                    "image_url": image_file
                }
                                
                response = jsonify(resultsJson)
                return response, 200
        except:
            return jsonify("Failed"), 500
        
@ai_models.route("/prediction/cancer", methods=["POST"])
class GetCHDPrediction(MethodView):
    @jwt_required()
    def post(self):
        now = datetime.now() # current date and time

        return
           
