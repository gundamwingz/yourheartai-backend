import os
import json
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
                                                YHAPatientDataSchema
                                                )

ai_models = Blueprint('ai_models', __name__, url_prefix="/api/yha")

@ai_models.route("/ai-prediction/cvd-mrcnn", methods=["POST"])
class GetCvdCnnPrediction(MethodView):
    @jwt_required()
    ## TODO: Update AI Schemas to match incoming data etc...
    # @ai_models.arguments(YHANewPostSchema, location="query")    
    def post(self):
        print("Routing working")
        
        # ai_img_raw_dir = "./yourheartai_api/static/ai_images/chd-mrcnn/raw"
        # ai_img_res_dir = "./yourheartai_api/static/ai_images/chd-mrcnn/results"
        # if not os.path.exists(ai_img_raw_dir):
        #     os.makedirs(ai_img_raw_dir)
        # if not os.path.exists(ai_img_res_dir):
        #     os.makedirs(ai_img_res_dir)

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
                path = current_app.root_path+'/static/ai_images/chd-mrcnn/raw'


                date_time = now.strftime("%Y%m%d_%H%M%S")
                print("date and time:",date_time)
            
                f_name, f_ext = os.path.splitext(aiFilename) #underscore used to discard variable
                aiFilename = f_name +"-"+ date_time + f_ext

                # aiFilename = aiFilename + date_time

                isExist = os.path.exists(path)
                print("path: ",path)
                if not isExist:
                    print("makine rcnn dir")
                    os.makedirs(path)
                elif isExist:                
                    file.save(os.path.join(current_app.root_path, 'static/ai_images/chd-mrcnn/raw', aiFilename))


                print("getStenosisPrediction: getting prediction!")
                stenPred =  getStenosisPrediction(aiFilename)
                print("f_ext",f_ext)
                finalStenPred = stenPred+f_ext
                print("stenPred w_ext: ",finalStenPred)

                image_file = url_for('static',filename='/ai_images/chd-mrcnn/results/'+stenPred+".jpg")    

                # image_file = url_for('static', filename='/ai_images/chd-mrcnn/raw/' + aiFilename)
                
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


        print("current_app: ",os.path.join(current_app.root_path))
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
                path = current_app.root_path+'/static/ai_images/cancer'


                date_time = now.strftime("%Y%m%d_%H%M%S")
                print("date and time:",date_time)
            
                f_name, f_ext = os.path.splitext(aiFilename) #underscore used to discard variable
                aiFilename = f_name +"-"+ date_time + f_ext

                # aiFilename = aiFilename + date_time
                print("Testing Cancerdir")
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
       
@ai_models.route("/prediction/chd", methods=["GET","POST"])
class UserAccount(MethodView):
    @jwt_required()
    def get(self):    
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(id=current_user_id).first() 
        current_user_jwt = user  

        ##TODO: Database needs to be updated to include patient one,
        # where it stores latest patient results     
        id  = current_user_jwt.id
        age = current_user_jwt.age
        gender = current_user_jwt.gender
        chestPain = current_user_jwt.chestPain
        restingBP = current_user_jwt.restingBP
        serumCholestrol = current_user_jwt.serumCholestrol
        fastingBloodSugar = current_user_jwt.fastingBloodSugar
        restingRElectro = current_user_jwt.restingRElectro
        maxHeartRate = current_user_jwt.maxHeartRate
        exerciseAngia = current_user_jwt.exerciseAngia
        oldPeak = current_user_jwt.oldPeak
        slope = current_user_jwt.slope
        noOfMajorVessels = current_user_jwt.noOfMajorVessels
        ##TODO: Crate Patient Database needs

        myjson = {
                    'id': id,
                    'age': age,
                    'gender': gender,
                    'chestPain' : chestPain,
                    'restingBP'  : restingBP,
                    'serumCholestrol'  : serumCholestrol,
                    'fastingBloodSugar'  : fastingBloodSugar,
                    'restingRElectro'  : restingRElectro,
                    'maxHeartRate'  : maxHeartRate,
                    'exerciseAngia'  : exerciseAngia,
                    'oldPeak'  : oldPeak,
                    'slope': slope,
                    'noOfMajorVessels': noOfMajorVessels,          
                }
        return jsonify(myjson), 200
     
    @ai_models.arguments(YHAPatientDataSchema, location="json")
    @jwt_required()
    def post(self, args): 
        data = args["data"]        
        data = json.loads(json.dumps(data))  

        age = data["age"]
        gender = data["gender"]
        chestPain = data["chestPain"]
        restingBP = data["restingBP"]
        serumCholestrol = data["serumCholestrol"]
        fastingBloodSugar = data["fastingBloodSugar"]
        restingRElectro = data["restingRElectro"]
        maxHeartRate = data["maxHeartRate"]
        exerciseAngia = data["exerciseAngia"]
        oldPeak = data["oldPeak"]
        slope = data["slope"]
        noOfMajorVessels = data["noOfMajorVessels"]

        current_user_id = get_jwt_identity()
        user = User.query.filter_by(id=current_user_id).first()
        current_user_jwt = user

        getCHDPrediction(data)

        ## TODO: Code below is to commit latest Patient data to db. Currently not implemented
        try:
            #Sets db details for match check
            current_user_jwt_email = current_user_jwt.email
            current_user_jwt_username = current_user_jwt.username     

            current_user_jwt.id = id
            current_user_jwt.age =  age
            current_user_jwt.gender = gender
            current_user_jwt.chestPain = chestPain
            current_user_jwt.restingBP = restingBP
            current_user_jwt.serumCholestrol = serumCholestrol
            current_user_jwt.fastingBloodSugar = fastingBloodSugar
            current_user_jwt.restingRElectro = restingRElectro
            current_user_jwt.maxHeartRate = maxHeartRate
            current_user_jwt.exerciseAngia = exerciseAngia
            current_user_jwt.oldPeak = oldPeak
            current_user_jwt.slope = slope
            current_user_jwt.noOfMajorVessels = noOfMajorVessels 
            
            # db.session.commit()# no database to commit
            return jsonify("CHD Prediction"), 200
        except:
            return jsonify("Failed"), 500