
import base64
import json
import os
import time
# from tkinter import Image
import uuid
from PIL import Image

from flask import render_template, url_for, flash, redirect, request, jsonify, make_response

from flask_login import login_user, current_user, logout_user, login_required
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required

from yourheartai_api import db, bcrypt
from yourheartai_api.models import User, Post
from yourheartai_api.users.forms import ResetPasswordForm
from yourheartai_api.users.utils import save_picture, send_reset_email

from flask_smorest import Blueprint, abort
from marshmallow import fields
from flask.views import MethodView

from werkzeug.datastructures import ImmutableMultiDict

from yourheartai_api.users.UserSchema import (
                                        YHALoginSchema, 
                                        YHARegisterDataSchema, 
                                        YHAUserAccountDataSchema, 
                                        YHAPasswordResetSchema, 
                                        YHAPasswordResetURLSchema,
                                        YHAPasswordResetTokenSchema,
                                        YHAUserPostsSchema,
                                        YHALogoutSchema ,
                                        YHAPasswordUpdateDataSchema
                                        )


users = Blueprint('users', __name__, url_prefix="/api/yha")

@users.route("/users/login", methods=["GET", "POST"])
class Login(MethodView):
    @users.arguments(YHALoginSchema, location="query") 
    def post(self, args):      
        email = args["email"]
        password = args["password"]
        # remember_bool = args["remember_bool"]
        try:
            # this code manages when user acceses login & register pages.
            if current_user.is_authenticated:
                return "Already Logged In", 200 

            if "@" in email:
                print("containts @ is email: ", email)
                user = User.query.filter_by(email=email).first()
                remember_me = True #if remember_bool==True else False
            else:
                print("is username: ", email)
                user = User.query.filter_by(username=email).first()
                remember_me = True #if remember_bool==True else False                

            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=remember_me) 

                access_token = create_access_token(identity=user.id,expires_delta=False)
                postsJson = {
                    "id": user.id,
                    "username": user.username,
                    "firstName": user.firstName,
                    "lastName": user.lastName,
                    "email": user.email,
                    "user.image_file": user.image_file,
                    "role": user.role,
                    "isLoggedIn": "true",
                    "token": access_token
                }        
                response = jsonify(postsJson)  
                return response, 200
            else:
                response = jsonify({'message':'Login Unssuccessful. Incorrect email or password'})
                return response, 401 
        except:            
            return jsonify("Failed"), 404
        
@users.route("/users/logout")
class Logout(MethodView): 
    @users.arguments(YHALogoutSchema, location="query")   
    def get(self, args):
        isLoggedIn = args["isLoggedIn"] 
        print("isLoggedIn",isLoggedIn)        
        try:
            if isLoggedIn == 'true':
                logout_user()
                return jsonify("Logged Out") , 200
            elif isLoggedIn == 'false':
                return jsonify("Not Logged In") , 200
        except:
            return jsonify("Failed"),404

def checkEmailExists(newUserEmail, currentUserEmail):
    print("Triggered, newUserEmail: ",newUserEmail) 
    try:
        if newUserEmail == currentUserEmail:
            return False
        newUser = User.query.filter_by(email=newUserEmail).first()
        print("Passed newUser: ",newUser )
        if newUser:
            return True
        ...
    except:
        print("Failed")
        return True 
    return False

def checkUsernameExists(newUserUsername, currentUserUsername):
    print("Tirggered newUsername Check:",newUserUsername) 
    try:
        if newUserUsername == currentUserUsername:
            return False
        newUser = User.query.filter_by(username=newUserUsername).first()
        print("Passed, newUser: ",newUser)
        if newUser:
            return True
        ...
    except:
        print("Failed")
        return True 
    return False

@users.route("/users/register", methods=["GET", "POST"])
class Register(MethodView):
    @users.arguments(YHARegisterDataSchema, location="json")
    def post(self, args): 
        data = args["data"]        
        data = json.loads(json.dumps(data))        

        username = data['username']
        firstName = data['firstName']
        lastName = data['lastName']
        role = data['role']
        company = data['company']
        address = data['address']
        city = data['city']
        country = data['country']
        postCode = data['postCode']
        aboutMe = data['aboutMe']
        email = data['email']
        password = data['password']

        try:
            if current_user.is_authenticated:
                return "Already Logged In", 200  
            
            user = User.query.filter_by(email=email).first()
            userUsername = User.query.filter_by(username=username).first()
            
            if username == "" or email == "":
                response = jsonify("Username and or Email can't be blank.")                
                return response, 409                

            if userUsername:
                response = jsonify("Username already taken.")
                print("Email address already exists.")
                return response, 409
            if user:
                response = jsonify("Email address already exists.")
                print("Email address already exists.")
                return response, 409
                       
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password) #expand User to include new fields on angular registration page
            db.session.add(user)
            db.session.commit()
            response = jsonify("New account created")
            return response, 200
        except:
            return jsonify("Failed"), 404

@users.route("/users/account", methods=["GET","POST"])
class UserAccount(MethodView):
    @jwt_required()
    def get(self):    
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(id=current_user_id).first() 
        current_user_jwt = user        
        id  = current_user_jwt.id

        username  = current_user_jwt.username        
        firstName = current_user_jwt.firstName
        lastName = current_user_jwt.lastName
        role  = current_user_jwt.role
        company  = current_user_jwt.company
        address  = current_user_jwt.address
        city  = current_user_jwt.city
        country  = current_user_jwt.country
        postCode  = current_user_jwt.postCode
        aboutMe  = current_user_jwt.aboutMe
        email = current_user_jwt.email
        
        myjson = {
                    'id': id,
                    'username': username,
                    'firstName': firstName,
                    'lastName' : lastName,
                    'role'  : role,
                    'company'  : company,
                    'address'  : address,
                    'city'  : city,
                    'country'  : country,
                    'postCode'  : postCode,
                    'aboutMe'  : aboutMe,
                    'email': email,
                    'image': [],          
                }
        return jsonify(myjson), 200
     
    @users.arguments(YHAUserAccountDataSchema, location="json")
    @jwt_required()
    def post(self, args): 
        data = args["data"]        
        data = json.loads(json.dumps(data))  

        username = data["username"]
        firstName = data["firstName"]
        lastName = data["lastName"]
        role = data["role"]
        company = data["company"]
        address = data["address"]
        city = data["city"]
        country = data["country"]
        postCode = data["postCode"]
        email = data["email"]
        aboutMe = data["aboutMe"]

        current_user_id = get_jwt_identity()
        user = User.query.filter_by(id=current_user_id).first()
        current_user_jwt = user
        try:
            #Sets db details for match check
            current_user_jwt_email = current_user_jwt.email
            current_user_jwt_username = current_user_jwt.username

            if checkEmailExists(email, current_user_jwt_email):
                print("checkEmailExists Triggered")
                response = jsonify("Email is Already Taken!")             
                return response, 409
            
            if checkUsernameExists(username, current_user_jwt_username):
                print("checkUsernameExists Triggered")
                response = jsonify("Username is Already Taken!")                
                return response, 409

            current_user_jwt.username = username 
            current_user_jwt.firstName = firstName
            current_user_jwt.lastName = lastName
            current_user_jwt.role = role
            current_user_jwt.company = company
            current_user_jwt.address = address
            current_user_jwt.city = city
            current_user_jwt.country = country
            current_user_jwt.postCode = postCode
            current_user_jwt.aboutMe = aboutMe                    
            current_user_jwt.email = email

            db.session.commit()
            return jsonify("Your account has been updated"), 200
        except:
            return jsonify("Failed"), 500
        
@users.route("/users/account-password-update", methods=["GET","POST"])
class UserAccountPassword(MethodView):    
    @jwt_required()
    @users.arguments(YHAPasswordUpdateDataSchema, location="json")
    def post(self,args): 
        data = args["data"] 
        data = json.loads(json.dumps(data))         
        password = data["password"]    
        oldPassword = data["oldPassword"] 
        
        current_user_id = get_jwt_identity()
        current_user_jwt  = User.query.filter_by(id=current_user_id).first()  

        if password == oldPassword:
            return jsonify("New Password cannot be the same as previous!"), 403

        if bcrypt.check_password_hash(current_user_jwt.password, oldPassword):
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            current_user_jwt.password = hashed_password
            db.session.commit()
            response = jsonify("Password Updated")
            return response, 200
        else:
            response = jsonify({'message':'Incorrect Previous Password'})
            return response, 401 


########################################
##### imaging routes begin #####
########################################

@users.route("/users/account-image", methods=["GET","POST"])
class UserAccountImage(MethodView): 
    @jwt_required()
    def get(self): 
        current_user_id = get_jwt_identity()
        current_user_jwt  = User.query.filter_by(id=current_user_id).first() 

        image_file = url_for('static', filename='profile_pics/' + current_user_jwt.image_file)
        print("image_file: ",image_file)
        
        return jsonify({'image_url': image_file})
        ...

    @jwt_required()
    def post(self):    
        # check if the post request has the file part
        if 'file' not in request.files:            
            response = jsonify({'message':'No Image Sent'})
            return response, 404 
        
        # data = dict(request.files)
        # print("data: ",data)
        file = request.files.get('file')

        current_user_id = get_jwt_identity()
        current_user_jwt  = User.query.filter_by(id=current_user_id).first()  

        img = Image.open(file.stream)

        picture_file = save_picture(file)
        print("picture_file: ",picture_file)
        current_user_jwt.image_file = picture_file 
        db.session.commit()

        return jsonify({'msg': 'success', 'size': [img.width, img.height]}), 200    

########################################
##### imaging routes end #####
########################################

@users.route("/test-url", methods=["GET","POST"])
class TestURL(MethodView):    
    # @login_required
    @jwt_required()
    def get(self): 
        print("current_user: ",current_user)
        return jsonify("Test URL working"), 200

# @users.route("/user/<string:username>")
@users.route("/user")
class UserPosts(MethodView): 
    @users.arguments(YHAUserPostsSchema, location="query")
    # def get(self,username):
    def get(self,args):        
        postsJsonArray = []
        page = args['page']
        username = args['username']

        page = request.args.get('page',1, type=int)
        user = User.query.filter_by(username=username).first_or_404()
        posts = Post.query.filter_by(author=user)\
            .order_by(Post.date_posted.desc())\
            .paginate(page=page, per_page=5) 
        
        for post in posts.items:
            postsJson = {
                    "id": post.id,
                    "title": post.title,
                    "date_posted": post.date_posted,
                    "content": post.content,
                    "user_id": post.user_id
                }
            postsJsonArray.append(postsJson)
        response = jsonify(postsJsonArray)
        return response, 200

@users.route("/user/<string:username>")
class UserMaxPostPages(MethodView): 
    @users.arguments(YHAUserPostsSchema, location="query")
    # def get(self,username):
    #posts.total for max posts in db
    def get(self,username): 
        return ...

@users.route("/reset_request", methods=["GET", "POST"])
class PasswordResetRequest(MethodView): 
    @users.arguments(YHAPasswordResetSchema, location="query")
    def get(self,args):      
        email = args['email']
        YHAurl = args['YHAurl']  
        if current_user.is_authenticated:
            return "Already Logged In", 200

        user = User.query.filter_by(email=email).first()
        send_reset_email(user, YHAurl)        
        return "An email has been sent with instruction to reset you password", 200

@users.route("/reset_request_token", methods=["GET", "POST"])
class PasswordResetRequestURL(MethodView): 
    @users.arguments(YHAPasswordResetTokenSchema, location="query")
    def get(self,args):        
        token = args['token'] 
        if current_user.is_authenticated:
            return "Already Logged In", 200
        user = User.verify_reset_token(token)
        if user is None:            
            return 'Invalid or expired token', 401
        return 'Valid Token', 200
    
    @users.arguments(YHAPasswordResetURLSchema, location="query")
    def post(self,args): 
        email = args['email']
        if current_user.is_authenticated:
            return "Already Logged In", 200

        form = ResetPasswordForm()
        if form.validate_on_submit():
            hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')        
            # user.password = hashed_password        
            db.session.commit()
            flash(f'Your passowrd has been updated! You are now able to log in', 'success')
            return redirect(url_for('users.login'))    
        return render_template('reset_token.html', title='Reset Password', form=form)

@users.route("/users/delete", methods=["GET","POST"])
class DeleteUser(MethodView):    
    @jwt_required()
    def get(self):   
        ...
