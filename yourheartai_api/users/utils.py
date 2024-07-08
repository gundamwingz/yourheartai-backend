import os
import secrets
from PIL import Image
from flask import url_for, current_app
from flask_mail import Message
from yourheartai_api import mail

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    # _, f_ext = os.path.splitext(form_picture.filename) #underscore used to discard variable
    # picture_fn = random_hex + f_ext
    
    picture_fn = form_picture.filename
    picture_path = os.path.join(current_app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125) #change to better size, or maybe consider cnn size also!
    i = Image.open(form_picture)
    # i.thumbnail(output_size)
    i.save(picture_path)
    
    return picture_fn


def send_reset_email(user, url):
    token = user.get_reset_token()
    print('token: ',token)
    YHAurl = url+""+ token
    # print("YHAurl 1: ",YHAurl)
    # YHAurl = "http://localhost:4401/" +""+ token
    # print("YHAurl 2: ",YHAurl)

    msg = Message('Password Reset Request',
                  sender='noreply@yourheartai.com',
                  recipients=[user.email])
                    
#     msg.body = f''' To reset your password, visit the following link:
# {url_for('users.reset_token', token=token, _external=True)}

# If you did not make this request then simply ignore this email and no changes will be made.
# '''
    msg.body = f''' To reset your password, visit the following link:
{YHAurl}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)
