from yourheartai_api import create_app 

app = create_app()

if __name__== '__main__':
    app.run(debug=True,host='127.0.0.1', port=5008)

# from yourheartai_api import app   
# if __name__== '__main__':
#     app.run(debug=True)