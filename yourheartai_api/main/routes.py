from flask import render_template, request, jsonify
from yourheartai_api.models import Post

from flask_smorest import Blueprint
from marshmallow import fields
from flask.views import MethodView
from flask import request

from yourheartai_api.main.MainSchema import YHAHomePostsSchema

main = Blueprint('main', __name__, url_prefix="/api/yha")

@main.route("/", methods=["GET","POST"])
@main.route("/home", methods=["GET","POST"])
class Home(MethodView):
    @main.arguments(YHAHomePostsSchema, location="query")
    def get(self,args):
        # posts = Post.query.all()
        postsJsonArray = []
        page = request.args.get('page',1, type=int)
        posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5) 

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

@main.route("/home_max_pages", methods=["GET","POST"])
class HomeMaxPages(MethodView):
    def get(self,args):
        """
        Calculate the total number of pages: Determine the total number of pages 
        by dividing the total number of items by the number of items per page. 
        If there is a remainder, round up to the nearest whole number.
        """
        ...

@main.route("/about")
def about():
    return ...
