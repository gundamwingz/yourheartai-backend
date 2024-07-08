# from flask import (render_template, url_for, flash,
#                    redirect, request, abort, Blueprint)
from flask import (render_template, url_for, flash,
                   redirect, request)
from flask_login import current_user, login_required
from yourheartai_api import db
from yourheartai_api.models import Post
from yourheartai_api.posts.forms import PostForm

from flask_smorest import Blueprint, abort
from marshmallow import fields
from flask.views import MethodView
from flask import request

from yourheartai_api.posts.PostSchema import (
    YHANewPostSchema, 
    YHAUpdatePostSchema
    )

posts = Blueprint('posts', __name__, url_prefix="/api/yha")


@posts.route("/post/new", methods=["POST"])
class NewPost(MethodView):
    @login_required
    @posts.arguments(YHANewPostSchema, location="query")
    def post(self, args):
        title = args["title"]
        content = args["content"]
        post = Post(title=title, content=content, author=current_user)
        db.session.add(post)
        db.session.commit()        
        return "Your post has been created", 400

##figure out how to implement in angular!, could be a get post id that is passed back to angular
# @posts.route("/post/<post_id>", methods=["GET", 'POST'])
@posts.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)

@posts.route("/post/<int:post_id>/update", methods=["GET", 'POST'])
class UpdatePost(MethodView):
    @login_required
    @posts.arguments(YHAUpdatePostSchema, location="query")
    def post(self, post_id):
        post = Post.query.get_or_404(post_id)
        if post.author != current_user:
            abort(403)
        form = PostForm()
        if form.validate_on_submit():
            post.title = form.title.data
            post.content = form.content.data
            db.session.commit()
            flash('Your post has been updated!', 'success')
            return redirect(url_for('posts.post', post_id=post.id))
        elif request.method == 'GET':
            form.title.data = post.title
            form.content.data = post.content
        return render_template('create_post.html', title='Update Post',
                            form=form, legend='Update Post')
    def get(self, args):
        ...

@posts.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('main.home'))