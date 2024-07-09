from marshmallow import (
    Schema,
    fields,
)
    
class YHANewPostSchema(Schema):
    title = fields.String(required=True)  
    content = fields.String(required=True)   

class YHAUpdatePostSchema(Schema):
    post_id = fields.Integer(required=True)    