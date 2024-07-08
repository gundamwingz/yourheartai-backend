from marshmallow import (
    Schema,
    fields,
)
    
class YHAHomePostsSchema(Schema):
    page = fields.Integer(required=True)    