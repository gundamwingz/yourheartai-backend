from marshmallow import (
    Schema,
    fields,
)

class YHAPatientSchema(Schema):
    aboutMe = fields.String(required=True)
    address = fields.String(required=True)
    city = fields.String(required=True)
    company = fields.String(required=True)
    country = fields.String(required=True)
    email = fields.String(required=True)
    firstName = fields.String(required=True)
    id = fields.String(required=True)
    image = fields.Raw(required=False)
    isLoggedIn = fields.Boolean(required=True)
    lastName = fields.String(required=True)
    password = fields.String(required=True)
    postCode = fields.String(required=True)
    role = fields.String(required=True)
    username = fields.String(required=True)
    token = fields.String(required=True)

class YHAPatientDataSchema(Schema):
    data = fields.Nested(YHAPatientSchema, required=False, default={})

class YHANewPostSchema(Schema):
    title = fields.String(required=True)  
    content = fields.String(required=True)   

class YHAUpdatePostSchema(Schema):
    post_id = fields.Integer(required=True)    