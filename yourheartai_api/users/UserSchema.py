from marshmallow import (
    Schema,
    fields,
)


#TODO: This is the same as YHAUserAccountSchema, refactor to only have one Schema
class YHARegisterSchema(Schema):
    id = fields.String(required=True)
    username = fields.String(required=True)
    firstName = fields.String(required=True)
    lastName = fields.String(required=True)
    role = fields.String(required=True)

    company = fields.String(required=True)
    address = fields.String(required=True)
    city = fields.String(required=True)
    country = fields.String(required=True)
    postCode = fields.String(required=True)
    aboutMe = fields.String(required=True)
    
    email = fields.String(required=True)    
    password = fields.String(required=True)
    isLoggedIn = fields.Boolean(required=True)
    token = fields.String(required=True)
    image = fields.String(required=False)
    # image = fields.Raw(required=False)

class YHARegisterDataSchema(Schema):
    data = fields.Nested(YHARegisterSchema, required=False, default={})

class YHALoginSchema(Schema):
    email = fields.String(required=True)
    password = fields.String(required=True)

class YHALogoutSchema(Schema):
    isLoggedIn = fields.String(required=True)    

#TODO: This is the same as YHARegisterSchema, refactor to only have one Schema
class YHAUserAccountSchema(Schema):
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

class YHAUserAccountDataSchema(Schema):
    data = fields.Nested(YHAUserAccountSchema, required=False, default={})

class YHAPasswordUpdateSchema(Schema):
    password = fields.String(required=True)
    oldPassword = fields.String(required=True)
class YHAPasswordUpdateDataSchema(Schema):
    data = fields.Nested(YHAPasswordUpdateSchema, required=False, default={})
    
class YHAPasswordResetSchema(Schema):
    email = fields.String(required=True)
    YHAurl = fields.String(required=True)

class YHAPasswordResetTokenSchema(Schema):
    token = fields.String(required=True)

class YHAPasswordResetURLSchema(Schema):
    password = fields.String(required=True)
    
class YHAUserPostsSchema(Schema):
    page = fields.Integer(required=True)
    username = fields.String(required=True)
    