from marshmallow import (
    Schema,
    fields,
)

class YHAPatientSchema(Schema):
    id = fields .String(required=False)
    age = fields.Number(required=True)
    gender = fields.Number(required=True)
    chestPain = fields.Number(required=True)
    restingBP = fields.Number(required=True)
    serumCholestrol = fields.Number(required=True)
    fastingBloodSugar = fields.Number(required=True)
    restingRElectro = fields.Number(required=True)
    maxHeartRate = fields.Number(required=False)
    exerciseAngia = fields.Number(required=True)
    oldPeak = fields.Number(required=True)
    slope = fields.Number(required=True)
    noOfMajorVessels = fields.Number(required=True)
class YHAPatientDataSchema(Schema):
    data = fields.Nested(YHAPatientSchema, required=False, default={})

class YHANewPostSchema(Schema):
    title = fields.String(required=True)  
    content = fields.String(required=True)   

class YHAUpdatePostSchema(Schema):
    post_id = fields.Integer(required=True)    