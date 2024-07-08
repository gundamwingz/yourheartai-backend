import graphene
from marshmallow import Schema as MarshmallowSchema
from graphene import relay, ObjectType, Schema
from graphene_sqlalchemy import SQLAlchemyObjectType
from .models import Document as DocumentModel


class Document(SQLAlchemyObjectType):
    class Meta:
        model = DocumentModel
        interfaces = (relay.Node,)


class Query(ObjectType):
    node = relay.Node.Field()
    documents = graphene.List(
        lambda: Document,
        uuid=graphene.String(),
        _type=graphene.List(graphene.String, name="type"),
    )

    # def resolve_documents(self, info, uuid=None, _type=None):
    #     query = Document.get_query(info)
    #     if uuid:
    #         query = query.filter(DocumentModel.uuid == uuid)
    #     if _type:
    #         query = query.filter(DocumentModel.type == " OR ".join(_type))
    #     return query.filter(
    #         DocumentModel.description.is_not(None), DocumentModel.active == 1
    #     ).all()


schema = Schema(query=Query)


def camel_case(s):
    parts = iter(s.split("_"))
    return next(parts) + "".join(i.title() for i in parts)


class CamelCaseSchema(MarshmallowSchema):
    """Schema that uses camel-case for its external representation
    and snake-case for its internal representation.
    """

    def on_bind_field(self, field_name, field_obj):
        field_obj.data_key = camel_case(field_obj.data_key or field_name)
