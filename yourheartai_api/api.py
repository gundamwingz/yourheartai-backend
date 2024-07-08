from collections import abc
from copy import deepcopy
from typing import OrderedDict

from flask_smorest import Api as FSApi  # type: ignore
from flask_smorest import Blueprint as FSBlueprint
from flask_smorest.utils import deepupdate  # type: ignore
from marshmallow import Schema
from marshmallow.utils import is_instance_or_subclass

from .route_guard import ROUTE_GUARD


class Api(FSApi):
    def __init__(self, app=None, *, spec_kwargs=None):
        spec_kwargs = spec_kwargs or {}
        super().__init__(app, spec_kwargs=spec_kwargs)
        self.spec.components.security_scheme(
            "ApiKeyAuth", {"type": "apiKey", "in": "header", "name": "X-API-KEY"}
        )


class Blueprint(FSBlueprint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_doc_cbks.append(self._prepare_auth_doc)

    def register_views_in_doc(self, api, app, spec, *, name, parameters):
        """Register views information in documentation

        If a schema in a parameter or a response appears in the spec
        `schemas` section, it is replaced by a reference in the parameter or
        response documentation:

        "schema":{"$ref": "#/components/schemas/MySchema"}
        """
        url_prefix_parameters = parameters or []
        # This method uses the documentation information associated with each
        # endpoint in self._docs to provide documentation for corresponding
        # route to the spec object.
        # Deepcopy to avoid mutating the source. Allows registering blueprint
        # multiple times (e.g. when creating multiple apps during tests).
        for endpoint, endpoint_doc_info in deepcopy(self._docs).items():
            endpoint_route_parameters = endpoint_doc_info.pop("parameters") or []
            endpoint_parameters = url_prefix_parameters + endpoint_route_parameters
            doc = OrderedDict()
            # Use doc info stored by decorators to generate doc
            for method_l, operation_doc_info in endpoint_doc_info.items():
                tags = operation_doc_info.pop("tags")
                operation_doc = {}
                operation = operation_doc_info.get("arguments")
                if operation:
                    op_parameters = [
                        p for p in operation["parameters"] if isinstance(p, abc.Mapping)
                    ]
                    for param in op_parameters:
                        schema = param["schema"]
                        if not is_instance_or_subclass(schema, Schema):
                            param["schema"] = Schema.from_dict(schema)
                operation = operation_doc_info.get("response")
                for func in self._prepare_doc_cbks:
                    operation_doc = func(
                        operation_doc,
                        operation_doc_info,
                        api=api,
                        app=app,
                        spec=spec,
                        method=method_l,
                        name=name,
                    )
                operation_doc.update(operation_doc_info["docstring"])
                # Tag all operations with Blueprint name unless tags specified
                operation_doc["tags"] = (
                    tags
                    if tags is not None
                    else [
                        name,
                    ]
                )
                # Complete doc with manual doc info
                manual_doc = operation_doc_info.get("manual_doc", {})
                doc[method_l] = deepupdate(operation_doc, manual_doc)

            # Thanks to self.route, there can only be one rule per endpoint
            full_endpoint = ".".join((name, endpoint))
            rule = next(app.url_map.iter_rules(full_endpoint))
            spec.path(rule=rule, operations=doc, parameters=endpoint_parameters)

    @staticmethod
    def _prepare_auth_doc(doc, doc_info, name, **kwargs):
        if not doc_info.get("security", False):
            for route in ROUTE_GUARD: ## fix value of Route guard...
                if route["name"] == name:
                    doc["security"] = [{"ApiKeyAuth": []}]
        return doc
