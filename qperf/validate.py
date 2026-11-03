"""A small JSON Schema subset validator (type incl. type lists, enum, required, properties, additionalProperties
as false or a schema, items, minimum, maximum, $ref to #/$defs) so the CLI, the API and the skill share one contract
without a dependency."""

from __future__ import annotations

import json
import os

_SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema")
_schemas = {}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "number": (int, float), "integer": int,
          "null": type(None)}


def _is(node, t):
    return isinstance(node, _TYPES[t]) and not (t in ("integer", "number") and isinstance(node, bool))


def _check(node, schema, root, path, errs):
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].split("/")[-1]]
    t = schema.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        if not any(_is(node, x) for x in ts):
            errs.append(f"{path or '$'}: expected {'|'.join(ts)}")
            return
    if "enum" in schema and node not in schema["enum"]:
        errs.append(f"{path}: {node!r} not in {schema['enum']}")
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in schema and node < schema["minimum"]:
            errs.append(f"{path}: {node} < {schema['minimum']}")
        if "maximum" in schema and node > schema["maximum"]:
            errs.append(f"{path}: {node} > {schema['maximum']}")
    if isinstance(node, dict):
        for r in schema.get("required", []):
            if r not in node:
                errs.append(f"{path or '$'}: missing '{r}'")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        for k, v in node.items():
            if k in props:
                _check(v, props[k], root, f"{path}.{k}" if path else k, errs)
            elif extra is False:
                errs.append(f"{path or '$'}: unknown field '{k}'")
            elif isinstance(extra, dict):
                _check(v, extra, root, f"{path}.{k}" if path else k, errs)
    if isinstance(node, list) and "items" in schema:
        for i, v in enumerate(node):
            _check(v, schema["items"], root, f"{path}[{i}]", errs)


def _validate(doc, name):
    if name not in _schemas:
        with open(os.path.join(_SCHEMA_DIR, name)) as f:
            _schemas[name] = json.load(f)
    errs = []
    _check(doc, _schemas[name], _schemas[name], "", errs)
    return errs


def validate_request(req):
    return _validate(req, "request.schema.json")


def validate_response(resp):
    return _validate(resp, "response.schema.json")
