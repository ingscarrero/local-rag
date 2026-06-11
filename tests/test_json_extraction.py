"""Unit tests for the forgiving JSON extractor used by agent nodes.

No model servers required — the helper lives in local_rag.utils which has
no heavy dependencies.
"""

from local_rag.utils import extract_json


def test_clean_json():
    assert extract_json('{"route": "retrieve"}') == {"route": "retrieve"}


def test_json_embedded_in_prose():
    text = 'Sure thing! Here is the answer: {"route": "direct"} Hope that helps.'
    assert extract_json(text) == {"route": "direct"}


def test_json_with_leading_trailing_whitespace():
    assert extract_json('  \n{"score": 1}\n  ') == {"score": 1}


def test_nested_json():
    assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_no_json_returns_empty_dict():
    assert extract_json("no json here at all") == {}


def test_empty_string_returns_empty_dict():
    assert extract_json("") == {}


def test_malformed_json_returns_empty_dict():
    assert extract_json("{not valid json}") == {}


def test_boolean_values():
    assert extract_json('{"relevant": true, "score": 0}') == {"relevant": True, "score": 0}


def test_integer_value():
    assert extract_json('{"n": 42}') == {"n": 42}


def test_list_value():
    assert extract_json('{"items": [1, 2, 3]}') == {"items": [1, 2, 3]}


def test_route_direct():
    assert extract_json('{"route": "direct"}').get("route") == "direct"


def test_grade_relevant_true():
    reply = 'The passage is relevant. {"relevant": true}'
    assert extract_json(reply).get("relevant") is True


def test_grade_relevant_false():
    reply = '{"relevant": false}'
    assert extract_json(reply).get("relevant") is False
