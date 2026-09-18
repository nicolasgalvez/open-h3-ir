"""The injected engine: a model handed over already loaded, from inside the same Python.

A ComfyUI loader node owns the file, the context and the GPU layers; this compiler owns the
words. The engine is duck-typed -- `create_chat_completion(**kwargs)` answering the OpenAI
shape -- so the tests here drive it with a plain fake and nothing imports llama-cpp-python.

What is under test is the contract around the handover: an engine Backend never opens a wire
(no HTTP client, no dialect probe, no model discovery), sends exactly the five kwargs an
engine reads and no thinking dialect (there is no wire for a dialect to be dropped from, which
is the failure that motivated the Ollama spelling), and keeps the wire path's failure
discipline -- the truncation ladder, and exceptions re-raised as their own types because the
fix loop keys on them.
"""
from __future__ import annotations

import pytest

from h3ir.backend import Backend, BackendError, TruncatedResponse
from h3ir.config import Config, LLMConfig


class FakeEngine:
    """Answers in the OpenAI chat shape, records every call, raises what it is handed."""

    model_path = "/comfyui/models/llm/GGUF/qwen.gguf"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _ok(content="a brief", finish="stop", completion_tokens=40):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
            "model": "fake"}


def _backend(engine, **llm):
    cfg = Config(llm=LLMConfig(**{"base_url": "http://unused:1/v1", "model": "", **llm}))
    return Backend(cfg, engine=engine)


# --------------------------------------------------------------------------- the contract

def test_an_injected_engine_sends_exactly_the_five_kwargs_an_engine_reads():
    """No `chat_template_kwargs`, no `reasoning_effort`, no `options`: those exist to talk a
    wire dialect past a server that drops what it does not know, and there is no wire. A
    dialect field on an engine call is a claim about a transport that is not there."""
    e = FakeEngine([_ok()])
    b = _backend(e)
    b.chat([{"role": "user", "content": "Say OK."}], thinking=False, retries=0,
           seed=7, stop=["\n\n"], temperature=0.3)
    assert e.calls == [{"messages": [{"role": "user", "content": "Say OK."}],
                        "max_tokens": 16000, "temperature": 0.3,
                        "seed": 7, "stop": ["\n\n"]}], \
        "the engine call carries anything beyond the five kwargs an engine reads"


def test_the_engine_reply_becomes_the_wire_reply():
    e = FakeEngine([_ok(content="  the brief  ", completion_tokens=123)])
    b = _backend(e)
    r = b.chat([{"role": "user", "content": "x"}], thinking=False, retries=0)
    assert r.content == "the brief"
    assert (r.prompt_tokens, r.completion_tokens) == (100, 123)
    assert r.finish_reason == "stop"
    assert r.model == "fake"
    assert r.wall_s >= 0.0


def test_a_truncated_engine_reply_walks_the_same_ladder_as_the_wire():
    """F15 is a property of the model, not of the transport: a reply that ate its budget is
    retried with a grown one, and a reply that keeps truncating surfaces as
    TruncatedResponse, its own type, because the fix loop grows on that type."""
    e = FakeEngine([_ok(content="half an answ", finish="length"),
                    _ok(content="half an answ", finish="length"),
                    _ok(content="half an answ", finish="length")])
    # A small budget start, so three rungs stay under the 2x16000 ceiling and the ladder is
    # what stops this, not the cap.
    b = _backend(e, max_tokens=1000)
    with pytest.raises(TruncatedResponse, match="max_tokens"):
        b.chat([{"role": "user", "content": "x"}], thinking=False, retries=2)
    budgets = [c["max_tokens"] for c in e.calls]
    assert budgets[1] == int(budgets[0] * 1.75), "the retry did not grow the budget"
    assert budgets[2] == int(budgets[1] * 1.75)
    assert len(e.calls) == 3, "a truncation past the last retry did not stop asking"


def test_an_engine_failure_is_retried_then_reraised_as_its_own_type():
    e = FakeEngine([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    b = _backend(e)
    with pytest.raises(RuntimeError, match="boom"):
        b.chat([{"role": "user", "content": "x"}], thinking=False, retries=2)
    assert len(e.calls) == 3


def test_no_wire_is_ever_opened_for_an_engine_backend():
    """Health, version and model naming never construct a client or probe a dialect: the
    base_url is a placeholder nobody should be able to reach, and nothing tries."""
    e = FakeEngine([_ok()])
    b = _backend(e)
    probe = b.health_probe()
    assert probe.ok and probe.via.endswith("qwen.gguf"), \
        "an injected engine is 'up' by being handed over, and via names the model file"
    assert "in-process" in b.server_version()
    assert b.model_id() == "qwen.gguf"
    assert b._is_ollama_endpoint() is False
    b.require_available()      # no cfg.model, no discovery, no HTTP -- and no error


def test_a_named_model_beats_the_engine_file_and_the_schema_refusal_still_applies():
    e = FakeEngine([_ok()])
    b = _backend(e, model="my-qwen")
    assert b.model_id() == "my-qwen"
    with pytest.raises(BackendError, match="thinking=False"):
        b.chat([{"role": "user", "content": "x"}], thinking=True, retries=0,
               response_format={"type": "json_schema", "json_schema": {"schema": {}}})


def test_json_call_drives_the_engine_and_parses_its_answer():
    """The structured path reaches the engine unchanged: the schema ask is appended to the
    message (guided decoding is off by default) and the reply is parsed as JSON."""
    e = FakeEngine([_ok(content='Before {"ok": true} after')])
    b = _backend(e)
    out = b.json_call([{"role": "user", "content": "give me json"}],
                      {"title": "T", "type": "object"}, required=("ok",), retries=0)
    assert out == {"ok": True}
    assert "response_format" not in e.calls[0], \
        "guided decoding is off by default, so the schema travels in the prompt"
    assert "json" in e.calls[0]["messages"][-1]["content"]
