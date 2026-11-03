"""A per-brief context-window override, carried on the request rather than the environment.

Ollama's default context is 4096 tokens, and a brief with reference pictures is usually larger:
measured against a live Ollama, an ~8k-token prompt came back empty (`finish_reason=length`,
261 completion tokens) without a larger window and was answered with one. Nobody wants to set an
environment variable on the machine the compiler happens to run on to fix that -- the caller who
knows the brief is the one who knows whether it needs more room -- so this is a field on the
request: `POST /v1/briefs {"llm": {"num_ctx": 32768}}`.

`{"options": {"num_ctx": n}}` on `/v1/chat/completions` was measured to have NO EFFECT against a
live Ollama (`GET /api/ps` kept reporting `context_length=4096` after the call): Ollama's
OpenAI-compatible handler unmarshals into `ChatCompletionRequest` (openai/openai.go), a Go struct
with no `options`/`num_ctx` field, and `encoding/json` drops unknown keys silently. Only the
NATIVE `POST /api/chat` honours `options.num_ctx` -- confirmed the same value there raises the
loaded context. So `Backend.chat` routes to `/api/chat` whenever `num_ctx > 0` and the endpoint
looks like Ollama (`H3IR_LLM_API`, default `"auto"`, detected via `GET {root}/api/version`; see
`Backend._wants_native_ollama`), and leaves `/v1/chat/completions` alone otherwise.

Three things are tested here. `Backend` itself: which wire path a call goes over, and that
`options.num_ctx` is only ever sent on `/api/chat`, on every chat call, because `num_ctx` is a
constructor argument rather than a per-call one and `chat()` is where every call in this file ends
up. Auto-detection: `H3IR_LLM_API=auto` probes `/api/version` once and caches the answer. And the
service: `POST /v1/briefs` builds the `Backend` `compile_brief` uses from `body.llm.num_ctx`, so
the override actually reaches the model rather than stopping at the wire model.
"""
from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

import h3ir.service as S
from h3ir.backend import Backend, probe
from h3ir.config import Config, LLMConfig

BASE = "http://endpoint:11434/v1"


def _backend(handler, *, num_ctx: int = 0, api_mode: str = "ollama") -> Backend:
    """`api_mode="ollama"` by default: these tests are about the wire shape `chat()` picks once
    it knows it is talking to Ollama, not about the `/api/version` auto-detection probe, which
    `test_auto_mode_detects_ollama_via_api_version` covers on its own."""
    cfg = Config(llm=LLMConfig(base_url=BASE, model="m", api_mode=api_mode))
    return Backend(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)), num_ctx=num_ctx)


def _ready(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                   "message": {"content": "ready"}}]})


def _ready_native(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"done_reason": "stop", "done": True,
                                      "message": {"content": "ready"}})


# --------------------------------------------------------------------------- Backend.chat


def test_no_options_are_sent_when_num_ctx_is_left_at_the_default():
    """num_ctx=0: `/v1/chat/completions`, unchanged, regardless of `api_mode` -- a caller that
    never asked for a larger window never triggers the native path or its detection probe."""
    bodies: list[dict] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        bodies.append(json.loads(request.content))
        return _ready(request)

    _backend(handler, api_mode="auto").chat([{"role": "user", "content": "hi"}])
    # The `/api/version` probe is the thinking-off dialect detection (`_is_ollama_endpoint`), not
    # a num_ctx concern; what matters here is the one chat path used.
    assert [p for p in paths if p != "/api/version"] == ["/v1/chat/completions"], paths
    assert "options" not in bodies[0], bodies[0]


def test_a_positive_num_ctx_goes_to_native_api_chat_with_options_num_ctx():
    """This is the wire-shape regression: num_ctx>0 against an Ollama-mode Backend must land on
    `/api/chat` with `options.num_ctx`, never on `/v1/chat/completions`, on every call."""
    bodies: list[dict] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        bodies.append(json.loads(request.content))
        return _ready_native(request)

    b = _backend(handler, num_ctx=32768)
    b.chat([{"role": "user", "content": "hi"}])
    b.chat([{"role": "user", "content": "again"}])
    assert paths == ["/api/chat", "/api/chat"], paths
    assert bodies[0]["options"]["num_ctx"] == 32768
    assert bodies[1]["options"]["num_ctx"] == 32768


def test_a_positive_num_ctx_never_reaches_v1_chat_completions_as_options():
    """The bug this fixes: sending `options.num_ctx` to `/v1/chat/completions` is a silent
    no-op against real Ollama (openai/openai.go's `ChatCompletionRequest` has no such field).
    `api_mode="openai"` forces the old path deliberately (a non-Ollama OpenAI-compatible
    endpoint) and must not carry `num_ctx` as an `options` key it cannot honour differently."""
    bodies: list[dict] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        bodies.append(json.loads(request.content))
        return _ready(request)

    _backend(handler, num_ctx=32768, api_mode="openai").chat([{"role": "user", "content": "hi"}])
    assert [p for p in paths if p != "/api/version"] == ["/v1/chat/completions"], paths
    assert all("options" not in b for b in bodies), bodies


def test_auto_mode_detects_ollama_via_api_version():
    """`api_mode="auto"` (the default): a positive num_ctx probes `{root}/api/version` once,
    and routes to `/api/chat` only when that answers -- the same signal `liveness_urls` uses to
    recognise Ollama."""
    bodies: list[dict] = []
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.33.3"})
        bodies.append(json.loads(request.content))
        return _ready_native(request)

    b = _backend(handler, num_ctx=32768, api_mode="auto")
    b.chat([{"role": "user", "content": "hi"}])
    b.chat([{"role": "user", "content": "again"}])
    assert paths[0] == "/api/version", paths  # probed once
    assert paths.count("/api/version") == 1, paths  # ... and cached, not re-probed on call two
    assert paths[1:] == ["/api/chat", "/api/chat"], paths
    assert bodies[0]["options"]["num_ctx"] == 32768


def test_auto_mode_falls_back_to_openai_when_api_version_is_absent():
    """A non-Ollama OpenAI-compatible endpoint (vLLM, llama.cpp) has no `/api/version`, so
    `auto` must not guess native and must leave `/v1/chat/completions` exactly as before."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(404)
        return _ready(request)

    _backend(handler, num_ctx=32768, api_mode="auto").chat([{"role": "user", "content": "hi"}])
    assert paths == ["/api/version", "/v1/chat/completions"], paths


def test_doctor_never_configures_num_ctx():
    """`probe()` -- what `h3ir doctor` prints -- builds its own `Backend` with no override, so
    doctor's own calls are unaffected by this feature existing."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        bodies.append(json.loads(request.content))
        return _ready(request)

    import h3ir.backend as B
    real = B.Backend

    class Wired(real):
        def __init__(self, c=None, client=None):
            super().__init__(c, httpx.Client(transport=httpx.MockTransport(handler)))

    B.Backend = Wired
    try:
        probe(Config(llm=LLMConfig(base_url=BASE, model="m")))
    finally:
        B.Backend = real
    assert bodies, "no chat request was made"
    assert all("options" not in body for body in bodies), bodies


# --------------------------------------------------------------------------- the service wiring


def _stub_compile(monkeypatch, seen: dict) -> None:
    """Bypass the compiler entirely: this file is about whether the override reaches the
    `Backend` the route builds, not about compilation. `compile_brief` and `_envelope` are the
    seam `create_brief` calls after the wiring this test is about, so both are replaced."""

    class FakeDoc:
        ok = True

    def fake_compile_brief(brief, *, backend=None, **kw):
        seen["num_ctx"] = backend.num_ctx
        backend.close()
        return FakeDoc()

    monkeypatch.setattr(S, "compile_brief", fake_compile_brief)
    monkeypatch.setattr(S, "_envelope", lambda *a, **kw: {})
    monkeypatch.setattr(S, "_remember", lambda *a, **kw: None)


def test_a_brief_with_no_llm_override_uses_a_backend_with_num_ctx_at_zero(monkeypatch):
    seen: dict = {}
    _stub_compile(monkeypatch, seen)
    r = TestClient(S.app).post("/v1/briefs", json={
        "intent": "a lighthouse keeper lights the lamp in a storm"})
    assert r.status_code == 201, r.json()
    assert seen["num_ctx"] == 0


def test_a_brief_that_asks_for_a_larger_context_gets_a_backend_configured_for_it(monkeypatch):
    seen: dict = {}
    _stub_compile(monkeypatch, seen)
    r = TestClient(S.app).post("/v1/briefs", json={
        "intent": "a lighthouse keeper lights the lamp in a storm",
        "llm": {"num_ctx": 32768}})
    assert r.status_code == 201, r.json()
    assert seen["num_ctx"] == 32768


def test_refining_a_brief_also_honours_a_llm_override(monkeypatch):
    seen: dict = {}

    def fake_refine(brief, change, *, backend=None, **kw):
        own = backend is None
        from h3ir.config import get_config
        backend = backend or Backend(get_config())
        seen["num_ctx"] = backend.num_ctx
        if own:
            backend.close()
        return object(), brief, ["intent"]

    monkeypatch.setattr(S, "refine", fake_refine)
    monkeypatch.setattr(S, "_envelope", lambda *a, **kw: {})
    monkeypatch.setattr(S, "_remember", lambda *a, **kw: None)
    S._STORE["b1"] = {"brief": object(), "doc": object(), "at": 0, "versions": 1}
    try:
        r = TestClient(S.app).patch("/v1/briefs/b1", json={
            "change": "make it longer", "llm": {"num_ctx": 32768}})
        assert r.status_code == 200, r.json()
        assert seen["num_ctx"] == 32768
    finally:
        S._STORE.pop("b1", None)


def test_an_unknown_llm_subfield_is_refused_by_name_rather_than_dropped():
    r = TestClient(S.app).post("/v1/briefs", json={
        "intent": "a lighthouse keeper lights the lamp", "llm": {"temperature": 0.9}})
    assert r.status_code == 422
    assert "temperature" in r.json()["detail"]["message"]
