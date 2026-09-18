"""Tests for the rules added after reading the prior-art sweep.

Each one exists because a specific failure was reported in the wild, or because a validator rule
I wrote turned out to be wrong. No model and no GPU.
"""
from __future__ import annotations

import re

import httpx
import pytest

from h3ir.backend import Backend, TruncatedResponse, _extract_json
from h3ir.config import Config, LLMConfig, get_config, set_config
from h3ir.draft import deterministic_draft
from h3ir.models import AssetCard, AssetKind, AssetRef, Brief, DialogueLine, Mode, Role
from h3ir.plan import ProfileOptions
from h3ir.render import render_ir
from h3ir.validate import Context, validate


def _rules(text, ctx=None, sev="ERROR", **kw):
    ctx = ctx or Context(**kw)
    return {f.rule for f in validate(text, ctx) if f.severity == sev}


def _ref2va(defs, summary, retention, desc, sound="Wind moves through the grass.",
            music="N/A") -> str:
    return (f"subject_definitions:\n{defs}\n\nsummary:\n{summary}\n\n"
            f"retention_analysis:\n{retention}\n\ndetailed_description:\n{desc}\n\n"
            f"overall_soundscape:\n{sound}\n\nnon_diegetic_music:\n{music}\n")


# --------------------------------------------------------------- L5, the inherited false positive

def test_a_source_line_is_legal_when_the_label_is_analysed_separately():
    """The spec forbids a standalone <Picture N> line only when the label 'will not be analyzed
    or used separately later'. Flagging it unconditionally is a false positive -- the exact bug
    found in the harness validator."""
    legal = _ref2va(
        defs=("<Subject 1> is the man in <Picture 1>, with dark hair.\n"
              "<Picture 1> is the reference for the opening composition."),
        summary="[keyframe completion] The target video opens on <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the dark hair is retained.\n"
                   "<Picture 1> ([Shot 1] first frame): fully_preserved - the composition is held."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    assert "L5-redundant-source-line" not in _rules(legal, n_pictures=1)


def test_a_source_line_with_no_retention_entry_is_still_an_error():
    illegal = _ref2va(
        defs=("<Subject 1> is the man in <Picture 1>, with dark hair.\n"
              "<Picture 1> is the reference for the opening composition."),
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the dark hair is retained.",
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    assert "L5-redundant-source-line" in _rules(illegal, n_pictures=1)


# --------------------------------------------------------------- reasoning leakage

@pytest.mark.parametrize("marker", ["<think>", "</think>", "assistantfinal"])
def test_leaked_reasoning_markers_are_rejected(marker):
    """vLLM #35221 puts in-progress reasoning into `content`; #39697 injects the reasoning-end
    string mid-content. On a thinking model this is expected input, not an exception."""
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.",
        desc=f"A style line.\n[Shot 1] {marker} The camera holds a static shot on <Subject 1>.")
    assert "G1-reasoning-leaked" in _rules(text, n_pictures=1)


@pytest.mark.parametrize("phrase", [
    "I will show the man walking",
    "Let me establish the wide shot",
    "The user wants a dramatic angle",
    "Okay, the man steps forward",
])
def test_self_narration_inside_the_deliverable_is_rejected(phrase):
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.",
        desc=f"A style line.\n[Shot 1] {phrase}. The camera holds a static shot on <Subject 1>.")
    assert "G2-model-self-narration" in _rules(text, n_pictures=1, sev="WARN")


@pytest.mark.parametrize("phrase", [
    "He gives an okay sign to the crew",
    "He ignores the request and turns away",
    "The prompt board is visible on the wall",
    "A sign reading OKAY hangs behind him",
])
def test_ordinary_description_is_not_mistaken_for_self_narration(phrase):
    """A false positive here costs a valid brief, because the penalty is falling back to the
    draft. An earlier draft of G2 matched bare 'okay' and 'the request' and would have."""
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.",
        desc=f"A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>. {phrase}.")
    found = _rules(text, n_pictures=1)
    assert "G2-model-self-narration" not in found, found


def test_dialogue_may_legitimately_say_i():
    """A character saying 'I' is not self-narration. The check must exempt verbatim spans or it
    would forbid the most ordinary line of dialogue there is."""
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.",
        desc=("A style line.\n[Shot 1] The camera holds a static shot as <Subject 1> (S1) says: "
              "<d>[English] I will not let go.</d>"))
    found = _rules(text, n_pictures=1)
    assert "G2-model-self-narration" not in found, found


# --------------------------------------------------------------- marker legality against role

def test_a_frame_anchor_must_be_fully_preserved():
    """The legality gap the sweep found unclaimed: the marker must match the declared role, not
    merely be a member of the enum."""
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[keyframe completion] The target video opens on <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.\n"
                   "<Picture 1> ([Shot 1] first frame): partially_preserved - mostly held."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    ctx = Context(n_pictures=1,
                  declared_roles=(("<Picture 1>", "frame_anchor_first", ""),))
    assert "R6-anchor-must-be-preserved" in _rules(text, ctx)


def test_a_picture_cannot_carry_attribute_transfer():
    text = _ref2va(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.\n"
                   "<Picture 1> ([Shot 1] anchor): attribute_transfer - the look moves across."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    ctx = Context(n_pictures=1, declared_roles=(("<Picture 1>", "subject", ""),))
    assert "R7-picture-cannot-transfer" in _rules(text, ctx)


def test_only_one_audio_can_be_the_complete_track():
    text = _ref2va(
        defs=("<Subject 1> is the man in <Picture 1>, with dark hair.\n"
              "<Audio 1> is a background music track.\n<Audio 2> is a background music track."),
        summary="[reference generation + audio reuse] The target video shows <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.\n"
                   "<Audio 1>: fully_copy - reused as the whole track.\n"
                   "<Audio 2>: fully_copy - also reused as the whole track."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    assert "R9-multiple-full-copies" in _rules(text, n_pictures=1, n_audios=2)


def test_ref2va_must_not_declare_a_keyframe_anchor():
    text = _ref2va(
        defs=("<Subject 1> is the man in <Picture 1>, with dark hair.\n"
              "<Picture 1> is the first frame of the target video."),
        summary="[reference generation] The target video shows <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.\n"
                   "<Picture 1> ([Shot 1] anchor): fully_preserved - held."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    assert "R10-mode-role-contamination" in _rules(text, Context(n_pictures=1,
                                                                forbid_keyframe_refs=True))


def test_a_paired_soundtrack_shorter_than_its_video_warns():
    text = _ref2va(
        defs=("<Subject 1> is the man in <Video 1>, with dark hair.\n"
              "<Audio 1> is the synchronized audio track of <Video 1>."),
        summary="[video editing + audio reuse] The target video is an edited version of <Video 1>. "
                "It shows <Subject 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.\n"
                   "<Video 1> (source video editing): fully_preserved - framing maintained.\n"
                   "<Audio 1>: partially_copy - reused beneath the dialogue."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    ctx = Context(n_pictures=0, n_videos=1, n_audios=1, generation_task=False,
                  paired_audio=(("<Audio 1>", "<Video 1>", 3.0, 5.0),))
    assert "A6-paired-audio-short" in _rules(text, ctx, sev="WARN")


# --------------------------------------------------------------- A2 measures, not matches

def test_a2_fires_on_real_duplication_and_not_on_the_word_sound():
    common = dict(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.")
    dup = _ref2va(
        **common,
        desc=("A style line.\n[Shot 1] The camera holds a static shot as heavy rain taps against "
              "the tin roof above him."),
        sound="Heavy rain taps against the tin roof above him.")
    assert "A2-sound-duplicated" in _rules(dup, n_pictures=1, sev="WARN")

    clean = _ref2va(
        **common,
        desc=("A style line.\n[Shot 1] The camera holds a static shot while the sound of a latch "
              "clicking can be heard as he turns."),
        sound="Distant traffic hums beyond the wall.")
    assert "A2-sound-duplicated" not in _rules(clean, n_pictures=1, sev="WARN")


# --------------------------------------------------------------- R20 attributes the motion

def _t2va(desc: str) -> str:
    return (f"integrated_multimodal_description:\n{desc}\n\n"
            "overall_soundscape:\nWind moves through the grass.\n\nnon_diegetic_music:\nN/A\n")


@pytest.mark.parametrize("desc", [
    # A push in on a retreating subject. Two things moving in opposite directions is a shot, and
    # the rule called it a contradiction because a regex spanned the actor boundary.
    "[Shot 1] Live-action, a wide shot as the camera pushes in with small amplitude at slow speed "
    "while he steps backward into the dark.",
    # The camera holds; the SUBJECT moves. Same boundary, different verb.
    "[Shot 1] Live-action, a wide shot holds a static shot as he moves toward the doorway.",
    # Sequence, not contradiction: two camera states in order.
    "[Shot 1] Live-action, a wide shot where the camera holds a static shot, then pans right with "
    "large amplitude at slow speed.",
])
def test_r20_does_not_fire_when_the_other_thing_moves_or_the_moves_are_sequential(desc):
    assert "R20-camera-contradiction" not in _rules(_t2va(desc), mode="t2va", duration_s=8.0)


@pytest.mark.parametrize("desc", [
    # The camera itself, in one clause, doing both. This is the case worth an ERROR: the model will
    # do one of the two and nothing states which.
    "[Shot 1] Live-action, a wide shot as the camera pushes in with small amplitude, moving "
    "backward away from the doorway.",
    "[Shot 1] Live-action, a wide shot as the camera pulls out with small amplitude, drifting "
    "closer to the doorway.",
])
def test_r20_still_fires_on_a_camera_contradicting_itself(desc):
    assert "R20-camera-contradiction" in _rules(_t2va(desc), mode="t2va", duration_s=8.0)


def test_the_sound_section_lengths_are_checked_against_base_en():
    """A1 was DELETED on my false confession that the 1-4 figure was mine. It is stated in
    base-en.txt 4.6, and 1-3 for music in 4.7; ref-en.txt 6 does not restate them because it
    explicitly defers to that guide. Silence in one document was read as the spec's silence.

    WARN, never a gate: the spec phrases both as instructions and length has never gated a run here."""
    common = dict(
        defs="<Subject 1> is the man in <Picture 1>, with short dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.",
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")

    six = _ref2va(**common, sound=("Distant traffic hums. A compressor cycles on. A floorboard "
                                   "settles. Pigeons shift outside. A tap drips. The vents breathe."))
    found = {f.rule: f.severity for f in validate(six, Context(n_pictures=1))}
    assert found.get("A1-soundscape-length") == "WARN", found

    four = _ref2va(**common, sound=("Distant traffic hums beyond the wall. A compressor cycles on. "
                                    "A floorboard settles upstairs. A tap drips into a metal sink."))
    assert "A1-soundscape-length" not in {f.rule for f in validate(four, Context(n_pictures=1))}

    loud = _ref2va(**common, music=("A low cello drone opens. Strings answer it. A drum enters at a "
                                    "slow tempo. The whole thing drops out at the end."))
    assert {f.rule: f.severity for f in validate(loud, Context(n_pictures=1))}.get(
        "A7-music-length") == "WARN"

    # and N/A is never too long
    assert not [f for f in validate(_ref2va(**common, music="N/A"), Context(n_pictures=1))
                if f.rule.startswith(("A1", "A7"))]


# --------------------------------------------------------------- the deterministic floor

def _card():
    return AssetCard(sha256="i1", kind=AssetKind.IMAGE, style="Live-action, cinematic",
                     lighting="warm afternoon light", environment="a stone tower stairwell",
                     subjects=[{"kind": "person", "descriptor": "the keeper",
                                "attributes": ["oilskin coat", "grey beard"]}])


@pytest.mark.parametrize("seconds,mode", [(5, Mode.T2VA), (10, Mode.T2VA), (15, Mode.REF2VA)])
def test_the_draft_is_valid_with_no_model_at_all(seconds, mode):
    """The draft is the product floor. If it can be invalid there is nothing to fall back to.

    T2VA gets NO assets here, and that is the fix rather than the fixture being weakened: text-to-video
    binds no reference labels, so an attached image is unreferencable and `L4-unused-media` is right to
    reject it. Mode inference never produces that pairing; a caller override can, and `compile_brief`
    now drops the assets with `X16-assets-dropped-for-mode` instead of publishing a manifest the text
    cannot bind."""
    assets = ([] if mode is Mode.T2VA else
              [AssetRef(kind=AssetKind.IMAGE, role=Role.SUBJECT, sha256="i1", px=(1024, 1024))])
    brief = Brief(intent="a lighthouse keeper lights the lamp in a storm", seconds=seconds,
                  assets=assets)
    plan = deterministic_draft(brief, mode, {"i1": _card()} if assets else {}, opts=ProfileOptions())
    res = render_ir(plan, ProfileOptions())
    counts = plan.label_counts()
    errs = [f for f in validate(res.prompt, Context(
        mode=mode.value, duration_s=plan.target.effective_seconds,
        n_pictures=counts["Picture"], n_videos=counts["Video"], n_audios=counts["Audio"],
        forbid_keyframe_refs=(mode is Mode.REF2VA)))
        if f.severity == "ERROR"]
    assert not errs, [str(e) for e in errs]


def test_the_draft_places_dialogue_verbatim_and_canonical_camera():
    brief = Brief(intent="two mechanics argue", seconds=5,
                  dialogue=[DialogueLine(text="That belt is done.")])
    plan = deterministic_draft(brief, Mode.T2VA, {}, opts=ProfileOptions())
    out = render_ir(plan, ProfileOptions()).prompt
    assert "<d>[English] That belt is done.</d>" in out
    assert "with small amplitude at slow speed" in out
    assert ".." not in out, "placeholder substitution must not double a full stop"


def test_the_draft_is_byte_reproducible():
    brief = Brief(intent="rain on a window", seconds=5)
    p1 = deterministic_draft(brief, Mode.T2VA, {}, opts=ProfileOptions())
    p2 = deterministic_draft(brief, Mode.T2VA, {}, opts=ProfileOptions())
    assert render_ir(p1, ProfileOptions()).prompt == render_ir(p2, ProfileOptions()).prompt


# --------------------------------------------------------------- backend guards

def test_json_is_extracted_from_a_fence_or_a_sentence():
    assert _extract_json('```json\n{"a":1}\n```') == '{"a":1}'
    assert _extract_json('Here you go: {"a": 1} — hope that helps') == '{"a": 1}'
    assert _extract_json('{"a":1}') == '{"a":1}'


def _canned(payload: dict) -> Backend:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)
    return Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_an_unclosed_think_marker_is_treated_as_truncation():
    """vLLM #35221: the parser cannot tell 'thinking off' from 'thinking unfinished', so an
    unclosed marker means the reply is reasoning. Trust the marker, not the field split."""
    b = _canned({"choices": [{"finish_reason": "stop",
                              "message": {"content": "<think> I should start by"}}],
                 "usage": {}})
    with pytest.raises(TruncatedResponse, match="unclosed"):
        b.chat([{"role": "user", "content": "x"}], thinking=False, retries=0)


def test_empty_content_raises_instead_of_returning_nothing():
    b = _canned({"choices": [{"finish_reason": "length",
                              "message": {"content": None, "reasoning": "x" * 500}}],
                 "usage": {"completion_tokens": 400}})
    with pytest.raises(TruncatedResponse):
        b.chat([{"role": "user", "content": "x"}], thinking=False, retries=0)


def test_a_schema_ask_never_stringifies_an_image():
    """A multipart content is a LIST holding a base64 data URL. Coercing it with str() puts the
    whole image into the prompt as literal text and looks exactly like the endpoint hanging.
    This is the test for a bug that cost ten minutes of a stalled run."""
    from h3ir.backend import _append_text, user_message
    import base64

    fake_png = base64.b64encode(b"\x89PNG" + b"Q" * 4000).decode()
    content = [{"type": "text", "text": "describe"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + fake_png}}]
    out = _append_text(content, "SCHEMA HERE")
    assert isinstance(out, list) and len(out) == 3
    assert out[2] == {"type": "text", "text": "SCHEMA HERE"}
    for part in out:
        if part.get("type") == "text":
            assert "base64" not in part["text"], "the image leaked into a text part"
            assert len(part["text"]) < 200
    # a plain string message still concatenates
    assert _append_text("hello", " world") == "hello world"


def test_the_truncation_ladder_is_capped():
    """Growing the budget is right when reasoning ate it and wrong when the model loops --
    unbounded growth turns a loop into minutes of waiting."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {"content": "x" * 50}}],
            "usage": {"completion_tokens": 9}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(TruncatedResponse):
        b.chat([{"role": "user", "content": "x"}], thinking=False, max_tokens=15000, retries=6)
    assert calls["n"] <= 3, f"ladder ran {calls['n']} times; it must stop at the ceiling"


def test_guided_decoding_is_off_by_default():
    """Deliberate: with a reasoning parser active vLLM can silently skip grammar enforcement
    (#39130) and llama.cpp reports the converse on this model family (#20345)."""
    assert get_config().llm.guided_decoding is False


def test_a_schema_call_with_thinking_on_omits_the_grammar():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
            "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = b.json_call([{"role": "user", "content": "give me json"}],
                      {"title": "T", "type": "object"}, required=("ok",), thinking=True)
    assert out == {"ok": True}
    assert "response_format" not in captured, "no grammar may be sent with reasoning enabled"
    assert captured.get("chat_template_kwargs") is None, "thinking must stay on"


# ------------------------------------------- turning thinking off, per server dialect

def _dialect_endpoint(version_status: int, captured: dict, probes: list) -> httpx.Client:
    """A fake OpenAI endpoint whose `/api/version` answers `version_status`, recording chat
    bodies into `captured` and every probe into `probes`."""
    import json as _json

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/version"):
            probes.append(str(request.url))
            return httpx.Response(version_status, json={"version": "0.34.2"})
        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
            "usage": {}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_an_ollama_endpoint_gets_the_reasoning_effort_spelling_too():
    """MEASURED (2026-09, Ollama 0.34.2): `chat_template_kwargs` is not a field of Ollama's
    OpenAI request struct, so it is dropped and the Qwen3 template's own default -- thinking on
    -- runs anyway: ~13k reasoning tokens per call, minutes per compile. The one `/v1` spelling
    that Ollama honours is `reasoning_effort: "none"`. Both spellings ride in one body: vLLM
    reads the one it knows, Ollama reads the other."""
    captured: dict = {}
    probes: list = []
    b = Backend(client=_dialect_endpoint(200, captured, probes))
    b.chat([{"role": "user", "content": "Say OK."}], thinking=False, retries=0)
    assert captured.get("chat_template_kwargs") == {"enable_thinking": False}
    assert captured.get("reasoning_effort") == "none", \
        "an Ollama endpoint was not told to stop thinking in the dialect it reads"
    assert len(probes) == 1, "the dialect must be probed once per Backend, not per call"
    b.chat([{"role": "user", "content": "Again."}], thinking=False, retries=0)
    assert len(probes) == 1, "the second call re-probed a dialect already settled"


def test_a_non_ollama_endpoint_gets_exactly_the_request_it_always_got():
    """`reasoning_effort` is a live field for hosted reasoning models and "none" is not one of
    their values, so it may only ever be sent to an endpoint that answered `/api/version`. A
    vLLM or hosted endpoint that 404s the probe keeps the byte-equal request it received before
    this existed."""
    captured: dict = {}
    b = Backend(client=_dialect_endpoint(404, captured, []))
    b.chat([{"role": "user", "content": "Say OK."}], thinking=False, retries=0)
    assert "reasoning_effort" not in captured, \
        "reasoning_effort was sent to an endpoint that never said it was Ollama"
    assert captured.get("chat_template_kwargs") == {"enable_thinking": False}


def test_a_probe_that_cannot_complete_keeps_the_old_request():
    """A gateway that blackholes `/api/version` is not Ollama's to answer for. The probe fails,
    the dialect reads as not-Ollama, and the request goes out unchanged."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/version"):
            raise httpx.ConnectError("blackholed")
        import json as _json
        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
            "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    b.chat([{"role": "user", "content": "Say OK."}], thinking=False, retries=0)
    assert "reasoning_effort" not in captured


def test_thinking_on_never_sends_either_spelling_on_any_dialect():
    captured: dict = {}
    b = Backend(client=_dialect_endpoint(200, captured, []))
    b.chat([{"role": "user", "content": "Think about it."}], thinking=True, retries=0)
    assert "reasoning_effort" not in captured
    assert "chat_template_kwargs" not in captured


# --------------------------------------------------------------- the manifest is the contract

def test_a_text_only_brief_publishes_an_empty_manifest():
    """The owner settled who owns the reference-mismatch check: "if our ir is caller-agnostic, then
    that's for the app to check isn't it?" -- correct, and it makes the manifest the contract. The app
    wires from it and verifies by hash; an empty manifest means attach nothing.

    So this layer's whole obligation is that the manifest never claims an asset the text does not bind,
    and never omits one it does. A text-only brief rendered with two images attached is what produced
    a video where the character was not kept: the assets were inert because nothing in the text
    referenced them."""
    from h3ir.draft import deterministic_draft
    from h3ir.models import Brief, Mode
    from h3ir.plan import ProfileOptions
    from h3ir.render import render_ir

    plan = deterministic_draft(Brief(intent="the man walks down the stone corridor", seconds=8.0),
                               Mode.T2VA, {})
    assert plan.manifest == [], "a brief with no assets must publish no manifest entries"
    prompt = render_ir(plan, ProfileOptions(name="standard")).prompt
    assert not re.findall(r"<(?:Picture|Video|Audio|Subject)\s+\d+>", prompt), \
        "and its text must bind no labels, or the manifest and the text disagree"


def test_every_label_in_the_text_has_a_manifest_entry_behind_it():
    """The other direction, and the one L3-phantom-media covers in the validator. Asserted here on the
    manifest itself so the two halves cannot drift apart."""
    from h3ir.analyse import sha256_file
    from h3ir.config import get_config
    from h3ir.draft import deterministic_draft
    from h3ir.models import AssetKind, AssetRef, Brief, Mode, Role
    from h3ir.plan import ProfileOptions
    from h3ir.render import render_ir

    p = get_config().paths.golden_dir / "assets" / "ref1.png"
    if not p.exists():
        pytest.skip("golden asset missing")
    brief = Brief(intent="the man walks down the corridor", seconds=8.0,
                  assets=[AssetRef(kind=AssetKind.IMAGE, role=Role.SUBJECT,
                                   sha256=sha256_file(p), path=str(p), note="the woman",
                                   px=(733, 1100))])
    plan = deterministic_draft(brief, Mode.REF2VA, {})
    published = {m.label for m in plan.manifest}
    prompt = render_ir(plan, ProfileOptions(name="standard")).prompt
    used = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", prompt))
    assert used <= published, f"text binds labels with no asset behind them: {used - published}"
    assert published, "an asset was handed in and the manifest is empty"


# --------------------------------------------------------------- audio: words yes, sound no

def test_a_transcript_never_fills_the_sonic_fields():
    """The toolbox's Whisper closes the DIALOGUE half of audio and nothing else. A transcript gives
    the words, not the timbre, the delivery, the tempo, or whether the thing is music at all — so
    `timbre` and `music` must stay empty no matter how good the transcript is. Letting a transcript
    look like it solved audio is the trap."""
    from h3ir.analyse import analyse_audio

    card = analyse_audio(
        AssetRef(kind=AssetKind.AUDIO, role=Role.VOICE_TIMBRE, sha256="a" * 64, path="/tmp/v.wav",
                 note="his own voice, calm and low", seconds=6.0),
        transcript="Something has been down here a long time.")
    assert card.transcript
    assert card.timbre == "" and card.music == ""
    assert card.characterisation == "his own voice, calm and low"


def test_the_callers_characterisation_reaches_the_definition_not_the_soundscape():
    """It used to arrive via the audio card's `summary`, which the draft appended to ambient sound —
    so `overall_soundscape` carried "a spoken vocal reference supplying voice timbre and delivery,
    described by the caller as: ... (6.00s)". That section is for the target video's ambience. And the
    definition line said "containing a spoken vocal layer", which tells the encoder nothing, while
    being the only channel it has for that audio."""
    from h3ir.analyse import analyse_audio
    from h3ir.render import render_ir

    ref = AssetRef(kind=AssetKind.AUDIO, role=Role.VOICE_TIMBRE, sha256="a" * 64, path="/tmp/v.wav",
                   note="his own voice, calm and low, with a slight rasp", seconds=6.0)
    brief = Brief(intent="the man speaks in the corridor", seconds=8.0, assets=[ref],
                  dialogue=[DialogueLine(text="Something has been down here a long time.")])
    plan = deterministic_draft(brief, Mode.REF2VA, {ref.sha256: analyse_audio(ref, "a line.")})
    out = render_ir(plan, ProfileOptions(name="standard")).prompt

    defs = out.split("subject_definitions:")[1].split("summary:")[0]
    sound = out.split("overall_soundscape:")[1].split("non_diegetic_music:")[0]
    assert "calm and low" in defs, defs
    assert "described by the caller" not in out
    assert "(6.00s)" not in sound and "calm and low" not in sound
    assert "for the speaker (S1)" in defs, "and it reads as English when no subject is known"


def test_an_uncharacterised_voice_reference_is_reported():
    """A role that claims a sonic property with nothing describing it is a reference carrying no
    information. Reported rather than guessed at, and it names what to supply."""
    from h3ir.analyse import analyse_audio
    from h3ir.compile import _assess
    from h3ir.plan import ProfileOptions as PO

    ref = AssetRef(kind=AssetKind.AUDIO, role=Role.VOICE_TIMBRE, sha256="a" * 64,
                   path="/tmp/v.wav", seconds=6.0)          # no note
    brief = Brief(intent="the man speaks", seconds=8.0, assets=[ref],
                  dialogue=[DialogueLine(text="a line.")])
    plan = deterministic_draft(brief, Mode.REF2VA, {ref.sha256: analyse_audio(ref, "")})
    _, findings, _ = _assess(plan, brief, Mode.REF2VA, PO(name="standard"), [])
    rules = {f.rule for f in findings}
    assert "X15-audio-uncharacterised" in rules, sorted(rules)

    with_note = AssetRef(kind=AssetKind.AUDIO, role=Role.VOICE_TIMBRE, sha256="a" * 64,
                         path="/tmp/v.wav", note="calm and low", seconds=6.0)
    brief2 = Brief(intent="the man speaks", seconds=8.0, assets=[with_note],
                   dialogue=[DialogueLine(text="a line.")])
    plan2 = deterministic_draft(brief2, Mode.REF2VA, {with_note.sha256: analyse_audio(with_note, "")})
    _, f2, _ = _assess(plan2, brief2, Mode.REF2VA, PO(name="standard"), [])
    assert "X15-audio-uncharacterised" not in {f.rule for f in f2}


def test_the_service_can_receive_transcripts_but_never_makes_them():
    """The plumbing existed and no caller could reach it: `compile_brief(transcripts=...)` was never
    exposed, so the app that owns the Whisper call had nowhere to put the result."""
    from h3ir.service import BriefIn

    assert "transcripts" in BriefIn.model_fields
    assert BriefIn(intent="x").transcripts == {}
    b = BriefIn(intent="x", transcripts={"abc": "the words"})
    assert b.transcripts == {"abc": "the words"}


def test_an_attached_reference_the_text_never_mentions_is_an_error():
    """`L4-unused-media` was guarded by `and used[kind]`, which skipped the check whenever a kind was
    used ZERO times — the worst case, not an exempt one. Found by compiling the first
    video-plus-audio brief anyone has looked at: it published a manifest entry for `<Audio 1>` that the
    app would wire into the render, against text that never mentions the asset. Silent."""
    text = _ref2va(
        defs=("<Subject 1> is the wall torch in <Video 1>, with a cage-style grate.\n"
              "<Video 1> is the source video for the target video continuation."),
        summary="[video continuation] The target video continues from <Video 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the grate is retained.\n"
                   "<Video 1> (continuation source): fully_preserved - the location is held."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1>.")
    ctx = Context(n_pictures=0, n_videos=1, n_audios=1, generation_task=False)
    errs = {f.rule for f in validate(text, ctx) if f.severity == "ERROR"}
    assert "L4-unused-media" in errs, errs

    # partially used stays a WARN: some bound, some not, which is a different and lesser mistake
    ctx2 = Context(n_pictures=0, n_videos=2, n_audios=0, generation_task=False)
    warns = {f.rule for f in validate(text, ctx2) if f.severity == "WARN"}
    assert "L4-unused-media" in warns


def test_the_ask_names_every_wired_label_not_only_the_subject_ones():
    """A model cannot bind a label it was not told exists. `_definition_lines` walks `subjects`, so an
    `<Audio N>` never appeared in the ask at all."""
    import json

    import httpx

    from h3ir.grid import Target
    from h3ir.prose import compose_brief

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}], "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    compose_brief(b, Brief(intent="continue this shot", seconds=8), [], {}, Target.build(8),
                  ("<Video 1>", "<Audio 1>"))
    sent = captured["messages"][-1]["content"]
    if isinstance(sent, list):
        sent = " ".join(p.get("text", "") for p in sent)
    assert "<Video 1>" in sent and "<Audio 1>" in sent
    assert "must be referenced" in sent


def test_a_caller_who_pins_t2va_with_assets_does_not_get_a_manifest_it_cannot_bind():
    """Mode inference never pairs assets with T2VA; an explicit `brief.mode` can, and nothing checked
    it. The manifest is the app's wiring contract, so publishing an entry the text cannot reference is
    the arm5/arm6 failure in reverse — an asset wired into the render against a prompt that never
    mentions it."""
    from h3ir.mode import infer_mode

    brief = Brief(intent="a lighthouse keeper lights the lamp", seconds=5, mode=Mode.T2VA,
                  assets=[AssetRef(kind=AssetKind.IMAGE, role=Role.SUBJECT, sha256="i1",
                                   px=(1024, 1024))])
    assert infer_mode(brief).rule_fired == "caller-override", "the override is still honoured"

    # and the draft for that brief, once the assets are dropped, is valid
    stripped = Brief(intent=brief.intent, seconds=5, mode=Mode.T2VA)
    plan = deterministic_draft(stripped, Mode.T2VA, {}, opts=ProfileOptions())
    assert plan.manifest == []
    errs = [f for f in validate(render_ir(plan, ProfileOptions()).prompt,
                                Context(mode="t2va", duration_s=plan.target.effective_seconds,
                                        n_pictures=0, n_videos=0, n_audios=0))
            if f.severity == "ERROR"]
    assert not errs, [str(e) for e in errs]


def test_an_invented_audio_provenance_claim_is_rejected():
    """From the first video-plus-audio brief anyone compiled: the model wrote "<Audio 1> is the ambient
    sound track from <Video 1>" for an audio asset wired STANDALONE. It could not know the wiring and
    guessed from both assets existing. The runtime pairing is a fact this layer holds, so the guess is
    checkable — and since nothing here can hear, an invented claim about audio is the exact class the
    audio path exists to refuse."""
    text = _ref2va(
        defs=("<Subject 1> is the wall torch in <Video 1>, with a cage-style grate.\n"
              "<Audio 1> is the ambient sound track from <Video 1>, containing the torch crackle."),
        summary="[video continuation + audio reuse] The target video continues from <Video 1>.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the grate is retained.\n"
                   "<Audio 1>: fully_copy - the ambience is reused."),
        desc="A style line.\n[Shot 1] The camera holds a static shot on <Subject 1> with <Audio 1>.")
    standalone = Context(n_pictures=0, n_videos=1, n_audios=1, generation_task=False,
                         standalone_audio=("<Audio 1>",))
    assert "R21-audio-provenance-invented" in {
        f.rule for f in validate(text, standalone) if f.severity == "ERROR"}

    # and when the wiring DOES pair them, the same sentence is simply true
    paired = Context(n_pictures=0, n_videos=1, n_audios=1, generation_task=False,
                     paired_audio=(("<Audio 1>", "<Video 1>", 5.0, 5.0),))
    assert "R21-audio-provenance-invented" not in {f.rule for f in validate(text, paired)}

    # AND the case that matters most: a caller who states nothing gets no finding. The first version
    # keyed off `paired_audio` being empty and fired on MiniMax's own published example, whose control
    # declares no pairing while the example's audio genuinely is that video's track.
    silent = Context(n_pictures=0, n_videos=1, n_audios=1, generation_task=False)
    assert "R21-audio-provenance-invented" not in {f.rule for f in validate(text, silent)}


def test_g2_is_a_warning_and_g1_is_an_error():
    """The severity heuristic, and it is a better test than "how confident is the detection":

        A check whose false positive is UNFIXABLE BY THE THING BEING CHECKED must not be an ERROR.

    ERROR means "the model can repair this". When G2 is right it can. When G2 is wrong there is nothing
    wrong in the text, so it cannot converge, the loop exhausts its rounds, and the entire written brief
    is lost to the fallback -- a false positive on a phrase blacklist over free prose costs the whole
    artifact. A false WARN costs nothing and the true positive is still reported.

    G1 stays ERROR because an explicit `<think>` marker cannot be a false positive."""
    common = dict(
        defs="<Subject 1> is the man in <Picture 1>, with dark hair.",
        summary="[reference generation] The target video shows <Subject 1>.",
        retention="<Subject 1> (appears in [Shot 1]): fully_preserved - the hair is retained.")

    narrated = _ref2va(**common, desc="A style line.\n[Shot 1] I'll show the man walking. The camera "
                                      "holds a static shot on <Subject 1>.")
    sev = {f.rule: f.severity for f in validate(narrated, Context(n_pictures=1))}
    assert sev.get("G2-model-self-narration") == "WARN", sev
    assert not [f for f in validate(narrated, Context(n_pictures=1))
                if f.rule.startswith("G2") and f.severity == "ERROR"]

    leaked = _ref2va(**common, desc="<think>let me plan</think>\nA style line.\n[Shot 1] The camera "
                                    "holds a static shot on <Subject 1>.")
    assert {f.rule: f.severity for f in validate(leaked, Context(n_pictures=1))}.get(
        "G1-reasoning-leaked") == "ERROR"
