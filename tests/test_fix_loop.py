"""write -> verify -> fix: the findings go back to the model, not to a bespoke editor.

The owner's design, and better than the code repairers it replaced: an editor per failure mode is
a small parser with its own bugs that only handles anticipated shapes, while the model already
holds the intent behind its own text. Validator findings are unusually good review feedback --
they name the rule, the offence and the line.

What stays in code is only what the model cannot know or must never re-type: label ordinals (the
wiring owns socket order), the task-type prefix (the roles of the attachments derive it), the
caller's dialogue, and unicode hazards.
"""
from __future__ import annotations

import httpx
import pytest

from h3ir.backend import Backend
from h3ir.grid import Target
from h3ir.models import Finding, Mode
from h3ir.prose import fix_with_findings
from h3ir.repair import repair
from h3ir.validate import Context, validate

GOOD = """subject_definitions:
<Subject 1> is the man in <Picture 1>, with short dark hair and a navy t-shirt.

summary:
[reference generation] The target video shows <Subject 1> walking down a corridor.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the short dark hair and navy t-shirt are retained.

detailed_description:
The target video is in live-action style with warm torchlight.
[Shot 1] A wide shot holds on <Subject 1> as he walks forward. The camera pushes in with small amplitude at slow speed, and the torchlight climbs his chest.

overall_soundscape:
Footsteps echo on stone throughout.

non_diegetic_music:
N/A
"""


def _canned(replies: list[str]) -> Backend:
    """A backend that returns the given replies in order."""
    seq = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        # Only the chat path is this fake's business. Anything else -- a dialect probe asking
        # /api/version, say -- gets the 404 a real non-Ollama endpoint answers with, rather than
        # silently consuming a reply from the sequence.
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": "not found"})
        body = seq.pop(0) if seq else ""
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": body}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_the_findings_are_sent_and_the_spec_is_not():
    """The spec is already in the composing call's system prompt. Resending it invites a wholesale
    rewrite instead of a surgical fix, and wastes the context."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": GOOD}}], "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    findings = [Finding("S3-preamble", "ERROR", "brief must begin with 'subject_definitions:'"),
                Finding("T2-shot1-timestamp", "ERROR", "[Shot 1] must not carry a timestamp")]
    out = fix_with_findings(b, "Here is the brief:\n" + GOOD, findings)

    sent = captured["messages"][-1]["content"]
    assert "S3-preamble" in sent and "T2-shot1-timestamp" in sent, "findings must be quoted"
    assert "brief must begin with" in sent, "the message, not just the rule id"
    assert "Reference Labels and Definitions" not in sent, "the spec must NOT be resent"
    assert len(sent) < 12000, "a correction pass should not carry the whole specification"
    assert "change nothing else" in sent.lower()
    assert out.strip().startswith("subject_definitions:")


def test_a_broken_brief_converges_in_one_round():
    from h3ir.compile import MAX_FIX_ROUNDS
    assert MAX_FIX_ROUNDS == 2, "capped, like the truncation ladder"

    broken = "Here is your brief:\n\n" + GOOD.replace("[Shot 1] A wide",
                                                      "[Shot 1] At 00:00.000, A wide")
    ctx = Context(n_pictures=1, duration_s=8.0)
    before = {f.rule for f in validate(broken, ctx) if f.severity == "ERROR"}
    assert before, "the fixture must actually be broken"

    b = _canned([GOOD])
    fixed = fix_with_findings(b, broken, [f for f in validate(broken, ctx)
                                          if f.severity == "ERROR"])
    after = {f.rule for f in validate(fixed, ctx) if f.severity == "ERROR"}
    assert not after, after


def test_a_finding_that_survives_is_recorded_as_a_signal_about_the_rule():
    """If the model cannot fix a finding after being told plainly, either the message is unclear or
    the rule is wrong. That case is worth its own record, not just another retry."""
    ctx = Context(n_pictures=1, duration_s=8.0)
    stubborn = GOOD.replace("<Subject 1> is the man", "<Subject 9> is the man")
    errs = [f for f in validate(stubborn, ctx) if f.severity == "ERROR"]
    assert errs

    # the model returns the SAME text both times
    b = _canned([stubborn, stubborn])
    out1 = fix_with_findings(b, stubborn, errs)
    still = {f.rule for f in validate(out1, ctx) if f.severity == "ERROR"}
    assert still == {f.rule for f in errs}, "unchanged output means the finding survived"


def test_only_errors_reach_the_model():
    """The boundary the rule audit turned on. An ERROR is a decidable fact; a WARN may be a
    judgement a competent director would argue with, and the fix loop is an instruction to change
    the writing. A WARN that reached here would have the model rewrite legitimate work."""
    import inspect
    import re as _re

    from h3ir import compile as compile_mod

    src = inspect.getsource(compile_mod.compile_brief)
    # `errors` is the list handed to fix_with_findings and the loop's own condition. Every
    # assignment to it must filter on ERROR; a single WARN slipping in is the whole failure.
    assigns = _re.findall(r"^\s*errors = \[.*?severity == \"(\w+)\"", src, _re.M)
    assert assigns, "the fix loop no longer builds `errors` the way this test assumes"
    assert set(assigns) == {"ERROR"}, assigns

    # And the artifact-level proof: a brief whose ONLY findings are warnings is not sent anywhere.
    ctx = Context(n_pictures=1, duration_s=8.0)
    found = validate(GOOD, ctx)
    assert not [f for f in found if f.severity == "ERROR"], [str(f) for f in found]
    assert [f for f in found if f.severity == "WARN"], "the fixture must carry warnings to prove it"


# ---------------------------------------------------------------- the minimal code layer

def test_code_still_owns_the_label_namespace():
    """Only the wiring knows socket order, so this cannot be delegated to the model."""
    text = GOOD.replace("<Picture 1>", "<Image 1>")
    res = repair(text, target=Target.build(8), mode=Mode.REF2VA, labels=("<Picture 1>",))
    assert "<Picture 1>" in res.text and "<Image 1>" not in res.text
    assert any("pointed at nothing" in r for r in res.repairs)


def test_code_still_owns_the_callers_dialogue():
    """Their words are never round-tripped through a model."""
    text = GOOD.replace(
        "[Shot 1] A wide shot holds on <Subject 1> as he walks forward.",
        "[Shot 1] <Subject 1> (S1) says, <d>[English] Something has been here a long time.</d>")
    res = repair(text, target=Target.build(8), mode=Mode.REF2VA, labels=("<Picture 1>",),
                 dialogue=("Something's been down here a long time.",))
    assert "<d>[English] Something's been down here a long time.</d>" in res.text
    assert any("exact line" in r for r in res.repairs)


def test_code_still_owns_unicode():
    res = repair(GOOD.replace("torchlight climbs his chest", "torchlight climbs his chest’s edge"),
                 target=Target.build(8), mode=Mode.REF2VA, labels=("<Picture 1>",))
    assert "’" not in res.text
    assert any("normalised" in r for r in res.repairs)


def test_code_no_longer_writes_prose_sections():
    """The bespoke synthesisers are gone. A brief missing retention_analysis is now the model's to
    fix, because inventing it in code was the thing that only ever handled anticipated shapes."""
    import re
    missing = re.sub(r"retention_analysis:.*?(?=detailed_description:)", "", GOOD, flags=re.S)
    res = repair(missing, target=Target.build(8), mode=Mode.REF2VA, labels=("<Picture 1>",),
                 definitions=("<Subject 1> is the man in <Picture 1>, with dark hair.",))
    assert "retention_analysis:" not in res.text, "code must not invent the section any more"
    errs = {f.rule for f in validate(res.text, Context(n_pictures=1, duration_s=8.0))
            if f.severity == "ERROR"}
    assert "S1-missing-section" in errs, "it is reported so the model can fix it"


# ---------------------------------------------------------------- the mode split

def test_the_fix_ask_carries_the_label_inventory_and_the_right_section_count():
    """Both were missing and both cost whole briefs.

    The first eval run of this loop fell back on five of six suite briefs. Every failure was a
    BINDING fault surviving both rounds -- an invented `<Subject 3>`, a `<Picture 4>` that does not
    exist -- and the fix call was being asked to repair a binding with the binding table removed.
    "<Subject 3> is used but never defined" is satisfiable by deleting it OR by defining it, and
    nothing said which labels are real. Separately the ask said "output the corrected six sections"
    unconditionally, which pushed a three-section base brief toward the full-reference format during
    the pass meant to correct it.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": GOOD}}], "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    fix_with_findings(
        b, GOOD, [Finding("L2-undefined-subject", "ERROR", "<Subject 3> is used but never defined")],
        labels=("<Picture 1>", "<Subject 1>"),
        sections=("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"))
    sent = captured["messages"][-1]["content"]
    assert "<Picture 1>" in sent and "<Subject 1>" in sent, "the label inventory must be resent"
    assert "never to define it" in sent, "the inventory must say which direction to fix in"
    assert "3 sections" in sent and "integrated_multimodal_description" in sent
    assert "six sections" not in sent, "a base-mode fix must not be told to emit six sections"
    # still not the spec
    assert "Reference Labels and Definitions" not in sent
    assert len(sent) < 12000


def test_the_composer_is_chosen_by_mode():
    """One composer for both modes was the root cause of a 0.167 clean rate: every base-mode brief
    was handed the full-reference guide, which teaches `<Subject N>`, and then failed L2 for using
    it. Five of six suite briefs fell back, and the sixth -- the only ref2va one -- passed clean."""
    from h3ir.compile import _sections_for
    from h3ir.models import Mode
    from h3ir.prose import load_prompt

    assert len(_sections_for(Mode.REF2VA)) == 6
    assert len(_sections_for(Mode.T2VA)) == 3
    assert len(_sections_for(Mode.I2VA)) == 3

    ref = load_prompt("compose.v2.txt")
    base = load_prompt("compose_base.v1.txt")
    assert "Full-Reference Mode" in ref
    assert "Full-Reference Mode" not in base.split("=== END SPECIFICATION ===")[0]
    assert "there are no `<subject n>` labels in this mode" in base.lower()
    assert "subject_definitions" in ref
    # the base composer names the six-section fields only to forbid them
    body = base.split("=== END SPECIFICATION ===")[1]
    assert "No `subject_definitions`" in body


def test_a_base_mode_written_brief_keeps_its_description():
    """The section split searched only the six-section names, so a base-mode brief came back with no
    `integrated_multimodal_description` at all. Silent, because the validator reads the prompt text
    rather than this dict, and invisible until the base composer stopped every base brief from
    falling back to the draft. The eval reported 0 words on four briefs that were written correctly."""
    from h3ir.compile import _split_written

    base = ("integrated_multimodal_description:\n[Shot 1] Live-action, a wide shot as the knight "
            "rides low over the field. The camera trucks right with small amplitude at slow speed.\n\n"
            "overall_soundscape:\nWingbeats and distant fire throughout.\n\n"
            "non_diegetic_music:\nN/A\n")
    out = _split_written(base)
    assert "integrated_multimodal_description" in out, out.keys()
    assert "[Shot 1]" in out["integrated_multimodal_description"]
    assert out["non_diegetic_music"].strip() == "N/A"

    # and the six-section shape still splits exactly as before
    ref = _split_written(GOOD)
    assert set(ref) == {"subject_definitions", "summary", "retention_analysis",
                        "detailed_description", "overall_soundscape", "non_diegetic_music"}
    assert "[Shot 1]" in ref["detailed_description"]


def test_a_base_mode_with_an_image_is_not_told_nothing_is_attached():
    """An i2va brief has a real `<Picture 1>` and no definition lines. Keying the message off the
    definition lines announced "nothing is attached" over a live reference."""
    import httpx

    from h3ir.grid import Target
    from h3ir.models import Brief
    from h3ir.prose import compose_brief

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": GOOD}}], "usage": {}})

    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    compose_brief(b, Brief(intent="animate this photo", seconds=8), [], {}, Target.build(8),
                  ("<Picture 1>",), prompt_name="compose_base.v1.txt")
    sent = captured["messages"][-1]["content"]
    if isinstance(sent, list):
        sent = " ".join(p.get("text", "") for p in sent)
    assert "<Picture 1>" in sent
    assert "nothing is attached" not in sent


def test_the_eval_harness_does_not_override_the_mode_split():
    """RunConfig.compose_prompt defaulted to an explicit "compose.v2.txt", which forced the
    six-section full-reference composer onto every base-mode brief -- so the harness built to verify
    the mode split measured the exact bug the split had just fixed, and reported clean_rate 0.167
    when the shipping configuration is 1.000.

    Fifth instance of one family: a harness measuring something other than what ships."""
    from h3ir.evalloop.suite import RunConfig

    assert RunConfig().compose_prompt is None, \
        "the suite must let compile_brief pick the composer by mode, as production does"
    assert RunConfig().creativity is None, "same for the dial: no forced position by default"


# ---------------------------------------------------------------- the loop's permanent fixture

def _golden(name: str) -> str:
    from h3ir.config import get_config
    return (get_config().paths.golden_dir / name).read_text(encoding="utf-8")


BASE_MODE_FAILURE = "base_mode_wrote_ref_format.txt"
BASE_CTX = Context(mode="t2va", duration_s=10.125)


def test_the_pinned_failure_is_a_real_failure():
    """A genuine model output, not a constructed one: `t2va_battle` composed against the
    full-reference guide, which is the misconfiguration that actually happened tonight. It came back
    in the six-section shape with `subject_definitions: N/A` — so the base-mode field is missing, the
    ref-mode field is present, and there is no shot in the field the validator reads.

    The repeated-shot brief the owner rejected CANNOT serve this purpose: the taste audit made it
    mechanically clean, so the loop has nothing to fire on. A safety net needs a fixture that still
    fails."""
    errs = {f.rule for f in validate(_golden(BASE_MODE_FAILURE), BASE_CTX)
            if f.severity == "ERROR"}
    assert errs == {"S1-missing-section", "S5-mode-field-crossed", "T1-no-shot",
                    "P4-no-camera-at-all"}, errs


def test_the_loop_fires_on_it_and_is_told_what_it_needs():
    """The wiring is the part that rots — a signature change, a dropped inventory, a WARN slipping
    into the list. The model's ability to repair does not rot from our edits, so this asserts the
    ask is complete and correct; the live repair is recorded in the design doc and re-runnable."""
    from h3ir.validate import BASE_SECTIONS

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": GOOD}}], "usage": {}})

    text = _golden(BASE_MODE_FAILURE)
    errs = [f for f in validate(text, BASE_CTX) if f.severity == "ERROR"]
    b = Backend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    fix_with_findings(b, text, errs, labels=(), sections=tuple(BASE_SECTIONS))

    sent = captured["messages"][-1]["content"]
    for rule in ("S1-missing-section", "S5-mode-field-crossed", "T1-no-shot"):
        assert rule in sent, rule
    # the section list is the information this failure specifically needs
    assert "integrated_multimodal_description" in sent and "3 sections" in sent
    assert "six sections" not in sent
    assert "change nothing else" in sent.lower()
    # and the caller's own material is in front of the model to preserve
    assert "Hold the line!" in sent


# ------------------------------------------------------- the task-type prefix is never the model's

def _repaired(text, task_types=("reference generation", "video editing", "audio reuse")):
    return repair(text, target=Target.build(8), mode=Mode.REF2VA,
                  labels=("<Picture 1>", "<Video 1>", "<Audio 1>"), task_types=task_types)


def test_the_shipped_prefix_is_the_one_the_wiring_derives():
    """Rule 1: structure is computed, never chosen by a model. Measured on two live renders with
    the same wiring shape -- a replacement picture, an edit_source clip and its paired soundtrack --
    where one shipped all three derived types and the other dropped `reference generation`, the one
    the picture is the whole reason for. Nothing compared the two."""
    res = _repaired(GOOD.replace("[reference generation]", "[video editing + audio reuse]"))
    assert "[reference generation + video editing + audio reuse]" in res.text
    assert "[video editing + audio reuse]" not in res.text.replace(
        "[reference generation + video editing + audio reuse]", "")
    assert any("task-type prefix" in r for r in res.repairs), res.repairs


def test_a_summary_that_forgot_the_prefix_is_given_the_derived_one():
    """Rather than spending a fix-loop round on M1 to be told something the code already knows."""
    res = _repaired(GOOD.replace("[reference generation] The target", "The target"))
    assert res.text.count("[reference generation + video editing + audio reuse] The target") == 1
    assert any("added the task-type prefix" in r for r in res.repairs), res.repairs


def test_a_prefix_that_already_agrees_is_left_alone_and_reported_as_nothing():
    """A repair line for a repair that did not happen is a report nobody can trust."""
    res = _repaired(GOOD, task_types=("reference generation",))
    assert res.text.count("[reference generation]") == 1
    assert not [r for r in res.repairs if "task-type" in r], res.repairs


def test_the_prefix_is_replaced_on_either_layout_of_the_section():
    """Both shapes come back from the writer: the prefix on the colon's line and on the next."""
    for head in ("summary:\n[video editing] The target", "summary: [video editing] The target"):
        text = GOOD.replace("summary:\n[reference generation] The target", head)
        res = _repaired(text, task_types=("reference generation",))
        assert "[reference generation] The target" in res.text, head


def test_only_the_opening_prefix_is_touched():
    """A second bracket group inside the summary is M9's business, and an editor that reached for
    it would take the evidence away from the rule that reports it.

    The fixture opens with PROSE and carries the bracket group later, which is the one input that
    tells the two readings apart. Written the obvious way -- a wrong prefix in front and a group
    behind it -- both a match-at-the-start and a search-anywhere pass, and the test proved nothing:
    it survived that mutation the first time it was run against one.
    """
    text = GOOD.replace("[reference generation] The target video shows",
                        "The target video shows [reference generation] and")
    res = _repaired(text, task_types=("reference generation",))
    assert res.text.count("[reference generation]") == 2, res.text
    body = res.text.split("summary:", 1)[1].lstrip()
    assert body.startswith("[reference generation] The target video shows"), body


def test_no_derived_types_means_the_prefix_is_left_exactly_as_written():
    """Every caller that predates this passes nothing, and gets what it always got."""
    res = _repaired(GOOD.replace("[reference generation]", "[video editing]"), task_types=())
    assert "[video editing]" in res.text
    assert not [r for r in res.repairs if "task-type" in r]


def test_the_rule_behind_it_catches_a_prefix_that_disagrees_with_the_wiring():
    ctx = Context(mode="ref2va", n_pictures=1, n_videos=1, n_audios=1,
                  task_types=("reference generation", "video editing", "audio reuse"))
    bad = GOOD.replace("[reference generation]", "[video editing + audio reuse]")
    assert "M16-task-prefix-not-derived" in [f.rule for f in validate(bad, ctx)]


def test_the_rule_abstains_when_the_wiring_did_not_say():
    ctx = Context(mode="ref2va", n_pictures=1)
    bad = GOOD.replace("[reference generation]", "[video editing]")
    assert "M16-task-prefix-not-derived" not in [f.rule for f in validate(bad, ctx)]


def test_a_reordered_prefix_is_not_a_defect():
    """The input that must NOT trip it. Which types apply is decidable from the roles; the order
    they print in is this pack's convention and not something the spec constrains."""
    ctx = Context(mode="ref2va", n_pictures=1, n_videos=1, n_audios=1,
                  task_types=("reference generation", "video editing"))
    text = GOOD.replace("[reference generation]", "[video editing + reference generation]")
    assert "M16-task-prefix-not-derived" not in [f.rule for f in validate(text, ctx)]
