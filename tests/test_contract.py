"""The contract is what the compiler promises anything that drives it. These are its own checks.

`h3ir/contract.py` publishes the field names, the vocabularies, the refusals and the limits that a
client has to agree with this build about. It is built from the authorities rather than restating
them wherever an authority exists as an importable object, and where one does not -- the wire field
names live on pydantic models this module may not import -- it holds a literal, and the literal is
pinned here.

This file belongs to the COMPILER. It asks whether the contract tells the truth about this build.
Whether a particular client agrees with it is `tests/test_contract_drift.py`, which belongs to the
node pack.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest
from starlette.testclient import TestClient

from h3ir import contract as C
from h3ir import director as D
from h3ir.models import Role
from h3ir.service import AssetIn, BriefIn, DialogueIn, app

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every section as it stands, so that changing one without saying the contract changed is red rather
# than a discovery somebody makes on a user's machine. A failure here is not a bug: it means the
# thing that crossed the boundary moved, and the fix is to bump `CONTRACT_VERSION` and paste the new
# digest in. That is the whole ceremony, and it exists so the version number means something.
PINNED_VERSION = 3
PINNED = {
    "asset_fields": "38190799fcfb6cf1",
    "brief_fields": "6c9305f7a9743085",
    "dialogue_fields": "a63047779f907e58",
    "error_codes": "19906fa6afaf0a20",
    "roles": "4b85814fed5254ab",
    "directors": "8e50d53980fd8777",
    "camera_moves": "e0ebdca0a4dcaea8",
    "limits": "78554cef21934947",
}


# ------------------------------------------------------ the contract describes THIS build, not one

@pytest.mark.parametrize("section,model", [("asset_fields", AssetIn), ("brief_fields", BriefIn),
                                           ("dialogue_fields", DialogueIn)])
def test_the_contract_names_exactly_the_fields_the_wire_models_take(section, model):
    """The one place the contract holds a literal, held against the thing it describes.

    It has to be a literal: `contract.py` is imported by a client running inside ComfyUI's Python,
    where fastapi and pydantic are not installed and must never be needed. So the list is written
    out over there and pinned here, on the compiler's own side, where both are present.

    Order matters as well as membership. The contract is a document people read to work out what to
    send, and a field list in a different order from the model's is a document that has been edited
    rather than regenerated.
    """
    declared = tuple(getattr(C, section.upper()))
    assert declared == tuple(model.model_fields), (
        f"{model.__name__} takes {tuple(model.model_fields)} and the contract publishes "
        f"{declared}. A client reads the contract and sends what it says.")


def test_a_field_the_model_does_not_take_is_refused_by_name_rather_than_dropped():
    """The quietest failure this service can have, and the one the whole contract exists around.

    pydantic ignores an unknown key by default. So a client one version ahead sent `replaces` on a
    picture, this build dropped it without a word, the compiler bound the swap to whoever the
    analyser happened to find in three sampled frames, and the brief came back valid. Nothing in
    the logs, the brief, the canvas or the message said anything had been lost.

    Refused now, with the two things a reader needs: which field, and that the halves are at
    different versions.
    """
    reply = TestClient(app).post("/v1/briefs", json={
        "intent": "a man crosses a wet yard",
        "assets": [{"path": "/nowhere.png", "kind": "image", "replacez": "the man in the plaid"}]})
    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert detail["code"] == "unknown-field"
    assert "replacez" in detail["message"], "the refusal does not name the field it refused"
    assert "GET /v1/contract" in detail["message"], "the refusal does not say where the list is"


def test_a_top_level_key_this_build_does_not_know_is_refused_too():
    """Same rule one level up. A whole setting silently ignored is worse than one field, not better."""
    reply = TestClient(app).post("/v1/briefs", json={"intent": "a man crosses a wet yard",
                                                     "grading": "teal and orange"})
    assert reply.status_code == 422
    assert reply.json()["detail"]["code"] == "unknown-field"
    assert "grading" in reply.json()["detail"]["message"]


def test_a_request_that_is_merely_malformed_says_so_in_the_same_shape():
    """Every client in this repository reads `detail` as an object with a code and a message.
    FastAPI's own 422 body is a list of pydantic error objects, which is precise and unreadable, so
    a bad type must not be the one refusal that arrives in a different shape."""
    reply = TestClient(app).post("/v1/briefs", json={"intent": "x", "seconds": "soon"})
    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert detail["code"] == "malformed-request"
    assert "seconds" in detail["message"]


# ------------------------------------------------------------------ the vocabularies are complete

def test_every_role_belongs_to_at_least_one_kind():
    """The bug this catches already shipped, in the other direction.

    `structure` was added to the enum and not to the roles-by-kind table, which was read only to
    build an error message. So a caller who typo'd a video role was handed a list of the roles a
    video can have, with a legal one missing, and was told the role they wanted does not exist.

    A published vocabulary makes that a fact somebody can check rather than a message nobody reads.
    """
    placed = {r for roles in C.ROLES_BY_KIND.values() for r in roles}
    missing = sorted(r.value for r in Role if r not in placed)
    assert not missing, (
        f"these roles are in the enum and on no kind: {missing}. A caller naming one is told it "
        "does not exist, and a client reading the contract cannot offer it.")


def test_no_kind_claims_a_role_that_is_not_a_role():
    every = set(Role)
    for kind, roles in C.ROLES_BY_KIND.items():
        assert set(roles) <= every, f"{kind} claims something that is not a Role"
        assert len(set(roles)) == len(roles), f"{kind} lists a role twice"


def test_the_contract_lists_every_refusal_the_service_can_raise():
    """A client branches on these to say something useful. A code raised over there with no branch
    over here is a caller that shrugs at a failure it could have explained.

    Read out of the source rather than listed twice, and out of BOTH files: `over-capacity` is
    raised by a `BriefRefused` subclass in `compile.py`, so a scan of `service.py` alone missed it
    once already.
    """
    src = (REPO / "h3ir" / "service.py").read_text(encoding="utf-8")
    chunks = re.split(r"\n(?=@app\.)", src)
    on_assets = [c for c in chunks if re.match(r'@app\.\w+\("/v1/assets', c)]
    assert on_assets, "no /v1/assets route was found, so the route half of this scan is blind"

    def codes(text):
        return set(re.findall(r'"code": "([a-z-]+)"', text))

    raised = {code: [] for code in codes(src)}
    for code in codes("\n".join(on_assets)):
        raised[code].append(C.ASSETS)
    for code in codes("\n".join(c for c in chunks if c not in on_assets)):
        raised[code].append(C.BRIEFS)
    # `BriefRefused` carries its own literal code and is re-raised by the brief route, so a scan of
    # service.py alone cannot see any of them. BOTH shapes are read here: raised directly with the
    # code as its first argument, and raised through a subclass that passes it to `super()`. The
    # version of this scan that looked only for `super()` saw one refusal out of twelve, and the
    # eleven it missed included every one about who a picture replaces.
    compiler = (REPO / "h3ir" / "compile.py").read_text(encoding="utf-8")
    refusals = set(re.findall(r'BriefRefused\(\s*\n?\s*"([a-z-]+)"', compiler))
    refusals |= set(re.findall(r'super\(\)\.__init__\("([a-z-]+)"', compiler))
    assert len(refusals) > 1, (
        "only one refusal was found in compile.py, which is what the scan this replaced reported "
        "while eleven others existed. The pattern has stopped matching.")
    for code in refusals:
        raised.setdefault(code, []).append(C.BRIEFS)

    assert raised, "no refusal codes were found in the source, so this scan is blind"
    assert not set(raised) - set(C.ERROR_CODES), (
        f"these refusals are raised and not published: {sorted(set(raised) - set(C.ERROR_CODES))}")
    assert not set(C.ERROR_CODES) - set(raised), (
        f"these refusals are published and nothing raises them: "
        f"{sorted(set(C.ERROR_CODES) - set(raised))}")
    for code, routes in raised.items():
        assert sorted(set(routes)) == sorted(C.ERROR_CODES[code]["on"]), (
            f"{code} is raised on {sorted(set(routes))} and the contract says "
            f"{sorted(C.ERROR_CODES[code]['on'])}. A client writes a different message per route.")


def test_every_published_refusal_states_a_status_and_a_route():
    """The two facts a client needs about a refusal before it can write a message for it."""
    for code, spec in C.ERROR_CODES.items():
        assert isinstance(spec.get("status"), int) and 400 <= spec["status"] < 600, code
        assert spec.get("on") and set(spec["on"]) <= {C.BRIEFS, C.ASSETS}, code


def test_the_directors_and_the_camera_moves_are_the_compilers_own_objects():
    """Not a copy of them. If this ever needs a transformation, the transformation is the drift."""
    published = C.snapshot()
    assert published["directors"] == [D.to_mapping(d) for d in D.DIRECTORS]
    assert published["camera_moves"] == list(D.CAMERA_MOVES)
    assert published["limits"]["director_notes_max_chars"] == D.MAX_NOTES_CHARS


def test_the_service_publishes_one_statement_of_the_lists_it_shares_with_the_contract():
    """`/v1/capabilities` used to hold its own copy of the aspects and the languages, beside the
    node pack's copy and the contract's. Three statements of one list."""
    caps = TestClient(app).get("/v1/capabilities").json()
    assert caps["aspects"] == list(C.ASPECTS)
    assert caps["dialogue_languages"] == list(C.DIALOGUE_LANGUAGES)
    assert caps["output"]["fps"] == C.snapshot()["limits"]["fps"]
    assert caps["contract"]["version"] == C.CONTRACT_VERSION, \
        "capabilities names a contract version that is not this build's"


# -------------------------------------------------------------------- the version means something

def test_the_version_moves_when_any_part_of_the_contract_moves():
    """The gate that stops `CONTRACT_VERSION` becoming a number nobody maintains.

    A failure here is not a defect. It means something crossing the boundary changed, which is
    allowed and expected -- and the contract has to SAY so, because a client compares the number
    before it compares anything else. Bump the version, paste the new digest above, and the two
    facts stay attached to each other.
    """
    live = C.digests()
    moved = {name: (PINNED[name], live[name]) for name in PINNED if PINNED[name] != live[name]}
    added = sorted(set(live) - set(PINNED))
    if moved or added:
        assert C.CONTRACT_VERSION > PINNED_VERSION, (
            f"the contract changed and its version did not. Moved: {moved}. New sections: "
            f"{added}. Bump CONTRACT_VERSION and update PINNED in this file.")
    else:
        assert C.CONTRACT_VERSION == PINNED_VERSION


def test_every_section_is_named_and_digested():
    """A section in the document with no digest is a section a client cannot compare, which is the
    same as it not being in the contract at all."""
    published = C.snapshot()
    for name in C.SECTIONS:
        assert name in published, f"{name} is named as a section and is not published"
        assert name in published["digests"], f"{name} is published with no digest"
    body = {k for k in published
            if k not in ("contract_version", "digests", "field_lists_describe")}
    assert body == set(C.SECTIONS), (
        f"these are published and are not comparable sections: {sorted(body - set(C.SECTIONS))}")


# ---------------------------------------------------------------------- the three ways it is read

def test_the_document_says_which_surface_its_field_lists_describe():
    """A reader who takes `asset_fields` for `AssetRef`'s fields builds the wrong thing, and the
    two lists are similar enough to be mistaken for each other. The document says so itself rather
    than leaving it to whoever finds the docstring."""
    published = C.snapshot()
    assert "POST /v1/briefs" in published["field_lists_describe"]
    assert "h3ir.models" in published["field_lists_describe"]
    # and the claim is true: the dataclass carries fields no caller may state
    from dataclasses import fields

    from h3ir.models import AssetRef

    assert {f.name for f in fields(AssetRef)} != set(C.ASSET_FIELDS), \
        "the two surfaces are identical, so this warning is now misleading"
    assert "role_stated" not in C.ASSET_FIELDS, \
        "a compiler-owned field reached the list of things a caller may send"


def test_the_endpoint_and_the_import_answer_the_same_thing():
    served = TestClient(app).get("/v1/contract").json()
    imported = C.contract()
    assert served == imported, "the HTTP surface and the import disagree about the contract"


def test_the_endpoint_says_which_build_is_answering_and_the_file_does_not():
    """`package_version` is the one field that describes an installation rather than a contract.

    It belongs in a live answer, so somebody can say "the service at that address is 0.2.0", and it
    must NOT be in the copy written into another repository: baking one install's metadata into a
    generated file makes regenerating it produce different bytes on different machines. Not
    hypothetical -- this checkout's own metadata and its pyproject disagree.
    """
    assert "package_version" in TestClient(app).get("/v1/contract").json()
    assert "package_version" not in C.snapshot()
    assert "package_version" not in json.loads(C.as_json())


def test_the_written_contract_is_the_same_bytes_every_time():
    assert C.as_json() == C.as_json()
    assert C.as_js() == C.as_js()


def test_the_command_prints_what_the_module_builds():
    """`h3ir contract` is how the node pack's copy gets regenerated, so it has to be the same bytes
    the module produces and not a rendering of them."""
    for args, expected in ((["contract"], C.as_json()), (["contract", "--js"], C.as_js())):
        out = subprocess.run([sys.executable, "-m", "h3ir.cli", *args], cwd=REPO,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout == expected, f"`h3ir {' '.join(args)}` printed something else"


# ------------------------------------------------------- the browser copy carries the prose intact

def test_the_generated_module_carries_every_profile_word_for_word():
    """The generator is the only thing standing between the compiler's prose and what the panel
    shows. A quoting bug in it would be invisible: the panel would draw a paragraph that looks
    right and hand the writer something else.

    Read back out of the emitted JavaScript as JSON, which is what it is: every value is written
    with `json.dumps`, so every escape in the prose round-trips or this fails.
    """
    js = C.as_js()
    notes = re.findall(r"^    notes: (\".*\"),$", js, re.MULTILINE)
    names = re.findall(r"^    name: (\".*\"),$", js, re.MULTILINE)
    ids = re.findall(r"^    id: (\".*\"),$", js, re.MULTILINE)
    assert len(notes) == len(D.DIRECTORS), \
        f"the generated module holds {len(notes)} directions and the compiler ships {len(D.DIRECTORS)}"
    for got_id, got_name, got_notes, real in zip(ids, names, notes, D.DIRECTORS):
        assert json.loads(got_id) == real.id
        assert json.loads(got_name) == real.name
        assert json.loads(got_notes) == real.notes, f"{real.id}: the prose did not survive the trip"


def test_the_generated_module_carries_the_moves_and_the_cap():
    js = C.as_js()
    block = js[js.index("export const CAMERA_MOVES = ["):js.index("export const DIRECTORS = [")]
    assert tuple(json.loads(m) for m in re.findall(r'("(?:[^"\\]|\\.)*")', block)) == D.CAMERA_MOVES
    assert re.search(rf"^export const MAX_NOTES = {D.MAX_NOTES_CHARS};$", js, re.MULTILINE)
    assert re.search(rf"^export const CONTRACT_VERSION = {C.CONTRACT_VERSION};$", js, re.MULTILINE)
    assert re.search(rf'^export const DIRECTORS_DIGEST = "{C.digests()["directors"]}";$', js,
                     re.MULTILINE)


def test_the_generated_module_says_it_is_generated_and_how_to_regenerate_it():
    """It sits in a repository beside hand-written files. Somebody will open it to fix a typo in a
    profile, and the file has to tell them where the typo actually lives."""
    js = C.as_js()
    assert "GENERATED by `h3ir contract --js`" in js
    assert "Do not edit" in js
