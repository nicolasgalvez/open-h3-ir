# OpenH3-IR

Open-source local Context-IR for MiniMax H3.

[![tests](https://github.com/ruashots/open-h3-ir/actions/workflows/ci.yml/badge.svg)](https://github.com/ruashots/open-h3-ir/actions/workflows/ci.yml)

**Write one prompt. Get better video out of MiniMax H3.**

![A man sandboarding down a dune with a giant white dragon running alongside him, ending on the title OpenH3-IR](https://raw.githubusercontent.com/ruashots/open-h3-ir/main/docs/media/openh3ir-title.webp)

*That title card was made with this compiler: plain prose in, three named reference pictures, and
H3 wrote the music in the same pass as the picture.* **Watch it with sound:**
[openh3ir-title.mp4](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/openh3ir-title.mp4).

## What this is

MiniMax H3 does not want a prompt. It wants a structured document: named sections in a fixed order, every
subject bound to a numbered picture label, cut times that land on a legal frame grid. That document is
what MiniMax calls the Context-IR, and writing it is the whole job here.

MiniMax open-sourced the model but not the stage that writes that document, saying only that
["H3-Context-IR is critical to the quality of the final output"](https://huggingface.co/MiniMaxAI/MiniMax-H3)
and that the way to get one is to call their hosted service. This is an independent open implementation of
that missing layer, running on your own machine, with a dial on top of it.

It talks to any OpenAI-compatible endpoint you already have running: Ollama, llama.cpp's server, LM
Studio, vLLM, or a hosted API. If you already run local models, there is nothing new to install. The
compiler needs no GPU of its own, because the weights live at the endpoint, and nothing here ever
calls MiniMax.

Three ways to reach it, and none of them is the poor relation:

- **In ComfyUI**, four nodes and a workflow that ships ready to run, from
  [their own repository](https://github.com/ruashots/ComfyUI-OpenH3-IR). Those nodes install this
  package and run it inside ComfyUI's own Python, so there is nothing to start.
- **Over HTTP**, one API for an application to call.
- **From the command line**, for trying things out and for scripting.

One thing worth saying plainly: this is a young project, built because I need it for another
application I am making, which is why it is shaped the way it is. Expect changes as I go, and pin a
commit if you build on it.

## The difference, a clip of something vs a performance

![The same request, sent raw on the left and compiled on the right](https://raw.githubusercontent.com/ruashots/open-h3-ir/main/docs/media/off-vs-on.webp)

*Same model, same seed, same reference image. The only difference is the words.*

Both halves are the same request: *"she walks out onto the wet gantry in the rain and stops when she
sees the city below."* On the left the prompt goes to H3 as typed. On the right it goes through
`h3ir` first. Nothing else moved between the two runs: same reference image, same seed, same 10.125
seconds, same settings.

The right side does what the prompt asked. She walks out along the gantry, and at five seconds it
cuts to a low-angle close-up of her looking down at the city.

The left side is a good-looking clip that never arrives. It cannot decide where she is going, walking
toward camera for the first half and away from it in the second. Nothing on that side is badly rendered, and that is the point:
the model was fine, the words were the problem.

That is one pair, but it is the pattern I keep seeing. Run the same prompt again and again at flat
defaults, dial untouched, and the character and the ambience come back the same either way, while the
compiled side keeps arriving with more direction in it: sometimes mild, sometimes a lot, never
overdone. The difference I care about is the one between a clip of the thing and a clip of the thing
that actually means something.

**Watch it with sound:** [off-vs-on.mp4](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/off-vs-on.mp4). H3 generates the rain and the
music in the same pass as the picture, and GitHub cannot play a repo-hosted mp4 inline, so the
animation above is the silent version and the file is the one with audio. The brief that produced the
right half is committed beside it:
[`off-vs-on.compiled-brief.txt`](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/off-vs-on.compiled-brief.txt).

## Quick start

```bash
pip install open-h3-ir
```

That is the compiler: the `h3ir` command and the HTTP service. Point it at a language model you
already run, and compile:

```bash
export H3IR_LLM_URL=http://your-endpoint:8000/v1   # your own OpenAI-compatible endpoint
export H3IR_LLM_MODEL=qwen3.8                      # which model on it, if it serves more than one

h3ir doctor                                        # says what is actually answering
h3ir compile "a lighthouse keeper lights the lamp in a storm" --seconds 10
```

**In ComfyUI you install none of that by hand, and you start no service.** Search for
**OpenH3-IR** in ComfyUI Manager, or clone [ComfyUI-OpenH3-IR](https://github.com/ruashots/ComfyUI-OpenH3-IR)
into `custom_nodes`. It depends on this package, installs it for you, and runs it inside ComfyUI's
own Python. Your language model's address goes on a node instead of in a variable.

Clone this repository to run the compiler from source, or to change anything:

```bash
git clone https://github.com/ruashots/open-h3-ir.git
cd open-h3-ir
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

**When your endpoint serves more than one model, set `H3IR_LLM_MODEL`.** Ollama usually serves
several. The compiler reads your reference pictures through the model, so it needs a model that can
look at images. The list of models an endpoint serves never says which of them can.

So the compiler does not take the first id on that list. It stops, prints the ids it found, and you
pick one. `h3ir doctor` then sends a test picture to the model you picked and says whether it saw
it. An endpoint that serves one model gives you nothing to pick, so leave the variable unset.

## A dial, for how far it goes

![the same request at restrained on the left and extreme on the right](https://raw.githubusercontent.com/ruashots/open-h3-ir/main/docs/media/dial-restrained-vs-extreme.webp)

*"the car rolls into the showroom and stops under the lights."* Run twice, changing one flag.

`restrained` stays on the car and keeps its hands still: one slow low-angle track, one cut at six
seconds to a held medium shot, no music, and the room never really revealed. `extreme` turns the same
prompt into a car commercial. It opens wide on the empty showroom, cuts at three and a half seconds
to a close-up panning along the front wheel, finishes on a low-angle push-in, and puts music under
all of it. Two shots became three, and the camera stopped being polite.

Both reference plates are committed, so this runs as written:

```bash
h3ir compile "the car rolls into the showroom and stops under the lights" \
  --image docs/media/plate-car.jpg \
  --image docs/media/plate-showroom.jpg \
  --seconds 10 --creativity extreme
```

Swap `extreme` for `restrained` and you get the left half. Four positions in all: `restrained`,
`balanced` (the default), `bold`, `extreme`. What changes is how much the writer introduces that
you never asked for, and an explicit "no dialogue" at `extreme` still means no dialogue.

**Watch it with sound:** [dial-restrained-vs-extreme.mp4](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/dial-restrained-vs-extreme.mp4).
Both briefs are committed too, so you can read exactly what the flag did:
[restrained](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/dial-restrained.brief.txt) and
[extreme](https://github.com/ruashots/open-h3-ir/blob/main/docs/media/dial-extreme.brief.txt).

## Who is directing

The dial sets how far the writing can go. A director sets whose taste it goes with. Both are off
until you turn them on.

A director is a name and a paragraph of plain prose. There is no form to fill in and no order to
follow. You are describing taste, in your own words, at whatever length you want. Camera, framing,
light, colour, performance, pace, sound: write about the ones you care about and skip the rest.

Seven come with the project as examples of the shape. Read all seven before you write your own.
This command needs no model and nothing running:

```console
$ h3ir directors
whose taste fills what your prompt and your references leave open

  cameron      James Cameron        The camera is mounted and travelling — it ri...
  tarantino    Quentin Tarantino    The camera sits still and keeps holding afte...
  anderson     Wes Anderson         The camera sits dead centre and moves only a...
  villeneuve   Denis Villeneuve     The camera stays still for a long time, and ...
  bigelow      Kathryn Bigelow      The camera is carried and reacting rather th...
  wong         Wong Kar-wai         The camera watches from just outside the mom...
  spielberg    Steven Spielberg     The camera advances steadily onto whoever is...

`h3ir directors <id>` prints the whole thing, which is what a profile is: prose.
shot count and cut times are never a director's; anything your prompt states explicitly outranks one.
```

```bash
h3ir compile "she steps off the train and looks for a face in the crowd" \
  --seconds 10 --director wong
```

**Two things a director never does.** You can build on both.

A director never sets how many shots there are or where they cut. Pin `shots` and the count is
yours. Leave `shots` on `auto` and the writer decides the edit. That is what it does with no
director at all.

A director never beats your own prompt. Anything you state outright wins, one thing at a time.
Write "a locked-off wide" and you get a locked-off wide, whoever is directing. The light and the
sound are still theirs.

**The name never reaches the writing model. Only the paragraph does.** Type "Wes Anderson" at a
model and it copies famous scenes back at you. That is imitation. The paragraph describes how he
works instead, and that is the part that can steer a scene he never shot. So the name stays here, on
the report and in your saved workflow, and travels no further.

For the same reason, none of the seven names a film, quotes a line or lays out a shot. They describe
habits instead. That is what makes them worth editing rather than copying. All seven are yours to rename, rewrite or
delete.

All three doors send the same paragraph. In ComfyUI it is the fourth node, **OpenH3-IR Director**.
That node is also where you write and keep your own. Leave it out of the graph and nothing steers.

Over HTTP, `director` names one of the seven by id, and `director_profile` carries a paragraph of
your own. From the command line, `--director` takes one of the seven ids.

## Call the service from your own code

The API is the product and the other two doors are its clients. No field that affects the output is
reachable from only one of them, and no path skips the validator. The install is the one in the
[quick start](#quick-start) above.

```bash
h3ir serve --port 8420
curl -s localhost:8420/v1/briefs -H 'content-type: application/json' \
  -d '{"intent":"a lighthouse keeper lights the lamp in a storm","seconds":10}'
```

`intent` is the only required field. Every response comes back in three layers, so a screen never has
to read a format it does not care about: `presentation` is plain language for showing a person, `plan`
is the creative decisions somebody wants to change, and `ir` is the document plus the manifest for
whoever wires the render. `GET /v1/capabilities` reports the legal durations, aspects and asset limits,
so a caller never hardcodes them. `GET /v1/contract` reports every field name, every role and every
refusal code this build takes. It carries a version number of its own, 3 in this release. A client
reads it and checks itself against the service before it sends anything. A field this
service does not know is refused by name, never dropped.

Set `"llm": {"num_ctx": 32768}` on a brief to raise the reasoning model's context window for that
request only. There is no environment variable for it, because the machine the compiler runs on
does not know how big any particular brief is going to be — the caller attaching six reference
pictures does. It is sent as `{"options": {"num_ctx": n}}` on every call that brief makes and
omitted entirely when left unset. Ollama's own default context is 4096 tokens, and a brief with
pictures routinely needs more than that; an endpoint that does not recognise the field ignores or
rejects it, so set it only when talking to one that honours it.

Attachments arrive two ways. A caller that shares a filesystem with the service names a path, and
nothing is copied. A caller on another machine sends the bytes to `PUT /v1/assets/{sha256}` and then
names the file by that hash. The same file is never sent twice.

Under-specification never fails. `{"intent":"make a video of my dog"}` and nothing else comes back
`201` with a complete, zero-error brief: five seconds, widescreen, the edit and the sound picked for
you. Routes, request shapes, and what the service guarantees against what it only attempts:
[`docs/calling-the-api.md`](https://github.com/ruashots/open-h3-ir/blob/main/docs/calling-the-api.md).

## Why this is a compiler, not a prompt enhancer

A prompt enhancer makes your words prettier and hopes. Every row below is a place where hoping is not
good enough, because the answer is either mechanically right or the render is wrong.

| what goes wrong when words are all you have | what happens here instead |
| --- | --- |
| A reference is described in prose and nothing ties that description to the actual file | Every attachment gets its own numbered label, and the label in the document is the one the render wires |
| You have to know which of H3's tasks you are asking for | The job is derived from what you attached, never from what you typed, so no screen has to ask |
| The duration gets rounded once for the words and again for the render | The length is snapped onto H3's frame grid once and that one number is used for both |
| Cut times land past the end of the clip, or so close together the cut reads as a glitch | Every cut time is checked against the real length of the clip and against the 1.2 seconds a shot needs to hold |
| Your exact line comes back paraphrased | Dialogue never passes through the writing model, and a brief that reworded a locked line is refused |
| Nothing states what has to stay the same about a reference | Each one carries a stated retention in the document, and that statement is validated |
| A model that writes a broken document is asked to try again | What is mechanical is corrected in place, what needs judgement is reported, and a document that still fails falls back to a deterministic draft |
| Every front end reimplements the rules slightly differently | One compiler behind all three doors, and no path around the validator |

## Ten seconds is not ten seconds

```console
$ h3ir budget --seconds 10
requested 10.0s -> 243 frames = 10.125s (nominal S.SS 10.13)
[…]
```

MiniMax H3 only makes clips whose frame count fits a fixed grid. Inside the range the model was trained on,
5.167s to 15.083s, there are exactly fifteen legal lengths, and **only one of them is a whole number
of seconds** (8.0s, at 192 frames). Ten is not on the grid, so 243 frames at 10.125s is the closest
the model can get.

Ask for a round number and you quietly get something else. It matters the first time you cut to music,
and it matters for every cut time inside the clip, which is why the compiler owns those and the writing
model never picks one.

That trained range is a note rather than a wall. Ask for a length outside it and it still renders, and
the report says so plainly instead of the surface pretending the option does not exist.

The lines cut off above price your references, which is the other thing that command is for. Words are
nearly free and attachments are what cost: one reference image at its full size costs roughly ten times
what the entire written brief does. Write long, attach few. The arithmetic is in
[`docs/design.md`](https://github.com/ruashots/open-h3-ir/blob/main/docs/design.md).

## References decide the job

Attach two images and two subjects come back, each with its own numbered label, its own stated promise
about what has to stay the same about it, and a mention in every shot it appears in. If an image is
ambiguous about which of several things in it you care about, `--image path.png:"the pilot"` says which,
straight to the model that looks at it.

MiniMax H3 does not have one mode, it has five, and each wants the document written differently. Which one a
request needs is settled by what you attached, because that is the only thing that can settle it
correctly. You never pick one and no screen built on this has to ask. The names show up in the report
if you are curious: `t2va`, `i2va`, `fl2va`, `l2va`, `ref2va`.

## Exact dialogue stays exact

```bash
h3ir compile "two engineers argue in a server room while an alarm blinks" --seconds 10 \
  --say "The backup never ran, Mei." \
  --say "Then we tell them tonight."
```

From what came back:

```
[…] The camera holds a static shot as the woman with a sharp, urgent voice (S1) says:
<d>[English] The backup never ran, Mei.</d> The man turns his head slightly toward her, his
expression serious, and replies with a calm, steady tone (S2): <d>[English] Then we tell them
tonight.</d> The red alarm continues to flash in the background […]
```

Your lines never pass through the writing model. It decides who speaks, casts a voice for each of
them, places the lines in the scene, and the renderer substitutes your words back byte for byte. In
ComfyUI the same guarantee is `@speaks("...")` inside the prompt.

## It validates what it writes

More than a hundred named rules, and a rule that cannot be made to fire is not a rule, so every one is
proved in both directions.

```console
$ h3ir controls
  [ok  ] MUST PASS: MiniMax official Ref2VA example (P5 exempt, see note)
  [ok  ] EXPECTED: the official example lacks a motion type
  […]
  [ok  ] MUST FAIL: <Image N> instead of <Picture N>
  […]
23 controls, 0 failing
```

MiniMax's own published examples are in the reference set and have to validate clean, because a rule
that fires on the spec's own artifact is a wrong rule. That direction already caught two rules here
and demoted them to guidance. There is one documented exemption, where MiniMax's example omits a camera
motion type.

Going the other way, sixteen mutants of that example each carry exactly one defect, and each has to
trip the rule that defect earns, by name. The whole gate runs in under a second and needs no model.

```console
$ pytest -q
[…]
990 passed, 1 skipped, 1 warning in 2.95s
```

That suite needs no model, no GPU and no network, which is the point: everything decidable without a
model is decided without one. The one skip is about this machine rather than a hole in the suite: it
wants an `ffprobe` that can measure a webp. Run it with `pip install -e ".[dev]"`.

Legality is not quality, so `h3ir eval` measures the writing separately: it scores six briefs and gates
a change against a stored baseline, because a prompt change can improve one and wreck the other.

## Your first brief from the command line

This is the exact command that produced the right-hand side of the comparison up in
[The difference, a clip of something vs a performance](#the-difference-a-clip-of-something-vs-a-performance),
and `ref1.png` ships in the repo, so you can run it now.

```console
$ h3ir compile "she walks out onto the wet gantry in the rain and stops when she sees the city below" \
    --seconds 10 --image h3ir/golden/assets/ref1.png

mode=ref2va  tokens=708  timings={…}
==========================================================================
ref2va IR
  -> PASS (with warnings)   0 error(s), 2 warning(s), 0 info
==========================================================================
  [WARN] P2-too-short: detailed_description is 265 words; spec guidance 350-500, official example 336
  [WARN] R15-wardrobe-not-restated: [Shot 2] names the subject but not the garments (jacket, shirt,
         t-shirt); wardrobe drifts between shots when it is only stated once

subject_definitions:
<Subject 1> is the woman in <Picture 1>, with short dark hair with shaved sides and a small top knot,
dark complexion, black tactical jacket with shoulder straps and buckles, black t-shirt, black cargo
trousers, black lace-up combat boots, slender build.
[…]
detailed_description:
The target video is in a cinematic, high-contrast style with realistic 3D character design, featuring
cool blue tones and wet, reflective surfaces.
[Shot 1] A medium-long tracking shot follows <Subject 1> from behind as she walks out onto a wet,
metallic gantry in the rain. The camera tracks slowly with small amplitude, keeping her centered in
the frame as she moves away from the viewer. […]
[Shot 2] At 00:05.000, the shot cuts to a close-up of <Subject 1> from a slightly low angle as she
stops at the edge of the gantry. The camera is static, focusing on her face and upper body. She looks
down, her expression shifting to one of quiet contemplation as she sees the city below. […]
[…]
```

A real run, cut at `[…]`, which is the mark every printout on this page uses where something was left
out. Nobody typed `<Subject 1>`, `<Picture 1>`, `00:05.000`, or any section name.
One image path went in with no description of what was in it, and the tactical jacket, the shaved sides
and the combat boots were read off the pixels.

Both findings are warnings rather than errors, so it compiled. The second one is the interesting kind:
Shot 2 names the woman but not her clothes, which is the exact omission that lets wardrobe drift
between cuts. No legality check can see that, so it is a named rule with a reason attached.

## What you need

| requirement | why | if you skip it |
| --- | --- | --- |
| Python 3.10, 3.11 or 3.12 | all three are covered by CI, on main and on every pull request | 3.13 is untested rather than known bad |
| An OpenAI-compatible endpoint | this is where the writing happens | nothing compiles, and `h3ir doctor` says so |
| A model that can also look at images | that is how reference pictures get read | text-only prompts still work, references do not. `h3ir doctor` reports `vision_ok`, so you do not have to guess |
| `ffmpeg` | reading reference clips, nothing else | only needed if you attach video |

No GPU for the compiler itself: the weights live behind the endpoint. Run `h3ir doctor` before you
debug anything else, because it says what is actually answering. It reports:

- Which address replied when it checked the endpoint was alive
- Every model id the endpoint serves, which one will be used, and why
- The context length of that model
- Whether that model can read a picture
- Whether ComfyUI is reachable, and which H3 nodes it has
- A tokenizer self-test.

Three commands need nothing running at all, so you can poke at it before you configure anything:
`h3ir controls`, `h3ir budget --seconds 10` and `h3ir directors`.

Every brief on this page was written by Qwen3.6 27B, 4-bit, served by vLLM at 262K context on two RTX
3090s, and MiniMax H3 rendered the videos from those briefs. That is what the project is proven against and the
bar to size your own box against: a 27B-class local model that can also look at images is enough.
`h3ir eval` is there to measure what a different endpoint does to brief quality rather than guess at it.

Every setting, with the reason for each default, is in [`.env.example`](https://github.com/ruashots/open-h3-ir/blob/main/.env.example).

## What OpenH3-IR does not do

- **It does not make the video itself.** It writes the words and hands over everything the render
  needs. Over HTTP that is a brief plus which file belongs where. In ComfyUI it is the wires that
  feed the Render box, and every box inside there is ComfyUI's own rather than ours.
- **It does not judge whether the writing is good.** It can tell you a shot dropped the wardrobe. It
  cannot tell you the edit is dull.
- **It cannot hear.** The model that reads your files looks and does not listen, and a model asked
  what a piece of music sounds like invents a confident answer rather than admitting that. So a sound is
  described from the line you type about it, plus its own file details, plus a transcript if you have
  one. The transcript is the channel for words, and you supply the rest.
- **It cannot guarantee H3 obeys every reference.** The brief binds the reference and states what must
  be preserved. Whether the model delivers is a render outcome, and the hardest case is `extreme`,
  which reaches for extreme close-ups.
- **It cannot hand back your own footage, and neither can MiniMax H3.** Add something to a clip, or swap
  what is there, and your file is never touched. H3 watches it and makes a new video that follows it
  closely: the same scene, doing the same thing, at the same moments, with your change in place. So
  an edit here is a very close remake, not a repaint of your frames. That is H3's design, not this
  compiler's choice. The brief asks in plain words for everything else to hold, and how close it
  lands is a render outcome like any other.
- **It is deliberately MiniMax H3 specific.** The rules, the frame grid and the section names are H3's.
  Pointing it at another video model is a new compiler target, not a config change.

## Where to go next

| file | what it is for |
| --- | --- |
| [`HANDOFF.md`](https://github.com/ruashots/open-h3-ir/blob/main/HANDOFF.md) | **installing it and verifying it works**, top to bottom, with a check on every step and what to do when one fails |
| [`AGENTS.md`](https://github.com/ruashots/open-h3-ir/blob/main/AGENTS.md) | **contributing**: the rules that are not preferences, which file owns what, the known gaps |
| [`docs/calling-the-api.md`](https://github.com/ruashots/open-h3-ir/blob/main/docs/calling-the-api.md) | driving the service from an application: what it guarantees, what it only attempts, what comes back |
| [`docs/design.md`](https://github.com/ruashots/open-h3-ir/blob/main/docs/design.md) | why every rule exists: what the encoder sees, the cost model, the contract between stages |
| [`docs/build-log.md`](https://github.com/ruashots/open-h3-ir/blob/main/docs/build-log.md) | a dated record of what the build measured, including the positions it reversed |
| [ComfyUI-OpenH3-IR](https://github.com/ruashots/ComfyUI-OpenH3-IR#readme) | **the other repository**: the four nodes in full, the tray, the `@` prompt, the ready-to-run workflow, every failure message |

## Licence

Apache 2.0. See [LICENSE](https://github.com/ruashots/open-h3-ir/blob/main/LICENSE), and [NOTICE](https://github.com/ruashots/open-h3-ir/blob/main/NOTICE) for what belongs to whom.

**That covers this compiler. It does not cover the model you point it at, and MiniMax H3's own licence is
more restrictive than most.** Three terms worth knowing before you build on this,
because none of them is guessable:

- **MiniMax H3 is not licensed for use in the European Union, the United Kingdom, the Republic of Korea or
  the United States of America.** Those are its Excluded Territories, and the grant is worldwide
  except for them. MiniMax invites people there to contact them for a licence.
- A commercial product or service using H3 **shall prominently display "MiniMax H3" in its user
  interface** (section IV.2).
- Commercial products earning **more than 20 million USD a year need separate written authorization**
  from MiniMax first (section IV.1).

Read the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
rather than trusting this summary. This project is independent and unofficial: it is not affiliated
with, endorsed by, or supported by MiniMax, and nothing in this repository is a MiniMax work. No model
code, no weights, no checkpoint.
