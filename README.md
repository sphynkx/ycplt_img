# ycplt_img

A local image generation/editing daemon built on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(via the `stable-diffusion-cpp-python` bindings). It works as a passive job
queue: it accepts a job over HTTP, stores it in a SQLite queue, a worker
processes jobs strictly sequentially on an already-loaded model, and the
client polls for status and fetches the result itself — the service never
pushes anything on its own initiative.

Intended for a dedicated machine with a modern CPU (**AVX2 is required** —
this will not build/run on old CPUs without it) and enough RAM/SSD for the
model checkpoint.

## Layout

```
app.py                    — entry point: preloads the default model, starts the worker and the HTTP server
conf/
  config.py                 — configuration via environment variables (host/port, model path, TTL, ...)
  models.py                  — model factory: picks/lazily loads the right checkpoint per job mode
                               (also the vision/caption model, moondream2 via llama-cpp-python)
  segmentation.py             — CLIPSeg: automatic object masks for "remove X" edits (mode="img2img" + remove_target)
db/
  db.py                      — SQLite job queue (schema, CRUD, TTL purge)
srv/
  server.py                  — HTTP JSON API on the standard library (no FastAPI/Flask)
  worker.py                  — background worker: one job at a time, on the already-loaded model
install/
  requirements.txt            — Python dependencies
  ycplt_img.service            — systemd unit template (adjust paths/user)
  .env.example                — template for .env (copy to project root as .env, not tracked by git)
models/
  .gitempty                   — placeholder so the (empty) directory is tracked by git; put the checkpoint here
.gitignore                  — ignores .env, .venv/, data/, model files, etc.
```

## Install

```bash
sudo dnf install -y gcc gcc-c++ cmake git python3

cd /opt
git clone https://github.com/sphynkx/ycplt_img
cd ycplt_img

python3 -m venv .venv
source .venv/bin/activate

pip install -r install/requirements.txt   # builds stable-diffusion-cpp-python from source
```

The model checkpoint (SD1.x, `.ckpt`/`.safetensors`) is not part of the
repository — download it separately:

```bash
wget -O models/sd-v1-4.ckpt "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
```

Optionally, SD1.5 is a slightly refined successor to SD1.4 (same
architecture and speed, marginally better general quality) and works as a
drop-in replacement — same `-m`/`model_path` loading, nothing else to
change:

```bash
wget -O models/v1-5-pruned-emaonly.safetensors "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
# then in .env: MODEL=v1-5-pruned-emaonly.safetensors
```

For meaningfully better quality at the same architecture/speed, see
"Editing and inpainting models" below — the recommended checkpoint there
(Realistic Vision V6.0 B1) is a drop-in replacement for `MODEL` too, not
just for inpainting.

For editing an uploaded image with a **mask** (`mode="inpaint"` — not yet
sent by the main app, which currently only sends whole-image `img2img`
edits, but supported end to end here), a base checkpoint gives noticeably
worse results than one fine-tuned for inpainting — see "Editing and
inpainting models" below for why, and a recommended checkpoint + download
link.

Permit port:
```bash
sudo firewall-cmd --permanent --add-port=4011/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

## Configuration (.env)

There are no hardcoded host/port/model values — everything lives in a single
`.env` file in the project root (not tracked by git), loaded via
[python-dotenv](https://pypi.org/project/python-dotenv/):

```bash
cp install/.env.example .env
# edit .env: YCPLT_HOST, YCPLT_PORT, MODEL=sd-v1-4.ckpt (must exist under models/)
```

`conf/config.py` resolves the checkpoint path as `YCPLT_MODELS_DIR/MODEL`
(`models/sd-v1-4.ckpt` by default). Several different checkpoints may be in
use over time — to switch models, drop the new checkpoint into `models/` and
change the `MODEL=` line, then restart; no code or systemd unit changes needed.

Priority order for every setting: a real process environment variable (e.g.
set by systemd) wins over `.env`, which wins over the hardcoded default —
this is `python-dotenv`'s default behavior (it never overrides variables
already present in the environment). See `install/.env.example` for the full
list of overridable variables.

Run (foreground, for a manual check):

```bash
python app.py
```

On first load the model is quantized on the fly according to `YCPLT_WTYPE`
(default `f16`) — this only takes extra time once; afterwards the service
keeps it resident in memory permanently. `f16` was chosen over a more
aggressive quantization (`q4_0`, `q8_0`, ...) specifically because this
project's own hardware assumption ("a dedicated machine with... enough
RAM/SSD for the model checkpoint", see top of this file) already has room
for it: stable-diffusion.cpp's own quantization comparison
([docs/sd.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sd.md))
shows `q4_0` as visibly the worst of the quantized options, while `f16` is
close to indistinguishable from full `f32` at roughly half the memory.
**This used to be a real, reported quality bug** — the default here was
`q4_0` for a while, which combined badly with the fallback below (a base,
non-inpainting checkpoint being asked to repaint a wide masked region at
`strength=0.95`) to produce literal color-noise/static instead of a
plausible inpaint on a real "remove the cat" job. Neither half of that
was a hardware or model-choice limitation — both were configuration gaps,
fixed below.

## Editing and inpainting models (model factory)

`conf/models.py` is a small factory that picks which checkpoint (and
optional VAE) handles a job based on its `mode`, instead of one model
being forced to do everything:

- **`txt2img` and `img2img`** both use `MODEL`/`YCPLT_WTYPE`/`YCPLT_VAE_PATH`
  — a base SD1.x checkpoint handles `img2img` directly (it starts
  denoising from the uploaded image instead of pure noise; see
  stable-diffusion.cpp's own img2img example, which reuses the exact same
  checkpoint as its txt2img example). No separate checkpoint is needed for
  whole-image edits, which is all the main app currently sends.
- **`inpaint`** — masked editing with an explicit, user-drawn mask
  (`init_image_b64` + `mask_image_b64` + a real edit prompt) — uses
  `INPAINT_MODEL`/`YCPLT_INPAINT_WTYPE`/`YCPLT_INPAINT_VAE_PATH` if set,
  falling back to the same model as everything else otherwise. This
  fallback exists so the service keeps working with zero extra
  configuration, but it is a real quality trap, not just "a bit worse":
  `stable-diffusion-cpp-python`'s own documentation is explicit that
  "inpainting with a base model gives poor results" — it was never
  trained with the extra mask/masked-image input channels a real
  inpainting checkpoint's UNet expects. **Setting `INPAINT_MODEL` is not
  optional polish for this job type — without it, masked edits are
  expected to look broken.**

  **Note:** `img2img` jobs carrying `remove_target` (see "Removing a
  named object" below) do NOT use this slot — plain "remove X" is
  handled by a separate model (LaMa) entirely, for reasons explained
  there. There is an EXPERIMENTAL, off-by-default exception
  (`reconstruct_prompt`, gated by `YCPLT_RECONSTRUCT_ENABLED=false` by
  default) that would route such a job through this slot instead — see
  "Describing what should replace the removed object" below for why it's
  disabled and what's still unresolved about it.

Two recommended checkpoint pairs (pick one; both use the same architecture
and speed as what's already here, so nothing about serving/queueing
changes):

**Option A — zero new download, if `sd-v1-5-inpainting.ckpt` is already on
disk.** The standard SD1.5 inpainting fine-tune:

```bash
wget -O models/sd-v1-5-inpainting.ckpt "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
```

```bash
# .env
INPAINT_MODEL=sd-v1-5-inpainting.ckpt
```

**Option B — noticeably better general quality, recommended (needs one
extra file, see below).**
[Realistic Vision V6.0 B1](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE)
is a popular photorealistic SD1.5 fine-tune, published as single-file fp16
`safetensors` for both a base checkpoint and a matching purpose-built
inpainting variant from the same author — same consistent style for
generation and object-removal alike:

```bash
wget -O models/Realistic_Vision_V6.0_NV_B1_fp16.safetensors \
  "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
wget -O models/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors \
  "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors"
```

**Important — both files above are "noVAE" releases** (see the repo name).
The model's own Hugging Face card says so explicitly: *"For version 6.0 it
is recommended to use with VAE (to improve generation quality and get rid
of artifacts)."* Skipping this is a real, confirmed failure mode here, not
a theoretical one — without a paired VAE, both `txt2img`/`img2img` and
especially masked-region output come back as a scrambled, striped
color-noise patch. (This is a different, simpler bug than a separate
stable-diffusion.cpp upstream issue that can produce a similar-looking
symptom on `mode="inpaint"` jobs specifically —
[PR #926](https://github.com/leejet/stable-diffusion.cpp/pull/926), fixed
by `pip install --upgrade --force-reinstall --no-cache-dir
stable-diffusion-cpp-python` if `stable-diffusion-cpp-python` predates
2025-11-01 — check this too if a genuine prompt-guided masked edit still
looks corrupted after confirming the VAE is correctly set.) Download the
matching VAE too:

```bash
wget -O models/vae-ft-mse-840000-ema-pruned.safetensors \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
```

```bash
# .env
MODEL=Realistic_Vision_V6.0_NV_B1_fp16.safetensors
INPAINT_MODEL=Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors
YCPLT_VAE_PATH=models/vae-ft-mse-840000-ema-pruned.safetensors
YCPLT_INPAINT_VAE_PATH=models/vae-ft-mse-840000-ema-pruned.safetensors
```

Option A's `sd-v1-5-inpainting.ckpt` bundles its own VAE and needs none of
this — the VAE requirement is specific to Realistic Vision's "noVAE" release.

Each distinct checkpoint is loaded at most once and kept resident — the
`txt2img`/`img2img` model loads eagerly at startup (so the daemon still
fails fast if it's missing, same guarantee as before this change); the
inpainting model, if configured, loads lazily on first use instead, so its
memory isn't spent unless an inpaint job actually arrives. If `INPAINT_MODEL`
is left unset, no second model is ever loaded at all.

`GET /health` reports `wtype`, `inpaint_wtype`, and
`inpaint_model_configured` (`false` if `INPAINT_MODEL_PATH` still equals
`MODEL_PATH`, i.e. nothing dedicated is set) precisely so this
misconfiguration is visible at a glance instead of only showing up as a
bad-looking result days later.

Adding a further slot for a StableDiffusion-family model later means: add
its path/wtype/vae variables to `conf/config.py`, add one line to
`_MODEL_SLOTS` in `conf/models.py`. Nothing else in the service needs to
change.

### Going further: SDXL

stable-diffusion.cpp supports SDXL directly (same `model_path=`-style
loading, no architecture-specific code path needed here) — a real option
if "relatively powerful hardware" means more than what SD1.5-family models
need, since SDXL's ~2.6B UNet at 1024x1024 (its native resolution, vs
SD1.5's 512x512) is meaningfully slower per image on CPU and each loaded
checkpoint takes more RAM. Given this project's own design tolerance for
generation taking "minutes to tens of minutes" (see "Design rationale"
below), that's not necessarily disqualifying — just worth testing before
committing to it as the default.

```bash
wget -O models/sd_xl_base_1.0.safetensors "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

# SDXL's own bundled VAE is known to produce NaN/black output in fp16 on
# some backends — the standard fix is loading a corrected VAE instead of
# the one baked into the checkpoint:
wget -O models/sdxl_vae-fp16-fix.safetensors "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
```

```bash
# .env
MODEL=sd_xl_base_1.0.safetensors
YCPLT_VAE_PATH=models/sdxl_vae-fp16-fix.safetensors
YCPLT_DEFAULT_WIDTH=1024
YCPLT_DEFAULT_HEIGHT=1024
```

There is no dedicated SDXL inpainting checkpoint recommendation here yet —
if `INPAINT_MODEL` is left pointing at an SD1.5-family inpainting
checkpoint while `MODEL` is switched to SDXL, `txt2img`/`img2img` and
`inpaint` simply become two differently-sized models loaded side by side
(same "cached per distinct checkpoint" behavior as today), which works,
just costs more resident memory than either alone.

## Understanding an uploaded image (mode="caption")

Generating/editing pixels and understanding what's *in* a picture are
different capabilities — stable-diffusion.cpp has no notion of image
content. Answering "what's in this picture?" needs a vision-language
model instead: `conf/models.get_vision_model()` loads
[moondream2](https://huggingface.co/vikhyatk/moondream2) (~1.4B, designed
for exactly this kind of lightweight/CPU use) via `llama-cpp-python` — a
separate dependency and code path from the StableDiffusion factory above,
since it's a different backend entirely.

This lives in the graphics service, not the main chat app: the chat app
only classifies whether a message is asking about an attached image and
submits a `mode="caption"` job here, the same shape as a generation/edit
job (`prompt` = the question, `init_image_b64` = the image), just with a
text answer back instead of a PNG.

**It's optional and off by default.** If the model files aren't present,
`conf/models.get_vision_model()` returns `None` and a `caption` job simply
fails with a clear `error_message` (visible via `GET /jobs/{id}` and
`GET /health`) — generation/editing jobs are entirely unaffected.

To enable it, download the official GGUF conversion
([ggml-org/moondream2-20250414-GGUF](https://huggingface.co/ggml-org/moondream2-20250414-GGUF),
~2.8 GB total) into this service's `models/` directory (not the main
chat app's — the vision model belongs here, alongside the SD checkpoints):

```bash
wget -O models/moondream2-text-model-f16_ct-vicuna.gguf "https://huggingface.co/ggml-org/moondream2-20250414-GGUF/resolve/main/moondream2-text-model-f16_ct-vicuna.gguf"
wget -O models/moondream2-mmproj-f16-20250414.gguf "https://huggingface.co/ggml-org/moondream2-20250414-GGUF/resolve/main/moondream2-mmproj-f16-20250414.gguf"
```

The default `VISION_MODEL`/`VISION_MMPROJ` filenames in `conf/config.py`
already match these exact names; override `VISION_MODEL`/`VISION_MMPROJ`
(or the `YCPLT_VISION_MODEL_PATH`/`YCPLT_VISION_MMPROJ_PATH` full-path
escape hatches) in `.env` only if you place them elsewhere.

Loads lazily on the first `caption` job, not at startup — most
deployments may never use it, and it's a further ~2-3 GB resident once
loaded. The first such job is noticeably slower (model load time); after
that it stays resident like every other model here. Check `GET /health`'s
`vision` field (`files_found`, `loaded`, `load_error`) if a caption job
keeps failing — it tells apart "files not downloaded" from "files present
but the model failed to load" without needing to read server logs.

## Removing a named object

**Plain removal no longer uses a diffusion model at all — see "Why not
StableDiffusion inpainting" below for why that changed.** "Remove X"
instructions (`mode="img2img"` with `remove_target` set — see the main
chat app's `utils/intent.get_removal_target_async`) go through two
CPU-sized, purpose-built models instead of `MODEL`/`INPAINT_MODEL`. There
is an experimental, **off-by-default** exception for jobs that also
carry a `reconstruct_prompt` — see "Describing what should replace the
removed object" below for why it's disabled and what remains unresolved.

1. [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined)
   (`conf/segmentation.py`) takes the object's English name alone (e.g.
   "cat") and produces a mask — no manually-drawn mask needed. The mask
   is dilated by `YCPLT_REMOVE_MASK_DILATE_PX` (default `12` px) to clear
   away the soft-edge remnants (fur, whiskers) CLIPSeg's confidence map
   tends to leave just outside the thresholded region.
2. [LaMa](https://github.com/advimman/lama) (`conf/models.get_lama_model()`)
   takes that mask plus the image and fills the hole — no text prompt at
   all. LaMa is a plain CNN (Fourier convolutions, single forward pass,
   no denoising loop) trained specifically to extend surrounding texture
   into a masked region, which is the actual "remove X" task.

Plain `img2img` can't do this on its own: it just partially re-renders the
*whole* image guided by the prompt text, so a command like "remove the
cat" doesn't actually make the cat disappear — the model has no mechanism
to understand "remove", it just restyles the picture, cat included.

**Both models are optional, like the vision model.** Neither is a file
you download manually — CLIPSeg is pulled from the Hugging Face Hub
(`transformers`/`torch`, ~600MB) and LaMa's checkpoint (~381MB) from
GitHub via `torch.hub`, both on first use, both cached locally afterward;
this needs internet access on first use only. If either isn't installed
or the download fails, the job simply errors out with a clear message
(check `GET /health`'s `segmentation` and `lama` fields: `loaded`,
`load_failed`, `load_error`) — other job modes are unaffected.

LaMa's loading/inference code lives in `conf/lama_model.py` rather than a
separate pip package — see that file's docstring: the natural PyPI
package for this (`simple-lama-inpainting`) pins an old Pillow/numpy
range with no prebuilt wheels for current Python versions, which forces
pip to compile Pillow from source and fails on any system missing
libjpeg's development headers. Vendoring the ~60 lines of actual logic
(credited there, Apache-2.0) avoids that entirely — it uses whatever
torch/numpy/Pillow are already installed for CLIPSeg above, nothing new
to install.

**Where the downloaded checkpoint is cached:** `YCPLT_TORCH_HOME`
(default `models/.torch_cache`) — set explicitly rather than left at
torch's own default location under `$HOME/.cache`, since a dedicated
systemd service user doesn't always have a stable `$HOME`, which would
otherwise make the ~381MB download silently repeat on every restart
instead of being reused. The download itself only ever happens once
either way; loading the cached file into the process's memory still
happens on every restart (unavoidable for any resident model, and fast —
seconds, not the original download time) — that's not the same thing as
re-downloading, even though both show a `[models] loading LaMa...` log
line.

### Describing what should replace the removed object (`reconstruct_prompt`) — experimental, off by default

LaMa is deliberately prompt-free: it only knows how to extend generic
surrounding texture into a hole, so it can't paint a *specific*,
describable thing (a particular pattern, material, or object) into the
gap — asked to reconstruct, say, the perforated metal panel that was
behind a removed object, it just produces plain background instead. It
was never trained to do more than that; this is a training-data
limitation (its Places2 dataset is general scenes, not mechanical
textures), not something a "smarter" model would automatically fix.

The main chat app's two-stage classifier
(`utils/intent.get_reconstruction_prompt_async`, run only after
`get_removal_target_async` already identified a removal) can detect when
the user's own instruction describes what should appear in the removed
object's place (not just "remove it" — something more specific, e.g.
"remove the cat, recreate the perforated metal panel that was behind
it") and send that description here as `reconstruct_prompt`, an English
text-to-image prompt fragment. `srv/worker.py`'s `_generate_removal_edit`
would then route the job through the `inpaint` slot (`INPAINT_MODEL` —
see "Editing and inpainting models" above) instead of LaMa, using
`reconstruct_prompt` as the actual generation prompt and full strength
(`1.0`).

**This path is disabled by default (`YCPLT_RECONSTRUCT_ENABLED=false`).**
Four separate rounds of real testing, each targeting a different,
individually-reasonable hypothesis, never changed the actual defect:

1. A missing VAE for the checkpoint's "noVAE" release (see "Why not
   StableDiffusion inpainting" below) — fixed, output unchanged.
2. Upstream stable-diffusion.cpp's masked-input corruption bug
   ([PR #926](https://github.com/leejet/stable-diffusion.cpp/pull/926))
   — upgraded past it, output unchanged.
3. A genuine width/height mismatch between the job's generation
   resolution and the mask/init_image's real size — fixed by aligning
   them, output unchanged (same striped/scrambled pattern, still
   confined to roughly the mask's shape).
4. That alignment fix generating at an uncapped, potentially huge real
   photo resolution, which separately crashed the worker process
   (~10GB+ UNet compute buffer) — fixed with a resolution cap
   (`YCPLT_RECONSTRUCT_MAX_DIMENSION`), output unchanged in shape, just
   smaller.

The masked region came back as the same striped/scrambled pattern across
all four rounds, at different resolutions, with different upstream
fixes applied — strong evidence the actual cause is something none of
these hypotheses addressed. The most likely remaining explanation: a
genuine incompatibility between this specific SD1.5-style 5-channel
inpainting checkpoint (Realistic Vision's inpainting variant) and this
build of `stable-diffusion-cpp-python`'s "unet inpainting concat"
handling — PR #926's own description mentions patching exactly that
mechanism, but evidently not for every checkpoint/version combination.
Confirming that (or finding the real cause) would need hands-on access
to the failing checkpoint/binding pair that isn't available here — e.g.
testing the plain `sd-v1-5-inpainting.ckpt` checkpoint (Option A in
"Editing and inpainting models" above) instead of Realistic Vision's
inpainting variant, or testing outside this Python service via
stable-diffusion.cpp's own CLI directly.

`YCPLT_RECONSTRUCT_MAX_DIMENSION` (default `512`, matching
`YCPLT_DEFAULT_WIDTH`/`YCPLT_DEFAULT_HEIGHT`) and the width/height
alignment logic (both described above) remain in place regardless — they
were real, independent fixes worth keeping if this path is ever
re-enabled for further investigation, even though neither turned out to
be the root cause of the visual corruption.

With `YCPLT_RECONSTRUCT_ENABLED` left at its default (`false`), every
`remove_target` job uses LaMa unconditionally — exactly the
previously-confirmed-working behavior — regardless of what
`reconstruct_prompt` the main app's classifier sends. Set it to `true`
only to deliberately re-enable this path for further investigation.

### Why not StableDiffusion inpainting (for plain removal)

This used to go through the `inpaint` slot (`INPAINT_MODEL`, a text
prompt like "empty background, no cat", and a high `strength`), the same
as a real prompt-guided masked edit. Real, repeated testing showed this
is architecturally the wrong tool for plain removal, not a tuning
problem: asked to fill a masked region guided only by an "empty
background" prompt, a diffusion model still has to generate *something*
from that prompt, and in practice it kept inventing content instead of
returning nothing —

- a different cat/kitten appearing in the same spot as the removed one
  (the model reading a barely-visible remnant of the original object at
  the mask's edge as "there's a partial cat here, finish the picture"),
- on a base (non-inpainting) checkpoint, entirely unrelated scenery (a
  staircase and door drawn into a photographed appliance interior) —
  because a base checkpoint lacks the extra mask/masked-image input
  channels an inpainting-tuned UNet needs to stay anchored to the
  surrounding context, so at high `strength` it fell back to something
  close to unguided text-to-image within the masked rectangle,
- and, on a checkpoint published without its own VAE (Realistic Vision's
  "noVAE" release, used without the separately-required VAE file),
  scrambled striped color-noise instead of a coherent fill.

None of mask dilation, `strength`, checkpoint choice, or the VAE fix
changed this outcome in a way that held up under further testing.
**Later found, separately:** every one of these tests also happened to
carry the width/height mismatch described above under "Describing what
should replace the removed object" (the job's width/height never
matched the actual photo's real size) — a real, independent bug in its
own right that produces exactly this kind of scrambled/misaligned
masked-region output on ITS OWN, regardless of prompt/VAE/checkpoint.
That mismatch is now fixed. It doesn't change the core architectural
conclusion below (a prompt-free model is still the right tool for plain
"make it disappear" removal, and LaMa still handles that case
correctly) — but it does mean the specific "scrambled color-noise" and
"unrelated scenery" symptoms above were likely compounded by two
separate bugs at once, not one. The common thread that DOES still hold
for the *prompt-quality* problem (as opposed to the resolution-mismatch
one) is that a *prompted* diffusion
model has no way to express "nothing, just continue the background" as
strongly as "no text prompt at all" does. LaMa isn't a smarter or bigger
model than what came before it; it's simply the correct category of tool
for "erase and extend", the same way CLIPSeg (not a general segmentation
LLM) is the correct tool for "find this named object". The `inpaint` slot
and `INPAINT_MODEL` remain fully in use for genuine prompt-guided masked
edits (a job with an explicit mask AND a real edit instruction, not just
"remove this") — see "Editing and inpainting models" above.

## systemd

The unit loads the same `.env` file directly via `EnvironmentFile=`, so
switching models is a one-line edit + restart, whether run manually or under
systemd.

```bash
sudo cp install/ycplt_img.service /etc/systemd/system/
sudo useradd -r -s /sbin/nologin ycplt   # if a dedicated system user is wanted
# adjust WorkingDirectory/ExecStart/User/EnvironmentFile paths in the unit file
sudo systemctl daemon-reload
sudo systemctl enable --now ycplt_img
sudo systemctl status ycplt_img
journalctl -u ycplt_img -f

# switching models later:
#   edit /opt/ycplt_img/.env (MODEL=...), then:
sudo systemctl restart ycplt_img
```

## API

Plain JSON over HTTP, no authentication (meant for a trusted local network).

**POST /jobs** — create a job.

```json
{
  "prompt": "photorealistic red apple on a wooden table, soft natural light",
  "negative_prompt": "blurry, low quality",
  "mode": "txt2img",
  "width": 512,
  "height": 512,
  "steps": 20,
  "cfg_scale": 7.5,
  "seed": 42
}
```

Response: `202 {"job_id": 7, "status": "queued"}`.

For `img2img`/`inpaint`, additionally pass `init_image_b64` (and
`mask_image_b64` for inpaint) — the source image, base64-encoded — and
`strength` (0.0-1.0, how strongly to deviate from the source).

For `img2img` specifically, an optional `remove_target` (a short English
object name, e.g. `"cat"`) switches the job to automatic segmentation +
inpainting instead of plain img2img — see "Removing a named object"
above. When set, `strength`/`negative_prompt` are ignored (the worker
sets its own for this path); `prompt` is only used for logging. A further
optional `reconstruct_prompt` (only meaningful alongside `remove_target`)
describes what should be painted into the removed object's place instead
of plain background — see "Describing what should replace the removed
object" above; when omitted, removal is handled by LaMa exactly as before.

For `mode="caption"` (image understanding — see "Understanding an
uploaded image" above), `prompt` is the question and `init_image_b64` is
the image; `width`/`height`/`steps`/`cfg_scale` are ignored but still
required by the request shape below (any value works, e.g. the defaults).

**GET /jobs/{id}** — status without image content:

```json
{"id": 7, "status": "done", "mode": "txt2img", "prompt": "a cat wearing a hat", "created_at": 1785e9, "started_at": ..., "finished_at": ..., "error_message": null, "result_text": null}
```

`status`: `queued` -> `processing` -> `done` | `error`. `result_text` is
populated only for a finished `mode="caption"` job — the text answer is
small enough to return inline here rather than needing a second request
the way an image result does.

**GET /jobs/{id}/result** — if `status == "done"`, returns `image/png`
directly (image-generating modes only — for `mode="caption"`, read
`result_text` from `GET /jobs/{id}` instead). If not ready yet — `409`
with the current status in the body; if the job doesn't exist — `404`.

**DELETE /jobs/{id}** — client acknowledges it has retrieved the result; the
row is removed from the queue. If not explicitly deleted, the row is still
purged automatically after `YCPLT_JOB_TTL_HOURS` hours (default 24) once
finished.

**GET /health** — diagnostics, no side effects:

```json
{"status": "ok", "model_path": "...", "wtype": "f16", "inpaint_model_path": "...", "inpaint_wtype": "f16", "inpaint_model_configured": true, "vision": {"model_path": "...", "mmproj_path": "...", "files_found": true, "loaded": false, "load_failed": false, "load_error": null}, "segmentation": {"model": "CIDAS/clipseg-rd64-refined", "loaded": false, "load_failed": false, "load_error": null}}
```

`inpaint_model_configured` is `false` when `INPAINT_MODEL` was never set
(i.e. `inpaint_model_path` still equals `model_path`) — see "Editing and
inpainting models" above for why that's worth checking rather than
assuming.

## Manual smoke test (curl)

```bash
curl -s -X POST http://192.168.7.7:4011/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 8}'
# -> {"job_id": 1, "status": "queued"}

curl -s http://192.168.7.7:4011/jobs/1
# -> {"id": 1, "status": "processing", ...}   (poll until "status": "done")

curl -s http://192.168.7.7:4011/jobs/1/result -o result.png

curl -s -X DELETE http://192.168.7.7:4011/jobs/1
```

## Design rationale

- **Passive queue + client-side polling, not push/webhooks** — generation
  takes minutes to tens of minutes; the latency difference between push and
  polling every few seconds is irrelevant against that. Push would also
  require the service to know the client's address and handle it being
  unreachable/restarted.
- **Model loaded once per checkpoint, resident for the daemon's lifetime** —
  jobs never pay the cost of reloading a checkpoint from disk. With the
  model factory (`conf/models.py`), this now applies per distinct
  checkpoint rather than globally: the `txt2img`/`img2img` model still
  loads eagerly at startup, while an optional dedicated inpainting model
  loads lazily on first use and is then cached the same way.
- **One job at a time, no parallelism** — on a single CPU-bound machine,
  running jobs in parallel buys no throughput (they'd split the same cores),
  only doubling memory usage.
- **SQLite, not Redis** — jobs survive a daemon restart, zero extra
  processes/dependencies for a single-user local service.
- **No FastAPI/Flask** — this is a headless daemon; four simple JSON
  endpoints are perfectly served by the standard library's `http.server`. The
  only real dependency is the generation library itself.
