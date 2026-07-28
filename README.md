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
(default `q4_0`) — this only takes extra time once; afterwards the service
keeps it resident in memory permanently.

## Editing and inpainting models (model factory)

`conf/models.py` is a small factory that picks which checkpoint handles a
job based on its `mode`, instead of one model being forced to do
everything:

- **`txt2img` and `img2img`** both use `MODEL`/`YCPLT_WTYPE` — a base SD1.x
  checkpoint handles `img2img` directly (it starts denoising from the
  uploaded image instead of pure noise; see stable-diffusion.cpp's own
  img2img example, which reuses the exact same checkpoint as its txt2img
  example). No separate checkpoint is needed for whole-image edits, which
  is all the main app currently sends.
- **`inpaint`** (masked editing — a job with both `init_image_b64` and
  `mask_image_b64`) uses `INPAINT_MODEL`/`YCPLT_INPAINT_WTYPE` if set,
  falling back to the same model as everything else otherwise. This
  fallback exists so the service keeps working with zero extra
  configuration, but `stable-diffusion-cpp-python`'s own documentation is
  explicit that "inpainting with a base model gives poor results" and
  recommends a model fine-tuned for it — worth setting up if inpainting
  jobs are actually going to be used.

Recommended inpainting checkpoint — the standard SD1.5 inpainting
fine-tune, same architecture and speed as the base model, drop-in via
`model_path`:

```bash
wget -O models/sd-v1-5-inpainting.ckpt "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
```

Then in `.env`:

```bash
INPAINT_MODEL=sd-v1-5-inpainting.ckpt
```

Each distinct checkpoint is loaded at most once and kept resident — the
`txt2img`/`img2img` model loads eagerly at startup (so the daemon still
fails fast if it's missing, same guarantee as before this change); the
inpainting model, if configured, loads lazily on first use instead, so its
memory isn't spent unless an inpaint job actually arrives. If `INPAINT_MODEL`
is left unset, no second model is ever loaded at all.

Adding a further slot for a StableDiffusion-family model later means: add
its path/wtype variables to `conf/config.py`, add one line to
`_MODEL_SLOTS` in `conf/models.py`. Nothing else in the service needs to
change.

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

For `mode="caption"` (image understanding — see "Understanding an
uploaded image" above), `prompt` is the question and `init_image_b64` is
the image; `width`/`height`/`steps`/`cfg_scale` are ignored but still
required by the request shape below (any value works, e.g. the defaults).

**GET /jobs/{id}** — status without image content:

```json
{"id": 7, "status": "done", "mode": "txt2img", "created_at": 1785e9, "started_at": ..., "finished_at": ..., "error_message": null, "result_text": null}
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
{"status": "ok", "model_path": "...", "inpaint_model_path": "...", "vision": {"model_path": "...", "mmproj_path": "...", "files_found": true, "loaded": false, "load_failed": false, "load_error": null}}
```

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
