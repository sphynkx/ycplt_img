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
app.py                    — entry point: loads the model once, starts the worker and the HTTP server
conf/
  config.py                 — configuration via environment variables (host/port, model path, TTL, ...)
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

git clone <repository-url-tbd> ycplt_img
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

**GET /jobs/{id}** — status without image content:

```json
{"id": 7, "status": "done", "mode": "txt2img", "created_at": 1785e9, "started_at": ..., "finished_at": ..., "error_message": null}
```

`status`: `queued` -> `processing` -> `done` | `error`.

**GET /jobs/{id}/result** — if `status == "done"`, returns `image/png`
directly. If not ready yet — `409` with the current status in the body; if
the job doesn't exist — `404`.

**DELETE /jobs/{id}** — client acknowledges it has retrieved the result; the
row is removed from the queue. If not explicitly deleted, the row is still
purged automatically after `YCPLT_JOB_TTL_HOURS` hours (default 24) once
finished.

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
- **Model loaded once at startup, resident for the daemon's lifetime** — jobs
  never pay the cost of reloading the checkpoint from disk.
- **One job at a time, no parallelism** — on a single CPU-bound machine,
  running jobs in parallel buys no throughput (they'd split the same cores),
  only doubling memory usage.
- **SQLite, not Redis** — jobs survive a daemon restart, zero extra
  processes/dependencies for a single-user local service.
- **No FastAPI/Flask** — this is a headless daemon; four simple JSON
  endpoints are perfectly served by the standard library's `http.server`. The
  only real dependency is the generation library itself.
