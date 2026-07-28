"""Minimal HTTP JSON API built on the standard library (no FastAPI/Flask) —
this is a headless daemon with no web UI, so a full framework isn't needed.

The service is passive: it only accepts jobs and serves status/results on
request. Nothing is generated in this thread — the heavy lifting happens in
a separate worker thread (srv/worker.py) using the already-loaded model, so
HTTP handlers stay light and responsive even during a multi-hour generation.

API:
  POST   /jobs              -> {job_id}                (create a job)
  GET    /jobs/{id}         -> {id, status, ...}        (status, no image; result_text for mode="caption")
  GET    /jobs/{id}/result  -> image/png                (finished image result)
  DELETE /jobs/{id}         -> {status: "ok"}           (client claimed the result)
  GET    /health            -> {status, model, vision}  (diagnostics, no side effects)
"""
import base64
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from conf import config, models, segmentation
from db import db

_JOB_RESULT_RE = re.compile(r"^/jobs/(\d+)/result$")
_JOB_RE = re.compile(r"^/jobs/(\d+)$")


class Handler(BaseHTTPRequestHandler):
    server_version = "ycplt_img/1.0"

    # ---------- helpers ----------
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    # ---------- POST /jobs ----------
    def do_POST(self):
        if self.path != "/jobs":
            self._send_json(404, {"error": "not found"})
            return
        try:
            data = self._read_json_body()
            prompt = data.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "'prompt' field is required"})
                return

            init_image = base64.b64decode(data["init_image_b64"]) if data.get("init_image_b64") else None
            mask_image = base64.b64decode(data["mask_image_b64"]) if data.get("mask_image_b64") else None

            job_id = db.create_job(
                prompt=prompt,
                mode=data.get("mode", "txt2img"),
                negative_prompt=data.get("negative_prompt"),
                width=int(data.get("width", 512)),
                height=int(data.get("height", 512)),
                steps=int(data.get("steps", 20)),
                cfg_scale=float(data.get("cfg_scale", 7.5)),
                seed=data.get("seed"),
                strength=data.get("strength"),
                init_image=init_image,
                mask_image=mask_image,
                remove_target=data.get("remove_target"),
            )
            self._send_json(202, {"job_id": job_id, "status": "queued"})
        except Exception as e:
            self._send_json(400, {"error": str(e)})

    # ---------- GET /health, /jobs/{id} and /jobs/{id}/result ----------
    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_path": config.MODEL_PATH,
                    "inpaint_model_path": config.INPAINT_MODEL_PATH,
                    "vision": models.vision_status(),
                    "segmentation": segmentation.status(),
                },
            )
            return

        m = _JOB_RESULT_RE.match(self.path)
        if m:
            job_id = int(m.group(1))
            image_bytes = db.get_job_result(job_id)
            if image_bytes is None:
                status = db.get_job_status(job_id)
                if status is None:
                    self._send_json(404, {"error": "job not found"})
                else:
                    self._send_json(409, {"error": "job not ready yet", "status": status["status"]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(image_bytes)))
            self.end_headers()
            self.wfile.write(image_bytes)
            return

        m = _JOB_RE.match(self.path)
        if m:
            job_id = int(m.group(1))
            status = db.get_job_status(job_id)
            if status is None:
                self._send_json(404, {"error": "job not found"})
            else:
                self._send_json(200, status)
            return

        self._send_json(404, {"error": "not found"})

    # ---------- DELETE /jobs/{id} ----------
    def do_DELETE(self):
        m = _JOB_RE.match(self.path)
        if not m:
            self._send_json(404, {"error": "not found"})
            return
        db.delete_job(int(m.group(1)))
        self._send_json(200, {"status": "ok"})


def run_server(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"ycplt_img listening on {host}:{port}")
    httpd.serve_forever()
