"""
ESP32-CAM Backend Server
========================
รวม Backend Server + AI Engine (Roboflow) + Mobile App API ในไฟล์เดียว

Flow:
  ESP32-CAM  -->  POST /trigger      --> บันทึก + ส่งไป AI
  Backend    -->  Roboflow Workflow  --> ผลลัพธ์
  Mobile App -->  GET  /state        --> ดึงข้อมูลล่าสุด

ติดตั้ง:
  pip install flask inference-sdk pillow requests

รัน:
  python server.py
"""

import os
import io
import time
import base64
import threading
import requests
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from inference_sdk import InferenceHTTPClient

# ─────────────────────────────────────────────
# CONFIG  (แก้ค่าตรงนี้)
# ─────────────────────────────────────────────
ROBOFLOW_API_KEY   = "7J862mQ3IMOQOIjmzeIq"
ROBOFLOW_API_URL   = "https://serverless.roboflow.com"  # เปลี่ยนจาก serverless
WORKSPACE_NAME     = "perapon"
WORKFLOW_ID        = "detect-and-classify"

# Model ID สำหรับ infer โดยตรง (ดูจาก Roboflow → Deploy → Endpoint)
DETECT_MODEL_ID    = "animals-ij5d2-1ghad-chdgk/1"

SERVER_HOST        = "0.0.0.0"   # รับจากทุก IP (ให้ ESP32 เชื่อมได้)
SERVER_PORT        = 5000

IMAGE_SAVE_DIR     = Path("captured_images")  # โฟลเดอร์เก็บรูป
MAX_SAVED_IMAGES   = 50                        # เก็บไว้กี่รูปสูงสุด

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
IMAGE_SAVE_DIR.mkdir(exist_ok=True)

# Roboflow client
rf_client = InferenceHTTPClient(
    api_url=ROBOFLOW_API_URL.strip(),
    api_key=ROBOFLOW_API_KEY.strip()
)

# ─────────────────────────────────────────────
# STATE  (เก็บในหน่วยความจำ — ง่าย ไม่ต้องใช้ DB)
# ─────────────────────────────────────────────
state = {
    "last_image_path": None,   # path ของรูปล่าสุด
    "last_image_time": None,   # เวลาที่รับรูป
    "detection_result": None,  # ผลจาก AI
    "detection_time": None,    # เวลาที่ AI ตอบ
    "status": "waiting",       # waiting | processing | done | error
    "error": None,
    "total_captures": 0,
}
state_lock = threading.Lock()


def update_state(**kwargs):
    with state_lock:
        state.update(kwargs)


# ─────────────────────────────────────────────
# AI ENGINE
# ─────────────────────────────────────────────
def run_ai(image_path: str):
    """ส่งรูปไป Roboflow infer โดยตรง (รองรับ yolov5 classification)"""
    log.info(f"[AI] ส่งรูปไป Roboflow: {image_path}")
    update_state(status="processing", error=None)
    try:
        # ใช้ infer() แทน run_workflow() — รองรับ yolov5v6n classification
        result = rf_client.infer(image_path, model_id=DETECT_MODEL_ID.strip())
        log.info(f"[AI] ผลลัพธ์: {result}")
        update_state(
            detection_result=result,
            detection_time=datetime.now().isoformat(),
            status="done"
        )
    except Exception as e:
        log.error(f"[AI] Error: {e}")
        update_state(status="error", error=str(e))


# ─────────────────────────────────────────────
# HELPER: บันทึกรูป + ลบรูปเก่าถ้าเกิน MAX
# ─────────────────────────────────────────────
def save_image_bytes(data: bytes) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = IMAGE_SAVE_DIR / f"{ts}.jpg"
    path.write_bytes(data)
    log.info(f"[IMG] บันทึกรูป: {path}")

    # ลบรูปเก่าถ้าเกินจำนวน
    imgs = sorted(IMAGE_SAVE_DIR.glob("*.jpg"))
    while len(imgs) > MAX_SAVED_IMAGES:
        imgs.pop(0).unlink()

    return str(path)


# ─────────────────────────────────────────────
# ROUTES: ESP32-CAM  →  Backend
# ─────────────────────────────────────────────

@app.route("/trigger", methods=["POST"])
def trigger():
    """
    ESP32-CAM POST มาที่นี่หลังตรวจพบ Motion
    รับภาพ 3 แบบ:
      1. multipart/form-data  (field: "image")
      2. raw JPEG bytes       (Content-Type: image/jpeg)
      3. JSON { "image": "<base64>" }
    """
    image_data = None

    # --- แบบ 1: multipart ---
    if "image" in request.files:
        image_data = request.files["image"].read()

    # --- แบบ 2: raw bytes ---
    elif request.content_type and "image" in request.content_type:
        image_data = request.data

    # --- แบบ 3: JSON base64 ---
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        b64 = body.get("image", "")
        # รองรับ "data:image/jpeg;base64,..." หรือ base64 ล้วน
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            image_data = base64.b64decode(b64)
        except Exception:
            return jsonify({"ok": False, "error": "base64 decode failed"}), 400

    if not image_data:
        return jsonify({"ok": False, "error": "ไม่พบข้อมูลรูปภาพ"}), 400

    # บันทึกรูป
    path = save_image_bytes(image_data)
    with state_lock:
        state["last_image_path"] = path
        state["last_image_time"] = datetime.now().isoformat()
        state["total_captures"] += 1

    # รัน AI ใน background thread (ไม่บล็อก ESP32)
    t = threading.Thread(target=run_ai, args=(path,), daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "รับรูปแล้ว กำลังวิเคราะห์"}), 202


@app.route("/image/<filename>")
def serve_image(filename):
    """ให้ Mobile App ดาวน์โหลดรูปผ่าน URL"""
    return send_from_directory(IMAGE_SAVE_DIR, filename)


# ─────────────────────────────────────────────
# ROUTES: Mobile App  →  Backend
# ─────────────────────────────────────────────

@app.route("/state", methods=["GET"])
def get_state():
    """
    Mobile App เรียกทุก 10 วินาที
    คืนค่า JSON:
    {
      "status": "done",
      "image_url": "http://<server>:5000/image/20250310_123456.jpg",
      "detection_result": [...],
      "detection_time": "2025-03-10T12:34:56",
      "last_image_time": "...",
      "total_captures": 5
    }
    """
    with state_lock:
        s = dict(state)

    class_map = {
    "------------------------------": "person",
    "animal_test": "animal",
    "car_test": "vehicle"
    }

    if s["detection_result"] and "predictions" in s["detection_result"]:
        for pred in s["detection_result"]["predictions"]:
            pred["class"] = class_map.get(pred["class"], pred["class"])

    # สร้าง URL ของรูปล่าสุด
    image_url = None
    if s["last_image_path"]:
        filename = Path(s["last_image_path"]).name
        host = request.host_url.rstrip("/")
        image_url = f"{host}/image/{filename}"

    return jsonify({
        "status":           s["status"],
        "image_url":        image_url,
        "last_image_time":  s["last_image_time"],
        "detection_result": s["detection_result"],
        "detection_time":   s["detection_time"],
        "total_captures":   s["total_captures"],
        "error":            s["error"],
    })


@app.route("/", methods=["GET"])
def dashboard():
    """หน้า Dashboard แสดงสถานะระบบ + รูปล่าสุด"""
    with state_lock:
        s = dict(state)
    class_map = {
    "------------------------------": "person",
    "VehicleCount - v1 2026-03-07 7:26pm": "car",
    "car_test": "vehicle"
    }

    if s["detection_result"] and "predictions" in s["detection_result"]:
        for pred in s["detection_result"]["predictions"]:
            cls = pred["class"]

            if "VehicleCount" in cls:
                pred["class"] = "car"
            elif cls == "------------------------------":
                pred["class"] = "person"

    image_url = None
    if s["last_image_path"]:
        filename = Path(s["last_image_path"]).name
        host = request.host_url.rstrip("/")
        image_url = f"{host}/image/{filename}"

    status_color = {"waiting": "#888", "processing": "#f90", "done": "#0f0", "error": "#f44"}.get(s["status"], "#888")
    result_text = str(s["detection_result"]) if s["detection_result"] else "ยังไม่มีผล"
    img_tag = f'<img src="{image_url}" style="max-width:100%;border:1px solid #333;border-radius:4px;margin-top:10px">' if image_url else "<p style='color:#555'>ยังไม่มีรูป</p>"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>ESP32-CAM Dashboard</title>
  <style>
    body {{ background:#0d1117; color:#c9d1d9; font-family:monospace; padding:30px; }}
    h1 {{ color:#58a6ff; }} h2 {{ color:#79c0ff; border-bottom:1px solid #30363d; padding-bottom:6px; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:16px; margin:12px 0; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; background:{status_color}22; color:{status_color}; border:1px solid {status_color}; }}
    .result {{ background:#0d1117; padding:12px; border-radius:4px; font-size:12px; overflow-x:auto; white-space:pre-wrap; word-break:break-all; }}
    table {{ width:100%; border-collapse:collapse; }} td {{ padding:6px 10px; border-bottom:1px solid #21262d; }}
    td:first-child {{ color:#8b949e; width:160px; }}
  </style>
</head>
<body>
  <h1>📷 ESP32-CAM Dashboard</h1>
  <p style="color:#555;font-size:12px">Auto-refresh ทุก 5 วินาที</p>

  <div class="card">
    <h2>สถานะระบบ</h2>
    <table>
      <tr><td>Status</td><td><span class="badge">{s["status"].upper()}</span></td></tr>
      <tr><td>รูปล่าสุด</td><td>{s["last_image_time"] or "-"}</td></tr>
      <tr><td>AI วิเคราะห์เมื่อ</td><td>{s["detection_time"] or "-"}</td></tr>
      <tr><td>รูปทั้งหมด</td><td>{s["total_captures"]} ภาพ</td></tr>
      {f'<tr><td>Error</td><td style="color:#f44">{s["error"]}</td></tr>' if s["error"] else ""}
    </table>
  </div>

  <div class="card">
    <h2>รูปล่าสุด</h2>
    {img_tag}
  </div>

  <div class="card">
    <h2>ผลการวิเคราะห์ AI</h2>
    <div class="result">{result_text}</div>
  </div>

  <div class="card" style="font-size:12px;color:#555">
    <b>Endpoints:</b><br>
    POST <a href="#" style="color:#58a6ff">/trigger</a> — ESP32-CAM ส่งรูป &nbsp;|&nbsp;
    GET <a href="/state" style="color:#58a6ff">/state</a> — Mobile App &nbsp;|&nbsp;
    GET <a href="/health" style="color:#58a6ff">/health</a> — Health check
  </div>
</body>
</html>"""
    return html


@app.route("/health", methods=["GET"])
def health():
    """ตรวจสอบว่า server ยังทำงานอยู่"""
    return jsonify({"ok": True, "time": datetime.now().isoformat()})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("  ESP32-CAM Backend Server")
    log.info(f"  Listening  : http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"  Roboflow   : {WORKSPACE_NAME}/{WORKFLOW_ID}")
    log.info(f"  Image dir  : {IMAGE_SAVE_DIR.resolve()}")
    log.info("=" * 50)
    log.info("Endpoints:")
    log.info(f"  POST /trigger  <-- ESP32-CAM ส่งรูปมาที่นี่")
    log.info(f"  GET  /state    <-- Mobile App ดึงข้อมูล")
    log.info(f"  GET  /health   <-- ตรวจสอบสถานะ")
    log.info("=" * 50)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)