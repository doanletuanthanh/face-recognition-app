import os
import requests
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

BACKEND_URL = "http://127.0.0.1:8000/"

@main.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@main.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file:
        flash("No file uploaded")
        return redirect(url_for("main.index"))

    res = requests.post(f"{BACKEND_URL}/process_frame", files={"file": file})
    return res.content, res.status_code, {'Content-Type': 'image/jpeg'}

@main.route("/add", methods=["POST"])
def add():
    image_data = request.form.get("image_data")
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or (not file and not image_data):
        flash("Image and name are required")
        return redirect(url_for("main.index"))

    if image_data:
        # image_data is a data URL: 'data:image/jpeg;base64,...'
        import base64, io
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        file = ("webcam.jpg", io.BytesIO(img_bytes), "image/jpeg")
        files = {"file": file}
    else:
        files = {"file": file}

    res = requests.post(f"{BACKEND_URL}/add_face?name={name}", files=files)
    if res.ok:
        flash("Face added successfully")
        print(f"[FLASK] Added face: {name}")
    else:
        error = res.json().get("error", "Unknown error")
        flash(f"Error: {error}")
        print(f"[FLASK] Failed to add face: {error}")

    return redirect(url_for("main.index"))
