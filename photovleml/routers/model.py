import os
import shutil
from flask import Blueprint, jsonify, request, send_file
from photovleml.service import PhotovleService


model_bp = Blueprint('model', __name__, url_prefix='/model')


@model_bp.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]

        if img.filename == "":
            return jsonify(False)

        if label.filename == "":
            return jsonify(False)

        img_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "JPEGImages")
        label_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "Annotations")

        if os.path.isdir(img_path):
            shutil.rmtree(img_path)

        if os.path.isdir(label_path):
            shutil.rmtree(label_path)

        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        img.save(os.path.join(img_path, img.filename))
        label.save(os.path.join(label_path, label.filename))

        PhotovleService.train(user_id=user_id)

        return jsonify(True)


@model_bp.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]

        if img.filename == "":
            return jsonify(False)

        if label.filename == "":
            return jsonify(False)

        img_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "predict", "JPEGImages")
        label_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "predict", "Annotations")

        if os.path.isdir(img_path):
            shutil.rmtree(img_path)

        if os.path.isdir(label_path):
            shutil.rmtree(label_path)

        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        img.save(os.path.join(img_path, img.filename))
        label.save(os.path.join(label_path, label.filename))

        return jsonify(PhotovleService.predict(user_id=user_id))

@model_bp.route("/video", methods=["POST"])
def get_predicted_video():
    if request.method == "POST":
        user_id = str(request.json["user_id"])

        PhotovleService.predict_video(user_id=user_id)

        return send_file(
            os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "output.avi"),
            # os.path.join(os.getenv("TEMP_DATA_PATH"), "video", "hand.mp4"),
            # attachment_filename='output.avi',
            # as_attachment=True,
            mimetype="video/x-msvideo"
        )