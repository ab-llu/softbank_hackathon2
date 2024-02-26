from flask import Flask, request, render_template, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename

import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No image selected for uploading"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join('static', filepath))
        
        # ここで画像加工処理を呼び出す
        processed_image_filepath = process_image(os.path.join('static', filepath))
        processed_image_url = processed_image_filepath
        
        return render_template('image_display.html', image_url=processed_image_url)
    else:
        return "Something went wrong"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR="/content/drive/MyDrive/softbank_hackathon/model_trained/tooth_test_2/" 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3


def process_image(image_path):
    # 画像に対する加工処理を行う
    # 例: Pillowを使って画像をグレースケールにする
    from PIL import Image
    img = Image.open(image_path).convert('L')
    processed_image_path = image_path.replace('.jpg', '_gray.jpg')
    img.save(processed_image_path)
    return processed_image_path

if __name__ == "__main__":
    app.run(debug=True)