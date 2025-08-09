"""
Flask app to stream annotated frames and export CSV logs.
Run: python app/app.py
"""

import os
import time
import threading
from flask import Flask, Response, send_file, render_template_string
import cv2
import yaml
import pandas as pd

# add src to path if needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detector import YoloV7Detector
from src.tracker import CentroidTracker
from src.speed_estimator import SpeedEstimatorMPP, SpeedEstimatorHomography
from src.utils import ensure_dirs, draw_annotations, load_config
from src.logger import CSVLogger

cfg = load_config()  # loads config.yaml from project root
ensure_dirs(['exports/logs', 'calib'])

VIDEO_SOURCE = cfg.get('VIDEO_SOURCE', 0)
WEIGHTS = cfg.get('WEIGHTS_PATH', 'weights/yolov7.pt')
DEVICE = cfg.get('DEVICE', 'cpu')
FILTER_CLASSES = set(cfg.get('FILTER_CLASSES', []))
CALIB_PATH = cfg.get('CALIB_PATH', 'calib/H.npy')
LOG_CSV_PATH = cfg.get('LOG_CSV_PATH', 'exports/logs/speed_log.csv')
FRAME_SKIP = int(cfg.get('FRAME_SKIP', 0))
FPS_OVERRIDE = cfg.get('FPS_OVERRIDE', None)
MIN_TRACK_LENGTH = int(cfg.get('MIN_TRACK_LENGTH', 2))
SMOOTHING = int(cfg.get('SMOOTHING', 3))

detector = YoloV7Detector(weights=WEIGHTS, device=DEVICE, img_size=640, conf_thres=0.35)
tracker = CentroidTracker(max_disappeared=50)
# choose estimator: homography if available else MPP
if os.path.exists(CALIB_PATH):
    estimator = SpeedEstimatorHomography(CALIB_PATH, smoothing=SMOOTHING)
    print("Using homography-based speed estimator.")
else:
    estimator = SpeedEstimatorMPP(meters_per_pixel=float(cfg.get('METER_PER_PIXEL', 0.02)), smoothing=SMOOTHING)
    print("Using MPP-based speed estimator.")

csv_logger = CSVLogger(LOG_CSV_PATH)

app = Flask(__name__)

# simple index page
INDEX_HTML = """
<html>
  <head><title>AI Speed Detection</title></head>
  <body>
    <h2>AI Speed Detection — Video Stream</h2>
    <img src="{{ url_for('video_feed') }}" width="960" />
    <p><a href="{{ url_for('export_csv') }}">Download log CSV</a></p>
  </body>
</html>
"""

def frame_generator():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if FPS_OVERRIDE:
        fps = float(FPS_OVERRIDE)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.5)
            continue
        frame_idx += 1
        if FRAME_SKIP and (frame_idx % (FRAME_SKIP + 1) != 0):
            continue

        timestamp = time.time()
        detections = detector.detect(frame)
        # filter vehicle classes
        boxes = []
        meta = []
        for d in detections:
            if d['name'] in FILTER_CLASSES:
                boxes.append(d['bbox'])
                meta.append(d)

        objects = tracker.update(boxes, timestamp=timestamp)
        # objects: id -> dict {'centroid':(x,y), 'bbox': [x1,y1,x2,y2]}
        # compute speeds and annotate
        annotations = []
        for oid, obj in objects.items():
            history = tracker.history.get(oid, [])
            speed_kmph = None
            if len(history) >= MIN_TRACK_LENGTH:
                if isinstance(estimator, SpeedEstimatorHomography):
                    speed_kmph = estimator.estimate_speed(oid, history)  # returns km/h
                else:
                    speed_kmph = estimator.estimate_speed(oid, history)
            # find bbox if available
            bbox = obj.get('bbox', None)
            if bbox is None:
                # try nearest bbox by centroid proximity
                bbox = obj.get('bbox', None)
            label = f"ID {oid}"
            if speed_kmph:
                label += f" {speed_kmph:.1f} km/h"
                # log event - you might want to change frequency of logging
                csv_logger.log({
                    'timestamp': time.time(),
                    'id': oid,
                    'speed_kmph': float(speed_kmph),
                    'bbox': ",".join(map(str,bbox)) if bbox else ""
                })
            annotations.append({
                'bbox': bbox,
                'centroid': obj['centroid'],
                'label': label
            })
        # draw
        annotated = draw_annotations(frame, annotations)
        ret2, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export.csv')
def export_csv():
    # ensure CSV exists
    if not os.path.exists(LOG_CSV_PATH):
        pd.DataFrame(columns=['timestamp','id','speed_kmph','bbox']).to_csv(LOG_CSV_PATH, index=False)
    return send_file(LOG_CSV_PATH, as_attachment=True, download_name='speed_log.csv')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
