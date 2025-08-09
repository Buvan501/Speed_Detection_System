"""
Simple YOLOv7 wrapper. Tries to load model via torch.hub (WongKinYiu/yolov7).
If you have custom local loader, adapt below.
"""

import torch
import numpy as np
import cv2
import os

class YoloV7Detector:
    def __init__(self, weights='weights/yolov7.pt', device='cpu', img_size=640, conf_thres=0.4):
        self.device = device
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.names = []
        # load coco names if available
        names_path = os.path.join('data', 'coco.names')
        if os.path.exists(names_path):
            with open(names_path, 'r') as f:
                self.names = [x.strip() for x in f.readlines()]
        # attempt to load via torch.hub
        try:
            # This will download repo code the first time
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights, source='github')
            model.to(device)
            model.eval()
            self.model = model
        except Exception as e:
            print("torch.hub load failed:", e)
            # try loading locally with torch.load (may not work for all weights)
            try:
                self.model = torch.load(weights, map_location=device)
                self.model.eval()
            except Exception as e2:
                raise RuntimeError(f"Failed to load YOLOv7 model via torch.hub and direct load: {e2}")

    def preprocess(self, frame):
        # model wrapper used supports raw BGR passthrough in many hub implementations; if not, adapt
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = img[:, :, ::-1]  # BGR->RGB
        return img

    def detect(self, frame):
        """
        Input: BGR frame (numpy)
        Output: list of detections: [{'bbox': [x1,y1,x2,y2], 'conf':float, 'class':int, 'name':str}]
        """
        # The Torch hub wrapper generally accepts numpy RGB frames directly with a size arg.
        try:
            results = self.model(frame[..., ::-1], size=self.img_size)  # many yolov7 wrappers accept this
            # results.xyxy[0] -> [x1,y1,x2,y2,conf,class]
            detections = []
            if hasattr(results, 'xyxy'):
                arr = results.xyxy[0].cpu().numpy()
                for row in arr:
                    x1,y1,x2,y2,conf,cls = row
                    if conf < self.conf_thres:
                        continue
                    cls = int(cls)
                    name = self.names[cls] if (cls < len(self.names)) else str(cls)
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': float(conf),
                        'class': cls,
                        'name': name
                    })
            else:
                # fallback parsing if wrapper differs
                results = results.pandas().xyxy[0]  # pandas DataFrame in some wrappers
                for _, r in results.iterrows():
                    if r['confidence'] < self.conf_thres:
                        continue
                    cls = int(r['class'])
                    name = self.names[cls] if (cls < len(self.names)) else str(cls)
                    detections.append({
                        'bbox': [int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax'])],
                        'conf': float(r['confidence']),
                        'class': cls,
                        'name': name
                    })
            return detections
        except Exception as e:
            print("Detection error:", e)
            return []
