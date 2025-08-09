import os
import yaml
import cv2

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def draw_annotations(frame, annotations):
    # annotations: list of {'bbox': [x1,y1,x2,y2], 'centroid':(x,y), 'label': str}
    img = frame.copy()
    for ann in annotations:
        bbox = ann.get('bbox', None)
        centroid = ann.get('centroid', None)
        label = ann.get('label', '')
        if bbox:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if centroid and not bbox:
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 4, (0,255,0), -1)
            cv2.putText(img, label, (int(centroid[0])+5, int(centroid[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img
