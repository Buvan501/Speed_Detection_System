#!/usr/bin/env bash
# basic runner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install YOLOv7 repo for torch.hub
if [ ! -d "yolov7" ]; then
  git clone https://github.com/WongKinYiu/yolov7.git
fi
pip install -r yolov7/requirements.txt
# Reminder for weights and coco.names
echo "Make sure you put weights/yolov7.pt and data/coco.names in place."
python app/app.py
