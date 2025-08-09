"""
Run this to collect image points and world points and compute homography.
Usage:
  python src/calibrate.py --image path/to/frame.jpg
You will click image points (N>=4) in order, then provide matching world coords (X,Y in meters) as CSV input.
Saves calib/H.npy
"""

import cv2
import numpy as np
import argparse
import json
import os

IMG_WINDOW = "Calibration - click image points (press 'q' when done)"

def collect_image_points(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image: " + image_path)
    pts = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img, (x, y), 4, (0,255,0), -1)
    cv2.namedWindow(IMG_WINDOW)
    cv2.setMouseCallback(IMG_WINDOW, mouse_cb)
    while True:
        display = img.copy()
        for i,p in enumerate(pts):
            cv2.putText(display, f"{i+1}", (p[0]+5, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow(IMG_WINDOW, display)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyWindow(IMG_WINDOW)
    return pts, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image (frame) to click points on')
    parser.add_argument('--out', default='calib/H.npy', help='Output homography path')
    args = parser.parse_args()

    image_path = args.image
    image_points, img = collect_image_points(image_path)
    n = len(image_points)
    if n < 4:
        print("Need at least 4 points; you clicked", n)
        return
    print(f"You clicked {n} image points. Now provide the corresponding {n} WORLD points (X,Y in meters).")
    world_points = []
    for i in range(n):
        s = input(f"Enter world X,Y for point {i+1} (comma separated, e.g. 0.0,0.0): ")
        x,y = s.split(',')
        world_points.append((float(x.strip()), float(y.strip())))
    img_pts = np.array(image_points, dtype='float32')
    world_pts = np.array(world_points, dtype='float32')
    H, status = cv2.findHomography(img_pts, world_pts)
    if H is None:
        print("Homography computation failed.")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, H)
    print("Saved homography to", args.out)

if __name__ == '__main__':
    main()
