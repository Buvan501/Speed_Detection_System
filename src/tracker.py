"""
Simple centroid tracker that stores bbox and history (timestamp, centroid).
"""

import numpy as np
import time
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()  # id -> centroid
        self.bboxes = OrderedDict()   # id -> bbox
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.history = {}  # id -> list of (timestamp, (x,y))

    def register(self, centroid, bbox, timestamp=None):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.history[oid] = []
        if timestamp is None:
            timestamp = time.time()
        self.history[oid].append((timestamp, tuple(centroid)))
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.bboxes:
            del self.bboxes[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]
        if objectID in self.history:
            del self.history[objectID]

    def update(self, rects, timestamp=None):
        """
        rects: list of bboxes [x1,y1,x2,y2]
        timestamp: epoch seconds
        returns objects dict mapping id -> {'centroid':(x,y), 'bbox':[x1,y1,x2,y2]}
        """
        if timestamp is None:
            timestamp = time.time()
        if len(rects) == 0:
            # increase disappeared counters
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            # build return mapping
            return {oid: {'centroid': self.objects[oid], 'bbox': self.bboxes.get(oid, None)} for oid in self.objects}
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]), rects[i], timestamp=timestamp)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = tuple(input_centroids[col])
                self.bboxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                # append history
                self.history[objectID].append((timestamp, tuple(input_centroids[col])))
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])) - usedRows
            unusedCols = set(range(0, D.shape[1])) - usedCols
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            for col in unusedCols:
                self.register(tuple(input_centroids[col]), rects[col], timestamp=timestamp)
        return {oid: {'centroid': self.objects[oid], 'bbox': self.bboxes.get(oid, None)} for oid in self.objects}
