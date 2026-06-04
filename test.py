import numpy as np
import torch
import torch.nn as nn
import math
import torchvision
import dlib
import cv2

PATH=DF40-test/10340.jpg

def crop_with_padding(pts, img, pad=5):
        x, y, w, h = cv2.boundingRect(pts)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        return img[y1:y2, x1:x2], (x1, y1)

def pts_to_mask(pts, shape_img):
        mask = np.zeros(shape_img[:2], dtype=np.uint8)
        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        return mask

def iris_mask(crop_bgr, eye_pts):
        center = np.mean(eye_pts, axis=0).astype(int)
        width = np.linalg.norm(eye_pts[0] - eye_pts[3])  
        pupil_r = int(width * 0.15)
        iris_r = int(width * 0.35)
        crop_center = (center[0] - 0, center[1] - 0) 
        h, w = crop_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center[0], center[1]), iris_r, 255, -1)
        return mask / 255.0
def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
def to_tensor(arr):
        if arr.ndim == 2:
            return torch.from_numpy(arr).unsqueeze(0).float()
        else:
            return torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0

def main():
	return 0
