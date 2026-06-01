#Данный файл содержит непосредственно описание модели

#CNN для распознавания дипфейков
#Архитектура для класса RotEyes:
#  Фото  -> CNN(1) -> Выделение области глаз, глазного яблока и радужки в три потока -> CNN по области глаз(2)(4 слоя) -> активация, субдескритизация, выпрямление, полносвязный -> 
#                                                                   -> CNN по радужке(2)(4 слоя)      -> активация, субдескритизация, выпрямление, полносвязный -> 
#																	-> CNN по области глаза(2)(4 слоя)-> активация, субдескритизация, выпрямление, полносвязный -> 
#        -> Если на входе видео, то проходим в несколько итераций, по полученному массиву значений, ReLU с bias=0,7 и (3)-> выводим на градиентный спуск три полученых значения и полученный (3), получаем результат
# (1) CNN обучается распознавать по области глаз и радужки отдельно
# (2) Обучение на итоговой выборке, нужна вероятность того, что данное изображение принадлежит дипфейку
# (3) Для видео мы можем получить динамику EAR, потому мы можем, по сути создать эрзац-свёрточную функцию, в которой мы применяем метод скользящего окна на 20 значений через Max Pooling, с шагом в 5, и подаём на Байесовский
#    обучение с учителем без заранее размеченных параметров

# Архитектура для класса RotCNN
#Фото  -> CNN(1) -> Выделение области глаз -> CNN по области глаз(2)(6 слоёв) -> активация, субдескритизация, выпрямление, полносвязный ->                           
#        -> Если на входе видео, то проходим в несколько итераций, отбрасывая кадры с EAR<0.25 и (3)-> выводим на градиентный спуск два полученых значения: вероятность для глаз(3), получаем результат

import numpy as np
import torch
import torch.nn as nn
import math
import torchvision
import dlib

IMGWIDTH=256
IMGHEIGHT=128

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

def to_tensor(arr):
        if arr.ndim == 2:
            return torch.from_numpy(arr).unsqueeze(0).float()
        else:
            return torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0


#Выделение массива с тремя областями из входного изображения
def see_eyes(image_bgr: np.ndarray,
    predictor_path: str
):
        area_m=list(1,17,18,21)
        eye_l_m = list(range(36, 42))
        eye_r_m = list(range(42, 48))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face = detector(gray, 1)
        if len(face) == 0:
                return [0,0,0,0,0]

        shape = predictor(gray, face[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        area_mask = pts_to_mask(region_pts, image_bgr)
        area = to_tensor(mask_eye_region)

        right_e, _ = crop_with_padding(right_pts, image_bgr, pad=10)
        eye_r = to_tensor(right_crop)

        left_e, _ = crop_with_padding(left_pts, image_bgr, pad=10)
        eye_l = to_tensor(left_crop)

        iris_r_mask = iris_mask(right_crop, right_pts)
        iris_r = torch.from_numpy(iris_r_mask).unsqueeze(0)

        iris_l_mask = iris_mask(left_crop, left_pts)
        iris_l = torch.from_numpy(iris_l_mask).unsqueeze(0)
#Область глаз, левый и правый глаза, левая и правая радужки
        return area, eye_l, eye_r, iris_l, iris_r


##############################################
#Область глаза, три ветви, 4 слоя
class RotEyes(nn.Module):
	
	def __init__(self, num_classes=1):
		super().__init__()
		self.RotArea=self._branch()
		self.RotEye=self._branch()
		self.RotIris=self._branch()
		self.fusion = nn.Sequential(
			nn.Linear(256 * 3, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(128, 1)
			)

	def _branch(self):
		return nn.Sequential(
	    #256x128 -> 128x64
		nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #128x64 -> 64x32
		nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #64x32 -> 32x16
		nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #32x16 -> 16x8
		nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
		nn.AdaptiveAvgPool2d((1, 1))
	)

	def forward(self, area, eye_l, eye_r, iris_l, iris_r):

		#area, eye_l, eye_r, iris_l, iris_r = see_eyes(image_bgr, predictor_path)
		f_area = self.RotArea(area).flatten(start_dim=1)

		f_eye_l  = self.RotEye(eye_l).flatten(start_dim=1)
		f_eye_r  = self.RotEye(eye_r).flatten(start_dim=1)
		f_iris_l  = self.RotIris(iris_l).flatten(start_dim=1)
		f_iris_r  = self.RotIris(iris_r).flatten(start_dim=1)

		f_eye  = (f_eye_l  + f_eye_r)  * 0.5
		f_iris = (f_iris_l + f_iris_r) * 0.5
		final = torch.cat([f_area, f_eye, f_iris], dim=1)
		return self.fusion(final)

##############################################
#Макет, на всю область глаз, 6 слоёв
class RotCNN6(nn.Module):
	def __init__(self, num_classes=1):
		super().__init__()
		self.features=nn.Sequential(
            #256x128 -> 128x64
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            #128x64 -> 64x32
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            #64x32 -> 32x16
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            #32x16 -> 32x16
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
	    #32x16 -> 16x8
			nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),	nn.MaxPool2d(2),
	    #16x8 -> 16x8
			nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),

			nn.AdaptiveAvgPool2d((1, 1))
			)

		self.classifier = nn.Sequential(
			nn.Dropout(0.4),
			nn.Linear(512, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.Linear(64, 1)
        		)

	def forward(self, area):
		#x, _ = see_eyes(image_bgr, predictor_path)
		x = self.features(area)
		x = torch.flatten(x, 1)  
		x = self.classifier(x)
		return x
##############################################

#Макет, на всю область глаз, 4 слоя
class RotCNN4(nn.Module):
	def __init__(self, num_classes=1):
		super().__init__()
		self.features=nn.Sequential(
            #256x128 -> 128x64
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            #128x64 -> 64x32
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
	    #64x32 -> 32x16
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            #32x16 -> 16x8
			np.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
			nn.AdaptiveAvgPool2d((1, 1))
        	)

		self.classifier = nn.Sequential(
			nn.Dropout(0.4),
			nn.Linear(256, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.Linear(64, 1)
        		)

	def forward(self, area):
		#x, _ = see_eyes(image_bgr, predictor_path)
		x = self.features(area)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
##############################################
