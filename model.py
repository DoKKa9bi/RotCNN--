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
import torchvision.models as models
import dlib
import cv2
IMGWIDTH=256
IMGHEIGHT=128
EMPTY_T = torch.zeros((3,128,256), dtype=torch.float32)


def pts_to_mask(pts, landmarks):
	land = []
	for po in pts:
		land.append(landmarks[po])
	land=np.array(land, dtype=np.int32)
	mask = cv2.convexHull(land)
	return mask


def iris_out(img_crop, landmarks, pts):
	iris_ma = [1,2,4,5]
	iris_md = []
	for iri in iris_ma:
		iris_md.append(pts[iri])

	iris_mask = pts_to_mask(iris_md, landmarks)
	x, y, w, h = cv2.boundingRect(iris_mask)
	if h<=5:
		return None

	pad_x = int(w * 0.2)
	pad_y = int(h * 0.3)

	img_h, img_w = img_crop.shape[:2]
	x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
	x2, y2 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
	area=img_crop[y1:y2, x1:x2]
	if area.size==0:
		top_y = min(pts[1], pts[2])
		bottom_y = max(pts[4], pts[5])
		gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		a, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		best_contour = None
		max_area = 0
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if 30 < area < 2500:  
				perimeter = cv2.arcLength(cnt, True)	
				if perimeter > 0:
					circularity = 4 * np.pi * area / (perimeter * perimeter)
					M = cv2.moments(cnt)
					if M["m00"] != 0:
						cy = int(M["m01"] / M["m00"])
						if top_y < cy < bottom_y and circularity > 0.4:
							if area > max_area:
								max_area = area
								best_contour = cnt
		if best_contour is not None:
			x, y, w, h = cv2.boundingRect(best_contour)
		else:
			x = ((pts[0] + pts[3]) / 2) - 30
			y = top_y
			w = 60
			h = bottom_y - top_y
		img_h, img_w = img_crop.shape[:2]
		x1,y1 = int(max(0, x - pad_x)), int(max(0, y - pad_y))
		x2,y2 = int(min(img_w, x + w + pad_x)), int(min(img_h, y + h + pad_y))
		area = img_crop[y1:y2, x1:x2]
		if area.size==0:
			return None

	ret = cv2.cvtColor(area, cv2.COLOR_BGR2RGB)
	result = cv2.resize(ret, (128,128))
	if result.ndim == 3:
		result = result #.transpose(2,0,1)
	return result


def crop_face(pts, img, pad=5):
	x, y, w, h = cv2.boundingRect(pts)
	x1, y1 = max(0, x - pad), max(0, y - pad)
	x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
	re = img[y1:y2, x1:x2]
	re = cv2.resize(re, (256,128))
	return cv2.cvtColor(re, cv2.COLOR_BGR2RGB)

def crop_eyes(pts_b, img, pad=5):
	x_b, y_b, w_b, h_b = cv2.boundingRect(pts_b)

	x1_b, y1_b = max(0, x_b - pad), max(0, y_b - pad)
	x2_b, y2_b = min(img.shape[1], x_b + w_b + pad), min(img.shape[0], y_b + h_b + pad)
	img_b = cv2.resize(img[y1_b:y2_b, x1_b:x2_b], (256,128))

	return  img_l, img_r, img_b

def draw_end(area: [], img):
	img_b = img.copy()
	for item in area:
		x, y, w, h = cv2.boundingRect(item)
		cv2.rectangle(img_b, (x, y), (x + w, y + h), (0, 255, 0), -1)
		cv2.addWeighted(img_b, 0.4, img, 0.6, 0, img)
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imshow("Area", img)


def to_tensor(arr):
	if arr.ndim == 2:
		re = torch.from_numpy(arr).float().unsqueeze(0)
		return re
	if arr.ndim == 3 :
		a = torch.from_numpy(arr).float()
		return torch.permute(a, (2,0,1))

#Выделение массива с тремя(5) областями из входного изображения
def see_eyes(image_bgr: np.ndarray,
    predictor_path: str
):
	area_m = [0,16,17,26]
	eye_r_m = [36,37,38,39,40,41]
	eye_l_m = [42,43,44,45,46,47]
	eye_b_m = list(range(36, 47))
	lips = [49, 51, 53, 58]

	faces=[]
	
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	face = detector(gray, 1)
	if len(face) == 0:
		return [None, None, None]
	for face_o in face:
		shape = predictor(gray, face_o)
		landmarks = np.array([[p.x, p.y] for p in shape.parts()])

		area_m_l = pts_to_mask(area_m, landmarks)
		area_mask = crop_face(area_m_l, image_bgr, pad=5)
		area=to_tensor(area_mask)

		lips_m_l = pts_to_mask(lips, landmarks)
		lips_mask=crop_face(lips_m_l, image_bgr, pad=5)
		lips=to_tensor(area_mask)
	
	#right_e_l = pts_to_mask(eye_r_m, landmarks)
	#left_e_l = pts_to_mask(eye_l_m, landmarks)
	
		eyes_b_l = pts_to_mask(eye_b_m, landmarks)
		eyes_e = crop_eyes(eyes_b_l,  image_bgr, pad=10)
		eyes = to_tensor(eyes_e)
#Область глаз, глаза, рот
		draw_end(area_m_l, lips, image_bgr)
		faces.append([area,eyes,lips)
		
	return faces


##############################################
#Область глаза, три ветви, 4 слоя
class RotEyes(nn.Module):

	def __init__(self, num_classes=1):
		super().__init__()
		self.RotArea=self._branch()
		self.RotEye=self._branch()
		self.RotIris=self._branch()

		self.fusion = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(768, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(128, 1)
			)

	def _branch(self):
		resnet = models.resnet34(pretrained=False)
		return nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool,
			resnet.layer1,
			resnet.layer2,
			resnet.layer3,
			nn.AdaptiveAvgPool2d((1, 1))
		)

	def _branch_s(self):
		resnet = models.resnet34(pretrained=False)
		return nn.Sequential(
			resnet.conv1, 
			resnet.bn1,
			resnet.relu,
			resnet.maxpool,
			resnet.layer1,
			resnet.layer2,
			nn.AdaptiveAvgPool2d((1, 1))
	)

	def forward(self, input):
		area, eyes, lips = torch.split(input, 256, dim = 3)

		#area, eye, iris = see_eyes(image_bgr, predictor_path)
		f_area = self.RotArea(area).flatten(start_dim=1)
		f_eye = self.RotEye(eyes).flatten(start_dim=1)
		f_lips = self.RotIris(lips).flatten(start_dim=1)

		final = torch.cat([f_area, f_eye, f_lips], dim=1)

		return self.fusion(final)
