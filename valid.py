import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import cv2
from model import see_eyes, RotEyes, RotCNN4, RotCNN6

PREDICTOR_PATH = "/content/models/shape_predictor_68_face_landmarks.dat"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

def check_iris(iris_in):
	if iris_in.ndim == 2:
		iris = iris_in.unsqueeze(0) #.permute(2,0,1)
		return iris
	else:
		return iris_in


def check_pic(path: str):
	img_np = cv2.imread(path)
	img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)
	area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
	if iris is not None and area is not None:
		return 1
	else:
		return 0

class ValDataset(Dataset):
	def __init__(self, path: str):
		data = torch.load(path)
		self.tensors = data['tensors']
		self.labels = data['labels']

	def __len__(self):
		return len(self.tensors)

	def __getitem__(self, n):
		return self.tensors[n], self.labels[n]

#
def map_data(data_dir: str) -> List[str]:
	dirs = []
	for name in os.listdir(data_dir):
		full_path = os.path.join(data_dir, name)
		if os.path.isdir(full_path):
			dirs.append(full_path)
	if not dirs:
		raise FileNotFoundError(f"\n\n???\n\n")
	return dirs
#
def load_data(data_dir: str):

	paths, labels, tensors, labels_l = [], [], [], []

	label_map = {"cdf": 0.0, "real": 0.0, "ff": 1.0, "fake": 1.0}
	image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

	for folder_name, label in label_map.items():
		folder_path = os.path.join(data_dir, folder_name)
		for root, dirs, files in os.walk(folder_path):
			for f in files:
				if f.lower().endswith(image_exts):
					path.append(os.path.join(root,f))
					labels.append(label)
	paths = np.array(paths)
	labels = np.array(labels)
	idx = np.random.permutation(len(paths))

	for i in range(len(paths)):
		paths[[i,idx[i]]]=paths[[idx[i],i]]
		labels[[i,idx[i]]]=labels[[idx[i],i]]

	for n in range(len(paths)):
		img_np = cv2.imread(paths[n])
		if img_np is not None:
			img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)
			faces = see_eyes(img_np, PREDICTOR_PATH)
			for face in faces:
				if face[0] is not None and face[1] is not None:
					result = torch.cat([face[0], face[1], face[2]], dim=2)
					tensors.append(result)
					labels_l.append(labels[n])

	torch.save({
		'tensors': tensors,
		'labels': labels_l
		}, os.path.join(data_dir, 'data.pt'))

	return (os.path.join(data_dir, 'data.pt'))
#
def valid_full(model_class, model_name: str, DDATA_DIR: str):

	model = model_class().to(DEVICE)
	model.eval()
	criterion = nn.BCEWithLogitsLoss()
	#hist = {"Method": [], "Accuracy": [], "Time_avg": []}
	hist = {"Method": [], "Accuracy": []}
	dirs=map_data(DDATA_DIR)
	start_time = time.perf_counter()

	path_m = []
	for d in dirs:
		path_m.append(load_data(d))

	for d in path_m:
		hist["Method"].append(d)
		valid_in = DataLoader(ValDataset(d), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
		correct, total = 0,0
		time_d = time.perf_counter()
		with torch.no_grad():
			for inputs, labels in valid_in:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				if outputs is not None:
					loss = criterion(outputs.squeeze(1),labels)
					preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
					correct += (preds == labels.float()).sum().item()
					total += labels.size(0)
		hist["Accuracy"].append(correct / total)
                #history["loss"].append(val_loss_sum / len(val_loader))
		#hist["Time_agv"].append((time.perf_counter() - time_d)/total)

	return hist

