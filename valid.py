import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import cv2
from model import RotEyes, RotCNN4, RotCNN6

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DATA_DIR = "DF40-train"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

def check_pic(path: str):
	img_np = cv2.imread(path)
	img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)
	area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
	if iris is not None and area is not None:
		return 1
	else:
		return 0

class ValDataset(Dataset):
	def __init__(self, paths: List[str], labels: List[float]):
		self.paths = paths
		self.labels = labels

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, n):
		img_np = cv2.imread(self.image_paths[n])
		if img_np is None:
			raise FileNotFoundError(f"\n\nError in: {self.image_paths[idx]}\n")
		img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

		result = []
		area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
		result=torch.cat([area, eyes, iris], dim=2)
		label = torch.tensor(self.labels[n], dtype=torch.float32)

		return result, label
#
def map_data(data_dir: str) -> List[str]:
	dirs = []
	for root, dir, _ in os.walk(data_dir):
		for f in dir:
			dirs.append(os.path.join(data_dir,dir))
	if not dirs:
		raise FileNotFoundError(f"\n\n???\n\n")
	return dirs
#
def load_data(data_dir: list[str]) -> Tuple[List[str], List[float]]:

	path, labels = [], []

	label_map = {"cdf": 0.0, "ff": 1.0}
	image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

	for folder_name, label in label_map.items():
		folder_path = os.path.join(data_dir, folder_name)
		for root, dirs, files in os.walk(folder_path):
			for f in files:
				if f.lower().endswith(image_exts):
#					a=os.path.join(root,dirs,f)
#					if check_pic(a) == 1:
					path.append(os.path.join(root,dirs,f))
					labels.append(label)
	paths = np.array(paths)
	labels = np.array(labels)
	idx = np.random.permutation(len(paths))

	for i in range(len(paths)):
		paths[[i,idx[i]]]=paths[[idx[i],i]]
		labels[[i,idx[i]]]=labels[[idx[i],i]]
	return (paths.tolist(), labels.tolist())
#
def valid_full(model_class, model_name: str):
	model = model_class().to(DEVICE)
	model.eval()
	criterion = nn.BCEWithLogitsLoss()
	#hist = {"Method": [], "Accuracy": [], "Time_avg": []}
	hist = {"Method": [], "Accuracy": []}
	dirs=map_data(DATA_DIR)
	start_time = time.perf_counter()

	for d in dirs:
		hist["Method"].append(d)
		input_t, label_t = load_data(d)
		valid_in = DataLoader(ValDataset(input_t, label_t), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
		correct, total = 0,0
		time_d = time.perf_counter()
		with torch.no_grad():
			for inputs, labels in valid_in:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				if outputs is not None:
					loss = criterion(outputs.squeese(1,labels),labels)
					preds = (torch.sigmoid(outputs) > 0.5).float.squeese(1)
					correct += (preds == labels).sum().item()
					total += labels.size(0)
		hist["Accuracy"].append(correct / total)
                #history["loss"].append(val_loss_sum / len(val_loader))
		#hist["Time_agv"].append((time.perf_counter() - time_d)/total)

	return hist

