#Тренировка по выбору трёх моделей: RotEye, RotCNN4, RotCNN6
#Проверить графики
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import see_eyes, to_tensor, RotEyes, RotCNN4, RotCNN6
from valid import map_data, load_data, valid_full
import sys
from typing import List, Dict, Tuple
import cv2
from model import see_eyes, RotEyes, RotCNN4, RotCNN6
from valid import map_data, load_data, valid_full
from google.colab import drive, files


PREDICTOR_PATH = "/RotCNN--/RotCNN--/shape_predictor_68_face_landmarks.dat"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
TORCH_E = torch.zeros((3,128,256), dtype=torch.float32)


#MODELS_CONFIG = [
#    ("RotEyes", RotEyes),
#    ("RotCNN4", RotCNN4),
#    ("RotCNN6", RotCNN6)
#]
################
def weight(pred, label):
	ret=pred-label
	if ret < 0:
		ret = 1+ret
	if label == 1.0 and ret >= 0.85:
		return 1
	if label == 0.0 and ret <= 0.20:
		return 1
	return 0
#################
def early_stop(acc: []):
	l = len(acc)
	gate = 0.01
	sum = 0
	if l < 6:
		return False
	sum = ((acc[l-1] - acc[l-2])+(acc[l-2] - acc[l-3])+(acc[l-3] - acc[l-4])+(acc[l-4] - acc[l-5])+(acc[l-5] - acc[l-6]))/5
	if sum <=gate:
		return True
	else:
		return False
##################
def check_pic(path: str):
	img_np = cv2.imread(path)
	img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)
	area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
	if iris is None or area is None:
		return 0
	else:
		return 1
##################
def load_data_split(data_dir_i: str) -> Tuple[List[str], List[float]]:

  paths, labels, tensors, labels_l = [], [], [], []
  label_map = {"cdf": 0.0, "real": 0.0, "fake": 1.0, "ff": 1.0}

  for data_dir in os.listdir(data_dir_i):
    for folder_name, label in label_map.items():
      folder_path = os.path.join(data_dir, folder_name)
      for root, _, files in os.walk(folder_path):
        for f in files:
          paths.append(os.path.join(root,f))
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
      area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
      if area is not None and eyes is not None:
        result = torch.cat([area, eyes, check_iris(iris)], dim=2)
        tensors.append(result)
        labels_l.append(labels[n])

  torch.save({
    'tensors': tensors,
    'labels': labels_l
  }, os.path.join(data_dir, 'data.pt'))

  return (os.path.join(data_dir, 'data.pt'))
####################
def check_iris(iris_in):
	if iris_in.ndim == 2:
		iris = iris_in.unsqueeze(0) #.permute(2,0,1)
		return iris
	else:
		return iris_in
#####################
class EyesDataset(Dataset):
	def __init__(self, path: str):
		data = torch.load(path)
		self.tensors = data['tensors']
		self.labels = data['labels']

	def __len__(self):
		return len(self.tensors)

	def __getitem__(self, n):
		return self.tensors[n], self.labels[n]
#####################
def train_model(model_class, train_loader: DataLoader, epochs: int, lr: float, model_name: str) -> Tuple[nn.Module, Dict, float]:
	model = model_class().to(DEVICE)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	history = {"loss": [],"acc_t": [], "acc_v": [], "time": []}
	start_time = time.perf_counter()
	print("\nModel", model_name, "...\n")

	for epoch in range(epochs):
		print("\nEpoch",epoch,"...")
		model.train()
		train_correct, train_total, train_loss_sum = 0, 0, 0.0
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(inputs)
			if outputs is not None:
				loss = criterion(outputs.squeeze(1), labels)
				loss.backward()
				optimizer.step()

				preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
				train_correct += (weight(preds,labels.float())).sum().item()
				train_total += labels.size(0)
				train_loss_sum += loss.item()
		history["loss"].append(train_loss_sum)
		#функционал для построения графиков точности. Эксклюзив для курсовой, в прод не ставить
		model.eval()
		val_correct, val_total, val_loss_sum = 0, 0, 0.0
		with torch.no_grad():
			for inputs, labels in train_loader:
				labels_t = labels.float
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				if outputs is not None:
					loss = criterion(outputs.squeeze(1),labels)
					preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
					val_correct += (weight(preds,labels.float())).sum().item()
					val_total += labels.size(0)

		history["acc_t"].append(train_correct / train_total)
		history["acc_v"].append(val_correct / val_total)

		history["time"].append(time.perf_counter() - start_time)
		if early_stop(history["acc_v"]) is True:
			print("\nStopping...")
			break
	training_time = time.perf_counter() - start_time
	return model, history, training_time
##################
def plot_accuracy_curves(all_histories: Dict[str, Dict], output_path: str):

	plt.figure(figsize=(20, 6))
	for name, hist in all_histories.items():
		epochs = range(1, len(hist["acc_v"]) + 1)
		plt.plot(epochs, hist["acc_v"], label=f"{name}(val)", marker='o', linewidth=2)
		plt.plot(epochs, hist["acc_t"], label=f"{name}(train)", linestyle='-', alpha=0.7)

	plt.xlabel("Epoch", fontsize=12)
	plt.ylabel("Accuracy", fontsize=12)
	plt.title("Accuracy per epoch", fontsize=14)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.show()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
##################
def plot_methods(all_histories: Dict[str, float], output_path: str):

	plt.figure(figsize=(10, 6))
	methods = list(all_histories.keys())
	accuracies = [max(hist["acc_v"]) for hist in all_histories.values()]
	bars = plt.bar(methods, accuracies, color='red', edgecolor='black')

	plt.xlabel("Method", fontsize=12)
	plt.ylabel("Accuracy", fontsize=12)
	plt.title("Accuracy", fontsize=14)

	plt.grid(True, alpha=0.3)
	plt.show()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
##################
def main_f(model_config, DATA_DIR, VAL_DIR):
###################################################################
	train_path = load_data_split(DATA_DIR)
	train_loader = DataLoader(EyesDataset(train_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	all_histories = {}
	all_metrics   = {}


	for model_name, model_class in model_config:
		print("\nStart: ")
		print(model_name)
		model, history, train_time = train_model(model_class, train_loader, EPOCHS, LEARNING_RATE, model_name)

		all_histories[model_name] = history
		all_metrics[model_name] = {
			"best_val_acc": max(history["acc_v"]),
			"training_time": train_time
			}

		os.makedirs("models", exist_ok=True)
		torch.save(model.state_dict(), f"/content/models/{model_name}-a.pt")
		files.download(f"/content/models/{model_name}-a.pt")
		print(f"Модель сохранена: models/{model_name}.pt")


	print("\nAccuracy")
	plot_accuracy_curves(all_histories, f"/content/models/{model_name}-curve.png")
	hist_full=valid_full(model_class, model_name, VAL_DIR)
	plot_methods(hist_full,f"/content/models/{model_name}_methods.png")
	files.download(f"/content/models/{model_name}-curve.png")
	files.download(f"/content/models/{model_name}-methods.png")
#####################################################################
