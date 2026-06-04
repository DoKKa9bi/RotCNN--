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

from typing import List, Dict, Tuple
import cv2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DATA_DIR = "DF40-train"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8



MODELS_CONFIG = [
    ("RotEyes", RotEyes),
    ("RotCNN4", RotCNN4),
    ("RotCNN6", RotCNN6)
]
#
def check_pic(path: str):
	img_np = cv2.imread(path)
	img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)
	area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
	if iris is not None and area is not None:
		return 1
	else:
		return 0

def load_data_split(data_dir: str) -> Tuple[List[str], List[float]]:

	paths, labels = [], []
    	# Под разметку в две папки, соответственно настоящие фото и сгенерированные.
	#Методы(июль-октябрь 2024): Midjourney, StyleGANXL, VQGAN, sd2.1
	label_map = {"cdf": 0.0, "ff": 1.0}

	for folder_name, label in label_map.items():
		folder_path = os.path.join(data_dir, folder_name)
		for root, _, files in os.walk(folder_path):
			for f in files:
				a=(os.path.join(root, f))
#				if check_pic(a) == 1:
				paths.append(a)
				labels.append(label)

	paths = np.array(paths)
	labels = np.array(labels)
	idx = np.random.permutation(len(paths))
	for i in range(len(paths)):
		paths[[i,idx[i]]]=paths[[idx[i],i]]
		labels[[i,idx[i]]]=labels[[idx[i],i]]
	return (paths.tolist(), labels.tolist())


#
class EyesDataset(Dataset):
	def __init__(self, paths: List[str], labels: List[float]):
		self.paths = paths
		self.labels = labels

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, n):
		img_np = cv2.imread(self.paths[n])
		if img_np is None:
			raise FileNotFoundError(f"\n\nError in: {self.image_paths[n]}\n")
		img_np = cv2.cvtColor( img_np, cv2.COLOR_BGR2RGB)

		area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)
		print("\n\t",iris.ndim,"-",iris.shape)
		result=torch.cat([area, eyes, iris], dim=2)

		label = torch.tensor(self.labels[n], dtype=torch.float32)

		return result, label
#
def train_model(model_class, train_loader: DataLoader, epochs: int, lr: float, model_name: str) -> Tuple[nn.Module, Dict, float]:
	model = model_class().to(DEVICE)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

    	#history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
	history = {"loss": [],"acc_t": [], "acc_v": [], "time": []}
	start_time = time.perf_counter()
	print("\nModel {model_name}...\n\n")

	for epoch in range(epochs):
		print("\nEpoch {epoch}...")
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
				train_correct += (preds >= labels-0.25).sum().item()
				train_total += labels.size(0)
			#train_loss_sum += loss.item()
		history["loss"].append(train_loss_sum())
		#функционал для построения графиков точности. Эксклюзив для курсовой, в прод не ставить
		model.eval()
		val_correct, val_total, val_loss_sum = 0, 0, 0.0
		with torch.no_grad():
			for inputs, labels in train_loader:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				loss = criterion(outputs.squeese(1,labels),labels)
				preds = (torch.sigmoid(outputs) > 0.5).float.squeese(1)
				val_correct += (preds >= labels-0.25).sum().item()
				val_total += labels.size(0)
				#val_loss_sum += loss.item()

		history["acc_t"].append(train_correct / train_total)
		history["acc_v"].append(val_correct / val_total)
#history["loss"].append(val_loss_sum / len(val_loader))
		history["time"].append(time.perf_counter() - start_time)

	training_time = time.perf_counter() - start_time
	return model, history, training_time
#
def plot_accuracy_curves(all_histories: Dict[str, Dict], output_path: str = "results/accuracy_curves.png"):

	plt.figure(figsize=(20, 6))
	for name, hist in all_histories.items():
		epochs = range(1, len(hist["acc_v"]) + 1)
		plt.plot(epochs, hist["acc_v"], label=f"{name} (val)", marker='o', linewidth=2)
		plt.plot(epochs, hist["acc_t"], label=f"{name} (train)", linestyle='-', alpha=0.7)

	plt.xlabel("Epoch", fontsize=12)
	plt.ylabel("Accuracy", fontsize=12)
	plt.title("Accuracy per epoch", fontsize=14)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.show()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
#
def plot_methods(all_histories: Dict[str, float], output_path: str = "results/curves.png"):

	plt.figure(figsize=(10, 6))
	methods = list(all_histories.keys())
	accuracies = [value[1] for value, _ in all_histories.values()]
	bars = plt.bar(methods, accuracies, color='red', edgecolor='black')

	plt.xlabel("Method", fontsize=12)
	plt.ylabel("Accuracy", fontsize=12)
	plt.title("Accuracy", fontsize=14)

	plt.grid(True, alpha=0.3)
	plt.show()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close()
#
def main():
	train_paths, train_labels = load_data_split(DATA_DIR)
	train_loader = DataLoader(EyesDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	all_histories = {}
	all_metrics   = {}


	for model_name, model_class in MODELS_CONFIG:
		print("\nStart: ")
		print(model_name)
		model, history, train_time = train_model(model_class, train_loader, EPOCHS, LEARNING_RATE, model_name)

		all_histories[model_name] = history
		all_metrics[model_name] = {
			"best_val_acc": max(history["acc_v"]),
			"training_time": train_time
			}

		os.makedirs("models", exist_ok=True)
		torch.save(model.state_dict(), f"models/{model_name}.pt")
		print(f"Модель сохранена: models/{model_name}.pt")


		print("\nAccuracy")
		plot_accuracy_curves(all_histories)
		hist_full=valid_full(model_class, model_name)
		
#####################################################################
if __name__ == "__main__":

	from model import see_eyes, RotEyes, RotCNN4, RotCNN6
	from valid import map_data, load_data, valid_full
	main()
