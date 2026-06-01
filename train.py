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
from tqdm import tqdm
from typing import List, Dict, Tuple
import cv2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DATA_DIR = "DF40-train"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4


MODELS_CONFIG = [
    {"name": "RotEyes", "class": RotEyes},
    {"name": "RotCNN4", "class": RotCNN4},
    {"name": "RotCNN6", "class", RotCNN6},
]
#
def load_data_split(data_dir: str) -> Tuple[List[str], List[float]]:

	paths, labels = [], []
    	# Под разметку в две папки. Забрасываем туда соответствующие изображения от методов []
	label_map = {"real": 0.0, "fake": 1.0}

	for folder_name, label in label_map.items():
		folder_path = os.path.join(data_dir, folder_name)
		for root, _, files in os.walk(folder_path):
			for f in files:
				paths.append(os.path.join(root, f))
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
	img_np = cv2.imread(self.image_paths[n])
        if img_np is None:
            raise FileNotFoundError(f"\n\nError in: {self.image_paths[idx]}\n")

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img_np, label
#
def train_model(model_class, train_loader: DataLoader, epochs: int, lr: float, model_name: str) -> Tuple[nn.Module, Dict, float]:
	model = model_class().to(DEVICE)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

    	#history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
	history = {"loss": [],"acc_t": [], "acc_v": [], "time": []}
	start_time = time.perf_counter()

	for epoch in range(epochs):
		model.train()
		train_correct, train_total, train_loss_sum = 0, 0, 0.0
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs.squeeze(1), labels)
			loss.backward()
			optimizer.step()

			preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(1)
			train_correct += (preds == labels).sum().item()
			train_total += labels.size(0)
			#train_loss_sum += loss.item()
		history["loss"].append(train_loss_sum())
		#функционал для построения графиков точности. Эксклюзив для курсовой, в прод не ставить
		model.eval()
		val_correct, val_total, val_loss_sum = 0, 0, 0.0
		with torch.no_grad():
			for inputs, labels in train_loader
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				loss = criterion(outputs.squeese(1,labels),labels)
				preds = (torch.sigmoid(outputs) > 0.5).float.squeese(1)
				val_correct += (preds == labels).sum().item()
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

    plt.figure(figsize=(10, 6))
    for name, hist in all_histories.items():
        epochs = range(1, len(hist["val_acc"]) + 1)
        plt.plot(epochs, hist["val_acc"], label=f"{name} (val)", marker='o', linewidth=2)
        plt.plot(epochs, hist["train_acc"], label=f"{name} (train)", linestyle='--', alpha=0.7)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation & Train Accuracy over Epochs", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
	train_paths, train_labels = load_data_split(DATA_DIR)
	train_loader = DataLoader(EyesDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	all_histories = {}
	all_metrics   = {}


    for config in MODELS_CONFIG:
        model_name = config["name"]
	model_class = config["class"]
	print("\nStart: {model_name}\n")
        model, history, train_time = train_model(model_class, train_loader, EPOCHS, LEARNING_RATE, model_name)

        all_histories[model_name] = history
        all_metrics[model_name] = {
            "best_val_acc": max(history["acc_v"]),
            "training_time": train_time
        }

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/{model_name}_final.pt")
        print(f"Модель сохранена: models/{model_name}_final.pt")

	#Добавить валидацию по всему датасету после определения нужного количества эпох
    print("\nAccuracy")
    plot_accuracy_curves(all_histories)
   # plot_summary_table(all_metrics)
   
#    print("\n" + "="*50)
#    print("PROFIT?")
#    print("="*50)
#    print(f"{'Model':<12} | {'Val Acc':<8} | {'Train Time':<10} )
#    print("-"*50)
#    for name, m in all_metrics.items():
#Добавить среднее время отработки по изображениям 
#       print(f"{name} | {['best_val_acc']} | {['training_time']}s")
#    print("="*50)


if __name__ == "__main__":

	from model import see_eyes, RotEyes, RotCNN4, RotCNN6
	from valid import map_data, load_data, valid_full
    main()
