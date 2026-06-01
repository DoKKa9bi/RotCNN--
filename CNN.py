#CNN для распознавания дипфейков
#Архитектура:
# model.py--распознание изображения и прогон через нейронку
# train.py--тренировочный файл, загрузка данных, прогон через нейронку, сохранение весов и вывод статистики
# CNN.py--файл с интерфейсом для исползования нейронки
# valid.py--файл с функциями для валидации вывода нейронки

##задачи в порядке приоритетности
#отчёт, обучение

import os
import time
import argparse
import torch
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

FORMAT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():

	parser_a = argparse.ArgumentParser(
		prog="RotEyes",
		description="Поиск дипфейков нa основе модульной архитектуры",
		epilog="")

	parser_a.add_argument('file', type=str, help="путь к проверяемому файлу")
	args = parser_a.parse_args()


	if (args.file).tolower.endswith(FORMAT):
		file=cv2.imread(args.file)
	else:
		raise FileFormatError (f"\nНеправильный формат! Введите изображение!\n")
		return -1
	img_np = cv2.ctvColor(img_np, cv2.COLOR_BGR2RGB)

	area, eye_l, eye_r, iris_l, iris_r = see_eyes(img_np, PREDICTOR_PATH)

	area=area.to_tensor()
	eye_l=eye_l.to_tensor()
	eye_r=eye_r.to_tensor()
	iris_l=iris_l.to_tensor()
	iris_r=iris_r.to_tensor()


	model=RotEyes()
	save = torch.load('models/RotEyes.pt', map_location=DEVICE)
	state_dict = save
	model.load_state_dict(state_dict)
	model.eval()
	model=RotEyes().to(DEVICE)


	pred=model(area, eye_l, eye_r, iris_l, iris_r)
	preds = (torch.sigmoid(pred) > 0.5).float().squeeze(1)

	print("\nМодель RotEyes: на {pred} процентов дипфейк")

if __name__ == "__main__":
	from model import see_eyes, to_tensor, RotEyes
main()
