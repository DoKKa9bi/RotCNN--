#CNN для распознавания дипфейков
#Архитектура:
# model.py--распознание изображения и прогон через нейронку
# train.py--тренировочный файл, загрузка данных, прогон через нейронку, сохранение весов и вывод статистики
# CNN.py--файл с интерфейсом для исползования нейронки
# valid.py--файл с функциями для валидации вывода нейронки

##задачи в порядке приоритетности
#1. Добавить распознание по углу наклона глаз?
#2. Интерфейс(в текущем файле)
#3. Распознавание на видео?(по тому, ка будет обучаться)

import os
import time
import argparse

MODELS_CONFIG = [
    {"name": "RotEyes", "class": RotEyes},
    {"name": "RotCNN4", "class": RotCNN4},
    {"name": "RotCNN6", "class", RotCNN6},
]


def main():

parser_a = argparse.ArgumentParser(
		prog="RotEyes",
		description="Поиск дипфейков нa основе модульной архитектуры",
		epilog="")

parser_a.add_argument('filename')
parser_a.add_argument('-c6')
parser_a.add_argument('-c4')



args = parser_a.parse_args()
if __name__ == "__main__":
	from model import see_eyes, RotEyes, RotCNN4, RotCNN6

main()
