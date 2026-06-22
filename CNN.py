import cv2
import os
import time
import argparse
import torch
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

FORMAT = ("jpg", "jpeg", "png", "bmp", "webp")


class neApp(toga.App):

	def startup(self):
		self.select_btn = toga.Button(
			text='Picture', 
			on_press=self.select_image, 
			style=Pack(padding=10, width=200)
        )

		self.result_label = toga.Label(
			text='Result?', 
			style=Pack(padding=20, font_size=14, text_align='center')
        )

		main_box = toga.Box(
			children=[self.select_btn, self.result_label],
			style=Pack(direction=COLUMN, padding=30, alignment=CENTER)
        )

		mode=RotEyes().to(DEVICE)
		save = torch.load('RotEyes.pt', map_location=DEVICE)
		mode.load_state_dict(save)
		mode.eval()
		self.model=mode

		self.main_window = toga.MainWindow(title="RotEyes")
		self.main_window.content = main_box
		self.main_window.show()

	async def select_image(self, widget):
		file_path = await self.main_window.open_file_dialog(
		title='Pic?',
		multiselect=False,
		file_types=FORMAT
        )
		if file_path:
			img_np=cv2.imread(str(file_path))

			img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

			result = await asyncio.to_thread(self.model_use, str(file_path))
			self.result_label.text = f'Модель RotEyes: на {result:.1f} процентов дипфейк'
		else:
			self.result_label.text = "Aborted"

	def model_use(self, path):
		area, eyes, iris = see_eyes(img_np, PREDICTOR_PATH)

		inputs = torch.cat([area, eyes, iris], dim=2)
		inputs = inputs.to(DEVICE)
		with torch.no_grad():
			pred = self.model(inputs)
			preds = (torch.sigmoid(pred) > 0.5).item()
		return preds*100

def main():

	return neApp(
		"ROTEyes",
		"com.e.neural",
		version="1.0.0"
    )

if __name__ == "__main__":
	from model import see_eyes, to_tensor, RotEyes
	app = main()
	app.main_loop()
#main()
