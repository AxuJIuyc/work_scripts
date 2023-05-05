from PIL import Image
import os

imgs_folder = "humans_mmpose"
stock = "stock"
new_folder = "192x256"

imgs = os.listdir(f"{imgs_folder}/{stock}")
i = 0
for img in imgs:
    print(img)
    i += 1
    image = Image.open(f"{imgs_folder}/{stock}/{img}")
    new_image = image.resize((192, 256))
    new_image.save(f"{imgs_folder}/{new_folder}/{i}.png")