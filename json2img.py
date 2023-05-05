import os
import io
import json
import base64
from PIL import Image
import argparse
import tqdm

# convert bytes to PIL
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil

# get PIL image from json
def get_img(name_img):
    with open(name_img, 'r') as f:
        json_obj = json.load(f)
    b64 = json_obj['imageData']
    img_data = base64.b64decode(b64)
    img_pil = img_data_to_pil(img_data)
    return img_pil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='+',
                        help='Filenames or dir of input images')
    args = parser.parse_args()
    return args

# save image.png
def deflate(name_img):
    img = get_img(name_img)
    name = name_img.split('.json')[0]
    try:
        img.save(f'{name}.png')
    except:
        print("json is broken")
        realpath = os.path.realpath(name_img)
        dir_path = os.path.dirname(realpath)
        dir_broken = dir_path + '/broken/'
        name_img = name_img.split('/')[-1]
        if not os.path.exists(dir_broken):
            os.mkdir(dir_broken)
        os.replace(realpath, f'{dir_broken + name_img}')

def main(args):
    paths = args.input
    for f_path in paths:
        if not os.path.exists(f_path):
            continue
        if os.path.isfile(f_path):
            deflate(f_path)
        elif os.path.isdir(f_path):
            listdir = os.listdir(f_path)
            progress_bar = tqdm.trange(len(listdir))
            for file, i in zip(listdir, progress_bar):
                if file.endswith('.json'):
                    deflate(f"{f_path}/{file}")
                else:
                    continue

if __name__ == "__main__":
    args = get_args()
    main(args)




