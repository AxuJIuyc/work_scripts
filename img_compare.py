import os
from PIL import Image
import imagehash
import tqdm
import json
import argparse


# def cleaning(dir_path):
#     listdir = os.listdir(dir_path)
#     progress_bar = tqdm.trange(len(listdir))
#     for file_1, i in zip(listdir, progress_bar):
#         file_1 = f'{dir_path}/{file_1}'
#         if file_1.endswith('.png'):
#             json_1 = file_1[:-3] + 'json' # .png -> .json
#             for file_2 in listdir[i:]:
#                 file_2 = f'{dir_path}/{file_2}'
#                 if file_1 != file_2 and file_2.endswith('.png'):
#                     json_2 = file_2[:-3] + 'json' # .png -> .json
#                     if compare(file_1, file_2):
#                         os.remove(file_2)
#                         os.remove(json_2)
#                     else:
#                         continue
#         else:
#             continue

# def compare(file_1, file_2):
#     try:
#         hash_1 = imagehash.average_hash(Image.open(file_1), 32)
#         hash_2 = imagehash.average_hash(Image.open(file_2), 32)
#         compare = hash_1 - hash_2
#         if compare < 250:
#             return True
#         else:
#             return False
#     except:
#         return False

# cleaning a directory from duplicates
def cleaning(dir_path, h_size=32, difference=250):
    listdir = os.listdir(dir_path)
    progress_bar = tqdm.trange(len(listdir))
    for file_1, i in zip(listdir, progress_bar):
        file_1 = f'{dir_path}/{file_1}'
        if os.path.exists(file_1) and file_1.endswith('.png'):
            json_1 = file_1[:-3] + 'json' # .png -> .json
            for file_2 in listdir[i:]:
                file_2 = f'{dir_path}/{file_2}'
                if (file_1 != file_2 
                    and file_2.endswith('.png') 
                    and os.path.exists(file_2)
                ):
                    json_2 = file_2[:-3] + 'json' # .png -> .json
                    if compare(file_1, file_2, json_1, json_2, h_size, difference):
                        os.remove(file_2)
                        os.remove(json_2)
                    else:
                        continue
        else:
            continue

# comparison of two images
def compare(img1, img2, json1, json2, hash_size=32, difference=250):
    bbox1 = search_box(json1)
    bbox2 = search_box(json2)
    crop1 = make_crop(img1, bbox1)
    crop2 = make_crop(img2, bbox2)
    hash_1 = imagehash.average_hash(crop1, hash_size)
    hash_2 = imagehash.average_hash(crop2, hash_size)
    compare = hash_1 - hash_2
    if compare < difference:
        return True
    else:
        return False

# search bbox of box
def search_box(json_path):
    with open(json_path, 'r') as f:
        json_obj = json.load(f)
    json_shapes = json_obj['shapes']
    for shape in json_shapes:
        if shape['label'] == 'box':
            return shape['points']

# make image crop around box
def make_crop(img_file, bbox):
    img = Image.open(img_file)
    crop = img.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
    return crop

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='Folder of images and jsons')
    parser.add_argument('--hash', type=int, default=32,
                        help='Hash size for  images compare')
    parser.add_argument('--hdif', '-hd', type=int, default=250,
                        help='Minimal hash difference between images')
    args = parser.parse_args()
    return args


def main(args):
    path = args.input
    h_size = args.hash
    h_dif = args.hdif
    if not os.path.exists(path):
        print("path does not exist")
    else:
        cleaning(path, h_size, h_dif)

if __name__ == "__main__":
    args = get_args()
    main(args)