# import os
# import json
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from albumentations import (
#     HorizontalFlip, VerticalFlip, RandomRotate90,
#     ShiftScaleRotate, RandomBrightnessContrast,
#     Blur, HueSaturationValue, Normalize,
#     Compose, BboxParams)

# # Загрузите изображения и файлы с аннотациями:
# img_dir = 'path/to/images'
# ann_dir = 'path/to/annotations'
#
# img_paths = sorted(os.listdir(img_dir))
# ann_paths = sorted(os.listdir(ann_dir))

# # Определите функцию для чтения файла с аннотациями:
# def read_annotation(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # Определите функцию для записи файла с аннотациями:
# def write_annotation(data, file_path):
#     with open(file_path, 'w') as f:
#         json.dump(data, f)

# # Определите функцию для применения аугментаций к изображению и соответствующим аннотациям:
# def augment_image(img_path, ann_path, save_dir):
#     # Load image and annotations
#     img = cv2.imread(os.path.join(img_dir, img_path))
#     ann = read_annotation(os.path.join(ann_dir, ann_path))
#
#     # Define augmentation pipeline
#     bbox_params = BboxParams(format='coco', label_fields=['category_ids'])
#     transform = Compose([
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
#         RandomBrightnessContrast(p=0.5),
#         Blur(p=0.5),
#         HueSaturationValue(p=0.5),
#         Normalize(),
#     ], bbox_params=bbox_params)
#
#     # Apply augmentation
#     transformed = transform(image=img, bboxes=ann['annotations'], category_ids=ann['categories'])
#     transformed_img = transformed['image']
#     transformed_ann = {'annotations': transformed['bboxes'], 'categories': transformed['category_ids']}
#
#     # Save augmented image and annotations
#     new_img_path = os.path.join(save_dir, f'aug_{img_path}')
#     cv2.imwrite(new_img_path, transformed_img)
#     new_ann_path = os.path.join(save_dir, f'aug_{ann_path}')
#     write_annotation(transformed_ann, new_ann_path)

# # Примените аугментации к каждому изображению и соответствующим аннотациям в директории и сохраните их:
# save_dir = 'path/to/save/augmented/data'
# os.makedirs(save_dir, exist_ok=True)
#
# for img_path, ann_path in zip(img_paths, ann_paths):
#     augment_image(img_path, ann_path, save_dir)







# import imgaug.augmenters as iaa
# from imgaug.augmentables import Keypoint, KeypointsOnImage
#

#
# # Загрузка аннотаций из файла
# annotations_file = "annotations.txt"
# with open(annotations_file, "r") as f:
#     annotations = f.readlines()
#
# # Создание списка объектов KeypointsOnImage
# keypoints_list = []
# for annotation in annotations:
#     # Разделение строки аннотации на координаты ключевых точек
#     coords = annotation.strip().split(",")
#     x = float(coords[0])
#     y = float(coords[1])
#     # Создание объекта Keypoint и добавление его в список ключевых точек
#     keypoint = Keypoint(x=x, y=y)
#     keypoints_list.append(keypoint)
# # Создание объекта KeypointsOnImage для всех изображений
# keypoints = KeypointsOnImage(keypoints_list, shape=image.shape)
#
# # Применение аугментаций к изображению и аннотациям
# aug = iaa.SomeAugmentation()
# image_aug, keypoints_aug = aug(image=image, keypoints=keypoints)







import json
import albumentations as A
from albumentations import KeypointParams
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform
import cv2
from matplotlib import pyplot as plt


# Определение класса для чтения аннотаций из json файла
class KeypointAnnotation:
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def to_albumentations(self, image_height, image_width):
        keypoints = []
        x_coords = self.keypoints[::3]
        y_coords = self.keypoints[1::3]
        for x, y in zip(x_coords, y_coords):
            keypoints.append((x, y))

        return KeypointsOnImage(keypoints, shape=(image_height, image_width, 3))



# +
# Создание класса для чтения и применения аугментаций к датасету
class KeypointDataset:
    def __init__(self, annotations_path, images_dir):
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.images_dir = images_dir

        # Определение параметров для работы с ключевыми точками
        self.keypoint_params = KeypointParams(format="xy")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
#         print(self.annotations["annotations"])
        self.annotations = self.json_restruct(self.annotations)
        # Чтение изображения
        image_path = os.path.join(self.images_dir, self.annotations["images"][idx]["file_name"])
        image = cv2.imread(image_path)

        # Чтение аннотации и преобразование ее в формат albumentations
#         keypoints = []
        for anno in self.annotations["annotations"][idx]:
#             print(anno)
        
#         print(self.annotations["annotations"][idx])
#         keypoints_annotation = KeypointAnnotation(self.annotations["annotations"][idx]["keypoints"])
#         keypoints = keypoints_annotation.to_albumentations(image.shape[0], image.shape[1])
            keypoints_annotation = KeypointAnnotation(anno["keypoints"])
            keypoints = keypoints_annotation.to_albumentations(image.shape[0], image.shape[1])
#         print("keypoints:\n", keypoints)

        # Создание аугментации для изображения и ключевых точек
        aug = A.Compose([
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.Transpose(),
            A.VerticalFlip(),
#             A.Resize(width=256, height=256),
            A.Normalize(),
        ], keypoint_params=self.keypoint_params)

        # Применение аугментации к изображению и ключевым точкам
        data = aug(image=image, keypoints=keypoints)

        # Извлечение аугментированного изображения и ключевых точек
        image_aug = data["image"]
        keypoints_aug = data["keypoints"]

        # Преобразование координат ключевых точек обратно в формат json
        keypoints_aug_json = []
        for keypoint in keypoints_aug:
            x, y = keypoint[0], keypoint[1]
            keypoints_aug_json.append({"x": int(x), "y": int(y)})

        # Возвращение аугментированного изображения и ключевых точек
        return image_aug, keypoints_aug_json
    
    def json_restruct(self, json_file):
        keypoints_list = []
        for x in json_file["annotations"]:
            if len(keypoints_list) == x["image_id"]:
                keypoints_list[x["image_id"]-1] += x
            else:
                keypoints_list.append([x])            
        json_file["annotations"] = keypoints_list
        return json_file


# +
path = "../data/pprofile_test_2"
annos_path = f"{path}/val.json"
images_dir = f"{path}/images"

KD = KeypointDataset(annos_path, images_dir)
# -

KD[0]

# +

N = 20
plt.figure(figsize=(N, N))
for i in range(N):
    base = KD[i]
    drawed_img = base[0]
    # print(KD[0][0])
#     print(base[1])

    for kp in base[1]:
        drawed_img = cv2.drawMarker(drawed_img, (kp['x'], kp['y']), color=(0,0,255), 
                                    markerType=cv2.MARKER_CROSS, thickness=2)
    plt.subplot(int(N), int(N), i+1)
    plt.imshow(drawed_img)
plt.show()
# -

a = [[0],[0],[0]]
b = [[1],[2]]
for x in b:
    a.append(x)


a


