import os
import json
import cv2
from matplotlib import pyplot as plt
import numpy as np


dest_folder = "test_files"
imgs_folder = "si_3"
output_folder = "di_4"
# js_file = "val_5.json"
js_file = "../metrics/result_keypoints.json"


# +
def draw_bbox(image, bbox):
    """ Наносит на изображение только один bbox"""
    center = bbox[:2]
    scale = bbox[2:]
#     1. Вычислить координаты левого верхнего угла bbox по центру и масштабу:
    x1 = center[0] - scale[0] * 0.5
    y1 = center[1] - scale[1] * 0.5

# 2. Вычислить координаты правого нижнего угла bbox по центру и масштабу:
    x2 = center[0] + scale[0] * 0.5
    y2 = center[1] + scale[1] * 0.5

# 3. Создать прямоугольник с координатами левого верхнего и правого нижнего углов:
    bbox = [x1, y1, x2, y2]

# 4. Нарисовать прямоугольник на изображении с помощью функции cv2.rectangle():
    return cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


# -

def draw_kpt(image, kpt):
    """ Наносит на изображение только один набор точек"""
    drawed_img = image
    for dot in kpt:
            drawed_img = cv2.drawMarker(drawed_img, tuple(map(int, kpt[:2])), color=(0,255,0), 
                                        markerType=cv2.MARKER_CROSS, thickness=2)
    return drawed_img



# +
def kpts_draw(dest_folder, imgs_folder, output_folder, js_file):
    with open(f"{js_file}", 'r') as f:
        jsf = json.load(f)
    
    imgs = os.listdir(f"{dest_folder}/{imgs_folder}")
    print(imgs)
    # imgs[1], imgs[2] = imgs[2], imgs[1] 
    imgs.sort()    
    for i in range(len(imgs)):
        if not imgs[i].endswith('.png'):
            imgs.pop(i)
            break
    print(imgs)
    
    
    plt.figure(figsize=(20, 20))
    drawed_imgs = []
    for img, anno, i in zip(imgs, jsf, range(len(imgs))):
        print(img)
        
        # get json anno info
        kpts = anno["keypoints"]
        bbox_center = anno["center"]
        bbox_scale = anno["center"]
        
        # draw
        drawed_img = cv2.imread(f"{dest_folder}/{imgs_folder}/{img}")
        drawed_img = draw_bbox(drawed_img, (bbox_center + bbox_scale))
        for kpt in kpts:
            drawed_img = cv2.drawMarker(drawed_img, tuple(map(int, kpt[:2])), color=(0,255,0), 
                                        markerType=cv2.MARKER_CROSS, thickness=2)
#         cv2.imwrite(f"{dest_folder}/{output_folder}/{img}", drawed_img)
    
#         j = i // 4 + 1
        plt.subplot(4, 4, i+1)
        plt.imshow(drawed_img)
        drawed_imgs.append(drawed_img)
    plt.show()
    return drawed_imgs
# -

dest_folder = "test_files"
imgs_folder = "si_3"
output_folder = "di_4"
# js_file = "val_5.json"
js_file = "../metrics/result_keypoints.json"
# f = cv2.imread("test_files/si_3/7.png")
drawed_imgs = kpts_draw(dest_folder, imgs_folder, output_folder, js_file)


# +
def kpts_draw_2(path2imgs, js_file, output_folder=None):
    with open(f"{js_file}", 'r') as f:
        jsf = json.load(f)
    
    file_type = "val.json"
    with open(f"{path2imgs}/{file_type}") as f:
        annotations = json.load(f)
    
    all_imgs = os.listdir(f"{path2imgs}/images")
    for name_img, i in zip(all_imgs, range(len(all_imgs))):
        for name_id in annotations["images"]:
            if name_id["file_name"] == name_img:
                all_imgs[i] = [name_img, name_id["id"]]
    
    imgs = []
    for i in range(len(all_imgs)):
        if len(all_imgs[i]) == 2:
            imgs.append(all_imgs[i])
    
    imgs.sort(key=lambda x: x[1])
    print(f"Число изображений для {file_type}:", len(imgs), '\n', imgs)
    
    
    plt.figure(figsize=(20, 20))
    drawed_imgs = []
    for img, anno, i in zip(imgs, jsf, range(len(imgs))):
#         print(img)
        
        # get json anno info
        kpts = anno["keypoints"]
        bbox_center = anno["center"]
        bbox_scale = anno["center"]
        
        # draw
        drawed_img = cv2.imread(f"{path2imgs}/images/{img[0]}")
        drawed_img = draw_bbox(drawed_img, (bbox_center + bbox_scale))
        for kpt in kpts:
            drawed_img = cv2.drawMarker(drawed_img, tuple(map(int, kpt[:2])), color=(0,255,0), 
                                        markerType=cv2.MARKER_CROSS, thickness=2)
#         cv2.imwrite(f"{dest_folder}/{output_folder}/{img}", drawed_img)
    
#         j = i // 4 + 1
        plt.subplot(4, 4, i+1)
        plt.imshow(drawed_img)
        drawed_imgs.append(drawed_img)
    plt.show()
    return drawed_imgs

# +
path2imgs = "../data/pprofile_test_2"
# kps_file = "../work_dirs/pprofile_test_3/result_keypoints.json"
kps_file = "../metrics/result_keypoints.json"
# path2imgs = "../data/new_human_test"
# kps_file = "../work_dirs/new_human_test/result_keypoints.json"

drawed_imgs = kpts_draw_2(path2imgs, kps_file)

# +

# drawed_img = cv2.drawMarker(drawed_imgs[1], (154, 218), color=(255,0,0), 
#                                 markerType=cv2.MARKER_CROSS, thickness=2)

plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
plt.imshow(drawed_imgs[7])
plt.subplot(3, 3, 2)
plt.imshow(drawed_imgs[8])
plt.subplot(3, 3, 3)
plt.imshow(drawed_imgs[9])
plt.subplot(3, 3, 4)
plt.imshow(drawed_imgs[10])
plt.subplot(3, 3, 5)
plt.imshow(drawed_imgs[11])
plt.subplot(3, 3, 6)
plt.imshow(drawed_imgs[12])


# with open(kps_file, 'r') as f:
#     f = json.load(f)
# print(len(f))

# drawed_imgs[0]

# +
images = "../data/pprofile_3"
# annotations = "../metrics/result_keypoints.json"
annotations = "../work_dirs/pprofile_3/result_keypoints.json"

""" val
[['12.png', 1], ['23.png', 2], ['26.png', 3], ['29.png', 4], ['30.png', 5], ['33.png', 6], 
['36.png', 7], ['38.png', 8], ['41.png', 9], ['5.png', 10], ['6.png', 11], ['7.png', 12], 
['id_0.png', 13], ['id_1.png', 14], ['id_2.png', 15], ['id_3.png', 16], ['id_4.png', 17], 
['id_5.png', 18], ['id_6.png', 19], ['id_7.png', 20], ['id_8.png', 21], ['id_9.png', 22], 
['id_10.png', 23], ['id_11.png', 24], ['id_12.png', 25], ['id_13.png', 26], ['id_14.png', 27], 
['id_15.png', 28], ['id_16.png', 29], ['id_17.png', 30], ['id_18.png', 31], ['id_19.png', 32], 
['id_20.png', 33], ['id_21.png', 34], ['id_22.png', 35], ['id_23.png', 36], ['id_24.png', 37], 
['id_25.png', 38], ['id_26.png', 39], ['id_27.png', 40], ['id_28.png', 41], ['id_29.png', 42], 
['id_30.png', 43], ['id_31.png', 44], ['id_32.png', 45], ['id_33.png', 46], ['id_34.png', 47], 
['id_35.png', 48], ['id_36.png', 49], ['id_37.png', 50], ['id_38.png', 51], ['id_39.png', 52], 
['id_40.png', 53], ['id_41.png', 54], ['id_42.png', 55], ['id_43.png', 56], ['id_44.png', 57], 
['id_45.png', 58], ['id_46.png', 59], ['id_47.png', 60]]
"""
""" train
['10.png', '13.png', '14.png', '15.png', '16.png', '19.png', '21.png', '22.png', '24.png', 
'25.png', '27.png', '28.png', '31.png', '34.png', '37.png', '39.png', '40.png', '42.png', 
'43.png', '45.png', '0.png', '1.png', '2.png', '3.png', '4.png', 'mask_3510x2550.png', 
'id_48.png', 'id_49.png', 'id_50.png', 'id_51.png', 'id_52.png', 'id_53.png', 'id_54.png', 
'id_55.png', 'id_56.png', 'id_57.png', 'id_58.png', 'id_59.png', 'id_60.png', 'id_61.png', 
'id_62.png', 'id_63.png', 'id_64.png', 'id_65.png', 'id_66.png', 'id_67.png', 'id_68.png', 
'id_69.png', 'id_70.png', 'id_71.png', 'id_72.png', 'id_73.png', 'id_74.png', 'id_75.png', 
'id_76.png', 'id_77.png', 'id_78.png', 'id_79.png', 'id_80.png', 'id_81.png', 'id_82.png', 
'id_83.png', 'id_84.png', 'id_85.png', 'id_86.png', 'id_87.png', 'id_88.png', 'id_89.png', 
'id_90.png', 'id_91.png', 'id_92.png', 'id_93.png', 'id_94.png', 'id_95.png', 'id_96.png', 
'id_97.png', 'id_98.png', 'id_99.png', 'id_100.png', 'id_101.png', 'id_102.png', 'id_103.png', 
'id_104.png', 'id_105.png', 'id_106.png', 'id_107.png', 'id_108.png', 'id_109.png', 'id_110.png', 
'id_111.png', 'id_112.png', 'id_113.png', 'id_114.png', 'id_115.png', 'id_116.png', 'id_117.png', 
'id_118.png', 'id_119.png', 'id_120.png', 'id_121.png', 'id_122.png', 'id_123.png', 'id_124.png', 
'id_125.png', 'id_126.png', 'id_127.png', 'id_128.png', 'id_129.png', 'id_130.png', 'id_131.png', 
'id_132.png', 'id_133.png', 'id_134.png', 'id_135.png', 'id_136.png', 'id_137.png', 'id_138.png', 
'id_139.png', 'id_140.png', 'id_141.png', 'id_142.png', 'id_143.png', 'id_144.png', 'id_145.png', 
'id_146.png', 'id_147.png', 'id_148.png', 'id_149.png', 'id_150.png', 'id_151.png']
"""

path = f"{images}/images/23.png"

img = cv2.imread(path)
with open(annotations, 'r') as f:
    anno = json.load(f)


img1 = img
for y in range(3, 7):
    for x in anno[y]['keypoints']:
        img1 = draw_kpt(img, x)
        center = anno[y]['center']
        scale = anno[y]['scale']
        img1 = draw_bbox(img1, (center + scale))

# plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.show()
# -

with open(f'{images}/train.json', 'r') as f:
    anno = json.load(f)['images']
ims = []
for im in anno:
    ims.append(im['file_name'])
print(ims)

# +
kps_file = "../metrics/result_keypoints.json"
path2imgs = "../data/pprofile_test_2"
with open(kps_file, 'r') as f:
    f = json.load(f)
with open(f"{path2imgs}/val.json") as jf:
    jf = json.load(jf)
names = []
for name in jf["images"]:
    names.append(name["file_name"])
print(names)

kpts_list = f[3:7]
im = cv2.imread(f"{path2imgs}/images/23.png")
for kpts in kpts_list:
    for kpt in kpts["keypoints"]:
        im = cv2.drawMarker(im, tuple(map(int, kpt[:2])), color=(255,0,255), 
                            markerType=cv2.MARKER_CROSS, thickness=2)

plt.figure(figsize=(20, 20))
plt.imshow(im)

# +
plt.figure(figsize=(20, 20))

x = 2
y = x
plt.subplot(x, y, 1)
plt.imshow(drawed_imgs[0])
plt.subplot(x, y, 2)
plt.imshow(drawed_imgs[9])
plt.subplot(x, y, 3)
plt.imshow(drawed_imgs[10])
plt.show()



# -

odir = f"{dest_folder}/{output_folder}"
cv2.imwrite(f'{odir}/test_0.png', drawed_imgs[0])
cv2.imwrite(f'{odir}/test_9.png', drawed_imgs[9])
cv2.imwrite(f'{odir}/test_10.png', drawed_imgs[10])


def draw_my_kpts(img, kpts):
    drawed_img = img
    x_coords = kpts[::3] # берем каждый 3-й элемент, начиная с 0-го
    y_coords = kpts[1::3] # берем каждый 3-й элемент, начиная с 1-го
    for x,y in zip(x_coords, y_coords):
        drawed_img = cv2.drawMarker(drawed_img, tuple(map(int, [x, y])), color=(0,0,255), 
                                    markerType=cv2.MARKER_CROSS, thickness=2)
    return drawed_img


def draw_my_bbox(img, bbox):
    x, y = list(map(int, bbox[:2]))
    w, h = list(map(int, bbox[2:]))
    drawed_img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(0,0,255), thickness=2)
    return drawed_img


def draw_my_anno(path2imgs, annotations, output_dir=None):
    all_imgs = os.listdir(path2imgs)
    for name_img, i in zip(all_imgs, range(len(all_imgs))):
        for name_id in annotations["images"]:
            if name_id["file_name"] == name_img:
                all_imgs[i] = [name_img, name_id["id"]]
    
    imgs = []
    for i in range(len(all_imgs)):
        if len(all_imgs[i]) == 2:
            imgs.append(all_imgs[i])
    
    imgs.sort(key=lambda x: x[1])
    print("Число изображений:", len(imgs), '\n', imgs)
    
    draw_list = []
    annos = annotations["annotations"]
    for img in imgs:
        drawed_img = cv2.imread(f"{path2imgs}/{img[0]}")
        del_annos = []
        for anno, i in zip(annos, range(len(annos))):
            if anno["image_id"] == img[1]:
                drawed_img = draw_my_bbox(drawed_img, anno["bbox"])
                drawed_img = draw_my_kpts(drawed_img, anno["keypoints"])
                del_annos.append(i)
        draw_list.append([drawed_img, img[0]])
        for i in del_annos[::-1]:
            annos.pop(i)
    print("Остались неразмеченными:", annos)
    
    size_dl = len(draw_list)
    plt.figure(figsize=(30, 30))
    for img, i in zip(draw_list, range(size_dl)):
        plt.subplot(15, 15, i+1)
        plt.imshow(img[0])
    plt.show()
    return draw_list


# +
path2imgs = "../data/building_angles_1/images"
path2annotations = "../data/building_angles_1/val.json"

with open(path2annotations, 'r') as f:
    annotations = json.load(f)
draw_list = draw_my_anno(path2imgs, annotations)
print("Complete")
# -

plt.figure(figsize=(20, 20))
plt.imshow(draw_list[8][0])

plt.figure(figsize=(24, 24))
plt.imshow(draw_list[74][0])


def draw_kps(image_path):
    # reading image using the imread() function
    imageread = cv2.imread(image_path)
    # input image is converted to gray scale image
    imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

    # using the SIRF algorithm to detect key
    # points in the image
    features = cv2.SIFT_create()
    print(features)

    keypoints = features.detect(imagegray, None)

    # drawKeypoints function is used to draw keypoints
    output_image = cv2.drawKeypoints(imagegray, keypoints, 0, (0, 0, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # displaying the image with keypoints as the
    # output on the screen
    plt.figure(figsize=(30,30))
    plt.imshow(output_image)

    # plotting image
    plt.show()
    return keypoints


# # !ls ../../
imp = "../../mmpose/data/pprofile_max/test/8.png"
print(os.path.exists(imp))
i1 = draw_kps(imp)

# imp = "../data/pprofile_test/images/mask_3510x2550.png"
imp = "../../mmpose/data/pprofile_max/mask_3510x2550.png"
print(os.path.exists(imp))
i2 = draw_kps(imp)


def draw_segmentation(img, segmentations):
    segmentation = []
    for x, y in zip(segmentations[0::2], segmentations[1::2]):
        segmentation.append((x, y))
    print(segmentation)
    # задание точек сегментации
#     segmentation = [[(10, 20), (30, 40), (50, 60)], [(70, 80), (90, 100), (110, 120)]]

    # создание изображения
    img = cv2.imread(img)

    # отрисовка сегментации на изображении
    pts = np.array(segmentation, np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts],True,(0,0,255),thickness=2)

    # отображение изображения с сегментацией
    plt.figure(figsize=(20,20))
    plt.imshow(img)

    # plotting image
    plt.show()


img = "../../mmpose/data/pprofile_max/val/1.png"
seg = [[1246.46,1003.59,1249.88,1002.83,1267.11,1002.32,1298.53,1001.69,1317.66,1001.56,1333.75,1001.69,1362.0,1000.8,1390.38,1000.67,1415.98,1000.17,1446.0,999.28,1477.42,998.52,1510.36,997.76,1535.45,997.63,1573.08,997.38,1602.6,996.87,1637.18,996.62,1674.69,996.49,1703.95,996.62,1749.69,996.75,1784.28,996.87,1816.71,997.0,1858.01,996.49,1905.4,996.75,1951.39,997.38,1991.17,997.76,2021.45,998.39,2062.5,999.41,2079.47,999.79,2095.18,1000.42,2104.05,1002.32,2106.59,1007.14,2107.47,1015.37,2108.49,1029.69,2111.15,1030.32,2124.96,1030.7,2151.56,1031.21,2151.94,1014.36,2153.84,1012.84,2159.04,1012.71,2160.68,1012.96,2162.58,1015.75,2162.96,1040.08,2163.22,1059.97,2162.58,1104.18,2160.68,1134.21,2159.04,1163.1,2153.72,1196.42,2144.09,1190.97,2144.97,1176.52,2128.25,1176.14,2120.78,1174.62,2113.68,1170.82,2111.27,1167.15,2108.23,1168.42,2106.71,1170.82,2103.54,1171.71,2102.78,1180.2,2100.63,1185.01,2095.82,1191.09,2087.71,1197.94,2083.4,1200.09,2077.95,1201.1,2064.02,1202.37,2027.78,1202.24,1998.77,1202.24,1967.35,1201.74,1936.56,1201.1,1908.69,1199.08,1896.27,1196.67,1888.29,1195.28,1831.28,1195.28,1827.35,1196.54,1805.94,1196.42,1782.25,1195.4,1766.92,1193.5,1759.57,1196.16,1751.08,1198.06,1734.49,1199.2,1716.24,1199.46,1687.23,1200.22,1671.39,1200.72,1646.81,1200.85,1623.37,1200.98,1595.63,1200.72,1564.34,1198.44,1554.07,1197.81,1496.17,1198.44,1437.26,1199.84,1427.51,1201.23,1416.23,1202.62,1388.61,1203.51,1349.46,1204.14,1318.68,1205.92,1294.1,1206.3,1253.81,1206.55,1237.97,1204.78,1236.7,1202.5,1235.94,1186.03,1234.93,1156.25,1234.3,1120.4,1234.93,1084.04,1235.94,1040.2,1236.96,1017.27,1239.87,1011.57,1242.41,1005.87]]
print(os.path.exists(img))
draw_segmentation(img, seg[0])


