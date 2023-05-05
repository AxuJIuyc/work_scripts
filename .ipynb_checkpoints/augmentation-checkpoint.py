import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np


def one_draw_kpts(img, kpts):
    drawed_img = cv2.drawMarker(img, tuple(map(int, kpts)), color=(0,255,0), 
                                markerType=cv2.MARKER_CROSS, thickness=2)
    return drawed_img



# +
dest_folder = "test_files"
imgs_folder = "si_3"
output_folder = "di_3"
js_file = "val_4.json"

with open(f"{dest_folder}/{js_file}", 'r') as f:
    jsf = json.load(f)
    annos = jsf["annotations"]
    print(annos)

img_bgr = cv2.imread(f'{dest_folder}/{imgs_folder}/7.png')
print(img_bgr.shape)


# -

def draw_fig(img_bgr, kpts):
    # create figure
    fig = plt.figure(figsize=(20, 20))

    # Adds a subplot at the 1st position
    fig.add_subplot(2, 2, 1)
    plt.imshow(img_bgr)

    for kpt in kpts:
        one_draw_kpts(img_bgr, kpt[:2])

    # Adds a subplot at the 2st position
    fig.add_subplot(2, 2, 2)
    plt.imshow(img_bgr)
    
    return img_bgr


# +
def hcv(img):
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print(image.shape)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(image[:, :, i], cmap='gray')
    plt.show()

contrast(img_bgr)


# +
def yuv(img):
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    print(image.shape)
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(image[:, :, i], cmap='gray')
    plt.show()
    
yuv(img_bgr)


# +
def conv_definition(img):
    # увеличение чёткости
    kernel = np.array([
        [-0.1, -0.1, -0.1],
        [-0.1,    2, -0.1],
        [-0.1, -0.1, -0.1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out)
    plt.show()
    
conv_definition(img_bgr)


# +
def conv_blackout(img):
    # затемнение
    kernel = np.array([
        [-0.1,  0.1, -0.1],
        [ 0.1,  0.5,  0.1],
        [-0.1,  0.1, -0.1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out)
    plt.show()
    
conv_blackout(img_bgr)


# +
def conv_light(img):
    # увеличение яркости
    kernel = np.array([
        [-0.1,  0.2, -0.1],
        [ 0.2,    1,  0.2],
        [-0.1,  0.2, -0.1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out)
    plt.show()

conv_light(img_bgr)


# +
def roberts(img):
    # фильтр Робертса
    kernel = np.array([
        [1,  0],
        [0, -1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    return img_out
    
plt.imshow(roberts(img_bgr))
plt.show()
    

# +
def previtt(img):
    # фильтр Превитт
    kernel = np.array([
        [-1,  0, 1],
        [-1,  0, 1],
        [-1,  0, 1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out, cmap='gray')
    plt.show()
    
previtt(img_bgr)


# +
def sobel(img):
    # фильтр Собеля
    kernel = np.array([
        [-1,  0, 1],
        [-2,  0, 2],
        [-1,  0, 1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out, cmap='gray')
    plt.show()
    
sobel(img_bgr)


# +
def sobelv(img):
    # фильтр Собеля вертикальный
    kernel = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ])
    img_out = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_out, cmap='gray')
    plt.show()
    
sobelv(img_bgr)


# +
def togray(img):
    # читать в цветовой модели Grayscale
    img_gray = img
    print(img_gray.shape)
    plt.imshow(img_gray, cmap='gray')
    plt.show()

img_gray = togray(img_bgr)


# +
def binaryzation(img):
    # Бинаризация
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
#     plt.imshow(thresh, cmap='gray')
#     plt.show()
    return thresh

# binaryzation(img_bgr)


# +
def contour_search(img):
    # Поиск контуров
    thresh = binaryzation(img)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours, hierarchy)
#     print(len(contours), contours[2])
    return contours
    
# contour_search(img_bgr)


# +
def contour_draw(img):
    # Рисование контуров
    contours = contour_search(img)
    image_out = np.zeros_like(img)
    image_out = cv2.drawContours(image_out, contours, -1, (0, 255, 0), 2)
    plt.imshow(image_out)
    plt.show()

contour_draw(img_bgr)


# -
def channel_shift(img):
    # Сдвиг по каналам
    rgb_shift = np.random.randint(-128, 128, size=3)
    result = img.astype(np.int32) + rgb_shift
    result = np.clip(result, 0, 255).astype(np.uint8)  # Обрезаем значения до диапазона 0 - 255

    return result



plt.imshow(channel_shift(img_bgr))
plt.show()



def new_annotation(img: str, annos: dict, count: int):
    # in images
    new_anno = None
    
    for i in range(len(annos["images"])):
        if annos["images"][i]["file_name"] == img:
            new_anno = annos["images"][i].copy()
            img_id = annos["images"][i]["id"]
            break
            
    assert new_anno is not None, f"Не нашлось совпадений для {img}"
    
    new_id = annos["images"][-1]["id"] + 1
#     new_name = f"id_{new_id}.png"
    new_name = f"id_{count}.png"
    new_anno["id"] = new_id
    new_anno["file_name"] = new_name
    annos["images"].append(new_anno)
        
    # in annotations
    new_annos = []
    new_anno = None
    j = 1
    for i in range(len(annos["annotations"])):
        if annos["annotations"][i]["image_id"] == img_id:
            new_anno = annos["annotations"][i].copy()
            new_anno["id"] = annos["annotations"][-1]["id"] + j
            new_anno["image_id"] = new_id
            new_annos.append(new_anno)
            j += 1
            
    annos["annotations"] += new_annos
    return annos 


# +
def augmentation(path2imgs, annos, out_dir, jname):
    imgs = os.listdir(path2imgs)

    img_formats = ['jpg', 'png', 'jpeg', 'bmp']
    del_list = []
    for i in range(len(imgs)):
        if not imgs[i].lower().split('.')[-1] in img_formats:
            del_list.append(i)
    for x in del_list[::-1]:
        imgs.pop(x)
    imgs.sort()

    assert len(annos["images"]) == len(imgs), \
            "Несовпадение количества изображений в папке и аннотациях"
    
    count = len(os.listdir(f"{out_dir}/images"))
    aug_foos = [roberts, channel_shift]
    for img in imgs:
        p2im = f"{path2imgs}/{img}"
        fig = cv2.imread(p2im)
        cv2.imwrite(f"{out_dir}/images/{img}", fig)
        for foo in aug_foos:
            if foo is channel_shift:
                for i in range(3):
                    new_img = foo(fig)
                    annos = new_annotation(img, annos, count)
                    fname = annos["images"][-1]["file_name"]
                    cv2.imwrite(f"{out_dir}/images/{fname}", new_img)
                    count += 1
            else:
                new_img = foo(fig)
                annos = new_annotation(img, annos, count)
                fname = annos["images"][-1]["file_name"]
                cv2.imwrite(f"{out_dir}/images/{fname}", new_img)
                count += 1
                
    with open(f"{out_dir}/{jname}", 'w') as f:
        json.dump(annos, f)
            
    
# -


imgs_folder = "si_3_train"
p2i = f"{dest_folder}/{imgs_folder}"
path2json = "../data/pprofile_test/train.json"
out_dir = "../data/new_pprofile_test"
with open(path2json, 'r') as f:
    annos = json.load(f)
augmentation(p2i, annos, out_dir, 'train.json')

a = "aBrA.gTp"
a.lower().split('.')[-1]


def augmentation_2(path2imgs, path2annos_list, out_dir, aug_foos: list):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    count = 0
#     count = len(os.listdir(out_dir))
    for path2annos in path2annos_list:
        with open(path2annos, 'r') as f:
            annos = json.load(f)
        jname = path2annos.split('/')[-1]

        all_imgs = os.listdir(path2imgs)

        for name_img, i in zip(all_imgs, range(len(all_imgs))):
            for name_id in annos["images"]:
                if name_id["file_name"] == name_img:
                    all_imgs[i] = [name_img, name_id["id"]]

        imgs = []
        for i in range(len(all_imgs)):
            if len(all_imgs[i]) == 2:
                imgs.append(all_imgs[i])

        imgs.sort(key=lambda x: x[1])
        print(f"Число изображений для {jname}:", len(imgs), '\n', imgs)

        for img in imgs:
            p2im = f"{path2imgs}/{img[0]}"
            fig = cv2.imread(p2im)
            cv2.imwrite(f"{out_dir}/images/{img[0]}", fig)
            for foo in aug_foos:
                if foo is channel_shift:
                    for i in range(3):
                        new_img = foo(fig)
                        annos = new_annotation(img[0], annos, count)
                        fname = annos["images"][-1]["file_name"]
                        cv2.imwrite(f"{out_dir}/images/{fname}", new_img)
                        count += 1
                else:
                    new_img = foo(fig)
                    annos = new_annotation(img[0], annos, count)
                    fname = annos["images"][-1]["file_name"]
                    cv2.imwrite(f"{out_dir}/images/{fname}", new_img)
                    count += 1

        with open(f"{out_dir}/{jname}", 'w') as f:
            json.dump(annos, f)


# +
aug_foos = [roberts, channel_shift]
path2workdir = "../data/pprofile_test"
path2imgs = f"{path2workdir}/images"
path2annos1 = f"{path2workdir}/val.json"
path2annos2 = f"{path2workdir}/train.json"
out_dir = "../data/pprofile_test_2"

augmentation_2(path2imgs, [path2annos1, path2annos2], out_dir, aug_foos)

# -


