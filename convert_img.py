import io
from PIL import Image
import cv2
import json
import base64
import numpy as np



def img_to_file(img_arr):
    """
    Convert OpenCV2 image to binary form
        Parameters:
            img_arr(numpy.ndarray) - OpenCV2 image
        Returns:
            img_bin(bytes) - Binary representation of image in PNG format
    """
    # img_arr = img_arr[:, :, ::-1]  # Convert BGR to RGB
    img_pil = Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    with open('1.txt', 'w') as r:
        r.write(str(img_bin))
    return img_bin

def img_arr_to_b64(img_arr):
    """
    Convert image to base64 string format
        Parameters:
            img_arr(numpy.ndarray) - OpenCV2 image
        Returns:
            img_b64(bytes) - Base64 representation of image
    """
    img_bin = img_to_file(img_arr)
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    with open('1-1.txt', 'w') as r:
        r.write(str(img_bin))
    return img_b64


def to_labelme(content, classes, frame_size, image_path=None):
    """
    Convert inference data to labelme format
        Parameters:
            content(tuple(numpy.ndarray, numpy.ndarray)) - Frameswith corresponding bounding boxes
            classes(list(str)) - List of neural network class names
            frame_size(tuple(int)) - Frame width and height
            image_path(str) - Relative image path
    """
    labelme_dict = {
        'version': '4.5.6',
        'flags': {},
        'imageHeight': frame_size[1],
        'imageWidth': frame_size[0],
        'imageData': img_arr_to_b64(content[0]).decode('utf-8')
    }
    if image_path is not None:
        labelme_dict['imagePath'] = image_path
    else:
        labelme_dict['imagePath'] = ''
    labelme_json = json.dumps(labelme_dict, indent=4)
    return labelme_json

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def main():
    file = cv2.imread('test_data/1.png')
    # cv2.imshow('image', file)
    # cv2.waitKey(0)
    img = img_arr_to_b64(file)
    img = img_b64_to_arr(img)
    # print(type(img))
    # cv2.imshow(img)


if __name__ == "__main__":
    main()