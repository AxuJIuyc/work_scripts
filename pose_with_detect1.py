# +
# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
from datetime import datetime
from calculations import Calculations
import tqdm
import cv2
import numpy as np
# import yolo_detect as yd

from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.models.heads.topdown_heatmap_base_head import decode

# -

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    if args.img == "all":
        image_names = os.listdir(args.img_root)
    else:
        image_names = [args.img]

    start_time = datetime.now()
    num = 0
    progress_bar = tqdm.trange(len(image_names))
    for name, _ in zip(image_names, progress_bar):
        if name.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'bmp']:
            continue
        num += 1
        image_name = os.path.join(args.img_root, name)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_name)
        
        # yolo detect
#         mmdet_results = run_yolo_detect(image_name)
#         print('===> mmdet_results:', mmdet_results)
        # separate results
#         mmdet_results = separate(mmdet_results)
        
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = True

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        print("===========")
#         print("pose_res")
#         print(pose_results)
        print("outputs")
        print(returned_outputs[0]['heatmap'].shape)
        
        #############################
#         img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        img = cv2.imread(image_name)
        
        img_height = img.shape[0]
        img_width = img.shape[1]
        aspect_ratio = img_width / img_height
        padding = 1.
        pixel_std = 200
        bbox = [1232,1290,927,218,1]
#         bbox = [1696,1399,927,218,1]
        center, scale = bbox_xywh2cs(
            bbox,
            aspect_ratio,
            padding,
            pixel_std)
        print("center, scale, bbox:", center, scale, bbox)
        dh = decode(returned_outputs[0]['heatmap'], center, scale, bbox[-1])
        box = list(map(int, dh['boxes'][0]))
        print(box)
#         check_img = cv2.rectangle(img,box[:2], box[2:4], (255,0,0), 4)
        check_img = cv2.rectangle(img,box[:2], box[2:4], (255,0,0), 4)
        for point in dh['preds'][0]:
            x, y, s = int(point[0]), int(point[1]), int(point[2]*100)
#             print(x, y, s)
            check_img = cv2.drawMarker(check_img, (x, y), color=(0,255,0), markerType=cv2.MARKER_CROSS)
        ############################
        
        print("+++++++++++++")
        
        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{name}')
        
        cv2.imwrite(out_file + '1.jpg', check_img)
       
        
        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)
    
        # calculations file write
        if out_file and pose_results:
            for results in pose_results:
                calc = Calculations(get_kps(results))
                params = calc.forward()
                path2file = out_file + '.csv'
                calc.writer(params, path2file)    
                calc.draw(out_file, params)
        
    end_time = datetime.now()
    delta = end_time - start_time
    print(f"Total Time for {num}:", delta)


def get_kps(pose_results):
    kps = []
    for kp in pose_results['keypoints']:
        kps.append((kp[:2]))
    return kps

def separate(mmdet_results):
#     print(mmdet_results[0])
#     print(np.shape(mmdet_results[1][0][0]))
#     print(mmdet_results[1][0][0]+mmdet_results[1][0][1]+mmdet_results[1][0][2])
    if len(mmdet_results[0][0]) <= 1:
        return mmdet_results
    else:
        dellist = []
        for i in range(1, len(mmdet_results[0][0])):
            IOU = iou(mmdet_results[0][0][i], mmdet_results[0][0][i-1])
#             print('IOU:', IOU)
            if IOU >= 0.3:
                dellist.append(i)
        ar = None
        maxbbox = None
        indx = 0
        for index in dellist[::-1]:
            if not ar:
                ar = mmdet_results[1][0][index]
            else:
                ar += mmdet_results[1][0][index]
            mmdet_results[1][0].pop(index)
            if maxbbox is None:
                maxbbox = mmdet_results[0][0][index]
            else:
                maxbbox = maxbox(maxbbox, mmdet_results[0][0][index])
            mmdet_results[0][0].pop(index)
            indx = index
        if dellist:
            mmdet_results[0][0][indx-1] = maxbox(maxbbox, mmdet_results[0][0][indx-1])
    return mmdet_results

def maxbox(box1, box2):
    box1[0] = min(box1[0], box2[0])
    box1[1] = min(box1[1], box2[1])
    box1[2] = max(box1[2], box2[2])
    box1[3] = max(box1[3], box2[3])
    box1[4] = 1
    return box1

def iou(box1, box2):
    # вычисляем координаты точек пересечения двух прямоугольников
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # вычисляем площади двух прямоугольников
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # вычисляем площадь пересечения двух прямоугольников
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # вычисляем площадь объединения двух прямоугольников
    union_area = area_box1 + area_box2 - intersection_area

    # вычисляем коэффициент пересечения/объединения (IOU)
    iou = intersection_area / union_area

    return iou

# def run_yolo_detect(image_name):    
# #     image = cv2.imread(image_name)
#     w = '../yolo_pprofile640.pt'
# #     source = '../data/pprofile_max/test'
#     s = image_name
#     ct = 0.6
#     lt = 2
    
#     bboxes = yd.run(weights=w, source=s, 
#                     conf_thres=ct, line_thickness=lt)
#     return bboxes

if __name__ == '__main__':
    main()

"""
python3 pose_with_detect.py openmmlab/pprofile_data/detect/stock_pprofile_max.py \
                            openmmlab/pprofile_data/detect/stock_epoch_30.pth \
                            openmmlab/pprofile_data/pose/pprofile_max.py \
                            openmmlab/pprofile_data/pose/best_PCK_epoch_340.pth \
                            --img-root openmmlab/pprofile_data/pose \
                            --img 8.png \
                            --out-img-root openmmlab/pprofile_data/results
                            
                            
python3 ../../../mmpose/custom_scripts/pose_with_detect.py \
                            ../pprofile_data/detect/stock_pprofile_max.py \
                            ../pprofile_data/detect/stock_epoch_30.pth \
                            ../pprofile_data/pose/pprofile_max.py \
                            ../pprofile_data/pose/best_PCK_epoch_340.pth \
                            --img-root ../pprofile_data/pose \
                            --img 8.png \
                            --out-img-root ../pprofile_data/results
                            
"""