# +
# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
    
from datetime import datetime
from calculations import Calculations
import tqdm
import cv2
# import yolo_detect as yd


# -

def main():
    """Visualize the demo images.
    """
    parser = ArgumentParser()
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


    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''

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
        image = cv2.imread(image_name)
        h, w, c = image.shape
        box = [0, 0, w, h]
        mmdet_results = box
        
        # keep the person class bounding boxes.
#         person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        person_results = mmdet_results
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
        print("+++++++++++++")
        
        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{name}')
     
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
