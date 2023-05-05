# deploy:
# python mmdeploy/tools/deploy.py \
#     mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#     mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#     checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     mmdetection/demo/demo.jpg \
#     --work-dir mmdeploy_model/faster-rcnn \
#     --device cuda \
#     --dump-info
#     
#     
# Export model to backends.
#
# positional arguments:
#   deploy_cfg            deploy config path
#   model_cfg             model config path
#   checkpoint            model checkpoint path
#   img                   image used to convert model model
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --test-img TEST_IMG [TEST_IMG ...]
#                         image used to test model
#   --work-dir WORK_DIR   the dir to save logs and models
#   --calib-dataset-cfg CALIB_DATASET_CFG
#                         dataset config path used to calibrate in int8 mode. If not specified, it will use "val" dataset in model config instead.
#   --device DEVICE       device used for conversion
#   --log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
#                         set log level
#   --show                Show detection outputs
#   --dump-info           Output information for SDK
#   --quant-image-dir QUANT_IMAGE_DIR
#                         Image directory for quantize model.
#   --quant               Quantize model to low bit.
#   --uri URI             Remote ipv4:port or ipv6:port for inference on edge device.

# python mmdeploy/tools/deploy.py \
#     mmdeploy/configs/mmpose/pprofile_rknn.py \
#     pprofile_data/pose/pprofile_max.py \
#     pprofile_data/pose/best_PCK_epoch_340.pth \
#     pprofile_data/pose/test/8.png \
#     --work-dir pprofile_data/deploy_results \
#     --device cuda \
#     --dump-info




