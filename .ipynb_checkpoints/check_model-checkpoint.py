import os
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
import inspect


pose_config = 'Projects/openmmlab/pprofile_data/pose/pprofile_max.py'
pose_checkpoint = 'Projects/openmmlab/pprofile_data/pose/epoch_340.pth'
device = 'cuda'
model = init_pose_model(pose_config, 
                             pose_checkpoint, 
                             device=device)
print(inspect.getfullargspec(model))
# print(model)