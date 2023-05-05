# import cv2
# import numpy as np
# from mmpose.pose_estimation import load_model
#


# path_to_model_file = "../work_dirs/pprofile_test/best_PCK_epoch_30.pth"
# path_to_image = "test_files/si_3/7.png"
#
# # Загрузка модели
# model = load_model(path_to_model_file)
#
# # Загрузка изображения
# image = cv2.imread(path_to_image)
#
# # Детектирование объектов на изображении
# result = model.detect(image)
#
# # Вывод результатов
# for obj in result:
#     print(obj.class_id, obj.confidence, obj.bbox)
#     cv2.rectangle(image, (obj.bbox[0], obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), (0, 255, 0), 2)
#
# cv2.imwrite('test_files/di_4/7.png', image)
# # cv2.imshow('image', image)
# # cv2.waitKey()


# +
import mmcv
from mmpose.apis import init_pose_model, inference_top_down_pose_model, inference_bottom_up_pose_model, vis_pose_result
# from mmpose.visualization import vis_pose_result


# Путь к конфигурационному файлу модели
config_file = '../config_files/pprofile_7_DeepFashionDataset.py'

# Путь к файлу весов модели
checkpoint_file = '../work_dirs/pprofile_3/best_PCK_epoch_120.pth'

# Инициализация модели
model = init_pose_model(config_file, checkpoint_file, device='cuda:0')

# 2. Загрузить изображение и подготовить его для детекции:

# Путь к изображению
img_path = '../data/pprofile_3/images/12.png'

# Загрузка изображения
img = mmcv.imread(img_path)

# Подготовка изображения для детекции
# result = inference_pose_model(model, img)
# result = inference_top_down_pose_model(model, img)
result = inference_bottom_up_pose_model(model, img)
print("result:", result)
print("result[0]:", result[0])
print("result[0][0]:", result[0][0])

# 3. Получить результаты детекции:

# Координаты ключевых точек на изображении
keypoints = result[0][0]['keypoints']

# Вероятности детекции каждой ключевой точки
# scores = result['pred_scores']

# Координаты ограничивающего прямоугольника позы на изображении
bbox = result[0][0]['bbox']

# 4. Визуализировать результаты детекции на изображении:
# Визуализация результатов детекции на изображении
vis_img = vis_pose_result(model, img, result)

# Сохранение визуализации в файл
mmcv.imwrite(vis_img, '../vis_results/pprofile_3/hand_detect/12.png')

# +
# import mmcv
# from mmpose.apis import init_pose_model, inference_top_down_pose_model, inference_bottom_up_pose_model

# ba(model, img_or_path, dataset='BottomUpCocoDataset', dataset_info=None, pose_nms_thr=0.9, return_heatmap=False, outputs=None)
# td(model, imgs_or_paths, person_results=None, bbox_thr=None, format='xywh', dataset='TopDownCocoDataset', dataset_info=None, return_heatmap=False, outputs=None)

# -


