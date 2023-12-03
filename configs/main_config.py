import sys
from configs.retinaface import cfg_re50

################ PATH PARAMETERS ################

path_to_masks_model = 'all_models/masks'
path_to_distance_model = 'all_models/distance'
sys.path.append(path_to_masks_model)
sys.path.append(path_to_distance_model)

################ VIDEO PARAMETERS ################

# videopath = 'data/video/metro/test_video.mp4'
# videopath = 'data/video/no_masks/vid2_nomask.mp4'
videopath = 'data/video/metro/1.avi'

################ MODELS PARAMETERS ################
resnet_configuration = cfg_re50
det_weights_path = 'weights/resnet50.pth'
cla_mask_path = 'weights/mask_classifier.pt'

yolo_labels_path='configs/coco.names'
yolo_weights_path='weights/yolov3.weights'
yolo_config_path='configs/yolov3.cfg'

###################################################