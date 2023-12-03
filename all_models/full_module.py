import torch
import cv2
import warnings
import all_models.distance.utils.utils as utils
import all_models.masks.masks_module as mask_module
from all_models.masks.data.config import cfg_re50
from all_models.distance.models import Darknet
from all_models.distance.sort import *
from all_models.distance.video_sdd_distance import SDD
from all_models.masks.models.retinaface import RetinaFace
import time
from sttstcs import evaluate_statistics, interval_timer
from all_models.masks.masks_classifier.models import get_model, check_face

warnings.filterwarnings("ignore", category=DeprecationWarning)


def vortex(
        sdd_config_path='configs/yolov3.cfg',
        sdd_weights_path='weights/yolov3.weights',
        sdd_additional_config='configs/ssd_model',
        retina_config_path=cfg_re50,
        retina_weights_path='weights/resnet50.pth',
        mask_weight_path='weights/mask_classifier.pt',
        video_path='data/video/metro/1.avi',
        statistic='disable'):
    frames = 0
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture('rtsp://admin:admin1234@31.13.129.186:4554/cam/realmonitor?channel=1&subtype=0')
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --------- SDD init ---------
    img_size = 608
    conf_thres = 0.5
    nms_thres = 0.3
    key_points_path = 'data/k_points.txt'

    model_sdd = Darknet(sdd_config_path, img_size=img_size)
    model_sdd.load_weights(sdd_weights_path)
    model_sdd.to(device)
    model_sdd.eval()

    mot_tracker = Sort(30, 5)
    homography = utils.get_matrix(key_points_path)

    peopleIDs_work = []

    # statistics data for sdd
    violationsIDs = []
    amount_people = 0
    violation_count = 0
    # --------- SDD init END ---------

    # --------- RetinaFace init ---------
    model_retina = RetinaFace(cfg=cfg_re50, phase='test')
    model_retina = mask_module.load_model(model_retina, retina_weights_path, False)
    model_retina.eval()
    model_retina.to(device)
    # --------- RetinaFace init END ---------

    # --------- Mask model init ---------
    model_mask = get_model(device=device)
    model_mask.load_state_dict(torch.load(mask_weight_path))
    model_mask.eval()

    peopleID_mask_work = []
    # statistic data for mask
    amount_classes = {'with_mask': 0,
                      'no_mask': 0}
    # --------- Mask model init END ---------

    # --------- Vortex start ---------
    distance_violation_frame = None
    no_mask_frame = None
    violation_frame_temp = None
    iteration_start_time = time.time()
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frames += 1

        # Skip frames to increase fps
        if frames % 2 == 0 or frames % 3 == 0:
            continue

        data = SDD(
            frame, img_size, model_sdd,
            conf_thres, nms_thres, mot_tracker,
            violationsIDs, peopleIDs_work,
            amount_people, violation_count, homography,
            violation_frame_temp)

        violation_count, amount_people, peopleIDs, peopleIDs_work, boxes_ssd, violation_frame_temp, frame_sdd = data

        if type(violation_frame_temp) is np.ndarray:
            distance_violation_frame = violation_frame_temp

        frame_retina, bb_faces = mask_module.face_detection(5, frame, peopleIDs,
                                                            boxes_ssd, device,
                                                            model_retina, retina_config_path)

        amount_classes, peopleID_mask_work, no_mask_frame_temp = mask_module.mask_classifier(frame,  bb_faces,
                                                                                        model_mask, amount_classes,
                                                                                        peopleID_mask_work, peopleIDs,
                                                                                        no_mask_frame)

        if type(no_mask_frame_temp) is np.ndarray:
            no_mask_frame = no_mask_frame_temp

        if interval_timer(iteration_start_time=iteration_start_time, interval=10):
            counters = evaluate_statistics(
                amount_people=amount_people,
                amount_classes=amount_classes,
                contraventions=violation_count,
                frame=distance_violation_frame,
                mask_frame=no_mask_frame
            )
            amount_people, amount_classes, violation_count, iteration_start_time = counters

        cv2.imshow("RetinaFace", frame_retina)
        cv2.imshow('SDD', frame_sdd)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    vid.release()
    cv2.destroyAllWindows()