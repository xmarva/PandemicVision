# from __future__ import print_function
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from all_models.masks.data.config import cfg_re50
from all_models.masks.layers.functions.prior_box import PriorBox
from all_models.masks.masks_classifier.models import get_model, check_face
from all_models.masks.models.retinaface import RetinaFace
from all_models.masks.utils.box_utils import decode, decode_landm
from all_models.masks.utils.nms.py_cpu_nms import py_cpu_nms
from configs.tresholds_config import tresholds
from sttstcs import interval_timer, null_data, evaluate_statistics


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# VIP
def analize_masks(
        retina_config_path=cfg_re50,
        retina_weights_path='weights/resnet50.pth',
        mask_weight_path='weights/mask_classifier.pt',
        video_path='data/video/metro/1.avi',
        statistic='disable'):
    torch.set_grad_enabled(False)

    net = RetinaFace(cfg=retina_config_path, phase='test')
    net = load_model(net, retina_weights_path, False)
    net.eval()

    mask_model = get_model(device='cuda')
    mask_model.load_state_dict(torch.load(mask_weight_path))
    print('Finished loading models!')

    # print(net)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture('rtsp://admin:admin1234@31.13.129.186:4554/cam/realmonitor?channel=1&subtype=0')
    # vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frames = 0

    counters = null_data(init=True)
    people_count, amount_classes, iteration_start_time = counters
    absolute_time = iteration_start_time = time.time()
    while True:
        ret, frame = vid.read()

        if not ret:
            break
        frames += 1

        # Skip frames to increase fps
        # if frames % 2 == 0 or frames % 3 == 0:
        #     continue

        img = np.float32(frame)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        resize = 1

        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(retina_config_path, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, retina_config_path['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, retina_config_path['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # print('im_detect: {:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(frames + 1, _t['forward_pass'].average_time, _t['misc'].average_time))

        for b in dets:
            if b[4] < 0.85:
                continue
            text = "{:.4f}".format(b[4])

            if b[0] > 10:
                x1 = int(b[0] - 10)
            else:
                x1 = 0
            if b[1] > 10:
                y1 = int(b[1] - 10)
            else:
                y1 = 0
            x2 = int(b[2] + 10)
            y2 = int(b[3] + 10)

            face = frame[y1: y2, x1: x2].copy()
            # cv2.imwrite('raw\\{}.jpg'.format(frames), face)
            face_confidence, face_class = check_face(mask_model, face)

            # по таймеру раз в n
            if statistic == 'enabled' and interval_timer(iteration_start_time, interval=5):
                counters = evaluate_statistics(
                    ret=ret,
                    frame=frame,
                    people_count=people_count,
                    amount_classes=amount_classes,
                    tresholds=tresholds,
                    iteration_start_time=iteration_start_time,
                    absolute_time=absolute_time,
                    interval=1
                )

                people_count, amount_classes, iteration_start_time = counters

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cx = int(b[0])
            cy = int(b[1] + 12)

            text2 = f'{face_class}, {{:.4f}}'.format(face_confidence)

            cv2.putText(frame, text2, (cx, y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.imshow("Social Distancing Detector", frame)
        # vid_writer.write(frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    vid.release()
    # vid_writer.release()
    cv2.destroyAllWindows()


def face_detection(delta, frame,  peopleIDs, boxes_ssd, device, net, config_path):
    """
    Face detection with RetinaFace model
    :param delta: int. bounding box padding for crop
    :param frame: np.ndarray of whole image
    :param peopleIDs: list of int people's ID in frame
    :param boxes_ssd: list of  bounding box np.ndarray
    :param device: device i v afrike device str
    :param net: torch Retina Face model
    :param config_path: str
    :return:
        frame: frame of whole image with face bounding boxes
        bb_faces: list of face bounding box np.nparrays
    """
    img_input = np.float32(frame)
    bb_images = []
    for x1, y1, box_w, box_h in boxes_ssd:
        if x1 > delta:
            x1 = int(x1 - delta)
        else:
            x1 = 0

        if y1 > delta:
            y1 = int(y1 - delta)
        else:
            y1 = 0

        x2 = x1 + box_w
        y2 = y1 + box_h
        if x2+delta < img_input.shape[1]:
            x2 = int(x2 + delta)
        else:
            x2 = img_input.shape[1]-1

        if y2+delta < img_input.shape[0]:
            y2 = int(y2 + delta)
        else:
            y2 = img_input.shape[0]-1

        bb_images.append(img_input[y1: y2, x1: x2].copy())

    bb_faces = []
    for i in range(len(bb_images)):
        im_height, im_width, _ = bb_images[i].shape
        scale = torch.Tensor([bb_images[i].shape[1], bb_images[i].shape[0],
                              bb_images[i].shape[1], bb_images[i].shape[0]])
        bb_images[i] -= np.array([104, 117, 123], dtype='uint8')
        bb_images[i] = bb_images[i].transpose(2, 0, 1)
        bb_images[i] = torch.from_numpy(bb_images[i]).unsqueeze(0)
        bb_images[i] = bb_images[i].to(device)
        scale = scale.to(device)

        loc, conf, landms = net(bb_images[i])  # forward pass
        priorbox = PriorBox(config_path, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, config_path['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, config_path['variance'])
        scale1 = torch.Tensor([bb_images[i].shape[3], bb_images[i].shape[2],
                               bb_images[i].shape[3], bb_images[i].shape[2],
                               bb_images[i].shape[3], bb_images[i].shape[2],
                               bb_images[i].shape[3], bb_images[i].shape[2],
                               bb_images[i].shape[3], bb_images[i].shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        for b in dets:
            if b[4] < 0.85:
                continue
            text = 'ID:' + str(peopleIDs[i])  # + ' ' + "{:.4f}".format(b[4])

            if b[0] > 10:
                x1 = int(b[0] - 10) + boxes_ssd[i][0]
            else:
                x1 = 0 + boxes_ssd[i][0]
            if b[1] > 10:
                y1 = int(b[1] - 10) + boxes_ssd[i][1]
            else:
                y1 = 0 + boxes_ssd[i][1]
            x2 = int(b[2] + 10) + boxes_ssd[i][0]
            y2 = int(b[3] + 10) + boxes_ssd[i][1]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cx = int(b[0]) + boxes_ssd[i][0]
            cy = int(b[1] + 12) + boxes_ssd[i][1]

            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            bb_faces.append((x1, y1, x2, y2))
            break
    return frame, bb_faces


def mask_classifier(frame, bb_faces, mask_model, mask_dict, peopleID_mask_work, peopleIDs, no_mask_frame):
    """
    :param frame: np.ndarray whole image
    :param bb_faces: list of face's bounding box np.ndarray in current frame
    :param mask_model: torch ResNe-34 model
    :param mask_dict: dict {'str(no_mask)': int(count), 'str(with_mask)': int(count)}
    :param peopleID_mask_work: int list of all people ID
    :param peopleIDs: int list of people id in frame
    :param no_mask_frame: np.ndarray BGR image with mask violation
    :return:
        mask_dict dict {'str(no_mask)': int(count), 'str(with_mask)': int(count)}
        peopleID_mask_work int list of all people ID
        no_mask_frame np.ndarray BGR image with mask violation
    """
    get_frame = False

    for i in range(len(bb_faces)):
        x1, y1, x2, y2 = bb_faces[i]
        face = frame[y1: y2, x1: x2].copy()
        face_confidence, face_class = check_face(mask_model, face)

        if peopleIDs[i] not in peopleID_mask_work:
            peopleID_mask_work.append(peopleIDs[i])
            if face_class == 0:
                mask_dict['no_mask'] += 1
            elif face_class == 1:
                mask_dict['with_mask'] += 1

                if not get_frame:
                    get_frame = True
                    no_mask_frame = frame
                    print('WARN: ' + str(type(no_mask_frame)))

    return mask_dict, peopleID_mask_work, no_mask_frame
