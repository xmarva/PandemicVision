import torch
import cv2
import time
from PIL import Image
from numba import njit
import all_models.distance.utils.utils as utils
from all_models.distance.models import Darknet
from torch.autograd import Variable
from torchvision import transforms
from all_models.distance.sort import *
from sttstcs import interval_timer, null_data, evaluate_statistics
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@njit
def get_img_pad(img, img_size):
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    return pad_x, pad_y, unpad_h, unpad_w


@njit
def append_a_b(bbox, homography):
    x = np.mean(np.array([bbox[0], bbox[0] + bbox[2]]))
    y = bbox[1] + bbox[3]
    point_orig = np.array([x, y, 1]).reshape((3, 1))
    point_hmgf = np.dot(homography, point_orig)
    x = point_hmgf[0] / point_hmgf[2]
    y = point_hmgf[1] / point_hmgf[2]
    return x, y


def detect_image(image, img_size, model_sdd, conf_thres, nms_thres):
    """
    :param image: input image (PIL Image)
    :param img_size: image size used in YOLO net (tuple)
    :param model_sdd: YOLO model (torch.model)
    :param conf_thres: confidence threshold
    :param nms_thres: non maximum suppression threshold
    :return: raw detection from net (torch.Tensor [detection numbers x object detection vector])
    """
    # scale and pad image
    ratio = min(img_size / image.size[0], img_size / image.size[1])
    imw = round(image.size[0] * ratio)
    imh = round(image.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor()])
    # convert image to Tensor
    image_tensor = img_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detection_prep = model_sdd(input_img)
        detection_prep = utils.non_max_suppression(detection_prep, 80, conf_thres, nms_thres)
    return detection_prep[0]


# VIP
def analize_distance(
        sdd_config_path='configs/yolov3.cfg',
        sdd_weights_path='weights/yolov3.weights',
        video_path='data/video/metro/1.avi',
        statistic='disable',
        additional_config='configs/ssd_model'
):
    # TODO: import as cfg
    # img_size = additional_config['img_size']
    # conf_thres = additional_config['conf_thres']
    # nms_thres = additional_config['nms_thres']
    # device = additional_config['device']
    # key_points_path = additional_config['key_points_path']

    img_size = 608
    conf_thres = 0.5
    nms_thres = 0.3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key_points_path = 'data/k_points.txt'

    model_sdd = Darknet(sdd_config_path, img_size=img_size)
    model_sdd.load_weights(sdd_weights_path)
    model_sdd.to(device)
    model_sdd.eval()
    torch.set_grad_enabled(False)

    vid = cv2.VideoCapture(video_path)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mot_tracker = Sort(30, 5)

    homography = utils.get_matrix(key_points_path)

    frames = 0
    counters = null_data(init=True)
    people_count, amount_classes, iteration_start_time = counters
    absolute_time = iteration_start_time = time.time()

    people_count = 0
    violation_count = 0
    peopleIDs = []
    violationsIDs = []
    while True:
        ret, frame = vid.read()

        if not ret:
            break
        frames += 1

        # Skip frames to increase fps
        if frames % 2 == 0 or frames % 3 == 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        output = detect_image(pilimg, img_size, model_sdd, conf_thres, nms_thres)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img = np.array(pilimg)
        pad_x, pad_y, unpad_h, unpad_w = get_img_pad(img, img_size)

        boxes_ssd = []
        classIDs = []
        objectIDs = []
        if output is not None:
            tracked_objects = mot_tracker.update(output.cpu())
            # unique_labels = output[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                if cls_pred == 0:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    boxes_ssd.append(np.array([x1, y1, box_w, box_h]))
                    classIDs.append(cls_pred)
                    objectIDs.append(int(obj_id))

                    if obj_id not in peopleIDs:
                        peopleIDs.append(int(obj_id))
                        people_count += 1

        ind = []
        for i in range(0, len(classIDs)):
            if classIDs[i] == 0:
                ind.append(i)

        a = []
        b = []
        if len(boxes_ssd) > 0:
            for i in range(len(boxes_ssd)):
                x, y = append_a_b(boxes_ssd[i], homography)
                a.append(x)
                b.append(y)

        # check distance
        distance = []
        nsd = []
        for i in range(0, len(a) - 1):
            for k in range(1, len(a)):
                if k == i:
                    break
                else:
                    x_dist = (a[k] - a[i])
                    y_dist = (b[k] - b[i])
                    d = np.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if d <= 100:
                        nsd.append(i)
                        nsd.append(k)

                        if str(objectIDs[i]) + ' ' + str(objectIDs[k]) not in violationsIDs or \
                                str(objectIDs[k]) + ' ' + str(objectIDs[i]) not in violationsIDs:
                            violationsIDs.append(str(objectIDs[i]) + ' ' + str(objectIDs[k]))
                            violationsIDs.append(str(objectIDs[k]) + ' ' + str(objectIDs[i]))
                            violation_count += 1

                    nsd = list(dict.fromkeys(nsd))
                    # print(nsd)

        # draw bb
        color = (0, 255, 0)
        if len(boxes_ssd) > 0:
            for i in range(len(boxes_ssd)):
                if i in nsd:
                    continue
                else:
                    (x, y) = (boxes_ssd[i][0], boxes_ssd[i][1])
                    (w, h) = (boxes_ssd[i][2], boxes_ssd[i][3])

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = 'OK. ID:' + str(int(objectIDs[i]))
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        color = (0, 0, 255)
        for i in nsd:
            (x, y) = (boxes_ssd[i][0], boxes_ssd[i][1])
            (w, h) = (boxes_ssd[i][2], boxes_ssd[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = 'Alert. ID:' + str(int(objectIDs[i]))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        text2 = 'People count:' + str(int(people_count))
        cv2.putText(frame, text2, (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        text3 = 'Violation count:' + str(int(violation_count))
        cv2.putText(frame, text3, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
        # show frame
        cv2.imshow("Social Distancing Detector", frame)

        # по таймеру раз в n
        # if (statistic == 'enabled' and interval_timer(iteration_start_time, interval=5)):
        #     counters = evaluate_statistics(
        #         ret,
        #         frame,
        #         people_count,
        #         amount_classes,
        #         tresholds,
        #         face_confidence,
        #         face_class,
        #         iteration_start_time,
        #         absolute_time,
        #         interval=1
        #     )
        #     people_count, amount_classes, iteration_start_time = counters

        # exit conditions
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    vid.release()
    cv2.destroyAllWindows()


def SDD(frame, img_size, model_sdd,
        conf_thres, nms_thres, mot_tracker,
        violationsIDs, peopleIDs_work,
        people_count, violation_count, homography,
        violation_frame=None):
        """
        Social distance detector
        :param frame: nd.array input BGR image
        :param img_size: int YOLO image input size
        :param model_sdd: torch.model
        :param conf_thres: float confidence threshold
        :param nms_thres: float non maximum suppression threshold
        :param mot_tracker: SORT tracker class
        :param violationsIDs: list of distance violations pairs
        :param peopleIDs_work: list of int for keeping all people ID for correct function work
        :param people_count: int number of unique people
        :param violation_count: int number of distance violation
        :param homography: np.ndarray homography matrix. [3 x 3]
        :param violation_frame: nd.array of whole frame with distance violation
        :return:
        violation_count: int number of distance violation
        people_count: int number of unique people
        objectIDs: list of int people ID for correct function work
        peopleIDs_work: list of int for keeping all people ID for correct function work
        boxes_ssd: list of bounding boxes np.ndarray
        violation_frame: nd.array of whole frame with distance violation
        frame: nd.array of whole frame with bounding boxes
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        output = detect_image(pilimg, img_size, model_sdd, conf_thres, nms_thres)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img = np.array(pilimg)
        pad_x, pad_y, unpad_h, unpad_w = get_img_pad(img, img_size)

        boxes_ssd = []
        classIDs = []
        objectIDs = []
        if output is not None:
            tracked_objects = mot_tracker.update(output.cpu())
            # unique_labels = output[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                if cls_pred == 0:
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    boxes_ssd.append(np.array([x1, y1, box_w, box_h]))
                    classIDs.append(cls_pred)
                    objectIDs.append(int(obj_id))

                    if obj_id not in peopleIDs_work:
                        peopleIDs_work.append(int(obj_id))
                        people_count += 1

        ind = []
        for i in range(0, len(classIDs)):
            if classIDs[i] == 0:
                ind.append(i)

        a = []
        b = []
        if len(boxes_ssd) > 0:
            for i in range(len(boxes_ssd)):
                x, y = append_a_b(boxes_ssd[i], homography)
                a.append(x)
                b.append(y)

        # check distance
        distance = []
        nsd = []
        for i in range(0, len(a) - 1):
            for k in range(1, len(a)):
                if k == i:
                    break
                else:
                    x_dist = (a[k] - a[i])
                    y_dist = (b[k] - b[i])
                    d = np.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if d <= 100:
                        nsd.append(i)
                        nsd.append(k)

                        if str(objectIDs[i]) + ' ' + str(objectIDs[k]) not in violationsIDs or \
                                str(objectIDs[k]) + ' ' + str(objectIDs[i]) not in violationsIDs:
                            violationsIDs.append(str(objectIDs[i]) + ' ' + str(objectIDs[k]))
                            violationsIDs.append(str(objectIDs[k]) + ' ' + str(objectIDs[i]))
                            violation_count += 1

                    nsd = list(dict.fromkeys(nsd))
                    # print(nsd)

        # draw bb
        color = (0, 255, 0)
        if len(boxes_ssd) > 0:
            for i in range(len(boxes_ssd)):
                if i in nsd:
                    continue
                else:
                    (x, y) = (boxes_ssd[i][0], boxes_ssd[i][1])
                    (w, h) = (boxes_ssd[i][2], boxes_ssd[i][3])

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = 'OK. ID:' + str(int(objectIDs[i]))
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        color = (0, 0, 255)
        for i in nsd:
            (x, y) = (boxes_ssd[i][0], boxes_ssd[i][1])
            (w, h) = (boxes_ssd[i][2], boxes_ssd[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = 'Alert. ID:' + str(int(objectIDs[i]))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # DEBUG INFO
        text2 = 'People count:' + str(int(people_count))
        cv2.putText(frame, text2, (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        text3 = 'Violation count:' + str(int(violation_count))
        cv2.putText(frame, text3, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        if len(nsd) != 0:
            violation_frame = frame

        return violation_count, people_count, objectIDs, peopleIDs_work, boxes_ssd, violation_frame, frame
