import os
from typing import List

import cv2
import matplotlib.pyplot as plt

from models.model import inception_iccv
import numpy as np
from models.model import datasets, detect_pedestrians
import torchvision.transforms as transforms
from PIL import Image
import torch
global args
from models.model.datasets import description, attr_nums
# create model
model_inception = inception_iccv(pretrained=True, num_classes=35)
model_rap = inception_iccv(pretrained=True, num_classes=attr_nums['rap'])
# get the number of model parameters
# print('Number of model parameters: {}'.format(
#     sum([p.data.nelement() for p in model_inception.parameters()])))

# for training on multiple GPUs.
# model_path = "weights/peta_epoch_31.pth.tar"
model_path = "weights/rap_epoch_9.pth.tar"
# checkpoint = torch.load(model_path, map_location=torch.device('cpu') )
# start_epoch = checkpoint['epoch']
# best_accu = checkpoint['best_accu']
# model_inception.load_state_dict(checkpoint['state_dict'])
# optionally resume from a checkpoint


class PeopleReIdentificator:
    def __init__(self, detector=None, classifier=None, class_threshold=0.55):
        self._detector = detector if detector is not None else detect_pedestrians.PedestrianSensor()
        self._classifier = classifier
        self._objects_buffer: List[objectInfo] = []
        self.class_threshold = class_threshold
        self.last_id = 0

    def detect(self, image):
        # image = cv2.resize(image, (256, 128), interpolation = cv2.INTER_AREA)

        # Detecting all the regions
        # in the Image that has a
        # pedestrians inside it
        (regions, _) = self._detector.nn_detect(image)

        # Drawing the regions in the
        # Image
        # (x, y, w, h)

        return regions, _


    def get_PAR_features(self, detection_image):
        model = torch.nn.DataParallel(self._classifier)  # .cuda()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_test = transforms.Compose([
            transforms.Resize(size=(256, 128)),
            transforms.ToTensor(),
            normalize
        ])
        # input = Image.fromarray(detection_image)
        # input = input.cuda(non_blocking=True)
        img = transform_test(detection_image)
        output = model(img.unsqueeze(0))
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
        output = torch.sigmoid(output).detach().cpu().numpy()
        output = np.where(output > self.class_threshold, 1, 0)
        output = output.reshape((output.shape[1], ))
        return output

    def _identify(self, frame):
        detections, _ = self.detect(frame)
        is_buffer = len(self._objects_buffer) > 0
        for detection in detections:
            input = Image.fromarray(cv2.cvtColor(im[detection[0]:detection[0] + detection[3],
                                                 detection[1]:detection[1] + detection[2]],
                                                 cv2.COLOR_BGR2RGB))
            features = self.get_PAR_features(input)
            detected_object = objectInfo(id=None, features=features, bbox=detection)
            # if is_buffer:
            new_object = True
            if is_buffer:
                for buff in self._objects_buffer:
                    if buff == detected_object:
                        buff.bbox = detected_object.bbox
                        buff.repeated = True
                        new_object = False
                        break
            if new_object:
                detected_object.id = self.last_id
                self._objects_buffer.append(detected_object)
                self.last_id += 1

        for buff in self._objects_buffer:
            if buff.lifetime >= buff.time_to_live:
                self._objects_buffer.remove(buff)
            if buff.repeated:
                buff.lifetime = 0
                buff.repeated = False
            else:
                buff.lifetime += 1

    def draw_detections(self, array, detections):
        for detection in detections:
            cv2.rectangle(array, (detection[1], detection[0]),
                          (detection[1] + detection[2], detection[0] + detection[3]), color=(255, 0, 0),
                          thickness=2)
    def draw_identifications(self, array):
        for obj in self._objects_buffer:
            detection = obj.bbox
            cv2.rectangle(array, (detection[1], detection[0]),
                          (detection[1] + detection[2], detection[0] + detection[3]), color=(0, 255, 0),
                          thickness=2)
            cv2.putText(array, f"ID: {obj.id}, Lifetime {obj.lifetime}", (detection[1], detection[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2)
class objectInfo:
    def __init__(self, id, features, bbox, **kwargs):
        self.id = id
        self.features = features
        self.bbox = bbox
        self.lifetime = 0
        self.time_to_live = kwargs.get('time_to_live', 3)
        self.iou_thr = kwargs.get('iou_threshold', 0.8)
        self.similarity_thr = kwargs.get('similarity_threshold', 0.7)
        self.description = description[kwargs.get('dataset', 'rap')]
        self.repeated = False




    def __eq__(self, other):
        is_overlaping = self.bb_intersection_over_union(self.bbox, other.bbox) >= self.iou_thr
        is_similar = self.is_feature_similar(other)
        return is_overlaping and is_similar



    def is_feature_similar(self, other):
        vector_conjunction = (self.features == other.features)
        return np.sum(vector_conjunction)/vector_conjunction.shape[0] >= self.similarity_thr

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[0]+boxA[3], boxB[0]+boxB[3])
        y2 = min(boxA[1] + boxA[2], boxB[1] + boxB[2])
        # compute the area of intersection rectangle
        intersection_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        # compute the area of both the prediction and ground-truth
        # rectangles
        union_area = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection_area

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / union_area
        # return the intersection over union value
        print(boxA, boxB, iou)
        return iou

    @property
    def features_string(self):
        return [self.description[where] for where in np.argwhere(self.features == 1).flatten()]

    def cropped_image(self, image):
        return image[self.bbox[0]:self.bbox[0]+self.bbox[3], self.bbox[1]:self.bbox[1]+self.bbox[2]]




if __name__ == "__main__":
    model = PeopleReIdentificator(None, model_rap, 0.6)
    for i in range(200, 1, -1):
        im = cv2.imread('./data/MOT20/test/MOT20-06/img1/{0:06d}.jpg'.format(i))
        # print('./data/MOT20/test/MOT20-06/img1/{0:06d}.jpg'.format(i))
        im_out = im.copy()
        detections, _ = model.detect(im)
        model.draw_detections(im_out, detections)
        model._identify(im)
        # print([(m.id, m.bbox) for m in model._objects_buffer])
        # print("drawing...")
        model.draw_identifications(im_out)
        write_name = './data/MOT20/test/MOT20-06/demo/{0:06d}.jpg'.format(i)
        cv2.imwrite(write_name, im_out)





