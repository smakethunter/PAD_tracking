import pickle
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
import imutils
import torchvision.transforms as T
import torchvision
class PedestrianSensor(object):
    def __init__(self):
        self.pedestrian_view = None
        self.detection = False
        self.recording = False
        # self.clf = pickle.load(open('/Users/smaket/Desktop/siapwpa/pedestrians - carla/clf_svm_pedestrians.bin', 'rb'))
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # self.sensor.listen(lambda image: PedestrianSensor._parse_mask_pedestrian(weak_self, image))
        self.nn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.nn.eval()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    def toggle_detection(self):
        self.detection = not self.detection

    def toggle_recording_detection(self):
        self.recording = not self.recording


    @staticmethod
    def sliding_window(image, stepSizeX, stepSizeY, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSizeY):
            for x in range(0, image.shape[1], stepSizeX):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    @staticmethod
    def pyramid(image, scale=1.5, minSize=(64, 128)):
        # yield the original image
        # yield image
        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            # h = int(image.shape[0] / scale)
            image = imutils.resize(image, width=w)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    def pedestrian_detection_svm(self, img):
        (winW, winH) = (64, 128)
        biasUp = 150
        biasDown = 100

        biasLeft = 300
        biasRight = 10
        image = np.copy(img)
        rects = []

        for resized in self.pyramid(image[biasUp:-biasDown, biasLeft:-biasRight, :], scale=1.05):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in self.sliding_window(resized, stepSizeX=32, stepSizeY=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                window_r = cv2.resize(window, (32, 64))
                fd = hog(window_r, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, channel_axis=-1)
                label = self.clf.predict([fd])
                print(label)
                if label[0] == "pedestrian":
                    scale = int(image.shape[0] / resized.shape[0])
                    xx = x * scale
                    yy = y * scale
                    rects.append([xx+biasLeft, yy+biasUp, xx + winW*scale + biasLeft, yy + winH*scale + biasUp])
                               # else:
                #     scale = int(image.shape[0] / resized.shape[0])
                #     xx = x * scale
                #     yy = y * scale
                #     cv2.rectangle(image, (xx + biasLeft, yy + biasUp),
                #                   (xx + winW * scale + biasLeft, yy + winH * scale + biasUp), (255, 0, 0), 2)
        picks = non_max_suppression(np.array(rects), probs=None, overlapThresh=0.2)
        for pick in picks:
            cv2.rectangle(image, (pick[0], pick[1]), (pick[2], pick[3]), (0, 255, 0), 2)

        return image

    def pedestrian_detection_cv2(self, img):
        image = np.copy(img)
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        return image

    def nn_detect(self, img):
        array = img.copy()
        y_max, x_max, _ = array.shape
        height, width = 96, 32
        xywhs = []
        bboxes, clss = self.get_prediction(array, 0.8)
        for bbox, cls in zip(bboxes, clss):
            if cls == "person":
                bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                xywhs.append([bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0], cls])

        # for detection in xywhs:
            # cv2.rectangle(array, (detection[1], detection[0]),
            #               (detection[1] + detection[3], detection[0] + detection[2]), color=(0, 255, 0), thickness=2)
            # cv2.putText(array, detection[4], (detection[1], detection[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            #             thickness=2)
            # cv2.imwrite(f"test{detection[1]}{detection[0]}.jpg", array[detection[0]:detection[0] + detection[2], detection[1]:detection[1] + detection[3]])
        return [(x,y,h,w) for (x,y,w,h,l) in xywhs], array

    def get_prediction(self, img, threshold):
        """
        get_prediction
          parameters:
            - img_path - path of the input image
            - threshold - threshold value for prediction score
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
              are chosen.
        """
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.nn([img])
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img_path = '../../data/1200x-1.jpeg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detector = PedestrianSensor()
    # img_out = detector.pedestrian_detection_svm(img)
    regions, img_out = detector.nn_detect(img)
    fig, ax = plt.subplots()
    ax.imshow(img_out)
    fig.show()
