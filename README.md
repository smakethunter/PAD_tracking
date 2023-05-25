# PAD_tracking

## Object detection
resnet-50 [pytorch]

## inception_iccv models - PAD classification
@inproceedings{tang2019improving,
  title={Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization},
  author={Tang, Chufeng and Sheng, Lu and Zhang, Zhaoxiang and Hu, Xiaolin},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4997--5006},
  year={2019}
}

PETA model:
https://drive.google.com/file/d/1cvX43Qn_vydzT_jnmgwYUUe9hIA161PH/view?usp=sharing

RAP model:
https://drive.google.com/file/d/15paMK0-rKDsuzptDPK5kH2JuL8QO0HyS/view?usp=sharing

## Object reidetification
Similarity metric (logical and) + iou threshold

New detection similar to the objects in buffer is associated with the one of the highest iou between it's bbox and buffer_object bbox. 

Each object with id has it's time_to_live parameter - nr of consecutive frames when object remains undetected but remains in buffer. If time_to_live exceeds threshold (ie 7 frames) the object is discarded from buffer.

## Test dataset MOT-20
https://motchallenge.net/data/MOT20.zip


