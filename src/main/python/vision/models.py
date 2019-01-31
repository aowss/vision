import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN

#ROOT_DIR = Path("../..")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Directory containing the trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "resources", "models", "mask_rcnn_coco.h5")

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


def get_trained_model(trained_model_path = COCO_MODEL_PATH, model_dir = MODEL_DIR):
    """
    trained_model_path: Path to the trained weights file
    config: A Sub-class of the Config class
    model_dir: Directory to save training logs and trained weights
    """

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(trained_model_path):
        mrcnn.utils.download_trained_weights(trained_model_path)

    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=model_dir, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(trained_model_path, by_name=True)

    return model

def detect_objects(model, rgb_image):

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    return r

def draw_boxes(rgb_image, results_dictionary):
    # Convert the image from RGB to BGR color (which OpenCV uses)
    bgr_image = rgb_image[:, :, ::-1].copy()
    boxes = results_dictionary['rois']

    index = 0
    for box in boxes:

        y1, x1, y2, x2 = box

        # Draw the box
        cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # Annotate
        #cv2.putText(bgr_image, str(results_dictionary['class_ids'][index]) + '[ ' + str(results_dictionary['scores'][index]) + ' ]', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(bgr_image, class_names[results_dictionary['class_ids'][index]], (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        index = index + 1

    return bgr_image[:, :, ::-1]
    # Show the frame of video on the screen
    #cv2.imshow('Result', rgb_image)

