from unittest import TestCase

import os
import skimage

from src.main.python.vision.models import get_trained_model, detect_objects, draw_boxes

import cv2

INPUT_DIR = os.path.join("..", "resources", "images")
OUTPUT_DIR = os.path.join("..", "resources", "annotated-images")

image = skimage.io.imread(os.path.join(INPUT_DIR, "crossing-1.jpg"))
section1 = skimage.io.imread(os.path.join(INPUT_DIR, "200-200-200-300.jpg"))
section2 = skimage.io.imread(os.path.join(INPUT_DIR, "750-100-150-100.jpg"))
section3 = skimage.io.imread(os.path.join(INPUT_DIR, "850-100-50-100.jpg"))

class Test(TestCase):

    def setUp(self):
        self.model = get_trained_model()

    def tearDown(self):
        cv2.destroyAllWindows()

    def test_full_image(self):
        results = detect_objects(self.model, image)
        annotated_image = draw_boxes(image, results)
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "crossing-1.jpg"), annotated_image)
        self.assertTrue(results.isupper())

    def test_section(self):
        results = detect_objects(self.model, section1)
        annotated_image = draw_boxes(section1, results)
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "200-200-200-300.jpg"), annotated_image)
        self.assertTrue(results.isupper())

    def test_small(self):
        results = detect_objects(self.model, section2)
        annotated_image = draw_boxes(section2, results)
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "750-100-150-100.jpg"), annotated_image)
        self.assertTrue(results.isupper())

    def test_tiny(self):
        results = detect_objects(self.model, section3)
        annotated_image = draw_boxes(section3, results)
        skimage.io.imsave(os.path.join(OUTPUT_DIR, "850-100-50-100.jpg"), annotated_image)
        self.assertTrue(results.isupper())