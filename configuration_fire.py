# import the necessary packages
import os

BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "fire_images"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "non_fire_images"])

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (224, 224)

MODEL_PATH = "fire_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

MIN_PROBA = 0.99