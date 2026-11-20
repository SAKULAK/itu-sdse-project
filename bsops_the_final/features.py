"""Image preprocessing: transforms, dataset statistics and small artifacts."""
import json
import logging

import numpy as np
from torchvision import transforms

from vial_classification.config import (
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    IMAGE_STATISTICS_PATH,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
)