import numpy as np
from PIL import Image

class BinarizeMask(object):
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def __call__(self, mask):
        mask_array = np.array(mask) / 255.0
        binarized_mask = (mask_array > self.threshold).astype(np.uint8) * 255
        return Image.fromarray(binarized_mask)