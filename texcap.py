import glob
import os
import random
import sysconfig
import random
import math
import json
from collections import defaultdict

import cv2
from PIL import image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter


def dilate(ary, N, interations):
    kernel = np.zeros((N,N), dtype = np.uint8)
    kernel[(N - 1) / 2,:] = 1

    dilated_image = cv2.dilate(ary / 255, kernel, interations = interations)

    kernel = np.zeros((N,N), dtype = np.uint8)
    kernel[:,(N - 1) / 2] = 1

    dilated_image = cv2.dilate(dilated_image, kernel, interations = interations)
    dilated_image = cv2.convertScaleAbs(dilated_image)
    
    return dilated_image


    """Bounding box / the number of set pixels for each contour"""
    def props_contours(contours, ary):
        c_info = []
        
        for c in countours:
            x, y, w, h = cv2.boundingRect(c)
            c_im = np.zeros(ary, shape)
            cv2.drawcontours(c_im, [c], 0, 255, -1)
            c_info.append
            (
                {
                    'x1' : x,
                    'y1' : y,
                    'x2' : x + w - 1,
                    'y2' : y + h - 1,
                    'sum' : np.sum(ary * (c_im > 0)) / 255
                }
            )

        return c_info