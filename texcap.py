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
    kernel[:, (N - 1) / 2] = 1

    dilated_image = cv2.dilate(dilated_image, kernel, interations = interations)
    dilated_image = cv2.convertScaleAbs(dilated_image)
    
    return dilated_image


"""Calculate bounding box / number of set pixels for each contour"""
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


"""union crops"""
def union_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2

    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)

            
def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2

    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)

def crop_area(crop):
    x1, y1, x2, y2 = crop

    return max(0, x2 - x1) * max(0, y2 - y1)

"""find components"""
def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h -1))

    return borders


def find_optimal_components_subset(contours, edges):
    c_info = props_for_contours(contours, edges)
    c_info.sort(key = lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area - edges.shape[0] * edges.shape[1]

    c - c_info[0]
    del c_info[0] 
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
    recall = 1.0 * covered_sum / total
    prec = 1 - 1.0 * crop_area(crop) / area
    f1 = 2 * (prec * recall / (prec + recall))

    for i, c in enumerate(c_info):
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        new_crop = union_crops(crop, this_crop)
        new_sum = covered_sum + c['sum']
        new_recall = 1.0 * new_sum / total
        new_prec = I - 1.0 * crop_area(new_crop) / area
        new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)
    
    # Add this crop if it reproves ft score,
    #'_or_' adds 259 of the remining pixels for 5% crop expansion.
    
    remaining_frac = c['sum'] / (total - covered_sum)
    new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1

    if new_f1 > f1 or (remaining_frac > 0.25 and new_area_frac < 0.15):
        print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %S' % (
                i, covered_sum, new_sum, total, remaining_frac,
                crop_area(crop), crop_area(new_crop), area, new_area_frac,
                f1, new_f1))

        crop = new_crop
        covered_sum = new_sum
        
        del c_info[i]

        changed = True
        break

    if not changed:
        break