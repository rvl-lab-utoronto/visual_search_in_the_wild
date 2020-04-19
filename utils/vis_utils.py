"""
Visualization utility.
"""

# BASIC
import os
import argparse
import glob
import random

# LOGGING
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
from classification_models.tfkeras import Classifiers


TARGET_WIDTH = 128
TARGET_HEIGHT = 128
GAUSSIAN_BLUR_PROB = 0.1
HSV_JITTER_PROB = 0.3
COLOR_FLIP_PROB = 0.1
SALT_PEPPER_PROB = 0.3

"""
    Segmentation legend
"""
def draw_legend(colors, classes):
    legend = np.zeros((600, 200, 3), np.uint8)

    legend[:,:,:] = (214,111,150)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    for class_id, class_name in classes.items():
        color = colors[class_id]
        y1 = int((class_id)*(legend.shape[0]/len(colors))) + 10
        y2 = y1 + 20
        cv2.rectangle(legend, (20, y1), (40, y2), color, -1)
        cv2.putText(legend, class_name, (60, y2), font, fontscale, color=(255,255,255))

    cv2.imshow('legend', legend)
    cv2.waitKey()

def generate_semantic_segmentation(img, heatmap_dict, classes, colors):

    first_key = list(heatmap_dict.keys())[0]
    heatmap_shape = heatmap_dict[first_key].shape

    merged_segmentation = np.zeros((heatmap_shape[0], heatmap_shape[1], 3), np.uint8)
    merged_seg_classids = np.zeros((heatmap_shape[0], heatmap_shape[1]), np.uint8)

    for y in range(heatmap_shape[0]):
        for x in range(heatmap_shape[1]):
            highest_class = (0, 'sand', 0.0)
            for class_id, heatmap in heatmap_dict.items():
                if heatmap[y,x] > highest_class[2]:
                    highest_class = (class_id, classes[class_id], heatmap[y,x])

            merged_segmentation[y,x] = colors[highest_class[0]]
            merged_seg_classids[y,x] = highest_class[0]


    return merged_segmentation, merged_seg_classids



def generate_weighted_heatmap(img, heatmap_dict, weights_dict, alpha=0.4, colormap=cv2.COLORMAP_JET):

    first_key = list(heatmap_dict.keys())[0]
    heatmap_shape = heatmap_dict[first_key].shape

    merged_heatmap = np.zeros(heatmap_shape, np.float32)

    for patch_class, heatmap in heatmap_dict.items():
        weights = float(weights_dict[patch_class])
        merged_heatmap += weights*heatmap

    max_heat = np.max(merged_heatmap)

    if max_heat == 0.:
        max_heat = 1e-20

    # merged_heatmap_normd = merged_heatmap/max_heat
    merged_heatmap_normd = merged_heatmap

    merged_heatmap_color = cv2.applyColorMap(np.uint8(255*merged_heatmap_normd), colormap)
    img_cpy = img.copy()
    cv2.addWeighted(merged_heatmap_color,
                    alpha,
                    img,
                    1.0-alpha,
                    0,
                    img_cpy)

    return merged_heatmap, img_cpy


def generate_merged_heatmap(img, heatmaps, alpha=0.4, colormap=cv2.COLORMAP_JET):

    heatmap_shape = heatmaps[0].shape

    merged_heatmap = np.zeros(heatmap_shape, np.float32)

    for hmap in heatmaps:
        merged_heatmap += hmap

    merged_heatmap = merged_heatmap/len(heatmaps)

    max_heat = np.max(merged_heatmap)

    if max_heat == 0.:
        max_heat = 1e-20

    merged_heatmap_normd = merged_heatmap/max_heat
    # merged_heatmap_normd = merged_heatmap

    merged_heatmap_color = cv2.applyColorMap(np.uint8(255*merged_heatmap), colormap)
    merged_heatmap_color_normd = cv2.applyColorMap(np.uint8(255*merged_heatmap_normd), colormap)
    img_cpy = img.copy()
    cv2.addWeighted(merged_heatmap_color,
                    alpha,
                    img,
                    1.0-alpha,
                    0,
                    img_cpy)
    img_cpy2 = img.copy()
    cv2.addWeighted(merged_heatmap_color_normd,
                    alpha,
                    img,
                    1.0-alpha,
                    0,
                    img_cpy2)

    return merged_heatmap, img_cpy, img_cpy2


def generate_heatmaps(image, saliency_dict, desc_shape=(8,8), patch_size=(128,128), alpha=0.4, mode='avg', colormap=cv2.COLORMAP_JET, binary=False, sliding=True, visualize=True, get_normed=False, wait_key_dur=0):

    h, w, c = image.shape

    if sliding:
        dot_heatmap = np.zeros((h, w, 1), np.float32)
        # print("Generating heatmap of shape: " + str(dot_heatmap.shape))

        for key, val in saliency_dict.items():
            x, y = key
            pred_lbl, pred_dot = val
            if binary:
                val_interest = pred_lbl
            else:
                val_interest = pred_dot

            print("Dot product 2: "+ str(pred_dot))
            if mode == 'avg':
                print(val_interest)
                # dot_heatmap[y:y+patch_size[1],x:x+patch_size[0]] = val_interest
                # dot_heatmap[y:+patch_size[1], x:x+patch_size[0]] = val_interest
                if np.sum(dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]]) == 0:
                    dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]] += val_interest
                else:
                    dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]] += val_interest
                    dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]] /= 2
            if mode == 'sum':
                dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]] += val_interest
            if mode == 'overwrite':
                dot_heatmap[y:y + patch_size[1], x:x + patch_size[0]] = val_interest


        if visualize:
            max_heat = np.max(dot_heatmap)

            if max_heat == 0.:
                max_heat = 1e-20

            dot_heatmap_nrm = dot_heatmap / max_heat
        # dot_heatmap = cv2.applyColorMap(np.uint8(255*dot_heatmap), colormap)

        dot_heatmap_upsampled = cv2.resize(dot_heatmap, (h,w), interpolation=cv2.INTER_NEAREST)
        hmap_no_colormap = dot_heatmap_upsampled
        dot_heatmap_upsampled = cv2.applyColorMap(np.uint8(255*dot_heatmap_upsampled), colormap)
        # dot_heatmap_upsampled = cv2.applyColorMap(np.uint8(dot_heatmap_upsampled), colormap) # MEANINGLESS
        dot_heatmap = dot_heatmap_upsampled

        if visualize:
            dot_heatmap_nrm_upsampled = cv2.resize(dot_heatmap_nrm, (h,w), interpolation=cv2.INTER_NEAREST)
            hmap_nrm_no_colormap = dot_heatmap_upsampled
            dot_heatmap_nrm_upsampled = cv2.applyColorMap(np.uint8(255*dot_heatmap_nrm_upsampled), colormap)
            dot_heatmap_nrm = dot_heatmap_nrm_upsampled

    else:
        dot_heatmap = np.zeros((desc_shape[1], desc_shape[0], 1), np.float32)
        dim = int(desc_shape[0]/4)
        # print("Generating heatmap of shape: " + str(dot_heatmap.shape))

        for key, val in saliency_dict.items():
            x, y = key
            pred_lbl, pred_dot = val

            if binary:
                val_interest = pred_lbl
            else:
                val_interest = pred_dot


            if mode == 'avg':
                # if np.sum(dot_heatmap[y:y+4, x:x+4]) == 0:
                #     dot_heatmap[y:y+4, x:x+4] = val_interest
                # else:
                #     dot_heatmap[y:y+4, x:x+4] += val_interest
                #     dot_heatmap[y:y+4, x:x+4] = dot_heatmap[y:y+4, x:x+4] / 2

                dot_heatmap[y,x] = val_interest
                # for j in range(dim):
                #     for i in range(dim):
                #         # if dot_heatmap[y+j, x+i] == 0:
                #             # dot_heatmap[y+j, x+i] = val_interest
                #         # else:
                #         dot_heatmap[y+j,x+i] += val_interest
                #         dot_heatmap[y+j,x+i] /= 2

            elif mode == 'overwrite':
                for j in range(dim):
                    for i in range(dim):
                        dot_heatmap[y+j, x+i] = val_interest

            elif mode == 'sum':
                for j in range(dim):
                    for i in range(dim):
                        dot_heatmap[y+j, x+i] += val_interest

        if visualize:
            max_heat = np.max(dot_heatmap)

            if max_heat == 0.:
                max_heat = 1e-20

            dot_heatmap_nrm = dot_heatmap / max_heat
        # dot_heatmap = cv2.applyColorMap(np.uint8(255*dot_heatmap), colormap)

        dot_heatmap_upsampled = cv2.resize(dot_heatmap, (h,w), interpolation=cv2.INTER_NEAREST)
        # dot_heatmap_upsampled = cv2.resize(dot_heatmap, (h,w), interpolation=cv2.INTER_LINEAR)
        hmap_no_colormap = dot_heatmap_upsampled
        dot_heatmap_upsampled = cv2.applyColorMap(np.uint8(255*dot_heatmap_upsampled), colormap)
        # dot_heatmap_upsampled = cv2.applyColorMap(np.uint8(dot_heatmap_upsampled), colormap) # MEANINGLESS
        dot_heatmap = dot_heatmap_upsampled

        if visualize:
            dot_heatmap_nrm_upsampled = cv2.resize(dot_heatmap_nrm, (h,w), interpolation=cv2.INTER_NEAREST)
            hmap_nrm_no_colormap = dot_heatmap_upsampled
            dot_heatmap_nrm_upsampled = cv2.applyColorMap(np.uint8(255*dot_heatmap_nrm_upsampled), colormap)
            dot_heatmap_nrm = dot_heatmap_nrm_upsampled

    if visualize:
        image_cpy1 = image.copy()
        image_cpy2 = image.copy()
        cv2.addWeighted(dot_heatmap,
                        alpha,
                        image,
                        1.0-alpha,
                        0,
                        image_cpy1)

        cv2.imshow('heatmap_dot', image_cpy1)

        cv2.addWeighted(dot_heatmap_nrm,
                        alpha,
                        image,
                        1.0-alpha,
                        0,
                        image_cpy2)

        cv2.imshow('heatmap_dot_norm', image_cpy2)
        if wait_key_dur >= 0:
            k = cv2.waitKey(wait_key_dur)

        if get_normed:
            return dot_heatmap, hmap_no_colormap, image_cpy2
        else:
            return dot_heatmap, hmap_no_colormap, image_cpy1

    return dot_heatmap, hmap_no_colormap, image


def visualize_trios(trios, window_name='trio_image', normalized=True):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    color = (0,0,255)
    image_cache = dict()

    for idx, trio in enumerate(trios):
        label = ['query', 'similar', 'different']

        if type(trio[0]) == str:
            image_cache, im1 = cached_imread(trio[0],
                                image_cache,
                                width=TARGET_WIDTH,
                                height=TARGET_HEIGHT)
        else:
            print(trio[0].shape)

            if normalized:
                im1 = np.float32(trio[0])
            else:
                im1 = np.float32(trio[0]/255.)

        if type(trio[1]) == str:
            image_cache, im2 = cached_imread(trio[1],
                                image_cache,
                                width=TARGET_WIDTH,
                                height=TARGET_HEIGHT)
        else:
            if normalized:
                im2 = np.float32(trio[1])
            else:
                im2 = np.float32(trio[1]/255.)

        if type(trio[2]) == str:
            image_cache, im3 = cached_imread(trio[2],
                                image_cache,
                                width=TARGET_WIDTH,
                                height=TARGET_HEIGHT)
        else:
            if normalized:
                im3 = np.float32(trio[2])
            else:
                im3 = np.float32(trio[2]/255.)

        image_trios = hconcat_resize_min([im1, im2, im3])
        for i in range(3):
            cv2.putText(image_trios,
                        label[i],
                        (i*TARGET_WIDTH + 25, 25),
                        font,
                        fontScale,
                        color,
                        lineType)

        cv2.imshow(window_name, image_trios)
        k = cv2.waitKey(0)

        if k == 27:
            break

    return

def visualize_pairs(lbld_pairs, predictions=None, window_name='paired_image', normalized=True):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 2
    color = (0,255,0)
    image_cache = dict()

    for idx, pair in enumerate(lbld_pairs):
        label = ''
        if len(pair) == 3:
            if pair[2] == '0':
                label = 'different'
            elif pair[2] == '1':
                label = 'similar'


        if type(pair[0]) == str:
            image_cache, im1 = cached_imread(pair[0],
                                             image_cache,
                                             width=TARGET_WIDTH,
                                             height=TARGET_HEIGHT)
        else:
            if normalized:
                im1 = np.float32(pair[0])
            else:
                im1 = np.float32(pair[0]/255.0)

        if type(pair[1]) == str:
            image_cache, im2 = cached_imread(pair[1],
                                             image_cache,
                                             width=TARGET_WIDTH,
                                             height=TARGET_HEIGHT)
        else:
            if normalized:
                im2 = np.float32(pair[1])
            else:
                im2 = np.float32(pair[1]/255.0)

        paired_image = hconcat_resize_min([im1, im2])

        cv2.putText(paired_image,
                    label,
                    (25, 25),
                    font,
                    fontScale,
                    color,
                    lineType)

        if predictions is not None:
            pred_index = predictions[idx]
            pred_label = 0
            if pred_index:
                pred_label = 'similar'
            else:
                pred_label = 'different'
            cv2.putText(paired_image,
                        pred_label,
                        (25, 50),
                        font,
                        fontScale,
                        (0,0,255),
                        lineType)

        cv2.imshow(window_name, paired_image)
        k = cv2.waitKey(0)

        if k == 27:
            break

    return


def preprocess_image(image_path,
                     seed,
                     datagen,
                     datagen_args,
                     image_cache,
                     width=128,
                     height=128,
                     training=True,
                     imagenet=False):
    np.random.seed(seed)
    X = np.zeros((1, width, height, 3))
    image_cache, image = cached_imread(image_path,
                                       image_cache,
                                       width=width,
                                       height=height)

    X[0] = image

    if training:
        X[0] = datagen.random_transform(X[0], seed)
        X[0] = color_flip(X[0], seed)
        X[0] = gaussian_blur(X[0], seed)
        X[0] = salt_pepper(X[0], seed)

    if not imagenet:
        X[0] = X[0]/255.
    else:
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        X[0] = preprocess_input(X[0])

    return image_cache, X


def cached_imread(image_path,
                  image_cache,
                  width=128,
                  height=128):
    logging.info("Cache reading image:" + str(image_path))

    if type(image_path) != str and type(image_path) != bytes:

        image = cv2.resize(image_path, (width, height))
        return image_cache, image

    else:

        if type(image_path) == bytes:
            image_path = image_path.decode('utf8')

        if image_path not in image_cache:

            if len(image_cache) > 10000:
                print("Reseting cache, reached 10000 images.")
                image_cache = dict()


            image = cv2.imread(image_path)
            image = cv2.resize(image, (width, height))
            image_cache[image_path] = image

        # image_cache = dict()
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (width, height))
        # image_cache[image_path] = image
        return {}, image_cache[image_path]


"""
Image tiling code from: https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
"""
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

"""
    Old school image augmentation
"""
def hsv_jitter(image, seed):
    """Function that takes an image, and adds saturation jitter.
    Args:
        image (cv2 image): image in question.
    Returns:
        cv2_image: Returns preprocessed cv2 image.
    """
    random.seed(seed)
    if random.uniform(0.0, 1.0) > HSV_JITTER_PROB:
        return image

    img = image.copy()

    hsv_jit = 0.6
    hsv_jit_pos = 0.05
    ch = random.randint(0, 2)

    hsv_data = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_data[:, :, ch] = hsv_data[:, :, ch]*random.uniform(1-hsv_jit, 1+hsv_jit_pos)
    # hsv_data[:, :, 1] = hsv_data[:, :, 1]*random.uniform(1-hsv_jit, 1)
    # hsv_data[:, :, 2] = hsv_data[:, :, 2]*random.uniform(1-hsv_jit, 1)

    hsv_data[:, :, 0] = np.clip(hsv_data[:, :, 0], 0, 179)
    hsv_data[:, :, 1] = np.clip(hsv_data[:, :, 1], 0, 255)
    hsv_data[:, :, 2] = np.clip(hsv_data[:, :, 2], 0, 255)

    img = cv2.cvtColor(hsv_data, cv2.COLOR_HSV2BGR)

    return img


def color_flip(image, seed):
    """Function that takes an image, and flips channels.
    Args:
        image (cv2 image): image in question.
    Returns:
        cv2_image: Returns preprocessed cv2 image.
    """
    random.seed(seed)
    if random.uniform(0.0, 1.0) > COLOR_FLIP_PROB:
        return image

    img = image.copy()

    img = img[:, :, ::-1]
    return img


def salt_pepper(image, seed):
    """Function that takes an image, sprinkles salt and pepper noise.
    Args:
        image (cv2 image): image in question.
    Returns:
        cv2_image: Returns preprocessed cv2 image.
    """
    random.seed(seed)
    if random.uniform(0.0, 1.0) > SALT_PEPPER_PROB:
        return image

    img = image.copy()

    row, col, _ = img.shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount*img.size*salt_vs_pepper)
    num_pepper = np.ceil(amount*img.size*(1.0-salt_vs_pepper))

    # Add salt
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
    img[coords[0], coords[1], :] = 255

    # Add pepper
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
    img[coords[0], coords[1], :] = 0

    return img


def gaussian_blur(image, seed):
    """Function that takes an image, shades it with gaussian noise.
    Args:
        image (cv2 image): image in question.
    Returns:
        cv2_image: Returns preprocessed cv2 image.
    """
    random.seed(seed)
    if random.uniform(0.0, 1.0) > GAUSSIAN_BLUR_PROB:
        return image

    img = image.copy()
    row, col, _ = image.shape

    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img
