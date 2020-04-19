import os
import cv2
import math
import numpy as np

from utils.logging_config import *

orb = cv2.ORB_create()

def numerical_sort(alist):
    # inspired by Alex Martelli
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52234
    indices = map(_generate_index, alist)
    decorated = zip(indices, alist)
    decorated.sort()
    return [ item for index, item in decorated ]

def tup2int(tup):
    new_tup = (int(tup[0]), int(tup[1]))
    return new_tup

def mask_from_bbox(bbox, img_shape):
    height = img_shape[0]
    width = img_shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(bbox.top):int(bbox.bottom), int(bbox.left):int(bbox.right)] += 255

    return mask

def isclose(a, b, tol=40):
    return abs(a-b) <= tol

def matches_nms(matches, kp, min_score=0.5, patch_dims=(128,128)):
    # sort boxes by score
    # compare in that order with the rest
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    filtered_matches = []

    for match in sorted_matches:
        duplicate = False
        center = kp[match.trainIdx].pt
        bbox = BBox(int(center[0]-patch_dims[0]/2), # xmin
                    int(center[1]-patch_dims[1]/2), # ymin
                    int(center[0]+patch_dims[0]/2), # xmax
                    int(center[1]+patch_dims[1]/2), # ymax
                    center,
                    score=match.distance,
                    id='test',
                    label='test')
        for great_match in filtered_matches:
            great_center = kp[great_match.trainIdx].pt
            great_bbox = BBox(int(great_center[0]-patch_dims[0]/2), # xmin
                              int(great_center[1]-patch_dims[1]/2), # ymin
                              int(great_center[0]+patch_dims[0]/2), # xmax
                              int(great_center[1]+patch_dims[1]/2), # ymax
                              great_center,
                              score=great_match.distance,
                              id='test2',
                              label='test2')
            if bbox.overlap_ratio(great_bbox) > 0.4:
                duplicate = True
                break

        if not duplicate:
            filtered_matches.append(match)

    return filtered_matches

def nms(bboxes, min_score=0.9):
    # sort boxes by score
    # compare in that order with the rest
    sorted_bboxes = sorted(bboxes, key=lambda x: x.score)
    filtered_bboxes = []

    for bbox in sorted_bboxes:
        duplicate = False
        for great_bbox in filtered_bboxes:
            if bbox.overlap_ratio(great_bbox) > 0.4:
                duplicate = True
                break

        if not duplicate and bbox.score > min_score:
            filtered_bboxes.append(bbox)

    return filtered_bboxes

def _generate_index(str):
    """
    Splits a string into alpha and numeric elements, which
    is used as an index for sorting"
    """
    #
    # the index is built progressively
    # using the _append function
    #
    index = []
    def _append(fragment, alist=index):
        if fragment.isdigit(): fragment = int(fragment)
        alist.append(fragment)

    # initialize loop
    prev_isdigit = str[0].isdigit()
    current_fragment = ''
    # group a string into digit and non-digit parts
    for char in str:
        curr_isdigit = char.isdigit()
        if curr_isdigit == prev_isdigit:
            current_fragment += char
        else:
            _append(current_fragment)
            current_fragment = char
            prev_isdigit = curr_isdigit
    _append(current_fragment)
    return tuple(index)

class ImageLineSegment(object):
    def __init__(self, p1, p2, color=(255,255,255)):
        self.p1 = p1
        self.p2 = p2
        self.color = color

    def __repr__(self):
        return '[%d %d, %d %d]' % (self.p1[0], self.p1[1], self.p2[0], self.p2[1])


class SimilarSequence(object):
    """ Class that defines sequences of images that are deemed similar in a
        video or temporal stream of images.
        Sequences are stored in a list which contains frame paths.

    """
    def __init__(self, id="n_a", running_score=0., w=300, h=300):
        self.id = id
        self.running_score = running_score # running avg matching percentage
        self.sequence = []
        self.frame_dict = {}
        self.w = w
        self.h = h

    def get_seq_len(self):
        return len(self.sequence)

    def load_sequence(self, input_path):
        # logging.info("Loading sequence.")
        base_fname = os.path.basename(input_path)

        image_paths = []

        with open(input_path, "r") as annot_file:
            for cnt, line in enumerate(annot_file):
                line = line.strip()
                image_paths.append(line)

        self.sequence = image_paths
        return self.sequence

    def get_annotation_str(self):
        annotations = ""

        for frame_path in self.sequence:
            annotations += frame_path + "\n"

        return annotations

    def get_patch_annotation_str(self, patch):
        annotations = ""

        for patch_deets in patch:
            xmin = np.maximum(patch_deets[1].left, 0)
            ymin = np.maximum(patch_deets[1].top, 0)
            # print(patch_deets[1].right)
            # print(self.w)
            xmax = np.minimum(patch_deets[1].right, self.w)
            ymax = np.minimum(patch_deets[1].bottom, self.h)
            annotations += patch_deets[0] + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + "\n"

        return annotations

    def write_annotation_file(self, input_prefix, output_path):
        annot_fname = os.path.join(output_path, input_prefix + "_sequence_" + str(self.id) + ".txt")
        with open(annot_fname, "w") as annot_file:
            annot_file.write(self.get_annotation_str())

    def write_patches_annotation_file(self, input_prefix, output_path, min_sighting=4):
        annot_fname = os.path.join(output_path, input_prefix + "_sequence_" + str(self.id) )

        for patch_id, val in self.frame_dict.items():
            if len(val) >= min_sighting:
                fname = annot_fname + '_patch_' + patch_id + ".txt"
                with open(fname, "w") as annot_file:
                    annot_file.write(self.get_patch_annotation_str(val))

    def save_patches(self, input_prefix, output_path, min_sighting=4, patch_dims=(128,128)):
        for patch_id, val in self.frame_dict.items():
            if len(val) >= min_sighting:
                patch_dir_path = os.path.join(output_path, patch_id)

                for frame in val[:30]:
                    xmin = int(np.maximum(frame[1].left, 0))
                    ymin = int(np.maximum(frame[1].top, 0))
                    xmax = int(np.minimum(frame[1].right, self.w))
                    ymax = int(np.minimum(frame[1].bottom, self.h))

                    if (ymax-ymin) < patch_dims[0]/1.5 or (xmax-xmin) < patch_dims[1]/1.5:
                        continue

                    if not os.path.exists(patch_dir_path):
                        os.makedirs(patch_dir_path)

                    frame_fname = os.path.basename(frame[0])
                    frame_name, ext = os.path.splitext(frame_fname)
                    fname_path = os.path.join(patch_dir_path, frame_name + '_' + frame[1].label + '_patch.png')

                    try:
                        if os.path.isfile(frame[0]):
                            entire_image = cv2.imread(frame[0])
                            entire_image = cv2.resize(entire_image, frame[2], interpolation=cv2.INTER_AREA)
                            if entire_image is not None:
                                cropped_patch = entire_image[ymin:ymax, xmin:xmax]
                                if cropped_patch is not None and ymin < ymax and xmin < xmax:
                                    # cv2.imshow('cropped_patch', cropped_patch)
                                    # cv2.waitKey(0)
                                    cv2.imwrite(fname_path, cropped_patch)
                    except AssertionError as error:
                        # Output expected AssertionErrors.
                        logging.exception(error)
                    except Exception as exception:
                        # Output unexpected Exceptions.
                        logging.exception(exception)

                # cv2.destroyAllWindows()


class BBox(object):
  def __init__(self, left, top, right, bottom, kp_center=None, sightings=0, id="n_a", score=0., color=(0,0,255), label=None):
    self.id = id
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom
    self.label = label
    self.score = score
    self.color = color
    self.kp_center = kp_center
    self.sightings = sightings
    self.sighting_date = 0
    self.no_longer_occluded_count = 0
    self.occluded_count = 0
    self.appearance_models = None
    self.status_occluded = False
    self.status_sufficiently_occluded = False
    self.status_detected = True  # detected (true), tracked (false)
    self.projection_anchor = None

  def __repr__(self):
    return '%s %d %d %d %d' % (self.label, self.left, self.top, self.right, self.bottom)

  def get_projection_anchor(self):
    if self.projection_anchor is None:
      return self.center()
    return self.projection_anchor

  def top_left(self):
    return int(self.left), int(self.top)

  def bottom_right(self):
    return int(self.right), int(self.bottom)

  def top_center(self):
    x = int((self.left + self.right) / 2.)
    y = self.top
    return x,y

  def center(self):
    x = int((self.left + self.right) / 2.)
    y = int((self.top + self.bottom) / 2.)
    return x,y

  def center_dist(self, other):
    return np.linalg.norm(np.array(self.center()) - np.array(other.center()))

  def width(self):
    return self.right - self.left

  def height(self):
    return self.bottom - self.top

  def size(self):
    return self.width() * self.height()

  def lines(self, color=None):
    if not color:
      color = self.color

    l = [
      ImageLineSegment([self.left, self.top], [self.right, self.top], color),
      ImageLineSegment([self.left, self.bottom], [self.right, self.bottom], color),
      ImageLineSegment([self.left, self.top], [self.left, self.bottom], color),
      ImageLineSegment([self.right, self.top], [self.right, self.bottom], color)
    ]
    return l

  def similarity_helper(self, bbox, edge_diff):
    # smaller number is more similar
    if type(bbox) is type(None):
      return sys.maxint # Very dissimilar
    abs_diff = np.absolute(edge_diff)
    return np.mean(abs_diff)

  def similarity(self, bbox):
    edge_diff = np.array([
      self.left - bbox.left,
      self.right - bbox.right,
      self.top - bbox.top,
      self.bottom - bbox.bottom,
    ])
    return self.similarity_helper(bbox, edge_diff)

  def compute_appearance_model(self, img):
    mask = mask_from_bbox(self, img.shape)
    kp, des = orb.detectAndCompute(img, mask)
    self.appearance_models = (kp, des)

  def bottom_less_sim(self, bbox):
    edge_diff = np.array([
      self.left - bbox.left,
      self.right - bbox.right,
      self.top - bbox.top,
    ])
    return self.similarity_helper(bbox, edge_diff)

  def horizontal_overlap_ratio(self,
                               bbox,
                               displacement=0.0):

    x1_min, y1_min = self.top_left()
    x1_min += displacement
    x1_max, y1_max = self.bottom_right()
    x1_max += displacement
    x2_min, y2_min = bbox.top_left()
    x2_max, y2_max = bbox.bottom_right()

    w1 = math.fabs(x1_max - x1_min)
    w2 = math.fabs(x2_max - x2_min)

    overlap = max(0, min(x1_max, x2_max)-max(x1_min, x2_min))
    intersection = overlap
    union = w1 + w2 - overlap
    iou = intersection / union

    return iou

  def vertical_overlap_ratio(self,
                               bbox,
                               displacement=0.0):

    x1_min, y1_min = self.top_left()
    y1_min += displacement
    x1_max, y1_max = self.bottom_right()
    y1_max += displacement
    x2_min, y2_min = bbox.top_left()
    x2_max, y2_max = bbox.bottom_right()

    w1 = math.fabs(y1_max - y1_min)
    w2 = math.fabs(y2_max - y2_min)

    overlap = max(0, min(y1_max, y2_max)-max(y1_min, y2_min))
    intersection = overlap
    union = w1 + w2 - overlap
    iou = intersection / union

    return iou

  def bottom_edge_diff(self,
                       bbox):

    diff = abs(self.bottom - bbox.bottom)

    return diff

  def overlap_ratio(self, bbox):
    x1_ul, y1_ul = self.top_left()
    x1_br, y1_br = self.bottom_right()
    x2_ul, y2_ul = bbox.top_left()
    x2_br, y2_br = bbox.bottom_right()

    a1 = math.fabs(x1_br - x1_ul) * math.fabs(y1_br - y1_ul)
    a2 = math.fabs(x2_br - x2_ul) * math.fabs(y2_br - y2_ul)
    overlap = max(0, min(x1_br, x2_br) - max(x1_ul, x2_ul)) * max(0, min(y1_br, y2_br) - max(y1_ul, y2_ul))

    overlap_ratio = overlap / (a1 + a2 - overlap)
    return overlap_ratio

  def contains(self, bbox, offset=15):
    if bbox.left+offset > self.left and bbox.right-offset < self.right and bbox.top+offset > self.top and bbox.bottom-offset < self.bottom:
        return True
    else:
        return False

  def outside(self, frame, offset=100):
    # Check if bbox is outside frame on the horizontal axis:
    if self.left+offset < frame.left or self.right-offset > frame.right or self.top+offset < frame.top or self.bottom-offset > frame.bottom:
        return True
    else:
        return False
