"""
Python script to get cluster accuracy and confusion matrices

"""

# BASIC
import os
import glob
import random
import argparse
import itertools
import pickle as pkl
from datetime import datetime
random.seed(42)

# Logging
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
from utils.vis_utils import visualize_pairs, generate_heatmaps

import tensorflow as tf
from tensorflow.keras import Model
from network.layers.losses import *
from network.layers.metrics import *
from tensorflow.keras import backend as K
from network.siamese_nets import SiameseNetTriplet
from classification_models.tfkeras import Classifiers
from generators.similar_trios_generator import image_trio_generator, build_lbld_trios

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def setup_network(weights, network, sliding=True, imagenet=False, freeze_until=None):
    """ Function to setup model and load its weights
        Args:
            network (str): String identifying the network architecture to use.
            weights (str): Path string to a .h5 weights file.
    """

    model = None
    siamese_net = None

    if network == 'triplet':
        if imagenet:
            print("    Loading imagenet weights")
        else:
            print("    Not loading imagenet weights")
        siamese_net = SiameseNetTriplet((None,None,3), arch='resnet18', sliding=sliding, imagenet=imagenet, freeze_until=freeze_until)
        model, vector_model = siamese_net.setup_weights(weights=weights, lr=0.0006, optimizer='adam')
        return siamese_net
    elif network == 'cvae':
        print("Loading cvae model")
        net_128, net_512 = setup_cvae(weights)
        return net_128

def cluster(ref_path, test_path, pred_path, weights, network='triplet', ref_patch_num=5, visualize=False, sliding=True, imagenet=False, freeze_until=None, finetuning=False, subset=False):
    """Testing function.
    Args:
        network (str): String identifying the network architecture to use.
        weights (str): Path string to a .h5 weights file.
    """
    if imagenet:
        finetuning = True

    prediction_net = setup_network(weights, network, sliding=sliding, imagenet=imagenet, freeze_until=freeze_until)
    patch_shape = (128, 128)
    patch_w = 128
    patch_h = 128

    annotations = dict()

    if network == 'triplet':
        siamese_net = prediction_net
        siamese_single_model = siamese_net.single_model


    classes = dict()
    class_names = []

    # Parse reference patches and their corresponding classes:
    patch_dirs = os.listdir(ref_path)
    print("Reference patch directories: " + str(patch_dirs))
    ref_dirs = dict()

    for path in patch_dirs:
        if os.path.isdir(os.path.join(ref_path, path)):
            class_id = path.split('_')[0]
            class_name = path.split('_')[1]
            class_names.append(path)
            classes[int(class_id)] = class_name
            ref_dirs[int(class_id)] = (class_name, os.path.join(ref_path, path))

    test_patch_dirs = os.listdir(test_path)
    print("Test patch directories: " + str(test_patch_dirs))
    test_dirs = dict()

    test_data = []

    for path in test_patch_dirs:
        extended_path = os.path.join(path, 'patches')
        if os.path.isdir(os.path.join(test_path, extended_path)):
            class_id = path.split('_')[0]
            class_name = path.split('_')[1]

            if int(class_id) not in classes.keys():
                continue

            test_dirs[int(class_id)] = (class_name, os.path.join(test_path, extended_path))

            test_images = sorted(glob.glob(os.path.join(os.path.join(test_path, extended_path), '*.png')))
            print("Number of images in class " + class_name + ": ", len(test_images))

            if subset:
                print("Only keeping 10 images per class")
                test_images = random.sample(test_images, 10)

            for test_img in test_images:
                test_data.append((test_img, class_id))

    class_count = len(patch_dirs)
    print("Class count: " + str(class_count))

    # Make prediction folders:
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y-%m-%d_%H%M%S")
    pred_path = os.path.join(pred_path, 'RPN' + str(ref_patch_num) + '_' + dt_string)

    for class_id, class_name in classes.items():
        os.makedirs(os.path.join(pred_path, str(class_id) + '_' + class_name))

    ref_patches = dict()
    ref_patches_desc = dict()

    print("Loading reference patches + computing their descriptors")
    for class_id, val in ref_dirs.items():
        class_name = val[0]
        path = val[1]
        patches = sorted(glob.glob(os.path.join(path, '*.png')))

        patches = random.sample(patches, ref_patch_num)

        # patches = patches[:ref_patch_num]

        if len(patches) > 0:
            ref_patches[class_id] = patches
            ref_patches_desc[class_id] = []

            for patch in patches:
                print('\t' + patch)
                ref_patch = cv2.imread(patch)
                ref_patch = cv2.resize(ref_patch, (128, 128))
                # cv2.imshow("ref_patch" + "_" + class_name, ref_patch)
                # k2 = cv2.waitKey(0)
                cv2.imwrite(os.path.join(pred_path, os.path.basename(patch)), ref_patch)

                ref_patch_batched = np.zeros((1, 128, 128, 3))
                if not finetuning:
                    ref_patch_batched[0] = ref_patch/255.
                else:
                    ResNet18, preprocess_input = Classifiers.get('resnet18')
                    ref_patch_batched[0] = preprocess_input(ref_patch)

                if network == 'triplet':
                    ref_patches_desc[class_id].append(siamese_single_model.predict(ref_patch_batched))

        cv2.destroyAllWindows()
    print("Done")

    pred_data = []
    y_true = []
    y_pred = []

    for test_idx, (test_img_path, label_id) in enumerate(test_data):
        if test_idx % 25 == 0:
            print("Processed " + str(test_idx) + "/" + str(len(test_data)))
        label_id = int(label_id)
        test_patch = cv2.imread(test_img_path)
        test_patch = cv2.resize(test_patch, (128, 128))
        test_patch_batched = np.zeros((1, 128, 128, 3))

        if not finetuning:
            test_patch_batched[0] = test_patch/255.
        else:
            ResNet18, preprocess_input = Classifiers.get('resnet18')
            test_patch_batched[0] = preprocess_input(test_patch)

        if network == 'triplet':
            test_patch_desc = siamese_single_model.predict(test_patch_batched)

        similarity = dict()
        similarity_mean = dict()
        for class_id, ref_descs in ref_patches_desc.items():
            # print("Comparing against class ", class_id)
            similarity[class_id] = []
            for idx, ref_desc in enumerate(ref_descs):
                anchor_vec = test_patch_desc[0].flatten()
                vec_a_norm = np.linalg.norm(anchor_vec)
                if vec_a_norm != 0:
                    anchor_vec = anchor_vec/vec_a_norm

                ref_vec = ref_desc[0].flatten()
                vec_p_norm = np.linalg.norm(ref_vec)
                if vec_p_norm != 0:
                    ref_vec = ref_vec/vec_p_norm

                dot_prod = np.dot(anchor_vec, ref_vec.T)
                eval_csim = cos_sim_pos(None, [anchor_vec, ref_vec], concat=False)
                # print(dot_prod)
                # print(np.array(np.abs(eval_csim)))
                similarity[class_id].append(dot_prod)

        # print("Summarize prediction for this image: ")
        highest = 0
        best_match_class = 0

        test_img_basename = os.path.basename(test_img_path)
        annotations['main_dir'] = pred_path
        annotations[test_img_basename] = dict()


        for class_id, dot_prods in similarity.items():
            similarity_mean[class_id] = np.mean(np.array(dot_prods))
            # print("Class " + str(class_id) + " mean dot prod: " + str(similarity_mean[class_id]))

            if similarity_mean[class_id] > highest:
                highest = similarity_mean[class_id]
                best_match_class = class_id

            # build annotation string
            annotations[test_img_basename][class_id] = similarity_mean[class_id]

        annotations[test_img_basename]['best_match_class'] = (best_match_class, classes[best_match_class])

        # print("Best match predicted class label: " + str(best_match_class))
        # print("Actual gt label: " + str(label_id))
        output_img_path = os.path.join(pred_path, str(best_match_class) + "_" + classes[best_match_class], os.path.basename(test_img_path))
        cv2.imwrite(output_img_path, test_patch)

        pred_data.append((test_img_path, best_match_class))
        y_true.append(label_id)
        y_pred.append(best_match_class)

    with open(os.path.join(pred_path, "RPN" +str(ref_patch_num) + "_similarity_annotations.pkl"), "wb") as handle:
        pkl.dump(annotations, handle, protocol=pkl.HIGHEST_PROTOCOL)

    target_names = sorted(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(classes.keys())))
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Confusion matrix: ")
    print(cm)


    return


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Clustering accuracy/performance.')
    parser.add_argument('-r','--ref-dir',
                        dest='refs_dir',
                        type=str,
                        help='<Required> Path to dir containing reference patches (e.g: /path/to/ref_patches/imgs1/)',
                        required=True)
    parser.add_argument('-d','--test-data',
                        dest='test_dir',
                        type=str,
                        help='<Required> Path to dir containing test clusters',
                        required=True)
    parser.add_argument('-o','--pred-data',
                        dest='pred_dir',
                        type=str,
                        help='<Required> Path to dir that will contain predictions',
                        required=True)
    parser.add_argument('-w', '--weights',
                        dest='weights',
                        help='Keras model',
                        default=None)
    parser.add_argument('-nw', '--network',
                        help='Network name',
                        dest='network',
                        default='triplet')
    parser.add_argument('-v --visualize',
                        dest='visualize',
                        action='store_true')
    parser.add_argument('-i --imagenet',
                        dest='imagenet',
                        action='store_true')
    parser.add_argument('-fu', '--freeze-until',
                        dest='freeze_until',
                        type=int,
                        default=None)
    parser.add_argument('-ft --finetuning',
                        dest='finetuning',
                        action='store_true')
    parser.add_argument('-ss --subset',
                        dest='subset',
                        action='store_true')
    parser.add_argument('-rpn','--num-ref-patch',
                        dest='ref_patch_num',
                        type=int,
                        help='<Required> Number of references to compare against per class',
                        required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    REF_PATH = args.refs_dir
    TEST_PATH = args.test_dir

    print("Reference directory: " + str(REF_PATH))
    print("Test data directory: " + str(TEST_PATH))

    cluster(REF_PATH,
            TEST_PATH,
            args.pred_dir,
            args.weights,
            network=args.network,
            ref_patch_num=args.ref_patch_num,
            visualize=args.visualize,
            imagenet=args.imagenet,
            freeze_until=args.freeze_until,
            finetuning=args.finetuning,
            subset=args.subset)
