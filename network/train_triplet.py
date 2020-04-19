"""
Python script to train (siamese with triplet loss)

"""

# BASIC
import os
import glob
import random
import argparse
import itertools
import pickle as pkl
from datetime import datetime

# LOGGING
from utils.logging_config import *

# VISION & ML
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from network.layers.losses import *
from network.layers.metrics import *
from tensorflow.keras.optimizers import Adam
from generators.similar_trios_generator import *
from tensorflow.keras.callbacks import TensorBoard
from network.siamese_nets import SiameseNetTriplet

BATCH_SIZE = 64
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
RANDOM_SEED = 42


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def train(train_lbld_trios, val_lbls_trios, network, weights, model_path, n_epochs, init_lr, optmzr_name, imagenet=False, freeze_until=None):
    """Training function: train a model of type 'network' over the data.
    Args:
        network (str): String identifying the network architecture to use.
        weights (str): Path string to a .cpkt weights file.
        model_path (str): Path string to a directory to save models in.
        n_epochs (int): Integer representing the number of epochs
                        to run training.
    """

    # Create a folder for saving trained models
    if os.path.isdir(model_path) is False:
        logging.info("Creating a folder to save models at: " + str(model_path))
        os.mkdir(model_path)

    starting_epoch = 0

    if network == 'SiameseNetTriplet':
        siamese_net = SiameseNetTriplet((128,128,3), arch='resnet18', sliding=True, imagenet=imagenet, freeze_until=freeze_until)
        optimizer = Adam(lr = 0.0006)
        model = siamese_net.build_model()
        loss_model = siamese_net.loss_model
        single_model = siamese_net.single_model

        if weights:
            print("Loading model at: " + str(weights))
            starting_epoch = int(weights.split('-')[1]) + 1
            model.load_weights(weights)

        model.compile(loss=triplet_loss, optimizer=optimizer, metrics=[cos_sim_pos, cos_sim_neg])

    # Load data and create data generator:
    train_ds = tf.data.Dataset.from_generator(
                image_trio_generator,
                args=[train_lbld_trios, True, False, False, imagenet],
                output_types=((tf.float32, tf.float32, tf.float32), tf.float32, tf.string),
                output_shapes=(((TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3)), (1, None), (3)))

    batched_train_ds = train_ds.batch(BATCH_SIZE) # shuffle(10000).batch

    valid_ds = tf.data.Dataset.from_generator(
                image_trio_generator,
                args=[val_lbld_trios, False, False, False, imagenet],
                output_types=((tf.float32, tf.float32, tf.float32), tf.float32, tf.string),
                # output_shapes=(((TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3)), (1, None)))
                output_shapes=(((TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3), (TARGET_WIDTH, TARGET_HEIGHT, 3)), (1, None), (3)))


    batched_valid_ds = valid_ds.batch(BATCH_SIZE)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                       histogram_freq=0,
                                       batch_size=BATCH_SIZE,
                                       write_graph=True,
                                       write_grads=True)

    tensorboard_callback.set_model(model)

    def named_logs(metrics_names, logs):
        result = {}
        for l in zip(metrics_names, logs):
            result[l[0]] = l[1]
        return result

    # Train model:
    steps_per_epoch = len(train_lbld_trios)//BATCH_SIZE
    val_steps_per_epoch =  len(val_lbld_trios)//BATCH_SIZE # steps_per_epoch//3
    best_val_loss = 1000
    best_train_loss = 1000

    batched_train_iter = iter(batched_train_ds)
    batched_val_iter = iter(batched_valid_ds)
    # batched_train_iter = batched_train_ds.make_one_shot_iterator()
    # batched_val_iter = batched_valid_ds.make_one_shot_iterator()

    for epoch in range(starting_epoch, n_epochs):

        cumm_csim_pos_tr = 0
        cumm_csim_neg_tr = 0
        cumm_tr_loss = 0
        cumm_csim_pos_val = 0
        cumm_csim_neg_val = 0
        cumm_val_loss = 0

        print('Epoch #' + str(epoch) + ':')

        for step in tqdm(range(steps_per_epoch)):
            train_inputs, train_y, train_seq_ids = next(batched_train_iter)
            #train_inputs, train_y = batched_train_iter.get_next()
            # tinrain_x1 = train_inputs['input_1']
            # train_x2 = train_inputs['input_2']
            train_x1 = train_inputs[0]
            train_x2 = train_inputs[1]
            train_x3 = train_inputs[2]
            # train_y = train_y['output']

            X_dict = {}
            seq_ids = []
            for idx, row in enumerate(train_seq_ids):
                for idx2, class_id in enumerate(row):
                    class_id = class_id.numpy().decode('utf8')
                    if class_id not in seq_ids:
                        seq_ids.append(class_id)
                    if class_id in X_dict:
                        X_dict[class_id].append(train_inputs[idx2][idx])
                    else:
                        X_dict[class_id] = []
                        X_dict[class_id].append(train_inputs[idx2][idx])


            triplets = get_batch_hard(model, train_inputs, seq_ids, BATCH_SIZE)
            loss, csim_pos_tr, csim_neg_tr = model.train_on_batch([triplets[0], triplets[1], triplets[2]], train_y[:BATCH_SIZE//2])
            cumm_tr_loss += loss
            cumm_csim_pos_tr += csim_pos_tr
            cumm_csim_neg_tr += csim_neg_tr

        cumm_csim_pos_tr = cumm_csim_pos_tr/steps_per_epoch
        cumm_csim_neg_tr = cumm_csim_neg_tr/steps_per_epoch
        cumm_tr_loss = cumm_tr_loss/steps_per_epoch

        # evaluate
        for step in tqdm(range(val_steps_per_epoch)):
            valid_inputs, val_y, val_seq_ids = next(batched_val_iter)
            # valid_inputs, valid_y = batched_val_iter.get_next()
            valid_x1 = valid_inputs[0]
            valid_x2 = valid_inputs[1]
            valid_x3 = valid_inputs[2]
            val_loss, csim_pos_val, csim_neg_val = model.test_on_batch([valid_x1, valid_x2, valid_x3], val_y)

            cumm_val_loss += val_loss
            cumm_csim_pos_val += csim_pos_val
            cumm_csim_neg_val += csim_neg_val

        cumm_csim_pos_val = cumm_csim_pos_val/val_steps_per_epoch
        cumm_csim_neg_val = cumm_csim_neg_val/val_steps_per_epoch
        cumm_val_loss = cumm_val_loss/val_steps_per_epoch

        print('Training loss: ' + str(cumm_tr_loss))
        print('Validation loss: ' + str(cumm_val_loss))
        print('* Cosine sim positive (train) for this epoch: %0.2f' % (cumm_csim_pos_tr))
        print('* Cosine sim negative (train) for this epoch: %0.2f' % (cumm_csim_neg_tr))
        print('* Cosine sim positive (valid) for this epoch: %0.2f' % (cumm_csim_pos_val))
        print('* Cosine sim negative (valid) for this epoch: %0.2f' % (cumm_csim_neg_val))

        metrics_names = ['tr_loss', 'tr_csim_pos', 'tr_csim_neg', 'val_loss', 'val_csim_pos', 'val_csim_neg']
        tensorboard_callback.on_epoch_end(epoch, named_logs(metrics_names, [cumm_tr_loss, cumm_csim_pos_tr, cumm_csim_neg_tr, cumm_val_loss, cumm_csim_pos_val, cumm_csim_neg_val]))

        model_filepath = os.path.join(model_path, "model-{epoch:03d}-{val_loss:.4f}.hdf5".format(epoch=epoch, val_loss=cumm_val_loss))

        if cumm_val_loss < best_val_loss*1.5:
            if cumm_val_loss < best_val_loss:
                best_val_loss = cumm_val_loss
            model.save(model_filepath) # OR model.save_weights()
            print("Best model w/ val loss {} saved to {}".format(cumm_val_loss, model_filepath))

    tensorboard_callback.on_train_end(None)

    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training script for learning visual similarity.')
    parser.add_argument('-p','--path-list',
                        dest='list_of_paths',
                        type=str,
                        nargs='+',
                        help='<Required> Paths to dirs or lists (.txt) of directories containing patches (e.g: /path/to/list1.txt /path/to/dir /path/to/list2.txt)',
                        required=True)
    parser.add_argument('-e','--tracked-list',
                        dest='list_tracked_vs_random',
                        type=str,
                        nargs='+',
                        help='<Required> list of 0/1 to indicate if patches were feature tracked (1) or randomly extracted (0) (e.g: 0 1 1). Must match number of different input paths were provided.',
                        required=True)
    parser.add_argument('-w', '--weights',
                        dest='weights',
                        help='Keras model',
                        default=None)
    parser.add_argument('-mp', '--model-path',
                        help='Path to save learned models in',
                        dest='model',
                        default='./models')
    parser.add_argument('-nw', '--network',
                        help='Network name',
                        dest='network',
                        default='SiameseNetTriplet')
    parser.add_argument('-ne', '--n-epochs',
                        help='Number of epochs',
                        dest='n_epochs',
                        type=int,
                        default=100)
    parser.add_argument('-g', '--gpu',
                        dest='gpu',
                        default='0')
    parser.add_argument('-lr', '--learning-rate',
                        dest='init_lr',
                        type=float,
                        default='0.001')
    parser.add_argument('-fu', '--freeze-until',
                        dest='freeze_until',
                        type=int,
                        default=None)
    parser.add_argument('-o', '--optimizer',
                        dest='optimizer',
                        default='adam')
    parser.add_argument('-i --imagenet',
                        dest='imagenet',
                        action='store_true')
    parser.add_argument('-d', '--dataset',
                        dest='dataset',
                        help='underwater or scott',
                        default=None)
    parser.add_argument('-c --curated',
                        dest='curated',
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    logging.info("Dataset: " + str(args.dataset))

    curated = args.curated

    if not args.dataset:
        logging.error("Dataset type not specified.")
        exit()

    PATHS_LIST = args.list_of_paths
    TRACKED_VS_RANDOM_LIST = args.list_tracked_vs_random

    if len(PATHS_LIST) != len(TRACKED_VS_RANDOM_LIST):
        logging.error("Length of paths and tracked_vs_random list do not match. Please provide a '1' or '0' for each input path provided.")
        exit()

    INPUT_PATHS = {}

    for idx, path in enumerate(PATHS_LIST):
        if os.path.isdir(path):
            patches_path = os.path.join(path, 'patches')

            if not os.path.isdir(patches_path):
                logging.error('Patches folder is missing from this directory.')
            else:
                # directory of patches
                INPUT_PATHS[idx] = [patches_path]

        else:
            # video or .txt containing list of videos
            base_name = os.path.basename(path)
            fname, ext = os.path.splitext(base_name)

            if ext == '.txt':
                dir_list = []
                with open(path, "r") as dir_list_file:
                    for cnt, line in enumerate(dir_list_file):
                        line = line.strip()
                        if os.path.isdir(line):
                            patches_path = os.path.join(line, 'patches')
                            if not os.path.isdir(patches_path):
                                logging.error('Patches folder is missing from this directory: ' + str(patches_path))
                            else:
                                # directory of patches
                                dir_list.append(patches_path)

                INPUT_PATHS[idx] = dir_list

    cluster_ids = list(INPUT_PATHS.keys())
    cluster_params = {}
    for cluster_id in cluster_ids:
        cluster_params[cluster_id] = int(TRACKED_VS_RANDOM_LIST[cluster_id])

    tracked_vs_random_dict = {1: 'tracked', 0: 'random'}

    # split video list into train/val (0.7/0.3) here:
    train_paths = {}
    val_paths = {}

    print("\nList of directories to process: ")
    for cluster_id, paths in INPUT_PATHS.items():
        print('    Cluster ' + str(cluster_id) + ' (' + str(tracked_vs_random_dict[cluster_params[cluster_id]]) + '):')
        train_val_cutoff = int(np.ceil(len(paths)*0.7))
        train_paths_list = []
        val_paths_list = []
        random.shuffle(paths)
        random.shuffle(paths)
        for idx, input_path in enumerate(paths):
            if curated:
                train_paths_list.append(input_path)
                print('        ' + str(idx) + '. ' + str(input_path))
            else:
                if idx < train_val_cutoff:
                    # training
                    train_paths_list.append(input_path)
                    print('        ' + str(idx) + '. ' + str(input_path) + ' [train]')
                else:
                    # validation
                    val_paths_list.append(input_path)
                    print('        ' + str(idx) + '. ' + str(input_path) + ' [valid]')

        train_paths[cluster_id] = train_paths_list
        val_paths[cluster_id] = val_paths_list

    print("\n")

    if args.dataset == 'underwater':
        train_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=2500, num_patch_folders=1000, curated=False)
        val_trios = build_lbld_trios(val_paths, cluster_params, max_neg_folder_pairs=2500, num_patch_folders=1000, curated=False)
    elif args.dataset == 'scott':
        train_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=2500, num_patch_folders=2000, curated=False)
        val_trios = build_lbld_trios(val_paths, cluster_params, max_neg_folder_pairs=2500, num_patch_folders=2000, curated=False)
    elif args.dataset == 'underwater_curated':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'scott_curated':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'underwater_agglomerative':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=3, num_patch_folders=6, curated=True)
    elif args.dataset == 'underwater_agglom_merged':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'scott_agglom_merged':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=50, num_patch_folders=50, curated=True)
    elif args.dataset == 'scott_agglomerative':
        all_trios = build_lbld_trios(train_paths, cluster_params, max_neg_folder_pairs=3, num_patch_folders=6, curated=True)


    train_lbld_trios = []
    val_lbld_trios = []

    num_examples = 10000000
    if curated:
        random.shuffle(all_trios)

        # Only take a subset for now:
        if len(all_trios) > num_examples:
            all_trios = all_trios[:num_examples]

        train_trios = all_trios[:int(len(all_trios)*0.7)]
        val_trios = all_trios[int(len(all_trios)*0.7):]

    else:
        random.shuffle(train_trios)
        random.shuffle(val_trios)

        # Only take a subset for now:
        if len(train_trios) > int(num_examples*0.7):
            train_trios = train_trios[:int(num_examples*0.7)]
        if len(val_trios) > int(len(train_trios)*0.3):
            val_trios = val_trios[:int(len(train_trios)*0.3)]

    train_lbld_trios.extend(train_trios)
    val_lbld_trios.extend(val_trios)

    print("[TRAIN] Number of trios: " + str(len(train_trios)))
    print("[VAL] Number of trios: " + str(len(val_trios)))
    random.shuffle(train_lbld_trios)
    random.shuffle(val_lbld_trios)

    # Optional, vizualize the trios
    visualize_trios(train_lbld_trios, normalized=False)
    visualize_trios(val_lbld_trios, normalized=False)

    train(train_lbld_trios,
          val_lbld_trios,
          args.network,
          args.weights,
          args.model,
          args.n_epochs,
          args.init_lr,
          args.optimizer,
          imagenet=args.imagenet,
          freeze_until=args.freeze_until)
