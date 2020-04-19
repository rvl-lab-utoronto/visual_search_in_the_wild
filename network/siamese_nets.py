"""
Python module defining our triplet siamese architecture.

"""

# BASIC
import time
import numpy as np
import pickle as pkl

# VIZ
from utils.vis_utils import *

# LOGGING
from utils.logging_config import *

# ML
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Multiply, Layer, Dot
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, ReLU, Input
from tensorflow.keras.layers import Lambda, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from classification_models.tfkeras import Classifiers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from network.layers.losses import *
from network.layers.metrics import *


# output dimensionality for various architectures
OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              : 512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'mnist_siamese'         : 256,
    'mobilenetv2'           : 1280,
}


def extract_features(input_shape, name='', archi='resnet18', imagenet=False, freeze_until=None):

    input = Input(shape=input_shape)

    if archi == 'resnet18':
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        if imagenet:
            base_model = ResNet18(input_tensor=input, input_shape=input_shape, weights='imagenet', include_top=False, version='v2')
        else:
            base_model = ResNet18(input_tensor=input, input_shape=input_shape, weights=None, include_top=False, version='v2')

        if freeze_until:
            print("Freezing network until layer: " + str(freeze_until))
            for layer in base_model.layers[:-freeze_until]:
                layer.trainable = False
            for layer in base_model.layers[-freeze_until:]:
                layer.trainable = True
        else:
            for idx, layer in enumerate(base_model.layers):
                layer.trainable = True

        x = base_model.output

    elif archi == 'mobilenetv2':
        MobNetV2, preprocess_input = Classifiers.get('mobilenetv2')
        if imagenet:
            base_model = MobNetV2(input_tensor=input, input_shape=input_shape, weights='imagenet', include_top=False)
        else:
            base_model = MobNetV2(input_tensor=input, input_shape=input_shape, weights=None, include_top=False)

        if freeze_until:
            for layer in base_model.layers[:-freeze_until]:
                layer.trainable = False
            for layer in base_model.layers[-freeze_until:]:
                layer.trainable = True
        else:
            for idx, layer in enumerate(base_model.layers):
                layer.trainable = True

        x = base_model.output

    return Model(input, x, name=name)


class SiameseNetTriplet():
    def __init__(self, input_shape, arch='resnet18', sliding=True, imagenet=False, freeze_until=None):
        self.input_shape = input_shape
        self.output_dim = OUTPUT_DIM[arch]
        self.arch = arch
        self.sliding = sliding
        self.imagenet = imagenet
        self.freeze_until = freeze_until

    def build_model(self):
        input_shape = self.input_shape

        if self.arch == 'resnet18':
            self.features = extract_features(input_shape, name='feats_extractor', archi='resnet18', imagenet=self.imagenet, freeze_until=self.freeze_until)
        elif self.arch == 'mobilenetv2':
            self.features = extract_features(input_shape, name='feats_extractor', archi='mobilenetv2', imagenet=self.imagenet, freeze_until=self.freeze_until)

        input_a = Input(shape=input_shape, name='input_a')
        input_b = Input(shape=input_shape, name='input_b')
        input_c = Input(shape=input_shape, name='input_c')

        features_a = self.features(input_a)
        features_b = self.features(input_b)
        features_c = self.features(input_c)

        if self.sliding:

            print("Setting up network [sliding window]")
            anchor_vec = Flatten(name="anchor_vec")(features_a)
            pos_vec = Flatten(name="pos_vec")(features_b)
            neg_vec = Flatten(name="neg_vec")(features_c)

            anchor_vec = Lambda(K.l2_normalize, name="anchor_vec_l2")(anchor_vec)
            pos_vec = Lambda(K.l2_normalize, name="pos_vec_l2")(pos_vec)
            neg_vec = Lambda(K.l2_normalize, name="neg_vec_l2")(neg_vec)

            merged_vecs = Concatenate(axis=1, name='vectors')([anchor_vec, pos_vec, neg_vec])
            loss = triplet_loss(None, merged_vecs)

            self.model = Model(inputs=[input_a, input_b, input_c], outputs=merged_vecs, name='siamese_network_triplet')
            self.loss_model = Model(inputs=[input_a, input_b, input_c], outputs=loss, name='siamese_network_triplet_loss')
            self.vector_model = Model(inputs=self.model.input, outputs=[anchor_vec, pos_vec, neg_vec], name='vector_model_triplet')
            self.single_model = Model(inputs=[input_a], outputs=[features_a], name='single_model')
            return self.model

        else:
            print("Setting up network [regular]")
            anchor_vec = Flatten(name="anchor_vec")(features_a)
            pos_vec = Flatten(name="pos_vec")(features_b)
            neg_vec = Flatten(name="neg_vec")(features_c)

            anchor_vec = Lambda(K.l2_normalize, name="anchor_vec_l2")(anchor_vec)
            pos_vec = Lambda(K.l2_normalize, name="pos_vec_l2")(pos_vec)
            neg_vec = Lambda(K.l2_normalize, name="neg_vec_l2")(neg_vec)

            self.model = Model(inputs=[input_a, input_b, input_c], outputs=[features_a, features_b, features_c], name='siamese_network_triplet')
            self.vector_model = Model(inputs=self.model.input, outputs=[anchor_vec, pos_vec, neg_vec], name='vector_model_triplet')
            self.single_model = Model(inputs=[input_a], outputs=[features_a], name='single_model')
            return self.model

    def setup_weights(self, weights=None, lr=0.0006, optimizer='adam'):
        optimizer = Adam(lr=lr)
        model = self.build_model()

        if weights:
            print("Loading model at: " + str(weights))
            starting_epoch = int(weights.split('-')[1]) + 1
            try:
                model.load_weights(weights)
            except Exception as e:
                print (e)

        else:
            logging.warning("Couldn't find a weights/model file to load.")

        model.compile(loss=triplet_loss, optimizer=optimizer, metrics=[cos_sim_pos, cos_sim_neg])
        print("Compiled model")

        self.model = model
        return (self.model, self.vector_model)

    def slide_descriptors(self, img_vec_1, img_vec_2):
        # slide patch descriptor over a larger image descriptor
        # and compute inner product to compute a heatmap
        pred_dict = dict()

        flattened_vec_2 = img_vec_2[0].flatten()
        vec2_norm = np.linalg.norm(flattened_vec_2)
        ref_vec_normd = flattened_vec_2
        if vec2_norm != 0:
            ref_vec_normd = flattened_vec_2/vec2_norm

        # example: 16x16 image descriptor, 4x4 patch descriptor
        bounds_y = (int(img_vec_2.shape[1]/2), int(img_vec_1.shape[1]-img_vec_2.shape[1]/2+1))
        bounds_x = (int(img_vec_2.shape[2]/2), int(img_vec_1.shape[2]-img_vec_2.shape[2]/2+1))
        orig_img_vec_1 = img_vec_1

        t4 = time.time()
        for y in range(bounds_y[0], bounds_y[1]):
            for x in range(bounds_x[0], bounds_x[1]):
                flattened_subvec = img_vec_1[0, y-bounds_y[0]:y+bounds_y[0], x-bounds_x[0]:x+bounds_x[0], :].flatten()
                subvec_norm_val = np.linalg.norm(flattened_subvec)
                subvec_normd = flattened_subvec
                if subvec_norm_val != 0:
                    subvec_normd = flattened_subvec/subvec_norm_val
                t2 = time.time()
                pred_dot = np.dot(subvec_normd, ref_vec_normd.T)
                t3 = time.time()

                pred_label = 0
                pred_dict[(x, y)] = (pred_label, pred_dot)

        t5 = time.time()
        return pred_dict, (orig_img_vec_1.shape[2], orig_img_vec_1.shape[1])

    def forward_pass_single(self, input_img):
        single_model = self.single_model
        test_x1 = input_img[0]
        img_vec_1 = single_model.predict(input_img)

        return img_vec_1


    def get_similarity(self, inputs, visualize=False):
        model = self.model
        single_model = self.single_model

        test_a = inputs[0]
        test_p = inputs[1]

        patch_w = test_p.shape[2]
        patch_h = test_p.shape[1]

        if self.sliding:
            # in sliding window mode, comparing patch to patch
            anchor_vec = single_model.predict(test_a)
            pos_vec = single_model.predict(test_p)

            anchor_vec = anchor_vec[0].flatten()
            vec_a_norm = np.linalg.norm(anchor_vec)
            if vec_a_norm != 0:
                anchor_vec = anchor_vec/vec_a_norm

            pos_vec = pos_vec[0].flatten()
            vec_p_norm = np.linalg.norm(pos_vec)
            if vec_p_norm != 0:
                pos_vec = pos_vec/vec_p_norm

            dot_prod_p = np.dot(anchor_vec, pos_vec.T)

            return dot_prod


        else:
            # comparing full images to patches, need to slide descriptors
            pred_dict = dict()

            img_vec_1 = single_model.predict(test_a)
            img_vec_2 = single_model.predict(test_p)

            slide_results = self.slide_descriptors(img_vec_1, img_vec_2)

            return slide_results

    def get_similarity_triple(self, inputs, visualize=False):
        model = self.model
        single_model = self.single_model

        test_a = inputs[0]
        test_p = inputs[1]
        test_n = inputs[2]

        patch_w = test_p.shape[2]
        patch_h = test_p.shape[1]

        if self.sliding:
            anchor_vec = single_model.predict(test_a)
            pos_vec = single_model.predict(test_p)
            neg_vec = single_model.predict(test_n)

            anchor_vec = anchor_vec[0].flatten()
            vec_a_norm = np.linalg.norm(anchor_vec)
            if vec_a_norm != 0:
                anchor_vec = anchor_vec/vec_a_norm

            pos_vec = pos_vec[0].flatten()
            vec_p_norm = np.linalg.norm(pos_vec)
            if vec_p_norm != 0:
                pos_vec = pos_vec/vec_p_norm

            neg_vec = neg_vec[0].flatten()
            vec_n_norm = np.linalg.norm(neg_vec)
            if vec_n_norm != 0:
                neg_vec = neg_vec/vec_n_norm

            dot_prod_p = np.dot(anchor_vec, pos_vec.T)
            dot_prod_n = np.dot(anchor_vec, neg_vec.T)

            return dot_prod_p, dot_prod_n

        else:
            pred_dict = dict()

            img_vec_1 = single_model.predict(test_a)
            img_vec_2 = single_model.predict(test_p)
            img_vec_3 = single_model.predict(test_n)

            slide_results_p = self.slide_descriptors(img_vec_1, img_vec_2)
            slide_results_n = self.slide_descriptors(img_vec_1, img_vec_3)

            return slide_results_p, slide_results_n
