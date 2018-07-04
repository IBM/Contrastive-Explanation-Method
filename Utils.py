## Utils.py -- Some utility functions 
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from keras.models import Model, model_from_json, Sequential
from PIL import Image

import tensorflow as tf
import os
import numpy as np


def load_AE(codec_prefix, print_summary=False):

    saveFilePrefix = "models/AE_codec/" + codec_prefix + "_"

    decoder_model_filename = saveFilePrefix + "decoder.json"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder.load_weights(decoder_weight_filename)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder

def save_img(img, name = "output.png"):

    np.save(name, img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)

def generate_data(data, id, target_label):
    inputs = []
    target_vec = []

    inputs.append(data.test_data[id])
    target_vec.append(np.eye(data.test_labels.shape[1])[target_label])

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec

def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n','')
    return prob, predicted_class, prob_str
