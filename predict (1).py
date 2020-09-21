#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image

import argparse
import numpy as np

IMAGE_SHAPE = (224, 224, 3)

def open_labels(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Label_map's index starts from 1 to 102, but oxford_flower102's indices are from 0 to 101
    label_names = dict()
    for name in class_names:
        label_names[str(int(name)-1)] = class_names[name]
    return label_names


def get_model(model_name):
    model = tf.keras.models.load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer})
    print("\n",model.summary(),"\n")
    return model

def process_image(image):
    
    image_tensor = tf.image.convert_image_dtype(image, dtype=tf.int32, saturate=False)
    image_resized = tf.image.resize(image,(224,224)).numpy()
    image_normalized = image_resized/255
    
    return image_normalized


def predict(image_path, model_file, top_k, class_labels):
   
    top_k = int(top_k)
    model = get_model(model_file)

    image = Image.open(image_path)
    image = np.asarray(image)

    # processing the image
    processed_image = process_image(image)
    
    # compute prediction probabilities
    
    pred = model.predict(np.expand_dims(processed_image,axis=0))
    preds = pred[0].tolist()
        
    k_values, k_indices = tf.math.top_k(preds, k=top_k)
    probs = k_values.numpy().tolist()
    classes = k_indices.numpy().tolist()
    
    print("\nTop 5 probabilities:",probs)
    print("\ntop 5 Classes:",classes)
    
    labels = [class_labels[str(index)] for index in classes]
    print('\nTop 5 class labels: ',labels)
    
    prob_dict = dict(zip(labels, probs))       
    print("\nTop 5 classes with respective probabilities : ",prob_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Predict flowers")
    parser.add_argument("image_path")
    parser.add_argument("keras_saved_model")
    parser.add_argument("--top_k_values", required = False, default = 5)
    parser.add_argument("--category_names", required = False, default = "label_map.json")
    
    args = parser.parse_args()

    class_labels = open_labels(args.category_names)

    predict(args.image_path, args.keras_saved_model, args.top_k_values, class_labels)








