import os
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

from euclid_morphology import cutouts

def load_and_predict_full(image, model_path) -> np.ndarray:
    """
    

    Args:
        image (_type_): _description_
        model_path (str): to folder including saved_model.pb (includes architecture)

    Returns:
        np.ndarray: _description_
    """

    model = tf.keras.models.load_model(model_path)
    return model.predict(image)

def load_and_predict(image, model_path):
    # model_path must be converted to .tflite

    # Load the TFLite model and allocate tensors.
    # num_threads=1 to keep things simple for Euclid pipeline
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)  
    interpreter.allocate_tensors()

    # Get input and output tensors.
    # can replace these once model fixed, for speed
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if image is None:
    # Test the model on random input data.
        input_shape = input_details[0]['shape']
        image = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # 0 to remove batch index


def main():


    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # model_path = repo_dir / 'data/models/dummy_mnist.tflite'
    model_path = os.path.join(repo_dir, 'data/models/zoobot_example.tflite')

    # catalogue_path = 'data/example_tile/EUC_MER_FINAL-CUTOUTS-CAT_TILE100158586-2F9FF9_20220829T221845.491503Z_00.00.fits'
    # mosaic_path = 'data/example_tile/EUC_MER_BGSUB-MOSAIC-VIS_TILE100158586-863FA9_20220829T190315.054985Z_00.00.fits'
    # segmentation_path = 'data/example_tile/EUC_MER_FINAL-SEGMAP_TILE100158586-CB5786_20220829T221845.491530Z_00.00.fits' 

    # # Loading and preprocessing data
    # image, seg, catalogue = cutouts.load_data_for_mosaic(catalogue_path, segmentation_path, mosaic_path)

    # # selecting a random source from the catalogue
    # idx = np.random.randint(0, len(catalogue)) 
    # cutout = cutouts.prepare_image(image, seg, catalogue, idx, mode='seg', m=1.5)
    # print(cutout.shape)

    cutout = np.random.rand(12, 300, 300, 1).astype(np.float32)

    prediction = load_and_predict(cutout, model_path)
    print(prediction)
    print(prediction.shape)

if __name__ == '__main__':
    main()