import tensorflow as tf
import numpy as np
from skimage.transform import resize


def prepare_image(image, catalog_radii):
    # TODO any other simple preprocessing specific to ML
    # TODO obviously this isn't optimised for speed

    # TODO make a smaller cutout based on catalog radii

    # scale to 0-1 interval - for now, with a simple log
    image = np.log10(image)
    image = image + image.min()
    image = image / image.max()

    # move channels last 
    image = np.transpose(image, axes=[1, 2, 0])

    # greyscale (won't be needed for Euclid, perhaps)
    image = np.mean(image, axis=2, keepdims=True)

    # resize to standard pixel shape, with aliasing
    image = resize(image, output_shape=(300, 300))

    # add batch dimension, which TFLite expects
    image = np.expand_dims(image, axis=0)

    return image


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
    return output_data


def main():

    # model_path = 'models/dummy_mnist.tflite'
    model_path = 'models/zoobot_example.tflite'

    image = None  # for testing, will make random data
    prediction = load_and_predict(image, model_path)
    print(prediction)
    print(prediction.shape)

if __name__ == '__main__':
    main()