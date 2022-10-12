import tensorflow as tf
import numpy as np

def load_and_predict(model_path, image=None):

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)  # must be converted to .tflite
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

    prediction = load_and_predict(model_path, image=None)
    print(prediction)
    print(prediction.shape)

if __name__ == '__main__':
    main()