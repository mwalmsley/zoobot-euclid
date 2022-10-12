# zoobot-euclid
Minimal Zoobot (or other) ML model for Euclid pipeline

Euclid presumably won't want to have an ML server running (even though this would be much faster). So can't use TF Serving.

Fully loading TensorFlow/model for every galaxy will probably be too slow to do tens of millions of times. 

Instead, plan on using Tensorflow Lite (TFLite).

SavedModel can be converted to TFLite: https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

Model can then be run in Python, requiring only tf.lite.Interpreter import:

    import numpy as np
    import tensorflow as tf

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python

TFLite is a module within Tensorflow so is included in EDEN 3.0, although it's quite an old version (TF 2.4.2). I assume this might change later.

Any pre-processing will not happen - that will need to be separate (standard) Python. But for inference, we don't need augmentations, so only minimal pre-processing required.