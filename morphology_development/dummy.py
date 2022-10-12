import tensorflow as tf

import tensorflow_datasets as tfds  # not needed for Euclid

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def get_dummy_mnist_model():
    # copied from https://www.tensorflow.org/datasets/keras_example

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='data'
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=1)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=1)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=2,
        validation_data=ds_test,
        verbose=1
    )
    
    return model


def main():

    saved_model_dir = 'models/dummy_mnist'
    tfline_model_path = 'models/dummy_mnist.tflite'

    # save as SavedModel
    model = get_dummy_mnist_model()
    model.save(saved_model_dir, save_format='tf')

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    # save as .tflite
    with open(tfline_model_path, 'wb') as f:
        f.write(tflite_model)

    

if __name__ == '__main__':

    main()

