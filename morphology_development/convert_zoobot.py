from os import getpid
import tensorflow as tf

# imports not needed for Euclid
from zoobot.tensorflow.training import train_with_keras
from zoobot.shared import label_metadata, schemas
from pytorch_galaxy_datasets.prepared_datasets import DecalsDR5Dataset

def main():

    saved_model_dir = 'models/zoobot_example'
    tfline_model_path = 'models/zoobot_example.tflite'

    dataset = DecalsDR5Dataset(root='/nvme1/scratch/walml/repos/_data/decals_dr5', download=False)
    dr5_catalog = dataset.catalog
    adjusted_catalog = dr5_catalog.sample(1000)

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # dr5
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)

    model = train_with_keras.train(
        catalog=adjusted_catalog,
        schema=schema,
        save_dir='models/zoobot_temp_training',
        batch_size=16,
        epochs=1,
        gpus=0,
        mixed_precision=False
    )
    model.save(saved_model_dir, save_format='tf')

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    # save as .tflite
    with open(tfline_model_path, 'wb') as f:
        f.write(tflite_model)

    

if __name__ == '__main__':

    main()

