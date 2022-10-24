import os
import tensorflow as tf

# imports not needed for Euclid
from zoobot.tensorflow.training import train_with_keras
from zoobot.shared import label_metadata, schemas
from pytorch_galaxy_datasets.prepared_datasets import DecalsDR5Dataset

def main():

    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    if os.path.isdir('/nvme1/scratch'):
        version = 'zoobot_latest'
        dataset = DecalsDR5Dataset(root=os.path.join(repo_dir, '../_data/decals_dr5'), download=False)
    elif os.path.isdir('/home/user'):
        version = 'zoobot_eden'
        dataset = DecalsDR5Dataset(root=os.path.join(repo_dir, 'data/decals_dr5'), download=False)

    saved_model_dir = os.path.join(repo_dir, 'data/models', version)
    tfline_model_path = os.path.join(repo_dir, f'data/models/{version}.tflite')


    dr5_catalog = dataset.catalog
    adjusted_catalog = dr5_catalog.sample(1000)

    question_answer_pairs = label_metadata.decals_dr5_ortho_pairs  # dr5
    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)

    model = train_with_keras.train(
        catalog=adjusted_catalog,
        schema=schema,
        save_dir=os.path.join(repo_dir, f'data/models/{version}_training'),
        batch_size=16,
        epochs=1,
        gpus=0,
        mixed_precision=False
    )

    # save keras model as SavedModel (includes architecture)
    model.save(
        saved_model_dir,
        save_format='tf',
        include_optimizer=False
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    # save as .tflite
    with open(tfline_model_path, 'wb') as f:
        f.write(tflite_model)

    

if __name__ == '__main__':

    main()

