# zoobot-euclid
Minimal Zoobot (or other) ML model for Euclid pipeline

This repo has two halves.
- ``morphology_production`` is the production code to measure detailed morphology in the Euclid pipeline.
- ``morphology_development`` creates a static dependency (the frozen ML model) for ``morphology_production``. It need not be installed or run Euclid-side. 

## Production Installation

``morphology_production`` depends only on numpy, TensorFlow, and scikit-image, all already included in EDEN 3.0. 

You will need the frozen ML model (``example_zoobot.tflite``) and optionally the example FITS images. Download from [Dropbox](https://www.dropbox.com/sh/4dz0vc980zi1s24/AACWGqcSJNbE4Igj0Q7vXTXca?dl=0).

## Production Use

``morphology_production`` includes ``entrypoint.py``, copied below. This illustrates the expected API.

    from morphology_production import inference

    def measure_morphology(fits_image, catalog_radii, model_path='models/zoobot_example.tflite'):
        ml_ready_image = inference.prepare_image(fits_image, catalog_radii)
        return inference.load_and_predict(ml_ready_image, model_path)

<!-- ## Development Installation

``morphology_development`` requires Zoobot and other standard PyData packages. It cannot be run within EDEN. -->

## Dev Note - Why TFLite?



Euclid presumably won't want to have an ML server running (even though this would be faster). So can't use TF Serving.

Fully loading TensorFlow/model for every galaxy will probably be too slow to do tens of millions of times. 

Instead, plan on using Tensorflow Lite (TFLite).

SavedModel can be [converted to TFLite](https://www.tensorflow.org/lite/api_docs/python/tf/lite/TFLiteConverter).

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

Model can then be [run](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python) in Python, requiring only tf.lite.Interpreter import

TFLite is a module within Tensorflow so is included in EDEN 3.0, although it's quite an old version (TF 2.4.2). I assume this might change later.

tf.keras.experimental.preprocessing layers (in particular, the crop/resize) are still part of the TFLite model.


## Dev Note - Environment

Run on EDEN 3.0 with [Dockeen](https://euclid.roe.ac.uk/projects/codeen-users/wiki/LODEEN_DOCKEEN_IDE) (redmine login required)

    docker login gitlab.euclid-sgs.uk:4567
    docker run -it --name dockeen --privileged gitlab.euclid-sgs.uk:4567/st-tools/ct_xodeen_builder/dockeen

gitlab credentials are the same as cas.cosmos. You can also use an [API token](https://euclid.roe.ac.uk/issues/20384) if needed.

    doocker build -tag zoobot-euclid .

Run in interactive mode with zoobot-euclid attached as a bindmount (-v) so it has the latest code and data available

    docker run --name dev -it --rm --privileged -v /Users/walml/repos/zoobot-euclid:/home/user/zoobot-euclid zoobot-euclid

For production use, would add directly into the Dockerfile. But data is large.

Inside the container:

    pip install --user -e zoobot-euclid
    python zoobot-euclid/euclid_morphology/inference.py

To remake the model

    pip install --user -e zoobot-euclid
    python zoobot-euclid/development/convert_zoobot.py


## Dev Note - TF Versions

Zoobot is trained on TF 2.8+. The current version is 2.10. EDEN uses TF 2.4.1. Models trained on TF 2.8 and converted with TFLite cannot be loaded in 2.4.1 (throws a helpful error). So either:

- Model must be trained and exported to TFLite in 2.4.1. Pray training is still possible. Current attempt.
- Model can be trained in TF 2.8+ and used as-is in 2.4.1, perhaps. Pray can be used as-is.
- Make EDEN change request to upgrade TF (unclear if any other projects are using it?)

### Do Not Use Conda

EDEN is not available on Conda (except perhaps locally at ESA?). Only an old version is available:

Add Eden channel

    conda config --add channels https://condaread:euclid_2020@codeen-repo.euclid-ec.org/nexus/repository/conda-euclid

Check available

    conda search -c https://condaread:euclid_2020@codeen-repo.euclid-ec.org/nexus/repository/conda-euclid --override-channels

Install as needed (but eden appears to be old on the channel?)

    conda install eden