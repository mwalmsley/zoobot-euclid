
from morphology_production import inference


def measure_morphology(fits_image: str, catalog_radius: float, model_path='models/zoobot_example.tflite'):
    """
    For a FITS image centered on a galaxy,
    - Resize and preprocess the image to match Zoobot expected input
    - Predict detailed morphology using Zoobot (frozen ML model)

    Zoobot predicts concentrations describing the posterior predictions for detailed morphology.
    See Walmsley 2022 "Galaxy Zoo DECaLS" for more.

    This is the key "black box" to add to the Euclid pipeline.

    Args:
        fits_image (str): path to cutout image to load.
        catalog_radius (float): estimate of galaxy radius, used to optionally further crop the cutout
        model_path (str, optional): TFLite model to use for predictions. Defaults to 'models/zoobot_example.tflite'.

    Returns:
        np.array: np.float32 vector of Dirichlet concentrations. \
            Length 34, corresponding to GZ DECaLS' 34 answers.
    """
    ml_ready_image = inference.prepare_image(fits_image, catalog_radius)
    return inference.load_and_predict(ml_ready_image, model_path)
