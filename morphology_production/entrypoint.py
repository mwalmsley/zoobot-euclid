# the main black box for Euclid

from morphology_production import inference


def measure_morphology(fits_image, catalog_radii, model_path):
    ml_ready_image = inference.prepare_image(fits_image, catalog_radii)
    return inference.load_and_predict(ml_ready_image, model_path)[0]  # 0 to remove batch index
