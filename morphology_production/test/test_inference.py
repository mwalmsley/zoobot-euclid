from morphology_production import inference

def test_prepare_image(fits_image, catalog_radii):
    inference.prepare_image(fits_image, catalog_radii)

def test_load_and_predict(prepared_image, model_path):
    inference.load_and_predict(prepared_image, model_path)
