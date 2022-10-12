import morphology_production.entrypoint as entrypoint


def test_measure_morphology(fits_image, catalog_radii, model_path):
    entrypoint.measure_morphology(fits_image, catalog_radii, model_path)
