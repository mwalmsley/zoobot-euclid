import pytest

from astropy.io import fits
import numpy as np

@pytest.fixture()
def fits_image():
    return fits.getdata('data/example_fits/J000000.80+004200.0.fits', ext=0)

@pytest.fixture()
def prepared_image():
    return np.random.rand(1, 300, 300, 1).astype(np.float32)

@pytest.fixture()
def catalog_radius():
    return 42

@pytest.fixture()
def model_path():
    return 'models/zoobot_example.tflite'