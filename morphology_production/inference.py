import tensorflow as tf
import numpy as np
from skimage.transform import resize
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import binary_dilation, generate_binary_structure
import matplotlib.patches as patches
from numpy import extract
from skimage import draw, transform, filters
from tqdm import tqdm
import matplotlib
matplotlib.rcParams.update({'font.size': 16})



# Output size ?
# Prepare image 300 x 300



#def prepare_image(image, catalog_radius):
    # TODO any other simple preprocessing specific to ML
    # TODO obviously this isn't optimised for speed

    # TODO make a smaller cutout based on catalog radii

    # scale to 0-1 interval - for now, with a simple log
#    image = np.log10(image)
#    image = image + image.min()
#    image = image / image.max()
    
    # move channels last 
#    image = np.transpose(image, axes=[1, 2, 0])

    # greyscale (won't be needed for Euclid, perhaps)
#   image = np.mean(image, axis=2, keepdims=True)

    # resize to standard pixel shape, with aliasing
#    image = resize(image, output_shape=(300, 300))

    # add batch dimension, which TFLite expects
#    image = np.expand_dims(image, axis=0)

#    return image


def fits_to_pandas(catalogue_file):
    # Read in the catalogue and convert to pandas
    hdul = fits.open(catalogue_file)
    catalogue = np.array(hdul[1].data)
    columns = hdul[1].columns
    catalogue = pd.DataFrame(catalogue.byteswap().newbyteorder(), columns=columns.names)
    return catalogue

def prepare_df(catalogue, header):
    # Extracts the WCS from the fits and converts the catalogue ra and dec to pixel coordinates
    catalogue = catalogue.copy()
    wcs = WCS(header)
    catalogue[['x', 'y']] =  wcs.all_world2pix(catalogue[
        ['RIGHT_ASCENSION', 'DECLINATION']], 0)
    catalogue[['x0', 'y0']] =  wcs.all_world2pix(catalogue[
        ['CORNER_0_RA', 'CORNER_0_DEC']], 0)
    catalogue[['x1', 'y1']] =  wcs.all_world2pix(catalogue[
        ['CORNER_1_RA', 'CORNER_1_DEC']], 0)
    catalogue[['x2', 'y2']] =  wcs.all_world2pix(catalogue[
        ['CORNER_2_RA', 'CORNER_2_DEC']], 0)
    catalogue[['x3', 'y3']] =  wcs.all_world2pix(catalogue[
        ['CORNER_3_RA', 'CORNER_3_DEC']], 0)
    return catalogue

def extract_params_by_idx(catalogue, idx):
    # Extracts the parameters for a given idx
    return catalogue.iloc[idx, :][['OBJECT_ID', 'x', 'y', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3',
                                     'SEMIMAJOR_AXIS', 'SEMIMINOR_AXIS', 'SEG_ID']]

def get_cut_from_idxs(seg, image, idx):
    # Extracts the indexes from the segmentation map and returns the squared cutout
    seg_idxs = np.where(seg == idx)
    y_min, y_max, x_min, x_max = np.min(seg_idxs[0]), np.max(seg_idxs[0]), np.min(seg_idxs[1]), np.max(seg_idxs[1])
    y_d = y_max - y_min
    x_d = x_max - x_min
    dif = np.abs(y_d - x_d) // 2
    if y_d > x_d:
        x_min -= dif
        x_max += dif
    else:
        y_min -= dif
        y_max += dif
    cutout = image[y_min - (y_max - y_min) // 2:y_max + (y_max - y_min) // 2 + 1, x_min - (y_max - y_min) // 2:x_max + (y_max - y_min) // 2 + 1]
    if cutout.shape[0] != cutout.shape[1]:
        if cutout.shape[0] > cutout.shape[1]:
            cutout = image[y_min - (y_max - y_min) // 2:y_max + (y_max - y_min) // 2 + 1, x_min - (y_max - y_min) // 2 - 1:x_max + (y_max - y_min) // 2 + 1]
        else:
            cutout = image[y_min - (y_max - y_min) // 2 - 1:y_max + (y_max - y_min) // 2 + 1, x_min - (y_max - y_min) // 2:x_max + (y_max - y_min) // 2 + 1]
    return cutout

def create_plots(image, seg, catalogue):
    # Utility function to create plots
    fig, ax = plt.subplots(ncols=8, nrows=30, figsize=(8 * 8, 8 * 30))
    for i in tqdm(range(30)):
        params = extract_params_by_idx(catalogue, np.random.randint(0, len(catalogue)))
        ref_seg = seg[int(params[2]) - 64: int(params[2]) + 64, int(params[1]) - 64: int(params[1]) + 64]
        ref_img = image[int(params[2]) - 64: int(params[2]) + 64, int(params[1]) - 64: int(params[1]) + 64]
        seg_cutout = get_cut_from_idxs(seg, image, params[-1])
        axes = params[['SEMIMAJOR_AXIS', 'SEMIMINOR_AXIS']].values
        rows, cols = draw.ellipse(r=params[1], c=params[2], r_radius=axes[0], c_radius=axes[1], shape=seg.shape)
        ellipse = np.zeros_like(seg)
        ellipse[cols, rows] = 1
        maj_cutout = get_cut_from_idxs(ellipse, image, 1)
        rows, cols = draw.ellipse(r=params[1], c=params[2], r_radius=3 * axes[0], c_radius=3 * axes[1], shape=seg.shape)
        ellipse = np.zeros_like(seg)
        ellipse[cols, rows] = 1
        maj3_cutout = get_cut_from_idxs(ellipse, image, 1)

        seg_resized = resize(seg_cutout, output_shape=(300, 300))
        maj_resized = resize(maj_cutout, output_shape=(300, 300))
        maj3_resized = resize(maj3_cutout, output_shape=(300, 300))

        ax[i, 0].imshow(ref_img, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 0].set_title('Reference Image Cutouts')
        ax[i, 1].imshow(ref_seg, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 1].set_title('Reference Segmentation Cutout')
        ax[i, 2].imshow(seg_cutout, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 2].set_title('Segmentation Derived Cutouts')
        ax[i, 3].imshow(maj_cutout, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 3].set_title('Major Axis Derived Cutouts')
        ax[i, 4].imshow(maj3_cutout, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 4].set_title('3x Major Axis Derived Cutouts')
        ax[i, 5].imshow(seg_resized, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 5].set_title('Segmentation Derived Cutouts Resized')
        ax[i, 6].imshow(maj_resized, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 6].set_title('Major Axis Derived Cutouts Resized')
        ax[i, 7].imshow(maj3_resized, origin='lower', cmap='viridis')
        if i == 0:
            ax[i, 7].set_title('3x Major Axis Derived Cutouts Resized')
    plt.tight_layout()
    plt.show()

def prepare_image(image, seg, catalogue, idx, mode='seg', m=1.5):
    # Prepares the image for the model, either by using the segmentation map or the major axis
    # INPUT PARAMETERS
    # image (numpy array M x M): The image from which the cutout will be extracted
    # seg (numpy array M x M): The segmentation map
    # catalogue (pandas DataFrame): The catalogue containing the parameters of the objects
    # idx (int): The index of the object to be extracted (from 0 to the length of the catalogue)
    # mode (str): The mode to be used for the cutout extraction. Either 'seg' or 'maj'
    # m (float): The factor to be used for the major axis cutout extraction
    # OUTPUTS
    # cutout (numpy array 300 x 300): The cutout of the image
    params = extract_params_by_idx(catalogue, idx)
    if mode == 'seg':
        cutout = get_cut_from_idxs(seg, image, params[-1])
    else:
        axes = params[['SEMIMAJOR_AXIS', 'SEMIMINOR_AXIS']].values
        rows, cols = draw.ellipse(r=params[1], c=params[2], 
                                  r_radius=m * axes[0], 
                                  c_radius=m * axes[1], shape=seg.shape)
        ellipse = np.zeros_like(seg)
        ellipse[cols, rows] = 1
        cutout = get_cut_from_idxs(ellipse, image, 1)
    #cutout = np.log10(cutout)
    cutout =  (cutout - cutout.min()) / (cutout.max() - cutout.min())
    image = resize(cutout, output_shape=(300, 300))
    return image[np.newaxis, :, :, np.newaxis]

def preprocess_data(catalogue_path, segmentation_path, mosaic_path):
    # Loads the image, the segmentation map and the catalogue, and makes necessary transformations on the 
    # catalogue rows 
    catalogue = fits_to_pandas(catalogue_path)
    header = fits.getheader(mosaic_path, ignore_blank=True)
    catalogue = prepare_df(catalogue, header)
    seg = fits.getdata(segmentation_path)
    image = fits.getdata(mosaic_path)
    unique_seg_idxs = np.unique(seg)[2:]
    catalogue['SEG_ID'] = unique_seg_idxs
    return image, seg, catalogue



def load_and_predict(image, model_path):
    # model_path must be converted to .tflite

    # Load the TFLite model and allocate tensors.
    # num_threads=1 to keep things simple for Euclid pipeline
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)  
    interpreter.allocate_tensors()

    # Get input and output tensors.
    # can replace these once model fixed, for speed
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if image is None:
    # Test the model on random input data.
        input_shape = input_details[0]['shape']
        image = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # 0 to remove batch index


def main():

    # model_path = 'models/dummy_mnist.tflite'
    model_path = 'models/zoobot_example.tflite'
    catalogue_path = '/Users/michele/Downloads/EUC_MER_FINAL-CUTOUTS-CAT_TILE100158586-2F9FF9_20220829T221845.491503Z_00.00.fits'
    mosaic_path = '/Users/michele/Downloads/EUC_MER_BGSUB-MOSAIC-VIS_TILE100158586-863FA9_20220829T190315.054985Z_00.00.fits'
    segmentation_path = '/Users/michele/Downloads/EUC_MER_FINAL-SEGMAP_TILE100158586-CB5786_20220829T221845.491530Z_00.00.fits' 
    # Loading and preprocessing data
    image, seg, catalogue = preprocess_data(catalogue_path, segmentation_path, mosaic_path)
    # selecting a random source from the catalogue
    idx = np.random.randint(0, len(catalogue)) 
    cutout = prepare_image(image, seg, catalogue, idx, mode='seg', m=1.5)
    print(cutout.shape)
    prediction = load_and_predict(cutout, model_path)
    print(prediction)
    print(prediction.shape)

if __name__ == '__main__':
    main()