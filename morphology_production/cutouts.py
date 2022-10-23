import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from skimage import draw
from skimage.transform import resize


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
