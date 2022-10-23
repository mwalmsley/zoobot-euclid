import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from skimage import draw
from skimage.transform import resize


def prepare_image(image, seg, catalog, source_index, mode='seg', m=1.5, resized_cutout_size=300):
    # Prepares single cutout image of  for the model, either by using the segmentation map or the major axis
    # INPUT PARAMETERS
    # image (numpy array M x M): The image from which the cutout will be extracted
    # seg (numpy array M x M): The segmentation map
    # catalog (pandas DataFrame): The catalog containing the parameters of the sources
    # source_index (int): The index of the source to be extracted (from 0 to the length of the catalog)
    # mode (str): The mode to be used for the cutout extraction. Either 'seg' or 'maj'
    # m (float): The factor to be used for the major axis cutout extraction
    # OUTPUTS
    # cutout (numpy array resized_cutout_size x resized_cutout_size): The cutout of the image

    # catalog column values for the source
    # TODO only read the required columns, and read here directly
    params = extract_params_by_source_index(catalog, source_index)
    if mode == 'seg':
        # changed from params[-1]
        # SEG_ID is the source index in the segmentation map
        cutout = get_cutout_from_seg_map(seg, image, source_index=params['SEG_ID'])
    elif mode == 'maj':
        ellipse_axes = params[['SEMIMAJOR_AXIS', 'SEMIMINOR_AXIS']].values
        
        # find the array indices enclosed by an ellipse  
        # - with those axes
        # - centered on the x, y array indices of the source center
        rows, cols = draw.ellipse(r=params['x'], c=params['y'], 
                                  r_radius=m * ellipse_axes[0], 
                                  c_radius=m * ellipse_axes[1], shape=seg.shape)

        # create a segmentation map with 1 inside the ellipse and 0 elsewhere
        dummy_segmentation_map = np.zeros_like(seg)
        dummy_segmentation_map[cols, rows] = 1
        # can then (re-)use get_cutout_from_seg_map to retrieve a cutout around the ellipse
        cutout = get_cutout_from_seg_map(dummy_segmentation_map, image, source_index=1)
    else:
        raise ValueError(mode)

    # rescale from 0 to 1
    # cutout = np.log10(cutout)
    cutout =  (cutout - cutout.min()) / (cutout.max() - cutout.min())

    # resize to desired output size
    # TODO specify resize method order
    image = resize(cutout, output_shape=(resized_cutout_size, resized_cutout_size))
    # return with batch and channel dimensions (TODO may change)
    return image[np.newaxis, :, :, np.newaxis]  


def load_data_for_mosaic(catalog_path, segmentation_path, mosaic_path):
    # Loads the mosaic image, the segmentation map and the catalog, and makes necessary transformations on the 
    # catalog rows 
    catalog = fits_to_pandas(catalog_path)
    header = fits.getheader(mosaic_path, ignore_blank=True)
    catalog = prepare_df(catalog, header)

    seg = fits.getdata(segmentation_path)
    mosaic = fits.getdata(mosaic_path)

    unique_seg_idxs = np.unique(seg)[2:]
    catalog['SEG_ID'] = unique_seg_idxs
    return mosaic, seg, catalog


def fits_to_pandas(catalog_file):
    # Read in the catalog and convert to pandas
    hdul = fits.open(catalog_file)
    catalog = np.array(hdul[1].data)
    columns = hdul[1].columns
    catalog = pd.DataFrame(catalog.byteswap().newbyteorder(), columns=columns.names)
    return catalog

def prepare_df(catalog, header):
    # Extracts the WCS from the fits and converts the catalog ra and dec to pixel coordinates
    catalog = catalog.copy()
    wcs = WCS(header)
    # x, y are pixel coordinates
    catalog[['x', 'y']] =  wcs.all_world2pix(catalog[
        ['RIGHT_ASCENSION', 'DECLINATION']], 0)
    # x0, etc, are corners of segmentation map as recorded
    # TODO these can surely be used directly instead of the current idx_to_seg_map function, 
    # if we split out the ellipse method and ensure equal axis length, but that's for another day
    catalog[['x0', 'y0']] =  wcs.all_world2pix(catalog[
        ['CORNER_0_RA', 'CORNER_0_DEC']], 0)
    catalog[['x1', 'y1']] =  wcs.all_world2pix(catalog[
        ['CORNER_1_RA', 'CORNER_1_DEC']], 0)
    catalog[['x2', 'y2']] =  wcs.all_world2pix(catalog[
        ['CORNER_2_RA', 'CORNER_2_DEC']], 0)
    catalog[['x3', 'y3']] =  wcs.all_world2pix(catalog[
        ['CORNER_3_RA', 'CORNER_3_DEC']], 0)
    return catalog

def extract_params_by_source_index(catalog, source_index):
    # Extracts the parameters for a given source_index
    return catalog.iloc[source_index, :][['OBJECT_ID', 'x', 'y', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3',
                                     'SEMIMAJOR_AXIS', 'SEMIMINOR_AXIS', 'SEG_ID']]

def get_cutout_from_seg_map(seg: np.ndarray, image: np.ndarray, source_index: int):
    """
    Extracts the indexes from the segmentation map and returns the squared cutout

    Args:
        seg (np.ndarray): M x M array where each value (pixel) is the index of the (multi-pixel) object at that location
        image (np.ndarray): M x M array of sky flux values
        source_index (int): source index in segmentation map

    Returns:
        _type_: _description_
    """
    # all segmentation array indices for the source with source_index
    seg_idxs = np.where(seg == source_index)

    # get highest and lowest indices - these will make a rectangle around the source
    y_min, y_max, x_min, x_max = np.min(seg_idxs[0]), np.max(seg_idxs[0]), np.min(seg_idxs[1]), np.max(seg_idxs[1])
    y_d = y_max - y_min
    x_d = x_max - x_min

    # TODO some tweaks?
    dif = np.abs(y_d - x_d) // 2
    if y_d > x_d:
        x_min -= dif
        x_max += dif
    else:
        y_min -= dif
        y_max += dif

    # slice for square cutout around segmentation map
    y_low = y_min - (y_max - y_min) // 2
    y_high = y_max + (y_max - y_min) // 2 + 1
    x_low = x_min - (y_max - y_min) // 2
    x_high = x_max + (y_max - y_min) // 2 + 1
    cutout = image[y_low:y_high, x_low:x_high]

    # adjust by a pixel if needed to ensure square
    if cutout.shape[0] != cutout.shape[1]:
        if cutout.shape[0] > cutout.shape[1]:
            cutout = image[y_low:y_high, x_low - 1:x_high]
        else:
            cutout = image[y_low - 1:y_high, x_low:x_high]

    return cutout
