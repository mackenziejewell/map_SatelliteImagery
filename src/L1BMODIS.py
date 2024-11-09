
# scripts to work with Level1B MODIS data

import numpy as np
from pyhdf.SD import SD, SDC





def get_hdf_data(file,dataset,attr):
    
    """Load data from hdf file (can load single attribute or full dataset).
    
INPUT:
- file: hdf filename with directory  
        (e.g. '/Users/kenzie/MOD021KM.A2000066.2255.061.2017171220013.hdf')
- dataset: desired data set within HDF file 
           (e.g. 'EV_250_Aggr1km_RefSB')
- attr: None OR desired attribute within dataset 
        (e.g. None, 'reflectance_scales')

OUTPUT:
- specified dataset or attribute

Latest recorded update:
11-07-2024

    """
    
    f = SD(file,SDC.READ)
    data = f.select(dataset)
    # if no attribute, grab full data
    if attr == None:            
        data_or_attr = data[:]
    # or grab attribute from full data
    else:                       
        index = data.attr(attr).index()
        data_or_attr = data.attr(index).get()
    f.end()
    
    return data_or_attr


def print_hdf_contents(file):

    """Load local hdf file.
    
INPUT:
- file: filename with directory 
        (e.g. '/Users/kenzie/MOD021KM.A2000066.2255.061.2017171220013.hdf')

OUTPUT:
- list of datasets in hdf file
Latest recorded update:
11-07-2024

    """

    f = SD(file, SDC.READ)

    datasets = f.datasets().keys()
    print(datasets)

    return datasets



def grab_attr(dataset, attr):

    """Grab attribute data from desired field in hdf file"""

    index = dataset.attr(attr).index()
    return dataset.attr(index).get()

def locate_band(hdf, band):

    # load all band nums
    band_info = {}

    band_lists = ['Band_250M', 'Band_500M', 'Band_1KM_RefSB', 'Band_1KM_Emissive']
    band_names = ['EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'EV_1KM_RefSB', 'EV_1KM_Emissive']

    for key, field in zip(band_lists, band_names):
        band_info[key] = {}
        band_info[key]['numbers'] = hdf.select(key)[:].astype(int)
        band_info[key]['field'] = field
        
    # identify which data field contains band
    # and its index in real data
    for key in band_info.keys():
        if band in band_info[key]['numbers']:
            FIELD = band_info[key]['field']
            INDEX = np.where(band_info[key]['numbers'] == band)[0][0]

    return FIELD, INDEX
        
        
def load_coords(hdf):
    
    latitude = hdf.select('Latitude')[:]
    longitude = hdf.select('Longitude')[:]
    
    return latitude, longitude


def grab_scaling(hdf, field, index):
    
    """Grab scaling info for desired data field in hdf file"""

    dataset = hdf.select(field)

    [validmin, validmax] = grab_attr(dataset, 'valid_range')
    fillval = grab_attr(dataset, '_FillValue')

    if 'Ref' in field:
        scales = grab_attr(dataset, 'reflectance_scales')
        offsets = grab_attr(dataset, 'reflectance_offsets')

    else:
        scales = grab_attr(dataset, 'radiance_scales')
        offsets = grab_attr(dataset, 'radiance_offsets')

    scaling = {}
    scaling['validmin'] = validmin
    scaling['validmax'] = validmax
    scaling['fillval'] = fillval
    scaling['scale'] = scales[index]
    scaling['offset'] = offsets[index]

    return scaling


def mask_bad_data(band_data, scaling, apply_mask = False):

    # make mask of data, elminating "bad" data  
    # ---------------------------------------------------------------------
    invalid = np.full(band_data.shape, False)

    # identify fill values or values outside valid range
    invalid[band_data > scaling['validmax']] = True
    invalid[band_data < scaling['validmin']] = True
    invalid[band_data == scaling['fillval']] = True

    
    # replace these ^ with NaNs
    band_data[invalid] = np.nan

    # apply offset and scale
    offset_data = (band_data - scaling['offset']) * scaling['scale']

    # make data a masked array to ignore NaNs in calculations
    if apply_mask:
        offset_data = np.ma.masked_array(offset_data, np.isnan(offset_data))

    return offset_data


def get_MODISgeo(geofile):

    """Load lat, lon arrays from MODIS geo (hdf) files.
    Reads in Terra/MODIS (MOD03) or Aqua/MODIS (MOD03) geo files,
    May also work for hdf geolocation files from other satellites.
    Returns longitudes in range (0, 360).
    
INPUT:
- geofile: filename with directory  
           (e.g. '/Users/kenzie/MOD03.A2000059.1745.061.2017171195808.hdf')
OUTPUT:
- geolat: array of lat values
- geolon: array of lon values

DEPENDENCIES:
from pyhdf.SD import SD, SDC

Latest recorded update:
06-03-2022

    """
    
    # open geo file
    #--------------   
    try:
        f = SD(geofile,SDC.READ) 
    # raise an error if file can't be opened 
    except Exception as e:
        print(e, ", error opening ", geofile, sep='')

    # open geo file
    #--------------
    geolat = f.select(0)[:]       # pull out lat    
    geolon = f.select(1)[:]       # pull out lon 
    
    # make all longitudes range (0,360)
    #----------------------------------
    geolon[geolon<0] += 360
                    
    # close geo file and return lat, lon
    #-----------------------------------
    f.end() 
    
    return geolon, geolat

def load_MODIS1KMband(file, band: int, apply_scaling = True, apply_mask = False):

    """Load a band from L1B 1km MODIS imagery (MD021KM/MYD021KM). Applies scale factor and offsets,
    makes mask for invalid/missing data values. Uses pyhdf.SD module.

INPUT:
- file: filename with directory 
        (e.g. '/Users/kenzie/MOD021KM.A2000066.2255.061.2017171220013.hdf')
- band: band number (int) (reference: https://modis.gsfc.nasa.gov/about/specifications.php)
- apply_scaling: apply scaling to data (True/False)
- apply_mask: apply mask to data (True/False)
"""
    # open file
    hdf = SD(file,SDC.READ)
    
    # grab field and index of band
    FIELD, INDEX = locate_band(hdf, band)

    # grab band data
    band_data = hdf.select(FIELD)[:][INDEX, :, :].astype(np.double)

    if apply_scaling:
        # grabd scaling info
        scaling = grab_scaling(hdf, FIELD, INDEX)

        # mask bad data
        scaled_data = mask_bad_data(band_data, scaling, apply_mask = apply_mask)

    else:
        scaled_data = band_data
    # load data coordinates
    # latitude, longitude = load_coords(hdf) # these are at 5 km res for some reason?

    hdf.end() 

    return scaled_data



def load_MODISband_old(file, dataset, band, refrad):

    """Load a band from MODIS imagery. Applies scale factor and offsets,
    makes mask for invalid/missing data values. Uses homemade get_hdf_data function.
    
INPUT:
- file: filename with directory 
        (e.g. '/Users/kenzie/MOD021KM.A2000066.2255.061.2017171220013.hdf')
- dataset: desired data set within HDF file 
           (e.g. 'EV_250_Aggr1km_RefSB')
- band: band number formatted as string 
        (e.g. '30')
- refrad: reflectance or radiance datatype
          ('reflectance' or 'radiance')

OUTPUT:
- band_data: ref or rad band data, "bad" data masked

Latest recorded update:
11-07-2024

    """
    
    # import data
    #---------------------------------------------------------------------
    band_names = get_hdf_data(file, dataset, 'band_names')
    band_data = get_hdf_data(file, dataset, None)[band_names.split(",").index(band), :, :].astype(np.double)
    if refrad == 'reflectance':
        scales = get_hdf_data(file, dataset, 'reflectance_scales')[band_names.split(",").index(band)]
        offsets = get_hdf_data(file, dataset, 'reflectance_offsets')[band_names.split(",").index(band)]
    elif refrad == 'radiance':
        scales = get_hdf_data(file, dataset, 'radiance_scales')[band_names.split(",").index(band)]
        offsets = get_hdf_data(file, dataset, 'radiance_offsets')[band_names.split(",").index(band)]   
    else:
        print('REFLECTANCE OR RADIANCE NOT SPECIFIED')
        
    validmin = get_hdf_data(file, dataset, 'valid_range')[0]
    validmax = get_hdf_data(file, dataset, 'valid_range')[1]
    fillval = get_hdf_data(file, dataset, '_FillValue')
    
    # make mask of data, elminating "bad" data  
    # ---------------------------------------------------------------------
    # identify fill values or values outside valid range
    invalid = np.logical_or(band_data > validmax, band_data < validmin)
    invalid = np.logical_or(invalid, band_data == fillval)
    # replace these ^ with NaNs
    band_data[invalid] = np.nan
    # apply offset and scale
    band_data = (band_data - offsets) * scales 
    # make data a masked array to ignore NaNs in calculations
    band_data = np.ma.masked_array(band_data, np.isnan(band_data))
    
    return band_data