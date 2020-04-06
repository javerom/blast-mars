#!/usr/bin/python3

import sys
import csv
import numpy as np 
import astropy.units as u
import pygetdata as gd

from datetime import datetime
from subprocess import call
from astropy.io import fits as pf
from astropy.time import Time, TimezoneInfo
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

def blob_border(pixellist, thresh, image, blobmap):
    """
    A recursion function for blobfinding. Generates a pixel list of candidate blob pixels.

    pixellist: A list containing all current candidate pixels. Must not be empty (i.e. at least have
        the start pixel.
    thresh: The value above which a pixel is considered a candidate pixel.
    image: Reference to the image data (2D array)
    blobmap: Reference to the truth map for pixels already considered (2D array, shape == image).
    """
    (x, y) = pixellist[-1]
    (h, w) = image.shape

    blobmap[y][x] = 0

    # Check next pixels (1 to the right and 3 below) 
    for n in range(4):
        xc = int(x + (n + 2) % 3 - 1)
        yc = int(y + (n + 2) / 3)
        if xc < w and yc < h and blobmap[yc][xc] and image[yc][xc] > thresh:
            pixellist.append((xc, yc))
            blob_border(pixellist, thresh, image, blobmap)

def manual_blob_finder(image, minsigma=3, minnpx=8):
    """
    A manual blob finder that simply looks for blobs with a certain number of sigma above the mean.

    image: Reference to the image data (2D array)
    minsigma: The minimum number of standard deviations above the background a blob must be.
    minpx: The minimum number of pixels per candidate blob to be considered a blob.
    """
    image = image - np.mean(image)
    sigma = np.std(image)
    thresh = sigma * minsigma
    blobmap = np.ones(image.shape)
    (h, w) = image.shape

    bloblist = {
            "cent_x": [],
            "cent_y": [],
            "width": [],
            "height": [],
            "flux": [],
            "max": []}

    for y in range(h):
        for x in range(w):
            if blobmap[y][x] and image[y][x] > thresh:
                pixellist = []
                pixellist.append((x, y))
                blob_border(pixellist, thresh, image, blobmap)

                if len(pixellist) < minnpx:
                    continue

                # compute centroid
                sum_px = 0
                sum_py = 0
                min_x = 20000
                max_x = 0
                min_y = 20000
                max_y = 0
                sum_p = 0
                max_p = 0

                for (xc, yc) in pixellist:
                    p = image[yc][xc]
                    sum_px = sum_px + (xc * p)
                    sum_py = sum_py + (yc * p)
                    sum_p = sum_p + p
                    if xc > max_x:
                        max_x = xc
                    if xc < min_x:
                        min_x = xc
                    if yc > max_y:
                        max_y = yc
                    if yc < min_y:
                        min_y = yc
                    if p > max_p:
                        max_p = p

                bloblist['cent_x'].append(sum_px / sum_p)
                bloblist['cent_y'].append(sum_py / sum_p)
                bloblist['width'].append(max_x - min_x + 1)
                bloblist['height'].append(max_y - min_y + 1)
                bloblist['flux'].append(sum_p)
                bloblist['max'].append(max_p)

    if len(bloblist['cent_x']) == 0:
        return None

    print(bloblist)   
    return bloblist



def find_brightest_biggest(filename, catname="sources.cat", config="config.sex", minsize=10,
        minflux=450000):
    """
    Returns the coordinates of the brightest, biggest star/object in the image relative to the center
    of the frame (top-left to bottom-right is increasing xy).
    
    filename: File name of the fits image in which the blob will be found
    catname: The name of the file to be written by sextractor when finding blobs
    config: The configuration file for sextractor
    minsize: The minimum size (width or height) of a blob to be considered big.
    minflux: The minimum integrated flux of a blub to be considered bright.
    """
    # Get dimensions of the image
    hdulist = pf.open(filename)
    (height, width) = hdulist[0].data.shape
    bloblist = manual_blob_finder(hdulist[0].data)
    hdulist.close()

    if bloblist is None:
        return None

    sort_ind = np.argsort(bloblist['max'])[::-1]
    blob_ind = None
    for ind in sort_ind:
        if bloblist['width'][ind] >= minsize and bloblist['flux'][ind] > minflux:
            blob_ind = ind
            break

    if blob_ind is None:
        return None

    return (bloblist['cent_x'][blob_ind] - (float(width) / 2.),
            bloblist['cent_y'][blob_ind] - (float(height) / 2.),
            bloblist['flux'][blob_ind],
            bloblist['width'][blob_ind],
            bloblist['max'][blob_ind])

    '''
    # Use sextractor to find blobs.
    # N.B. may be tuning of parameters, but this was mostly unreliable and noisy.

    hdulist.close()

    # Source extract
    call(["sextractor", filename, "-c", config, "-CATALOG_NAME", catname])

    # Load the catalog file
    srclist = pf.open(catname)
    srctable = srclist[2].data
    sort_ind = np.argsort(srctable['FLUX_MAX'])[::-1]
    blob_ind = None
    for ind in sort_ind:
        if (srctable['FLUX_RADIUS'][ind] > minsize and srctable['FLUX_MAX'][ind] > minflux and
                srctable['FLUX_RADIUS'][ind] < maxradius and srctable['FLUX_MAX'][ind] < maxflux):
            blob_ind = ind;
            break
    if blob_ind is None:
        return None
    return (srctable['X_IMAGE'][blob_ind] - (float(width) / 2.), 
            srctable['Y_IMAGE'][blob_ind] - (float(height) / 2.),
            srctable['FLUX_MAX'][blob_ind],
            srctable['FLUX_RADIUS'][blob_ind],
            srctable['SNR_WIN'][blob_ind])
    '''

def compute_delta_position(centroid, pixelscale=6.628, rotation=0.):
    """
    Convert the centroid coordinates to a delta (XEL, EL)

    pixelscale: pixel scale for the image [arcsec/px]
    rotation: the field rotation of the image w.r.t. the local horizontal frame [deg]
    """
    rotation = rotation * np.pi/180.0
    cent =  ((+centroid[0]*np.cos(rotation) + centroid[1]*np.sin(rotation)) * pixelscale * u.arcsec,
             (-centroid[0]*np.sin(rotation) + centroid[1]*np.cos(rotation)) * pixelscale * u.arcsec)
    return cent

def get_mars_ephemeris(timedate):
    """
    Get the ephemeris of Mars given a particular julian date
    """
    t = Time(timedate)
    with solar_system_ephemeris.set('builtin'):
        mars = get_body('mars', t) 
    return mars 

def extract_obstime_from_name(filename, tz=13):
    """
    Talks a filename in XSC convention and generates a Time object based on it.

    filename: the XSC convention filename from which time and date will be extracted
    tz: the timezone that the date and time are referenced to (+13 for NZT)
    """
    name = filename.split("/")[-1]
    datebits = name.split("--")
    (Y,M,D) = datebits[0].split("-")
    (h,m,s) = datebits[1].split("-")
    ms = datebits[2].split(".")[0]
    tz = TimezoneInfo(utc_offset=tz*u.hour)
    t = datetime(int(Y), int(M), int(D), int(h), int(m), int(s), 1000*int(ms), tzinfo=tz)
    obstime = Time(t)
    obstime.format = 'unix'
    return obstime

# Defaults
dirfilename = "/data6/fc1/extracted/master_2020-01-06-06-21-22/"
imglist = "imglist.txt"
coordlist = "coordlist.txt"
pixelscale = 6.628
timezone = 13
xsc = 1
fieldrotation = 2.914 
minflux = 450000
minsize = 10

# Parse arguments
for arg in sys.argv[1:]:
    (option, value) = arg.split("=", 1)
    if option == "dirfile":
        dirfilename = value
    elif option == "imglist":
        imglist = value
    elif option == "pixelscale":
        pixelscale = float(value)
    elif option == "timezone":
        timezone = int(value)
    elif option == "output":
        coordlist = value
    elif option == "xsc0":
        xsc = 0
    elif option == "xsc1":
        xsc = 1
    elif option == "fieldrotation":
        fieldrotation = float(value)
    elif option == "minflux":
        minflux = float(value)
    elif option == "minsize":
        minsize = float(value)
    else:
        print("Unrecognized option " + option)
        sys.exit()

# Load GPS data from dirfile
df = gd.dirfile(dirfilename, gd.RDONLY)
TIME = df.getdata("TIME",
        first_frame=0,
        first_sample=0,
        num_frames=df.nframes-1,
        num_samples=0,
        return_type=gd.FLOAT64)
LAT = df.getdata("LAT",
        first_frame=0,
        first_sample=0,
        num_frames=df.nframes-1,
        num_samples=0,
        return_type=gd.FLOAT64)
LON = df.getdata("LON",
        first_frame=0,
        first_sample=0,
        num_frames=df.nframes-1,
        num_samples=0,
        return_type=gd.FLOAT64)

output = open(coordlist, "w")
writer = csv.writer(output)
writer.writerow(("# Az [deg]", "El [deg]", "UTC [s]", "Lat [deg]", "Lon [deg]", "X [px]", "Y [px]", "Flux []", "Flux Radius [px]", "Metric"))

filelist = open(imglist, "r")
prev_centroid = None

for filename in filelist.readlines():
    filename = filename.strip()

    # Get the EL, XEL coordinates for Mars w.r.t. to the center of the frame
    # filename = "/data6/xsc1/images/2020-01-07/2020-01-07--07-15-59--380.fits"
    centroid = find_brightest_biggest(filename, minsize=minsize, minflux=minflux)
    if centroid is None: # or (prev_centroid is not None and abs(centroid[1] - prev_centroid[1]) > 10):
        continue
    prev_centroid = centroid
    (d_xel, d_el) = compute_delta_position(centroid, pixelscale=pixelscale, rotation=fieldrotation)

    # Extract date/time from filename 
    obstime = extract_obstime_from_name(filename, tz=timezone)

    # Get GPS location
    time_ind = np.argmin(abs(TIME - obstime.value))
    lat = LAT[time_ind] * u.deg
    lon = LON[time_ind] * u.deg 
    if abs(TIME[time_ind] - obstime.value) > 10:
        print("Imprecise GPS time > 10 s")
    location = EarthLocation.from_geodetic(lon, lat) 

    # Get Mars ephemeris
    with solar_system_ephemeris.set('builtin'):
        mars = get_body('mars', obstime, location).transform_to(AltAz(obstime=obstime, location=location))

    # Compute the centre of frame
    center_coords = (
            (mars.az - d_xel.to(u.deg) / np.cos(mars.alt)).value, 
            (mars.alt - d_el.to(u.deg)).value,
            TIME[time_ind],
            lat,
            lon,
            centroid[0],
            centroid[1],
            centroid[2],
            centroid[3],
            centroid[4])

    # Write to file
    writer.writerow(center_coords)
    output.flush()
    print(filename)
    print("\033[0;32m***** COORDINATES " + str(center_coords) + "*******\033[0m")
