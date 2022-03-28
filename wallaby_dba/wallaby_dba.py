import os
import io
import sys
import math

import numpy as np

import django
from django.db import models

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import PercentileInterval
from astroquery.skyview import SkyView
from astropy.utils.data import clear_download_cache

Run, Instance, Detection, Product, Source  = None, None, None, None, None
SourceDetection, Comment, Tag, TagDetection, TagSourceDetection = None, None, None, None, None



# utils
def _write_bytesio_to_file(filename, bytesio):
    """Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())
            

def _write_zipped_fits_file(filename, product, compress=True):
    """Compress a .fits file as .fits.gz for a data product.
   
    """
    with io.BytesIO() as buf:
        buf.write(product)
        buf.seek(0)
        if not os.path.isfile(filename):
            _write_bytesio_to_file(filename, buf)
            if compress:
                os.system(f'gzip {filename}')


def _write_products(products, prefix):
    _write_zipped_fits_file('%s_cube.fits' % (prefix), products.cube)
    _write_zipped_fits_file('%s_chan.fits' % (prefix), products.chan)
    _write_zipped_fits_file('%s_mask.fits' % (prefix), products.mask)
    _write_zipped_fits_file('%s_mom0.fits' % (prefix), products.mom0)
    _write_zipped_fits_file('%s_mom1.fits' % (prefix), products.mom1)
    _write_zipped_fits_file('%s_mom2.fits' % (prefix), products.mom2)

    # Open spectrum
    with io.BytesIO() as buf:
        buf.write(b''.join(products.spec))
        buf.seek(0)
        spec_file  = '%s_spec.txt' % (prefix)
        if not os.path.isfile(spec_file):
            _write_bytesio_to_file(spec_file, buf)


# Retrieve catalogue by tag
def get_catalog(tag):
    tag = str(tag)
    if tag == "":
        sys.stderr.write("Please specify a tag to extract a source catalogue, e.g.:\ntable = get_catalog(tag=\"NGC 5044 DR1\")\n")
        return None
    
    table = Table()
    
    # Get field names
    detection_field_names = [field.name for field in Detection._meta.fields if not isinstance(field, models.ForeignKey)]
    detection_field_names.remove("name")
    detection_field_names.remove("unresolved")
    detection_field_names.remove("v_opt")
    detection_field_names.remove("v_app")
    detection_field_names.remove("v_rad")
    detection_field_names.remove("l")
    detection_field_names.remove("b")
    detection_field_names.remove("v_opt_peak")
    detection_field_names.remove("v_app_peak")
    detection_field_names.remove("v_rad_peak")
    detection_field_names.remove("l_peak")
    detection_field_names.remove("b_peak")
    source_field_names = [field.name for field in Source._meta.fields if not isinstance(field, models.ForeignKey)]
    source_field_names.remove("id")
    
    # Get sources and detections
    sources = [
        Source.objects.get(id=sd.source_id) for sd in [
            SourceDetection.objects.get(id=tsd.source_detection_id) for tsd in 
                TagSourceDetection.objects.filter(tag_id=Tag.objects.get(name=tag).id)
        ]
    ]
    detections = [
        Detection.objects.get(id=sd.detection_id) for sd in [
            SourceDetection.objects.get(id=tsd.source_detection_id) for tsd in 
                TagSourceDetection.objects.filter(tag_id=Tag.objects.get(name=tag).id)
        ]
    ]
    
    # Add columns to the table
    for field in source_field_names:
        if field == 'name':
            table[field] = [getattr(s, field) for s in sources]
        else:
            table[field] = np.array([getattr(s, field) for s in sources], dtype=float)
    for field in detection_field_names:
        table[field] = np.array([getattr(d, field) for d in detections], dtype=float)
    
    # Extract and add comments, if any
    column_comments = []
    for i in range(len(table)):
        column_comments.append([])
        comments = Comment.objects.filter(detection=table["id"][i])
        for comment in comments:
            column_comments[i].append(comment.comment + " (" + comment.author + ")")
    table.add_column(col=column_comments, name="comments")
    
    # Extract and add tags, if any
    column_tags = []
    for i in range(len(table)):
        column_tags.append([])
        tags = TagSourceDetection.objects.filter(source_detection_id=SourceDetection.objects.get(detection_id=table["id"][i]))
        for tag in tags:
            column_tags[i].append(Tag.objects.get(id=tag.tag_id).name)
    table.add_column(col=column_tags, name="tags")
    
    return table


def save_catalog(tag, *args, **kwargs):
    """Write catalog of tagged sources. Remove object columns for write to file.

    """
    table = get_catalog(tag)
    table.remove_columns(['comments', 'tags'])
    table.write(*args, **kwargs)


def save_products_for_source(tag, source_name, *args, **kwargs):
    """Save source finding output products for a given source name.

    """
    table = get_catalog(tag)
    try:
        idx = list(table['name']).index(source_name)
        row = table[idx]
    except Exception as e:
        sys.stderr.write("Could not find source with provided name in tagged data.")
        return None
    detection = Detection.objects.get(id=row['id'])
    products = Product.objects.get(detection=detection)

    name = source_name.replace(' ', '_')
    parent = f'{name}_products'
    if not os.path.isdir(parent):
        os.mkdir(parent)
    
    # Write fits files
    _write_products(products, f'{parent}/{name}')
    
    return


def save_products(tag, *args, **kwargs):
    """Save source finding output products for a given tag

    """
    table = get_catalog(tag)
    parent = '%s_products' % tag.replace(' ', '_')
    if not os.path.isdir(parent):
        os.mkdir(parent)

    for row in table:
        name = row['name'].replace(' ', '_')
        if not os.path.isdir(f'{parent}/{name}'):
            os.mkdir(f'{parent}/{name}')
        detection = Detection.objects.get(id=row['id'])
        products = Product.objects.get(detection=detection)
        _write_products(products, f'{parent}/{name}/{name}')

    os.system(f'tar -czf {parent}.tar.gz {parent}')

    return


# Print list of supported tags
def print_tags():
    tags = Tag.objects.all()
    for tag in tags:
        print("{:20s}\t{:s}".format("\"" + tag.name + "\"", tag.description))
    return


# Retrieve FITS image from database
def get_image(product):
    with io.BytesIO() as buf:
        buf.write(product)
        buf.seek(0)
        hdu = fits.open(buf)[0]
        return hdu.data, hdu.header


# Retrieve spectrum from database
def get_spectrum(product):
    with io.BytesIO() as buf:
        buf.write(b"".join(product))
        buf.seek(0)
        return np.loadtxt(buf, dtype="float", comments="#", unpack=True)


# Retrieve DSS image from Skyview
def retrieve_dss_image(longitude, latitude, width, height):
    hdulist = SkyView.get_images(
        position="{}, {}".format(longitude, latitude),
        survey=["DSS"],
        coordinates="J2000",
        projection="Tan",
        width=width*u.deg,
        height=height*u.deg,
        cache=None
    )
    return hdulist[0][0]


# Create overview plot
def overview_plot(id):
    interval = PercentileInterval(95.0)
    plt.rcParams["figure.figsize"] = (16, 12)
    fig = plt.figure()
    
    # Retrieve products from database
    products = Product.objects.get(detection=id)
    
    # Open moment 0 image
    mom0, header = get_image(products.mom0)
    mom1, header = get_image(products.mom1)
    spectrum = get_spectrum(products.spec)
    wcs = WCS(header)
    
    # Extract coordinate information
    nx = header["NAXIS1"]
    ny = header["NAXIS2"]
    lon, lat = wcs.all_pix2world(nx/2, ny/2, 0)
    tmp1, tmp3 = wcs.all_pix2world(0, ny/2, 0)
    tmp2, tmp4 = wcs.all_pix2world(nx, ny/2, 0)
    width = np.rad2deg(math.acos(math.sin(np.deg2rad(tmp3)) * math.sin(np.deg2rad(tmp4)) + math.cos(np.deg2rad(tmp3)) * math.cos(np.deg2rad(tmp4)) * math.cos(np.deg2rad(tmp1 - tmp2))))
    tmp1, tmp3 = wcs.all_pix2world(nx/2, 0, 0)
    tmp2, tmp4 = wcs.all_pix2world(nx/2, ny, 0)
    height = np.rad2deg(math.acos(math.sin(np.deg2rad(tmp3)) * math.sin(np.deg2rad(tmp4)) + math.cos(np.deg2rad(tmp3)) * math.cos(np.deg2rad(tmp4)) * math.cos(np.deg2rad(tmp1 - tmp2))))
    
    # Plot DSS image with HI contours
    try:
        hdu_opt = retrieve_dss_image(lon, lat, width, height)
        wcs_opt = WCS(hdu_opt.header)
        
        bmin, bmax = interval.get_limits(hdu_opt.data)
        ax = plt.subplot(2, 2, 2, projection=wcs_opt)
        ax.imshow(hdu_opt.data, origin="lower")
        ax.contour(mom0, transform=ax.get_transform(wcs), levels=np.logspace(2.0, 5.0, 10), colors="lightgrey", alpha=1.0)
        ax.grid(color="grey", ls="solid")
        ax.set_xlabel("Right ascension (J2000)")
        ax.set_ylabel("Declination (J2000)")
        ax.tick_params(axis="x", which="both", left=False, right=False)
        ax.tick_params(axis="y", which="both", top=False, bottom=False)
        ax.set_title("DSS + Moment 0")
        ax.set_aspect(np.abs(wcs_opt.wcs.cdelt[1] / wcs_opt.wcs.cdelt[0]))
    except:
        sys.stderr.write("Failed to retrieve DSS image.\n")
        pass
    
    # Plot moment 0
    ax2 = plt.subplot(2, 2, 1, projection=wcs)
    ax2.imshow(mom0, origin="lower")
    ax2.grid(color="grey", ls="solid")
    ax2.set_xlabel("Right ascension (J2000)")
    ax2.set_ylabel("Declination (J2000)")
    ax2.tick_params(axis="x", which="both", left=False, right=False)
    ax2.tick_params(axis="y", which="both", top=False, bottom=False)
    ax2.set_title("Moment 0")
    
    # Add beam size
    ax2.add_patch(Ellipse((5, 5), 5, 5, 0, edgecolor="grey", facecolor="grey"))

    # Plot moment 1
    bmin, bmax = interval.get_limits(mom1)
    ax3 = plt.subplot(2, 2, 3, projection=wcs)
    ax3.imshow(mom1, origin="lower", vmin=bmin, vmax=bmax, cmap=plt.get_cmap("gist_rainbow"))
    ax3.grid(color="grey", ls="solid")
    ax3.set_xlabel("Right ascension (J2000)")
    ax3.set_ylabel("Declination (J2000)")
    ax3.tick_params(axis="x", which="both", left=False, right=False)
    ax3.tick_params(axis="y", which="both", top=False, bottom=False)
    ax3.set_title("Moment 1")
    
    # Plot spectrum
    xaxis = spectrum[1] / 1e+6
    data  = 1000.0 * np.nan_to_num(spectrum[2])
    xmin = np.nanmin(xaxis)
    xmax = np.nanmax(xaxis)
    ymin = np.nanmin(data)
    ymax = np.nanmax(data)
    ymin -= 0.1 * (ymax - ymin)
    ymax += 0.1 * (ymax - ymin)
    ax4 = plt.subplot(2, 2, 4)
    ax4.step(xaxis, data, where="mid", color="royalblue")
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Flux density (mJy)")
    ax4.set_title("Spectrum")
    ax4.grid(True)
    ax4.set_xlim([xmin, xmax])
    ax4.set_ylim([ymin, ymax])
    
    fig.canvas.draw()
    plt.tight_layout()
    
    return plt


def save_overview(tag, *args, **kwargs):
    """Save overview plots for tagged sources

    """
    table = get_catalog(tag)
    parent = '%s_overview' % tag.replace(' ', '_')
    if not os.path.isdir(parent):
        os.mkdir(parent)    

    for row in table:
        name = row['name'].replace(' ', '_')
        p = overview_plot(row['id'])
        p.savefig(f"{parent}/{name}_overview.png")
        p.close()

    os.system(f'tar -czf {parent}.tar.gz {parent}')

    return


def parse_spectrum_to_table(spectrum):
    """Takes the spectrum stored in the database and parses the object to an Astropy.table
    Columns: Channel, Frequency, Flux density, Pixels
    
    """
    array = []
    with io.BytesIO() as buf:
        buf.write(b''.join(spectrum))
        buf.seek(0)
        text = buf.getbuffer().tobytes().decode('utf-8')
        lines = text.strip().split('\n')
        for line in lines:
            if not line.startswith('#'):
                chan, freq, flux, pix = line.strip().split()
                array.append(np.array([int(chan), float(freq), float(flux), int(pix)]))

    t = Table(
        np.array(array),
        names=('Channel', 'Frequency', 'Flux Density', 'Pixels'),
        dtype=(int, np.float32, np.float32, int)
    )
    return t


def casda_export_products(table, directory):
    """Export data products for sources in an astropy.Table object
    to an output directory in a format compatible for CASDA ingest.

    """
    for row in table:
        name = row['name'].replace(' ', '_')
        detection = Detection.objects.get(id=row['id'])
        products = Product.objects.get(detection=detection)
        filename_prefix = f'{directory}/{name}'
        
        # write .fits files
        _write_zipped_fits_file('%s_cube.fits' % (filename_prefix), products.cube, compress=False)
        _write_zipped_fits_file('%s_chan.fits' % (filename_prefix), products.chan, compress=False)
        _write_zipped_fits_file('%s_mask.fits' % (filename_prefix), products.mask, compress=False)
        _write_zipped_fits_file('%s_mom0.fits' % (filename_prefix), products.mom0, compress=False)
        _write_zipped_fits_file('%s_mom1.fits' % (filename_prefix), products.mom1, compress=False)
        _write_zipped_fits_file('%s_mom2.fits' % (filename_prefix), products.mom2, compress=False)

        # write spectrum as fits file
        spectrum_table = parse_spectrum_to_table(products.spec)
        spectrum_table.write('%s_spec.fits' % (filename_prefix), format='fits')

    return

# Connect to WALLABY database
def connect():
    from django.conf import settings
    global Run, Instance, Detection, Product, Source
    global SourceDetection, Comment, Tag, TagDetection, TagSourceDetection
  

    settings.configure(
        DATABASE_ENGINE = 'postgresql_psycopg2',
        DATABASE_NAME = 'wallabydb',
        DATABASE_USER = 'wallaby_user',
        DATABASE_PASSWORD = 'wallaby_user',
        DATABASE_HOST = '161.111.167.192',
        DATABASE_PORT = '18020',
        TIME_ZONE = 'Europe/Madrid',
        DJANGO_SETTINGS_MODULE = "api.settings",
        DJANGO_SECRET_KEY="-=(gyah-@e$-ymbz02mhwu6461zv&1&8uojya413ylk!#bwa-l",
        DJANGO_ALLOW_ASYNC_UNSAFE="True"
    )

    django.setup()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
    from tables.models import Run, Instance, Detection, Product, Source
    from tables.models import SourceDetection, Comment, Tag, TagDetection, TagSourceDetection
    return

def Aconn():
    global Run, Instance, Detection, Product, Source
    global SourceDetection, Comment, Tag, TagDetection, TagSourceDetection
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "api.settings")
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    import django
    django.setup()
    from tables.models import Run, Instance, Detection, Product, Source
    from tables.models import SourceDetection, Comment, Tag, TagDetection, TagSourceDetection

