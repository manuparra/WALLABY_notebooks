import io
import tarfile
from django.http import HttpResponse
from urllib.request import pathname2url
from api.utils.io import tarfile_write
from api.decorators import basic_auth
from tables.models import Product, Instance, Detection, Run


PRODUCTS = ['mom0', 'mom1', 'mom2',
            'cube', 'mask', 'chan', 'spec']


@basic_auth
def instance_products(request):
    """Download products for all detections of an instance
    from the Admin portal.

    """
    # TODO(austin): This check could be wrapped as a decorator
    if not request.user.is_authenticated:
        return HttpResponse('Unauthorized', status=401)

    instance_id = request.GET.get('id', None)
    if not instance_id:
        return HttpResponse('id does not exist.', status=400)
    try:
        instance_id = int(instance_id)
    except ValueError:
        return HttpResponse('id is not an integer.', status=400)

    # TODO(austin): refactor into function
    instance = Instance.objects.filter(id=instance_id).first()
    if not instance:
        return HttpResponse('instance not found.', status=404)

    detections = Detection.objects.filter(instance=instance)
    if not instance:
        return HttpResponse('no detections for this instance.', status=404)

    fh = io.BytesIO()
    with tarfile.open(fileobj=fh, mode='w:gz') as tar:
        for d in detections:
            name = f"{d.run.name}_{d.instance.id}_{d.name}"
            name = pathname2url(name.replace(' ', '_'))
            product = Product.objects.filter(detection=d).first()
            if product is None:
                return HttpResponse(
                    f'no products for detection {d.id}.',
                    status=404
                )
            folder = f'{d.name}'.replace(' ', '_')
            tarfile_write(tar, f'{folder}/{name}_mom0.fits', product.mom0)
            tarfile_write(tar, f'{folder}/{name}_mom1.fits', product.mom1)
            tarfile_write(tar, f'{folder}/{name}_mom2.fits', product.mom2)
            tarfile_write(tar, f'{folder}/{name}_cube.fits', product.cube)
            tarfile_write(tar, f'{folder}/{name}_mask.fits', product.mask)
            tarfile_write(tar, f'{folder}/{name}_chan.fits', product.chan)
            tarfile_write(tar, f'{folder}/{name}_spec.txt', product.spec)

    data = fh.getvalue()
    size = len(data)
    response = HttpResponse(data, content_type='application/x-tar')
    response['Content-Disposition'] = f'attachment; \
        filename={instance.run.name}_{instance.filename}.tar'
    response['Content-Length'] = size
    return response


@basic_auth
def detection_products(request):
    detect_id = request.GET.get('id', None)
    if not detect_id:
        return HttpResponse('id does not exist.', status=400)

    try:
        detect_id = int(detect_id)
    except ValueError:
        return HttpResponse('id is not an integer.', status=400)

    product_arg = request.GET.get('product', None)
    if product_arg is not None:
        product_arg = product_arg.lower()
        if product_arg not in PRODUCTS:
            return HttpResponse('not a valid detection product.', status=400)

    if product_arg is None:
        product = Product.objects.filter(detection=detect_id)\
            .select_related(
                'detection',
                'detection__instance',
                'detection__run')\
            .only(
                'detection__name',
                'detection__instance__id',
                'detection__run__name',
                'mom0',
                'mom1',
                'mom2',
                'cube',
                'mask',
                'chan',
                'spec')\
            .first()

        if not product:
            return HttpResponse('Products not found.', status=404)

        detection = product.detection
        name = f"{detection.run.name}_{detection.instance.id}_{detection.name}"
        name = pathname2url(name.replace(' ', '_'))

        fh = io.BytesIO()
        with tarfile.open(fileobj=fh, mode='w:gz') as tar:
            tarfile_write(tar, f'{name}_mom0.fits', product.mom0)
            tarfile_write(tar, f'{name}_mom1.fits', product.mom1)
            tarfile_write(tar, f'{name}_mom2.fits', product.mom2)
            tarfile_write(tar, f'{name}_cube.fits', product.cube)
            tarfile_write(tar, f'{name}_mask.fits', product.mask)
            tarfile_write(tar, f'{name}_chan.fits', product.chan)
            tarfile_write(tar, f'{name}_spec.txt', product.spec)

        data = fh.getvalue()
        size = len(data)

        response = HttpResponse(data, content_type='application/x-tar')
        response['Content-Disposition'] = f'attachment; filename={name}.tar.gz'
        response['Content-Length'] = size
        return response

    else:
        product = Product.objects.filter(detection=detect_id)\
            .select_related(
                'detection',
                'detection__instance',
                'detection__run')\
            .only(
                'detection__name',
                'detection__instance__id',
                'detection__run__name',
                product_arg)
        if not product:
            return HttpResponse('products not found.', status=404)

        detect_name = product[0].detection.name
        run_name = product[0].detection.run.name
        inst_name = product[0].detection.instance.id
        name = f"{run_name}_{inst_name}_{detect_name}"
        name = pathname2url(name.replace(' ', '_'))

        data = getattr(product[0], product_arg)
        size = len(data)

        content_type = 'image/fits'
        ext = "fits"
        if product_arg == "spectrum":
            content_type = "text/plain"
            ext = "txt"

        response = HttpResponse(data, content_type=content_type)
        response['Content-Disposition'] = f'attachment; \
            filename={name}_{product_arg}.{ext}'
        response['Content-Length'] = size
        return response


@basic_auth
def run_products(request):
    """Download products for all detections of a run
    from the Admin portal.

    """
    if not request.user.is_authenticated:
        return HttpResponse('Unauthorized', status=401)

    run_id = request.GET.get('id', None)
    if not run_id:
        return HttpResponse('id does not exist.', status=400)

    try:
        run_id = int(run_id)
    except ValueError:
        return HttpResponse('id is not an integer.', status=400)

    run = Run.objects.filter(id=run_id).first()
    if not run:
        return HttpResponse('run not found.', status=404)

    detections = Detection.objects.filter(run=run)
    if detections is None:
        return HttpResponse('no detections for this run.', status=404)

    fh = io.BytesIO()
    with tarfile.open(fileobj=fh, mode='w:gz') as tar:
        for d in detections:
            name = f"{d.run.name}_{d.instance.id}_{d.name}"
            name = pathname2url(name.replace(' ', '_'))
            product = Product.objects.filter(detection=d).first()
            if product is None:
                return HttpResponse(
                    f'no products for detection {d.id}.',
                    status=404
                )
            folder = f'{d.name}'.replace(' ', '_')
            tarfile_write(tar, f'{folder}/{name}_mom0.fits', product.mom0)
            tarfile_write(tar, f'{folder}/{name}_mom1.fits', product.mom1)
            tarfile_write(tar, f'{folder}/{name}_mom2.fits', product.mom2)
            tarfile_write(tar, f'{folder}/{name}_cube.fits', product.cube)
            tarfile_write(tar, f'{folder}/{name}_mask.fits', product.mask)
            tarfile_write(tar, f'{folder}/{name}_chan.fits', product.chan)
            tarfile_write(tar, f'{folder}/{name}_spec.txt', product.spec)

    data = fh.getvalue()
    size = len(data)

    response = HttpResponse(data, content_type='application/x-tar')
    response['Content-Disposition'] = f'attachment; \
        filename={run.id}_{run.name}_products.tar'
    response['Content-Length'] = size
    return response


def _build_detection(detection):
    det = \
        f'<TR>\n' \
        f'<TD>{detection.name}</TD>\n' \
        f'<TD>{detection.id}</TD>\n' \
        f'<TD>{detection.x}</TD>\n' \
        f'<TD>{detection.y}</TD>\n' \
        f'<TD>{detection.z}</TD>\n' \
        f'<TD>{detection.x_min}</TD>\n' \
        f'<TD>{detection.x_max}</TD>\n' \
        f'<TD>{detection.y_min}</TD>\n' \
        f'<TD>{detection.y_max}</TD>\n' \
        f'<TD>{detection.z_min}</TD>\n' \
        f'<TD>{detection.z_max}</TD>\n' \
        f'<TD>{detection.n_pix}</TD>\n' \
        f'<TD>{detection.f_min}</TD>\n' \
        f'<TD>{detection.f_max}</TD>\n' \
        f'<TD>{detection.f_sum}</TD>\n' \
        f'<TD>{"" if detection.rel is None else detection.rel}</TD>\n' \
        f'<TD>{detection.flag}</TD>\n' \
        f'<TD>{detection.rms}</TD>\n' \
        f'<TD>{detection.w20}</TD>\n' \
        f'<TD>{detection.w50}</TD>\n' \
        f'<TD>{detection.ell_maj}</TD>\n' \
        f'<TD>{detection.ell_min}</TD>\n' \
        f'<TD>{detection.ell_pa}</TD>\n' \
        f'<TD>{detection.ell3s_maj}</TD>\n' \
        f'<TD>{detection.ell3s_min}</TD>\n' \
        f'<TD>{detection.ell3s_pa}</TD>\n' \
        f'<TD>{"" if detection.kin_pa is None else detection.kin_pa}</TD>\n' \
        f'<TD>{detection.err_x}</TD>\n' \
        f'<TD>{detection.err_y}</TD>\n' \
        f'<TD>{detection.err_z}</TD>\n' \
        f'<TD>{detection.err_f_sum}</TD>\n' \
        f'<TD>{"" if detection.ra is None else detection.ra}</TD>\n' \
        f'<TD>{"" if detection.dec is None else detection.dec}</TD>\n' \
        f'<TD>{"" if detection.freq is None else detection.freq}</TD>\n' \
        f'<TD>{"" if detection.l is None else detection.l}</TD>\n' \
        f'<TD>{"" if detection.b is None else detection.b}</TD>\n' \
        f'<TD>{"" if detection.v_rad is None else detection.v_rad}</TD>\n' \
        f'<TD>{"" if detection.v_opt is None else detection.v_opt}</TD>\n' \
        f'<TD>{"" if detection.v_app is None else detection.v_app}</TD>\n' \
        f'</TR>\n'
    return det


# TODO(austin): linting on this stuff
def _build_catalog(detections, date, version):
    cat = \
        f'<?xml version="1.0" ?>\n' \
        f'<VOTABLE version="1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
        f'xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n' \
        f'<RESOURCE>\n' \
        f'<DESCRIPTION>Source catalogue created by the Source Finding Application(SoFiA)</DESCRIPTION>\n' \
        f'<PARAM name="Creator" datatype="char" arraysize="*" value="{"" if version is None else version}"' \
        f' ucd="meta.id;meta.software"/>\n' \
        f'<PARAM name="Time" datatype="char" arraysize="*" value="{date}" ' \
        f'ucd="time.creation"/>\n' \
        f'<TABLE ID="SoFiA_source_catalogue" name="SoFiA source catalogue">\n' \
        f'<FIELD arraysize="32" datatype="char" name="name" unit="" ucd="meta.id"/>\n' \
        f'<FIELD datatype="long" name="id" unit="" ucd="meta.id"/>\n' \
        f'<FIELD datatype="double" name="x" unit="pix" ucd="pos.cartesian.x"/>\n' \
        f'<FIELD datatype="double" name="y" unit="pix" ucd="pos.cartesian.y"/>\n' \
        f'<FIELD datatype="double" name="z" unit="pix" ucd="pos.cartesian.z"/>\n' \
        f'<FIELD datatype="long" name="x_min" unit="pix" ucd="pos.cartesian.x;stat.min"/>\n' \
        f'<FIELD datatype="long" name="x_max" unit="pix" ucd="pos.cartesian.x;stat.max"/>\n' \
        f'<FIELD datatype="long" name="y_min" unit="pix" ucd="pos.cartesian.y;stat.min"/>\n' \
        f'<FIELD datatype="long" name="y_max" unit="pix" ucd="pos.cartesian.y;stat.max"/>\n' \
        f'<FIELD datatype="long" name="z_min" unit="pix" ucd="pos.cartesian.z;stat.min"/>\n' \
        f'<FIELD datatype="long" name="z_max" unit="pix" ucd="pos.cartesian.z;stat.max"/>\n' \
        f'<FIELD datatype="long" name="n_pix" unit="" ucd="meta.number;instr.pixel"/>\n' \
        f'<FIELD datatype="double" name="f_min" unit="Jy/beam" ucd="phot.flux.density;stat.min"/>\n' \
        f'<FIELD datatype="double" name="f_max" unit="Jy/beam" ucd="phot.flux.density;stat.max"/>\n' \
        f'<FIELD datatype="double" name="f_sum" unit="Jy*Hz" ucd="phot.flux"/>\n' \
        f'<FIELD datatype="double" name="rel" unit="" ucd="stat.probability"/>\n' \
        f'<FIELD datatype="long" name="flag" unit="" ucd="meta.code.qual"/>\n' \
        f'<FIELD datatype="double" name="rms" unit="Jy/beam" ucd="instr.det.noise"/>\n' \
        f'<FIELD datatype="double" name="w20" unit="Hz" ucd="spect.line.width"/>\n' \
        f'<FIELD datatype="double" name="w50" unit="Hz" ucd="spect.line.width"/>\n' \
        f'<FIELD datatype="double" name="ell_maj" unit="pix" ucd="phys.angSize"/>\n' \
        f'<FIELD datatype="double" name="ell_min" unit="pix" ucd="phys.angSize"/>\n' \
        f'<FIELD datatype="double" name="ell_pa" unit="deg" ucd="pos.posAng"/>\n' \
        f'<FIELD datatype="double" name="ell3s_maj" unit="pix" ucd="phys.angSize"/>\n' \
        f'<FIELD datatype="double" name="ell3s_min" unit="pix" ucd="phys.angSize"/>\n' \
        f'<FIELD datatype="double" name="ell3s_pa" unit="deg" ucd="pos.posAng"/>\n' \
        f'<FIELD datatype="double" name="kin_pa" unit="deg" ucd="pos.posAng"/>\n' \
        f'<FIELD datatype="double" name="err_x" unit="pix" ucd="stat.error;pos.cartesian.x"/>\n' \
        f'<FIELD datatype="double" name="err_y" unit="pix" ucd="stat.error;pos.cartesian.y"/>\n' \
        f'<FIELD datatype="double" name="err_z" unit="pix" ucd="stat.error;pos.cartesian.z"/>\n' \
        f'<FIELD datatype="double" name="err_f_sum" unit="Jy*Hz" ucd="stat.error;phot.flux"/>\n' \
        f'<FIELD datatype="double" name="ra" unit="deg" ucd="pos.eq.ra"/>\n' \
        f'<FIELD datatype="double" name="dec" unit="deg" ucd="pos.eq.dec"/>\n' \
        f'<FIELD datatype="double" name="freq" unit="Hz" ucd="em.freq"/>\n' \
        f'<FIELD datatype="double" name="l" unit="deg" ucd="pos.galactic.lon"/>\n'\
        f'<FIELD datatype="double" name="b" unit="deg" ucd="pos.galactic.lat"/>\n' \
        f'<FIELD datatype="double" name="v_rad" unit="m/s" ucd="spect.dopplerVeloc.radio"/>\n' \
        f'<FIELD datatype="double" name="v_opt" unit="m/s" ucd="spect.dopplerVeloc.opt"/>\n' \
        f'<FIELD datatype="double" name="v_app" unit="m/s" ucd="spect.dopplerVeloc"/>\n' \
        f'<DATA>\n' \
        f'<TABLEDATA>\n' \
        f'{"".join([_build_detection(detection) for detection in detections])}' \
        f'</TABLEDATA>\n' \
        f'</DATA>\n' \
        f'</TABLE>\n' \
        f'</RESOURCE>\n' \
        f'</VOTABLE>\n'
    return cat


def run_catalog(request):
    if not request.user.is_authenticated:
        return HttpResponse('Unauthorized', status=401)

    run_id = request.GET.get('id', None)
    if not run_id:
        return HttpResponse('id does not exist.', status=400)

    try:
        run_id = int(run_id)
    except ValueError:
        return HttpResponse('id is not an integer.', status=400)

    detections = Detection.objects.filter(run=run_id)
    if not detections:
        return HttpResponse('no detections found.', status=404)

    instance = Instance.objects.filter(run=run_id)\
        .order_by('-run_date')\
        .first()
    if not instance:
        return HttpResponse('no instance found.', status=404)

    cat = _build_catalog(detections, instance.run_date, instance.version)

    name = f'{run_id}_{detections[0].run.name}.xml'

    response = HttpResponse(cat, content_type='text/xml')
    response['Content-Disposition'] = f'attachment; filename={name}'
    response['Content-Length'] = len(cat)
    return response
