#!/usr/bin/env python3


import os
import sys
import astropy
import astropy.io.fits as pyfits
import numpy
import scipy
import scipy.ndimage
import skimage
import skimage.measure
import pandas
import astropy.wcs

import logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="a", format='%(asctime)s %(message)s')

# from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x,y):
    return 0.5*numpy.abs(numpy.dot(x,numpy.roll(y,1))-numpy.dot(y,numpy.roll(x,1)))

if __name__ == "__main__":

    logger = logging.getLogger("W")
    logger.setLevel(logging.DEBUG)

    fn = sys.argv[1]
    print(fn)

    master_df = None

    for fn in sys.argv[1:]:
        hdulist = pyfits.open(fn)
        hdulist.info()
        img = hdulist[0].data
        print(img.shape)

        header = hdulist[0].header
        magzero = 2.5*numpy.log10(header['FLUXMAG0'])
        print("zeorpoint: %2f" % (magzero))
        logger.info("zeorpoint: %2f" % (magzero))

        pixelscale = numpy.fabs(header['CD1_1'] * 3600.)
        print(pixelscale)

        i_mag25 = numpy.power(10., -0.4*(25.0 - magzero))
        pi_mag25 = i_mag25 * pixelscale**2
        print(i_mag25, pi_mag25)

        # find all pixels close to i_mag25
        delta = 0.05*pi_mag25
        contour_pixels = (img > pi_mag25-delta) & (img < pi_mag25+delta)
        #img[~contour_pixels] = numpy.NaN

        #pyfits.PrimaryHDU(data=img, header=header).writeto("contours.fits", overwrite=True)

        contours = skimage.measure.find_contours(img.T, pi_mag25)

        basename, ext = os.path.splitext(fn)

        reg = open("%s_contours.reg" % (basename), "w")
        print("""# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """, file=reg)
        ellipses = open("%s_ellipses.reg" % (basename), "w")
        print("""# Region file format: DS9 version 4.1
    global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """, file=ellipses)

        i=0
        fit_results = []

        for c in contours:
            if (len(c) < 50):
                continue
            i+=1
            if (i%100 == 0):
                print(i)
            sys.stdout.write(".")
            sys.stdout.flush()

            print("polygon(%s)" % (",".join(["%.3f" % p for p in c.flatten()])), file=reg)

            points = numpy.array(c)
            medi_xy = numpy.median(points, axis=0)

            rel_xy = points - medi_xy
            X = rel_xy[:,0:1]
            Y = rel_xy[:,1:2]

            area = PolyArea(points[:,0], points[:,1])

            # x2 = numpy.sum(X**2)
            # y2 = numpy.sum(Y**2)
            # xy = numpy.sum(X*Y)
            # A2 = (x2 + y2) / 2 + numpy.sqrt( ((x2-y2)/2)**2 + xy**2 )
            # B2 = (x2 + y2) / 2 - numpy.sqrt( ((x2-y2)/2)**2 + xy**2 )
            # tan2theta = 2 * xy / (x2 - y2)
            # theta = 0.5*numpy.arctan(tan2theta)
            # print(numpy.sqrt(A2), numpy.sqrt(B2), numpy.rad2deg(theta))
            # scale=0.05
            # print("""ellipse(%f, %f, %f, %f, %f""" % (medi_xy[0], medi_xy[1], numpy.sqrt(A2)*scale, numpy.sqrt(B2)*scale, -1*numpy.rad2deg(theta)), file=ellipses)

            # taken from https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
            # b = numpy.ones_like(X)
            X = points[:, 0:1]
            Y = points[:, 1:2]
            zero = numpy.zeros_like(X)
            Am = numpy.hstack([X**2, X * Y, Y**2, X, Y])
            bm = numpy.ones_like(X)

            use_for_fit = numpy.isfinite(X[:,0])
            ds9format = ['color={blue}', 'color={orange}', 'color={red} width=3']
            for iter in range(3):

                #print(A)
                _Am  = Am[use_for_fit, :]
                _bm = bm[use_for_fit]
                # print(_Am.shape, _bm.shape)

                x = numpy.linalg.lstsq(_Am, _bm)[0].squeeze()
                A = x[0]
                B = x[1]
                C = x[2]
                D = x[3]
                E = x[4]
                F = -1.
                if (((B**2) - 4*A*C) >= 0):
                    continue

                a = -numpy.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( A + C + numpy.sqrt((A-C)**2 + B**2))) / (B**2 - 4*A*C)
                b = -numpy.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( A + C - numpy.sqrt((A-C)**2 + B**2))) / (B**2 - 4*A*C)
                x0 = (2*C*D - B*E) / (B**2 - 4*A*C)
                y0 = (2*A*E - B*D) / (B**2 - 4*A*C)
                theta = numpy.rad2deg( numpy.arctan( (1/B) * (C - A - numpy.sqrt((A-C)**2 + B**2)) ) )
                # print(a,b,x0,y0,theta, medi_xy[0], medi_xy[1])
                print("""ellipse(%f, %f, %f, %f, %f) # %s""" % (x0, y0, a, b, theta, ds9format[iter]), file=ellipses)

                r = numpy.sum(Am*x, axis=1)
                # print(r)
                # print(r.shape)
                _stats = numpy.nanpercentile(r[use_for_fit], [16,50,84])
                _median = _stats[1]
                _sigma = 0.5*(_stats[2]-_stats[0])
                bad = (r > (_median+3*_sigma)) | (r < (_median-3*_sigma))
                use_for_fit[bad] = False

            area_ellipse = numpy.pi * a * b
            # break

            fit_results.append([a,b,theta,x0,y0, area, area_ellipse])

        df = pandas.DataFrame(
            data=fit_results,
            columns=['A', 'B', 'theta', 'x0', 'y0', 'area_contours', "area_ellipse"]
        )

        wcs = astropy.wcs.WCS(header)
        radec = wcs.all_pix2world(df[['x0', 'y0']].to_numpy(), 0)
        print(radec)
        df.loc[:, ['ra', 'dec']] = radec
        df.loc[:, 'filename'] = fn
        df.loc[:, ['a_arcsec', 'b_arcsec']] = df[['A','B']].to_numpy() * pixelscale
        df.loc[:, 'area_error'] = (df['area_ellipse']-df['area_contours']) / df['area_contours']
        df.loc[:, 'good_fit'] = numpy.fabs(df['area_error']) < 0.05
        df.info()

        if (master_df is None):
            master_df = df
        else:
            master_df = master_df.append(df, ignore_index=False)

    master_df.info()
    master_df.to_csv("master_df.csv")

    # also write master region file
    ellipses = open("master_ellipses.reg", "w")
    print("""# Region file format: DS9 version 4.1
    global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    """, file=ellipses)
    for i,row in master_df.iterrows():
        print("""ellipse(%f, %f, %f", %f", %f)""" % (row['ra'], row['dec'], row['a_arcsec'], row['b_arcsec'], row['theta']), file=ellipses)

