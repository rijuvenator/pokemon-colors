from __future__ import annotations
import math
import random
import logging
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans
import svg

# define logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# image analysis and runner constants
NCHANNELS = 3
RGBMAX = 255
CHANNEL_A = 3
ONE_HUNDRED = 100

# geometry svg constants
COS60 = math.cos(math.pi/3)
SIN60 = math.sin(math.pi/3)
CIRCLE_DEG = 360
HALFCIRCLE_DEG = 180

# the basic class for opening an image, pre-processing the pixels, running k-means,
# storing the cluster centers (main colors), and computing histogram/summary counts
# to keep this code useful beyond its application here, it is as generic as possible
# requiring only some set of images named 1.png, 2.png, ... and an override of the getImage function
class ImageAnalysis():
    def __init__(self, index, nclusters=3, lumi_cutoff=0.2, alpha_cutoff=200):
        self.index = index
        self.nclusters = nclusters
        self.lumi_cutoff = lumi_cutoff
        self.alpha_cutoff = alpha_cutoff
        self.data_raw = None
        self.data_rgb = None
        self.data_lab = None
        self.kmeans = None
        self.main_rgb = None
        self.counts = None

    # input is RGBA pixel data; alpha channel is the fourth channel
    # for some reason it's not just 0 and 255, but OK, I'll put in a soft cutoff
    # after this, transparent pixels are cut off and the data is RGB
    def cutoffAlpha(self, arr, alpha_cutoff=None):
        if alpha_cutoff is None: alpha_cutoff = self.alpha_cutoff
        result = arr[arr[:,CHANNEL_A] > alpha_cutoff][:,0:NCHANNELS]
        return result

    # input is RGB pixel data
    # using one of the algebraic formulae from here: https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
    # compute perceptual luma; used by cutoffLumi so that dark colors can be cut off
    # output is scalars of the same size as the array
    # this multiplication correctly scalar multiplies each "column" channel of the data given
    def lumi(self, arr):
        luma_sdtv = [0.299, 0.587, 0.114]
        luma_adobe = [0.212, 0.701, 0.087]
        luma_hdtv = [0.2126, 0.7152, 0.0722]
        luma_uhdtv = [0.2627, 0.6780, 0.0593]
        lumivec = luma_adobe
        return (lumivec * arr/RGBMAX).sum(axis=1)

    # input is RGB pixel data
    # after this, low-lumi pixels are cut off
    def cutoffLumi(self, arr, lumi_cutoff=None):
        if lumi_cutoff is None: lumi_cutoff = self.lumi_cutoff
        result = arr[self.lumi(arr) > lumi_cutoff]
        return result

    # encapsulate the image open so that it can be overridden for some other application
    # images courtesy of https://veekun.com/dex/downloads
    # found via reddit https://www.reddit.com/r/pokemon/comments/2f49dr/request_zip_file_of_official_art_of_all_pokemon/
    # open the image, get the data
    def getImage(self):
        im = Image.open(f'pokemon/sugimori/{self.index}.png')
        return im

    # cutoff the alpha so that transparent pixels aren't involved
    # cutoff the darkest pixels so that the main colors aren't too dark
    # convert to LAB so that Euclidean distance for k-means is perceptually uniform
    def computeData(self):
        im = self.getImage()
        self.data_raw = np.array(im.getdata())
        self.data_rgb = self.cutoffAlpha(self.data_raw)
        self.data_rgb = self.cutoffLumi(self.data_rgb)
        self.data_lab = skcolor.rgb2lab(self.data_rgb/RGBMAX)
        # Uncomment this line and comment the one above if you want to try pure RGB without LAB
        #self.data_lab = self.data_rgb/RGBMAX

    # perform the k-means algorithm; save the results as well
    def computeMainColors(self):
        self.kmeans = KMeans(n_clusters = self.nclusters).fit(self.data_lab)
        self.main_rgb = (skcolor.lab2rgb(self.kmeans.cluster_centers_)*RGBMAX).astype(np.uint8)
        # Uncomment this line and comment the one above if you want to try pure RGB without LAB
        #self.main_rgb = (self.kmeans.cluster_centers_*RGBMAX).astype(np.uint8)

    def computeCounts(self):
        # predict to get a list of labels from the original data, then histogram it for weights
        # np.unique gives unique values with optional counts
        # we will save as pandas because it makes manipulations much simpler
        labels, counts = np.unique(self.kmeans.predict(self.data_lab), return_counts=True)
        pdf = pd.DataFrame.from_dict({'label':labels, 'counts':counts})

        # Implementation of Largest Remainder Method as found on StackOverflow
        # in this answer: https://stackoverflow.com/a/13483710
        # basically: normalize, then round everything down
        # figure out the error; this is how many 1's to add back
        # keep track of the decimal parts
        # sort the decimal parts downwards as a priority list for where to add back 1's
        # the number of 1's to add is "error", and the length - error is the number of 0's
        pdf['label'] = pdf['label'].astype(int)
        pdf['normed'] = pdf['counts']/pdf['counts'].sum()
        pdf['rounded'] = (pdf['normed']*ONE_HUNDRED).astype(int)
        pdf['decimals'] = (pdf['normed']*ONE_HUNDRED) - pdf['rounded']
        error = ONE_HUNDRED - pdf['rounded'].sum()
        pdf = pdf.sort_values(by='decimals', ascending=False)
        corrections = [1 for _ in range(error)] + [0 for _ in range(pdf.shape[0] - error)]
        pdf['pixels'] = pdf['rounded'] + corrections
        pdf = pdf.sort_values(by='counts', ascending=False)
        self.counts = pdf

    def run(self):
        self.computeData()
        self.computeMainColors()
        self.computeCounts()
        logging.info(f'Completed image analysis #{self.index} with {self.nclusters} colors')

# this class runs several image analyses and collects the results multithreaded
# also has a couple functions for preview images, used for development before jumping into svg
# leave **iakwargs to pass through to ImageAnalysis as extra args: nclusters, lumi_cutoff, and alpha_cutoff
class ImageRunner():
    def __init__(self, indices=(1, 27), imageScaling=25, **iakwargs):
        self.indices = indices
        self.imageScaling = imageScaling
        self.data = []
        self.nindices = self.indices[1] - self.indices[0] + 1
        self.iakwargs = iakwargs

    # this is the function that will get run multithreaded
    # it returns an ImageAnalysis instance
    @staticmethod
    def mapfunc(argTuple):
        idx, iakwargs = argTuple
        p = ImageAnalysis(idx, **iakwargs)
        p.run()
        return p

    def run(self):
        with Pool(32) as pool:
            data = pool.map( self.mapfunc, [(idx, self.iakwargs) for idx in range(self.indices[0], self.indices[1]+1)] )
        self.data = data

    def showGrid(self):
        # the main grids are (nclusters, nchannels); putting together the array and reshaping
        # to the desired (nindices, nclusters) grid, each with (nchannels) deep
        # then the first dimension becomes x, so scaling means take nclusters * scaling
        # the second dimension becomes y, so scaling means take nindices * scaling
        nclusters = self.data[0].nclusters
        final_array = np.array([analysis.main_rgb for analysis in self.data]).reshape(self.nindices, nclusters, NCHANNELS)
        im = Image.fromarray(final_array).resize((nclusters*self.imageScaling, self.nindices*self.imageScaling), resample=Image.NEAREST)
        im.show()

    def showWeightedGrid(self):
        # for each analysis, counts are sorted by counts already
        # get the label (an integer) and the pixels (computed earlier)
        # the color is the direct integer mapping from the cluster centers -> main_rgb
        # tile( (pixels, 1) ) makes (pixels) copies of the 3-element color array
        # take the WHOLE tile and add to "pockets"; we will then concatenate all the clusters
        # finally, this concatenating represents a single row; append it to data
        data = []
        for analysis in self.data:
            pockets = []
            for row in analysis.counts.itertuples():
                label = row.label
                pixels = row.pixels
                color = analysis.main_rgb[label]
                tile = np.tile(color, (pixels, 1))
                pockets.append(np.tile(color, (pixels, 1)))
            pockets = np.concatenate(pockets)
            data.append(pockets)
        final_array = np.array(data)

        # then the first dimension (tiling dimension) becomes x, so it's ONE_HUNDRED pixels; no need to scale, or scale lightly
        # the second dimension becomes y, so scaling means take nindices * scaling
        im = Image.fromarray(final_array).resize((ONE_HUNDRED * 2, self.nindices*self.imageScaling), resample=Image.NEAREST)
        im.show()

    # a simple structure for unpacking for making hexagons
    # needed a simple way of converting the uint8 values to color, so going with hex codes
    # svg supports a few, this is the simplest I guess
    # https://www.w3.org/TR/css-color-3/#numerical
    def getDataForHexagons(self):
        data = []
        for analysis in self.data:
            breakdown = []
            for row in analysis.counts.itertuples():
                label = row.label
                pct = float(row.normed)
                color = analysis.main_rgb[label]
                hexcolor = '#{0:x}{1:x}{2:x}'.format(*color)
                breakdown.append((pct, hexcolor))
            data.append((analysis.index, breakdown))
        return data

# class for abstracting and holding a hexagon center
# but instantiating it this way also provides a unified coordinate system
# the entire document then only depends on the radius and top/left margins
@dataclass
class Center:
    x: int
    y: int
    r: int

    # you'll have to trust me on the math, but basically
    # the 0, 0 hexagon's center is x0, y0
    # then you can add some values to get the new y and new x
    # note in particular the use of i%2, since the y coordinates are staggered
    @classmethod
    def fromIJR(cls, i, j, r, MARGINX=10, MARGINY=0):
        x0 = MARGINX + r
        y0 = MARGINY + r*SIN60

        y = y0 + r * 2*SIN60   *j + (r*SIN60)*(i%2)
        x = x0 + r * (1+COS60) *i

        obj = cls(x, y, r)
        obj.i = i
        obj.j = j
        return obj

        # I guess I'm keeping the hacked-in i and j here
        # There are methods that rely on them, so be careful
        # but mostly I expect to just be using IJR
        #return cls(x, y, r)

# enum for defining 3 possible ways of assigning a starting rotation angle
# zero means they all start at 0 (pointing right)
# index means use the index number as the starting index
# random means get a random index
class RotationStart(Enum):
    ZERO = 0
    RANDOM = 1
    INDEX = 2

# the major unit holding the information and methods to create svg polygons
# with colors and clip masks, as well as organizing the information
# the index should be the same as the index for ImageAnalysis
class Hexagon:
    def __init__(self, center, index, rotStart=RotationStart.RANDOM):
        self.center = center
        self.index = index

        # it is easy to add a bit of interest by starting the "0" line somewhere else
        # can be index, in which case the line will slowly shift around the hexagon
        # can be random, so generate a random int between 0 and 360 and everything abstracts nicely with no work
        # because I did a decent job on the sector calculation part
        if rotStart == RotationStart.ZERO:
            self.currentRotDeg = 0
        elif rotStart == RotationStart.INDEX:
            self.currentRotDeg = self.index
        elif rotStart == RotationStart.RANDOM:
            self.currentRotDeg = random.randint(0, CIRCLE_DEG)
        else:
            raise ValueError('Received unknown value {rotStart}')

    # compute the points of the hexagon given the center and "radius"
    # since we have a unified system for coordinates and centers, no parameters necessary
    # these started as functions though and still look cleaner with x, y, and r, so they are retained
    def getHexagonPolygon(self):
        x, y, r = self.center.x, self.center.y, self.center.r
        points = [
            (x + r      , y          ),
            (x + r*COS60, y + r*SIN60),
            (x - r*COS60, y + r*SIN60),
            (x - r      , y          ),
            (x - r*COS60, y - r*SIN60),
            (x + r*COS60, y - r*SIN60),
            (x + r      , y          )
        ]
        polygon = svg.Polygon(points=points)
        return polygon

    # clip paths need to be referred to, so each one needs an ID
    def getClipPathID(self):
        return f'cliphex{self.index}'

    # the clip-path property specified by clip_path uses this url(#id) to refer to clip path
    def getClipPathURL(self):
        return f'url(#{self.getClipPathID()})'

    # make a clip path object
    def getClipPath(self):
        clipPath = svg.ClipPath(id=self.getClipPathID(), elements=[self.getHexagonPolygon()])
        return clipPath

    # given a point tuple, origin tuple, and rotation angle, return rotated point
    # need to subtract off the origin, rotate, and then put it back
    # needed for manual rotation of polygon points; see Sector
    @staticmethod
    def RotatePoint(point, origin, theta_deg):
        theta = theta_deg * math.pi/HALFCIRCLE_DEG
        px, py = point
        ox, oy = origin
        x, y = (px-ox, py-oy)
        rx, ry = x*math.cos(theta) - y*math.sin(theta), x*math.sin(theta) + y*math.cos(theta)
        return (rx+ox, ry+oy)

    # create a polygon (that will be masked) to emulate the "pie chart" sector
    # since we have a unified system for coordinates and centers, no coordinate parameters necessary
    # these started as functions though and still look cleaner with x, y, and r, so they are retained
    # rot_deg is there for atomic purposes in case the code is to be repurposed somehow, but
    # this whole thing was designed with the idea that you'd start somewhere, and then add pct incrementally
    # to get all the way around the circle. so you need to rotate the NEXT polygon by the PREVIOUS pct
    # that gives the "pie chart" type thing
    def getSectorPolygon(self, pct, color='red', rot_deg=None):
        angle = pct * 2 * math.pi
        tang = abs(math.tan(angle))
        xsgn = 1 if math.cos(angle)>0 else -1
        ysgn = 1 if math.sin(angle)>0 else -1

        x, y, r = self.center.x, self.center.y, self.center.r

        points = [(x, y)]

        # the specific polygon that needs to be drawn depends on the angle
        # here, pct is "percent of the way around the circle", which helps determine quadrant
        # check each octant one by one to see whether extra points are needed
        # at the same time, one can determine the "penultimate" point, which is the angled segment
        # a given number will stop updating penult at some point, and the final points can be
        # added in after that
        if pct > 0:
            points.append( (x+r, y            ) )
            penult =       (x+r, y+r*ysgn*tang)
        if pct > 1/8.: 
            points.append( (x+r          , y+r) )
            penult =       (x+r*xsgn/tang, y+r)
        if pct > 3/8.:
            points.append( (x-r, y+r          ) )
            penult =       (x-r, y+r*ysgn*tang)
        if pct > 5/8.:
            points.append( (x-r          , y-r) )
            penult =       (x+r*xsgn/tang, y-r)
        if pct > 7/8.:
            points.append( (x+r, y-r          ) )
            penult =       (x+r, y+r*ysgn*tang)

        points.extend([penult, (x,y)])

        # Instead of using the built-in SVG transform, I have to rotate every point myself
        # because SVG applies the clip path BEFORE the rotation, which means the rotation rotates the clip path too
        if rot_deg is None:
            rot_deg = self.currentRotDeg
            self.currentRotDeg += pct * CIRCLE_DEG
        rotated_points = [Hexagon.RotatePoint(p, (x,y), rot_deg) for p in points]

        polygon = svg.Polygon(points=rotated_points, fill=color, clip_path=self.getClipPathURL())
        return polygon

    # breakdown is list[(pct, '#hexcolor')]
    # comes from getDataForHexagons, which is list[(index, breakdown)]
    def getSectorsFromBreakdownData(self, breakdown):
        return [self.getSectorPolygon(pct, color) for pct, color in breakdown]

# wrapper class for defining and rendering an svg canvas
# hexes is expected to be a list of Hexagon instances
class Canvas():
    def __init__(self, hexes=None, width=480, height=550):
        self.width = width
        self.height = height
        self.hexes = hexes if hexes is not None else []

    def boundingBox(self):
        return svg.Rect(x=0, y=0, width=self.width, height=self.height, stroke='black', fill_opacity=0)

    def render(self, bounding_box=False, draw_coords=False, fname='test.svg'):
        elements = [svg.Defs(elements=[h.getClipPath() for h in self.hexes])]

        if draw_coords:
            elements.append(svg.Style(text='.small { font: 10px sans-serif; }'))

        if bounding_box:
            elements.append(self.boundingBox())

        for h in self.hexes:
            # some testing code that I'll leave in for understandability purposes
            # getSectorsFromBreakdownData basically does this, but with real data
            #elements.extend([h.getSectorPolygon(pct, color) for pct, color in [(.2, 'red'), (.5, 'blue'), (.3, 'green')]])
            #elements.extend([h.getSectorPolygon(pct, color) for pct, color in [(1.0, 'red')]])

            # relies on having run getSectorsFromBreakdownData when initializing the hexagons
            # since the data is "external", this is probably fine. it will crash if it hasn't been run, which is OK
            elements.extend(h.computedSectors)

            # get the hexagon polygon, but instead of using as a clip path, just draw an outline
            hp = h.getHexagonPolygon()
            hp.stroke = 'white'
            hp.fill_opacity=0
            elements.append(hp)

            # relies on having used fromIJR constructors so that i and j are available
            if draw_coords:
                txt = svg.Text(x=h.center.x-12, y=h.center.y+2, class_=['small'], text=f'{h.center.i}, {h.center.j}')
                elements.append(txt)

        canvas = svg.SVG(
            width=self.width,
            height=self.height,
            elements=elements
        )
        self.writeCanvas(canvas, fname=fname)

    # I used this temporarily for testing purposes when I just wanted to pass in some elements to draw
    def renderTemp(self, els, fname='test.svg'):
        canvas = svg.SVG(
            width=self.width,
            height=self.height,
            elements=els + [self.boundingBox()]
        )
        self.writeCanvas(canvas, fname=fname)

    def writeCanvas(self, canvas, fname='test.svg'):
        with open(fname, 'w') as f:
            f.write(str(canvas))
            logging.info(f'{fname} written')

# enum for defining 3 possible ways of getting hexagon data
# NORMAL means process the data and use the proportions; any of the rotation starts are good
# EQUAL means process the data and make all proportions equal; most interesting to use in conjunction with rotStart=RotationStart.ZERO
# DUMMY means skip the image processing and use some dummy data instead
class DataMode(Enum):
    NORMAL = 0
    EQUAL = 1
    DUMMY = 2

# the final mega runner class. all parameters except constants and MARGINX and MARGINY for Center can be controlled here
# (in the future if desired, add MARGINX and MARGINY alongside radius to the MainRunner class and wrangle with getCenters())
# the basics are the indices and the radius; then there are parameters for ImageRunner and ImageAnalysis
# kwargs will get forwarded to ImageRunner. If it contains imageScaling, it will get used by ImageRunner
# ImageRunner forwards any unknown keyword arguments forward to ImageAnalysis, allowing for nclusters, lumi_cutoff, and alpha_cutoff
# The renderCanvas method collects all of the actual SVG hexagon center data breakdown declaration rendering code
class MainRunner():
    def __init__(self, indices=(1,151), radius=20, **kwargs):
        self.indices = indices
        self.radius = radius
        self.kwargs = kwargs
        self.runner = None
        logging.info(f'Starting run for image indices {indices[0]}-{indices[1]}')

    # a general guard for the runner that is also resource respectful
    # once MainRunner is instantiated, runAnalyses() will only do the computation once
    # then any of the display functions can be called at will without recomputing
    def runAnalyses(self):
        if self.runner is None:
            logging.info('Starting all image analysis...')
            self.runner = ImageRunner(indices=self.indices, **self.kwargs)
            self.runner.run()
            logging.info('Completed all image analysis')

    # ensures the runner and shows grid
    def showGrid(self):
        self.runAnalyses()
        self.runner.showGrid()

    # ensures the runner and shows weighted grid
    def showWeightedGrid(self):
        self.runAnalyses()
        self.runner.showWeightedGrid()

    # ensures the runner and runs the dedicated data function
    # provides a few other modes for quick testing and variants
    def getDataForHexagons(self, mode=DataMode.NORMAL):
        logging.info(f'Getting hexagon data, {mode.name} data mode')
        if mode == DataMode.DUMMY:
            data = []
            for idx in range(self.indices[0], self.indices[1]+1):
                data.append( (idx, [(.2, 'red'), (.5, 'blue'), (.3, 'green')]) )
        else:
            self.runAnalyses()
            data = self.runner.getDataForHexagons()
            if mode == DataMode.NORMAL:
                pass
            elif mode == DataMode.EQUAL:
                data_equal = []
                for idx, breakdown in data:
                    nclusters = len(breakdown)
                    data_equal.append((idx, [(1./nclusters, color) for pct, color in breakdown]))
                data = data_equal
        return data

    # all right, some explanation needed. keep in mind the hexagon coordinate system
    # so, I need to arrange 151 hexagons in a relatively pleasing way
    # here you can see the number of hexagons in a grid:
    # https://boardgames.stackexchange.com/questions/54818/calculating-the-hexagons-in-a-hexagon-of-hexagons
    # it turns out n = 7 (8 hexagons on a side) is 169 hexagons. so I need to get rid of 18
    # very fortunately, I have 6 sides, so I can just remove 3 in a sane way from each side
    # so first, I will generate the full 169 hexagon grid
    # to prevent negative indices, I will not draw the 0, 0 hexagon, and instead start part of the way down at 0, 4
    # and draw 8 hexagons downward. then the next column starts at 1, 3 with 9 hexagons downwards, etc.
    # if you look at the grid coordinates (turn them on with draw_coords), it should be more clear
    # so config specifies the "top" hexagon in each column, and then how many downward
    # to save redundancy, the "right" half is the "left" half reversed, but only the column numbers need to go from i -> 14-i
    # this is "other_config"
    # now the weird part, which really is difficult without coordinates: need to remove 18 hexes
    # so first I removed the outermost "corners", which are the first 6
    # now I need to remove 2 more from each side. I tried a few different ways and settled on this one that gives a snowflake pattern
    # but I left them commented in. now you can generate the i, j coordinates you want, skipping the VETO18
    def getCenters(self, radius=None):
        if radius is None:
            radius = self.radius

        if (self.indices[1] - self.indices[0] + 1) > 151:
            raise ValueError('More than 151 indices specified; this implementation specifies 151 hexagons. Change the index range to cover 151 indices or less, or modify the getCenters() method to produce the desired set of hexagon centers.')

        logging.info(f'Getting centers, radius {radius}')
        centers = []

        config = [
            (0, 4, 8 ),
            (1, 3, 9 ),
            (2, 3, 10),
            (3, 2, 11),
            (4, 2, 12),
            (5, 1, 13),
            (6, 1, 14),
            (7, 0, 15),
        ]
        other_config = [(14-i, j, n) for i, j, n in reversed(config[:-1])]

        VETO18 = [
            (0, 4),
            (7, 0),
            (14, 4),
            (14, 11),
            (7, 14),
            (0, 11),

            #(0, 5),
            #(1, 3),
            #(6, 1),
            #(8, 1),
            #(13, 3),
            #(14, 5),
            #(14, 10),
            #(13, 11),
            #(8, 14),
            #(6, 14),
            #(1, 11),
            #(0, 10),

            (3, 2),
            (4, 2),
            (10,2),
            (11,2),
            (14,7),
            (14,8),
            (11,12),
            (10,13),
            (4,13),
            (3,12),
            (0,8),
            (0,7),
        ]
        for i1, j1, num in config + other_config:
            for dj in range(num):
                ii, jj = i1, j1 + dj
                if (ii, jj) not in VETO18:
                    centers.append(Center.fromIJR(ii, jj, radius))

        # finally, so that there is no ambiguity at all, map the LIST of centers
        # directly onto a DICTIONARY indexed by the actual "indices", the common label
        # to Hexagon, ImageAnalysis, etc.
        # instead of just doing "return centers"
        return dict(zip(range(self.indices[0], self.indices[1]+1), centers))

    # get the hexagons, which involves getting centers and getting the data
    # getting the data involves running the image runner, which involves running many image analyses...
    # once the centers and data are obtained, loop over the data, get the center, declare the hex,
    # compute the sectors and save them in a .computedSectors member variables for access later
    def getHexagons(self, mode=DataMode.NORMAL, rotStart=RotationStart.RANDOM):
        centers = self.getCenters()
        sectorData = self.getDataForHexagons(mode=mode)

        logging.info('Declaring hexagons and computing sectors')
        hexes = []
        for idx, breakdown in sectorData:
            center = centers[idx]
            h = Hexagon(center, idx, rotStart=rotStart)
            h.computedSectors = h.getSectorsFromBreakdownData(breakdown)
            hexes.append(h)

        return hexes

    # final function that does everything
    # to avoid copying many keywords, I'll allow any keywords and split them up into arg dicts
    # get the hexagons, declare the canvas with the hexagons, and render
    def renderCanvas(self, **kwargs):
        hexagonArgs, canvasArgs, renderArgs = {}, {}, {}
        for key in ('mode', 'rotStart'):
            if key in kwargs:
                hexagonArgs[key] = kwargs[key]
        for key in ('width', 'height'):
            if key in kwargs:
                canvasArgs[key] = kwargs[key]
        for key in ('bounding_box', 'draw_coords', 'fname'):
            if key in kwargs:
                renderArgs[key] = kwargs[key]

        hexes = self.getHexagons(**hexagonArgs)
        logging.info('Rendering final canvas')
        canvas = Canvas(hexes, **canvasArgs)
        canvas.render(**renderArgs)

if __name__ == '__main__':
    main = MainRunner(
        #indices=(1,151),
        #radius=20,
        #imageScaling=25,
        #nclusters=3,
        #lumi_cutoff=0.2,
        #alpha_cutoff=200
    )
    main.renderCanvas(
        #mode=DataMode.NORMAL,
        #rotStart=RotationStart.RANDOM,
        #width=480,
        #height=550,
        #bounding_box=False,
        #draw_coords=False,
        fname='pokemon.svg'
    )
