from cvbot.io import read_img
from cvbot.colors import threshold, count_clr
from cvbot.images import Image
from cvbot.match import mse
from math import prod
import cv2 as cv
import numpy as np


def init_reader(pth, clr, slant=True, custom=range(10), nos=None):
    """
    str, Color | None, bool -> None
    Initiate reader by loading number images
    from given path 'pth'
    Color of the number to read,
    if 'clr' is None use automatic color detection
    'slant' is True if digit reading includes a slant
                between the numbers
    """
    files = []


    if files == []:
        for i in custom:
            files.append(read_img("{}{}.png".format(pth, i), "grey"))

        if slant:
            slim = read_img("{}slant.png".format(pth), "grey")
        else:
            slim = None

        #print(len(files))

        tclr = clr


    def reader(img, clr_var=0, cstmclr=None, ignoredigs=[], unrot=False):
        """
        Image, int, Color, int | None -> int 
        Read the number shown in given image
        """
        nonlocal files, tclr, slant, custom, slim, nos

        if cstmclr is None:
            clr = tclr
        else:
            clr = cstmclr

        dst  = threshold(img, clr, clr_var)
        losim = segment(dst, slant, slim, nos, unrot)
        if ignoredigs:
            flosim = filter(lambda x: not (losim.index(x) in ignoredigs), losim)
        else:
            flosim = losim

        losmb = [parse(s, files) for s in flosim]
        losmb = [str(custom[s]) for s in filter(lambda x: not (x is None), losmb)]

        if not (None in losim):
            result = "".join(losmb)
            if result == "":
                return None
            else:
                return result if type(custom[0]) != int else int(result)
        else:
            return None

    return reader

def trim_rows(bim):
    """
    Binary Image -> Binary Image
    Remove all the black rows in the given
    binary image
    """
    fi, li = 0, -1
    brks = [False, False]

    for r in range(bim.size[1]):

        if not brks[0]:
            rim = bim.img[r, :]
            if 255 in rim:
                brks[0] = True
            else:
                fi = r

        if not brks[1]:
            lr = -r-1
            rim = bim.img[lr, :]
            if 255 in rim: 
                brks[1] = True
            else:
                li = lr

        if brks[0] and brks[1]:
            break

    return Image(bim.img[fi:li, :])

def trim_cols(bim):
    """
    Binary Image -> Binary Image
    Remove all the black columns in the given
    binary image 
    """
    w, _ = bim.size
    fin = 0
    tin = w

    fwts = False
    for x in range(0, w):
        clim = Image(bim.img[:, x])
        wts = count_clr(clim, 255)
        if wts == 0:
            fwts = True 
            continue
        elif fwts:
            fin = x

        break

    fwts = False
    if w >= 2:
        for x in range(w - 1, 0, -1):
            clim = Image(bim.img[:, x])
            wts = count_clr(clim, 255)
            if wts == 0:
                fwts = True
                continue
            elif fwts:
                tin = x + 1
            break

        return Image(bim.img[:, fin:tin])
    else:
        return bim
    

def parse(bim, files):
    """
    Binary Image -> int | None
    Read the digit in given binary image
    """
    #bim.show()
    data = []
    
    for num in files:
        try:
            narea, darea = prod(num.size), prod(bim.size)
            aratio = narea / darea
            if aratio > 1:
                num = num.resize(bim.size)
                cimg = bim.copy()
            else:
                cimg = bim.resize(num.size)
        except Exception as e:
            #print(e, num.size, bim.size)
            return None 

        sqdf = mse(cimg, num)
        szdf = abs(1 - aratio) * (sqdf / 10)
        sqdf += szdf
        data.append(sqdf if sqdf != 0 else 0.01)

    #--------Print result and show image-----------
    #for i, d in enumerate(data):
    #    print(i, "{}%".format(int(d * 1000) / 10))
    # 
    #print("--------------------")
    #----------------------------------------------

    #print(data)
    res = data.index(min(data))
    return res

def get_dig(bim):
    """
    Binary Image -> Binary Image, Binary Image
    Return the first digit in the given binary image
    of numbers, and return the remaining numbers
    as a new binary image
    """
    w, _ = bim.size
    ind = w

    for x in range(w):
        clim = Image(bim.img[:, x])
        wts = count_clr(clim, 255)
        if wts == 0:
            ind = x
            break

    return Image(bim.img[:, :ind]), None if ind == w else Image(bim.img[:, ind:])

def slntind(dst, sim):
    """
    Binary Image -> int
    Find slant in 'dst' and 
    return its index in 'dst'
    """
    slant = sim
    sw, _ = slant.size
    w, _  = dst.size

    rec = []

    for x in range(w):
        if (x + sw) <= w:
            pim = Image(dst.img[:, x:x+sw])
            sdf = mse(pim, slant)
            rec.append(sdf)

    return rec.index(min(rec))


def crptoslant(dst, sim):
    """
    Binary Image -> Binary Image
    Find the slant "/" in given binary image "dst"
    And crop the image upto the slant(not including the slant)
    """
    ind = slntind(dst, sim)

    return Image(dst.img[:, :ind])

def cnt_angle(cnt):
    """
    Contour -> float, float, float
    Given a contour return its angle
    of rotation, along with points
    of rotation
    """
    box = cv.minAreaRect(cnt)
    (cx, cy), (_, _), angle = box
    return box, cx, cy, angle

def nomcols(bim):
    """
    Binary Image -> int
    Return the number of black columns
    in the middle of the given image
    """
    bim = trim_cols(bim)
    return len(np.where(bim.img.sum(axis=0)==0)[0])

def unrotate(dst):
    """
    Image -> Image
    If the given image is rotated
    rotate it back to the natural position
    and return it 
    """
    dst.img = np.pad(dst.img, ((20, 20), (20, 20)))
    cim = dst.img

    locnt, _ = cv.findContours(cim, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #print("Length of contours", len(locnt))
    if len(locnt) < 2:
        return dst

    cntr = None

    for cnt in locnt:
        if cntr is None:
            cntr = cnt
        else:
            cntr = np.vstack([cntr, cnt])

    _, cx, cy, _ = cnt_angle(cntr)
    cx = int(cx)
    cy = int(cy)
    w, h = dst.size
    tdst = dst.copy()
    bangle = None
    for angle in range(7):
        M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
        tdst.img = cv.warpAffine(dst.img, M, (w, h))
        cols = nomcols(tdst)

        #print(angle, cols, bangle)
        if bangle is None or cols > (bangle[1] * 2):
            bangle = angle, cols

        M = cv.getRotationMatrix2D((cx, cy), -angle, 1.0)
        tdst.img = cv.warpAffine(dst.img, M, (w, h))
        cols = nomcols(tdst)

        #print(-angle, cols, bangle)
        if cols > (bangle[1] * 2):
            bangle = -angle, cols

        if bangle[1] > 2:
            break

    #print(bangle)
    #dst.show()
    M = cv.getRotationMatrix2D((cx, cy), bangle[0], 1.0)
    dst.img = cv.warpAffine(dst.img, M, (w, h))
    #dst.show()

    return dst
    #print(len(ncntr))

    #Image(cim).show()

    #if not (cntr is None):
    #    angles.append(angle if angle < 20 else angle - 90)
    #    print("Angles", angles)
    #    abangls = [abs(a) for a in angles]
    #    angle = angles[abangls.index(min(abangls))]
    #    print("Best angle", angle)
    #    pnts = cv.boxPoints(box)
    #    pnts = np.int0(pnts)
    #    dst.img = cv.drawContours(dst.img, [pnts], 0, 255)
    #    if angle > 75:
    #        angle -= 90
    #    elif 15 > abs(angle) > 1:
    #        #for pnt in pnts:
    #        #    dst.img = cv.circle(dst.img, pnt, 2, 255, 2)
    #        pass
    #    else:
    #        return dst


def segment(dst, slant, sim=None, nos=None, unrot=False):
    """
    Binary Image, bool -> [Binary Image][1+]
    Partition given binary image of numbers
    into image of each digit from left to right
    """
    digs = []

    if unrot:
        dst = unrotate(dst)

    if slant:
        dst = crptoslant(dst, sim)

    if nos == 1:
        dig = trim_rows(trim_cols(dst))
        digs.append(dig)
    else:
        while not (dst is None):
            dst = trim_cols(dst)
            dig, dst = get_dig(dst)
            dig = trim_rows(dig)
            if dig.size == (0, 0):
                break
            digs.append(dig.copy())

    return digs
