import cv2 as cv
import numpy as np
import heapq
from os import read
from time import time
from random import choice
from cvbot.match import look_for, sift_find, mse, find_all, remove_close
from cvbot.io import read_img, save_img
from cvbot.images import Image
from cvbot.ai import read, init_ocr
from cvbot.yolo.ai import Model
from cvbot._screen import MON
from thefuzz import fuzz
from math import prod, dist, atan, pi
from os import listdir

# Testing
import logging
#read = lambda a, b: ""
#init_ocr = lambda a: None
#----------------------------------
print("\n Initiating Neural Nets... \n")
logger = logging.getLogger(__name__)
chk_model = None
map_model = None
map_img   = None
game_mode = None
supported_maps = ("SAHA", "SHSH", "DRCA", "JUTE", "FRRE", 
                  "FOCI", "STST", "ARBA")
static_maps    = ("JUTE", "DECA", "FRRE", "FOCI", "STST", 
                  "ARBA", "SAHA", "SHSH", "VAVA", "DRCA")
# ---------------- Classes ------------------
class Weapon:
    NORMAL   = 0
    CARVER   = 1
    FLAME    = 2
    GATTLING = 3
    RAILGUN  = 4
    SANTA    = 5
    TORTOISE = 6
    TREX     = 7
    XBOW     = 8
    SHOTGUN  = 9

    WPIMS = {
        read_img("src/weps/{}".format(wpn), "grey"):wpn[:-4].upper()
        for wpn in listdir("src/weps")
    }

    TPND = {
        0:"Normal",
        1:"Carver",
        2:"Flame",
        3:"Gattling",
        4:"Rail Gun",
        5:"Santa",
        6:"Tortoise",
        7:"Trex",
        8:"Xbow",
        9:"Shotgun"
    }

    TPTR = {
        NORMAL  :700,
        CARVER  :300,
        FLAME   :250,
        GATTLING:700,
        RAILGUN :700,
        SANTA   :700,
        TORTOISE:700,
        TREX    :700,
        XBOW    :900,
        SHOTGUN :400
    }

    TPTHT = {
        NORMAL  :0.0,
        CARVER  :600,
        FLAME   :600,
        GATTLING:9.0,
        RAILGUN :2.5,
        SANTA   :0.0,
        TORTOISE:1.0,
        TREX    :3.0,
        XBOW    :1.0,
        SHOTGUN :0.0
    }

    def __init__(self, tp) -> None:
        if (type(tp) is int) and (0 <= tp < 9):
            self.type = tp
        else:
            self.type = Weapon.NORMAL

    def has_rt(self):
        """
        Self -> bool
        Return True if current
        weapon has a reload delay
        after a long shoot
        """
        return self.type in (Weapon.XBOW, Weapon.TORTOISE,
                             Weapon.RAILGUN)

    def tp_to_name(tp):
        return Weapon.TPND[tp]

    def hold_time(tp, dste=None):
        if tp in (Weapon.TORTOISE, ):
            factor = Weapon.TPTR[tp] / dste
            return Weapon.TPTHT[tp] / factor
        else:
            return Weapon.TPTHT[tp]

    def lower_part(img):
        """
        Image -> numpy img
        Return the lower part of the given
        image, corresponding to the part 
        where the weapon should be
        """
        w, h = img.size
        lby = -(h // 4)
        lbsx = (w // 3)
        lblx = -lbsx
        lbim = img.img[lby:, lbsx:lblx]
        return lbim

    def find_wep(img): 
        """
        numpy img -> numpy img | None
        Find weapon in given image
        and return the image of the
        weapon
        Returns None if not found
        """
        dst = cv.inRange(img, (0, 0, 0, 255),
                              (10, 10, 10, 255))
        cnts, _ = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts):
            larea = None
            scnt = None
            for cnt in cnts:
                area = cv.contourArea(cnt)
                if larea is None or area > larea:
                    larea = area
                    scnt = cnt

            x, y, w, h = cv.boundingRect(scnt)
            area = w * h
            #print(area)
            if area > 1500:
                wpim = img[y:y+h, x:x+w]
                return wpim

    def parse_wep(wpim):
        """
        numpy img -> Weapon Type | None
        Recognize the weapon in the
        given image and return its type
        if not known it will return None
        """
        wpim = Image(wpim)
        wpim.convert("grey")
        w, h = wpim.size

        bt = None
        bdf = None
        for rwpim, tp in Weapon.WPIMS.items():
            rw, rh = rwpim.size

            if (w, h) != (rw, rh):
                if w > rw:
                    cwpim = wpim.resize((rw, rh))
                else:
                    rwpim = rwpim.resize((w, h))
                    cwpim = wpim
            else:
                cwpim = wpim

            df = mse(cwpim, rwpim)
            #print(df)
            if df < 1000:
                if bdf is None or df < bdf:
                    bdf = df
                    bt = tp

        if not (bt is None):
            return eval("Weapon." + bt)

    def update(self, img):
        """
        Image -> None
        Update the current weapon
        type to the weapon visible
        in given image
        """
        # Crop image to area of weapon
        lpim = Weapon.lower_part(img)
        # Crop image to the actual weapon
        wepim = Weapon.find_wep(lpim)
        # Determine weapon type
        if not (wepim is None):
            wpntp = Weapon.parse_wep(wepim)
            if not (wpntp is None):
                self.type = wpntp
            else:
                self.type = Weapon.NORMAL

# ------------- Constants -------------------
VICTORY  = 1
DEFEAT   = 0
ENMCLR   = (99,   98, 255, 255)
ALYCLR   = (255, 216,  98, 255)
SLFCLR   = (115, 237,  69, 255)
INDCLR   = (240, 183,  12, 255), (245, 190, 15, 255)
CHKNCLR  = (255, 169, 47, 255)
HPCLR    = ((31,  252, 246, 255),
            (32,  255, 249, 255))
ENRCLR   = ((1,   245, 255, 255),
            (1,   249, 255, 255))
OG_RZ    = prod((1600, 900))
OGMSZ    = prod((1920, 1080))
fullscreen = False
weapon   = Weapon(None) 
rltv_sz  = OG_RZ
rltv_szu = 1600, 900
center   =  800, 450 
# TEST CONSTANTS
#CRSCLR = (0, 0, 255, 255)
# --------------

def new_win(wsz):
    """
    tuple(int, int) -> None
    Given current window size
    reset the factor variable
    based on it
    """
    global rltv_sz, rltv_szu, fullscreen, center
    fullscreen = wsz[0] == MON['width']
    if not fullscreen:
        w, h = (wsz[0] - 16), (wsz[1] - 39)
    else:
        w, h = wsz
    rltv_sz  = w * h
    rltv_szu = w, h
    center   = wsz[0] // 2, wsz[1] // 2

def recal(n, ogsz=None, reverse=False, wonly=False):
    """
    int | Image, optional* bool, optional* bool -> float 
    Given a number return the relative
    new number based on the current resolution
    """
    if ogsz is None:
        ogsz = OG_RZ
        
    if wonly:
        denom = rltv_szu[0]
    else:
        denom = rltv_sz

    if reverse:
        factor = (ogsz / denom)
    else:
        factor = (denom / ogsz)

    if type(n) is Image:
        nszw = round(n.size[0] * factor)
        nszh = round(n.size[1] * factor)
        return n.resize((nszw, nszh))
    else:
        if reverse:
            return n * factor
        else:
            return n * factor

def recal_wh(sz):
    """
    (int, int) -> (int, int)
    recalibrate given width and height of
    an object to the expected size
    based on the current game window
    """
    return round(recal(sz[0], OGMSZ)), round(recal(sz[1], OGMSZ))

def process(nimg):
    """
    NA Image -> NA Image
    Dilate and Blur white pixels in given image
    """
    nimg = cv.dilate(nimg, (21, 21), iterations=10) 
    nimg = cv.blur(nimg, (21, 21))
    return nimg

def cnt_to_pos(cnt):
    """
    Contour -> Point
    Turn given contour of player name
    to center of tank point
    """
    x, y, w, _ = cv.boundingRect(cnt)
    nx, ny = (x + (w//2)), y + int(recal(100))
    return nx, ny

def locate_player(bnimg):
    """
    Binary N Image -> list(Point)
    Locate player location in given binary image
    and return center points in a list
    """
    nimg = process(bnimg)

    cnts, _ = cv.findContours(nimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    lopl = []
    for cnt in cnts:
        if cv.contourArea(cnt) > 1000:
            lopl.append(cnt_to_pos(cnt))

    #------ Test Code ---------
    #pcrclr = (0, 255, 0, 255)
    #for pos in lopl:
    #    if pcrclr != CRSCLR:
    #        pos = pos[0] - 20, pos[1]
    #    img.img = cv.circle(img.img, pos, 50,
    #                            CRSCLR, 5)
    #--------------------------
    return lopl 

def allies(img):
    """
    Image -> list(Point)
    Find allies in given image
    """
    nimg = cv.inRange(img.img, ALYCLR, ALYCLR)
    return locate_player(nimg)

def enemies(img):
    """
    Image -> list(Point)
    Find enemies in given image
    """
    nimg = cv.inRange(img.img, ENMCLR, ENMCLR)
    return locate_player(nimg)

def player(img):
    """
    Image -> Point
    Find player location in given image
    """
    nimg = cv.inRange(img.img, SLFCLR, SLFCLR)
    return locate_player(nimg)

def aly_inds(img):
    """
    Image -> list(Point)
    Return locations of the allies indicators
    """
    loai = []

    dst = cv.inRange(img.img, INDCLR[0], INDCLR[1])
    dst = process(dst)
    cnts, _ = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv.contourArea(cnt) > 500:
            x, y, w, h = cv.boundingRect(cnt)
            pos = (x + (w//2), y + (h//2))
            loai.append(pos)

    return loai

def hp(img):
    """
    Image -> int[0-100]
    Return HP percentage
    """
    hsvim = cv.cvtColor(img.img, cv.COLOR_BGR2HSV)
    dst = cv.inRange(hsvim, HPCLR[0], HPCLR[1])
    wlocs = np.where(dst==255)
    if len(wlocs[1]):
        ww = wlocs[1].max() - wlocs[1].min()
    else:
        return 0

    ov = 215
    factor = recal(ov)
    if factor > ov:
        factor *= 0.9
    elif factor != ov:
        factor /= 0.82

    rhp = round((ww / factor) * 100)
    return rhp

def energy(img):
    """
    Image -> int[0-00]
    Return energy percentage
    """
    dst = cv.inRange(img.img, ENRCLR[0], ENRCLR[1])
    wlocs = np.where(dst==255)
    if len(wlocs[1]):
        ww = wlocs[1].max() - wlocs[1].min()
    else:
        return 0
    dst[dst == 255] = 1
    of = 224
    factor = recal(of)
    if factor > of:
        factor *= 0.86
    elif factor < of:
        factor /= 0.86

    return round((ww / factor) * 100)

def box_valid(box):
    """
    Rect -> bool
    Return True if given rectangle 'box'
    is a valid rectangle on screen
    else return False
    """
    x, y, w, h = box
    return ((7680 >= x >= 0) and
            (4320 >= y >= 0) and
            (7680 >= w >  0) and
            (4320 >= h >  0))


def find_n_compare(ima, imb, sft_qual=30, mdf=1000):
    """
    Image -> rect | None
    Return rectangle if 'ima' is found in 'imb'
    and the result of the MSE comparison is
    a reasonable difference
    else return None
    """
    box = sift_find(ima, imb, sft_qual)

    if not (box is None) and box_valid(box):
        x, y, w, h = box
        try:
            cimb = imb.copy()
            cimb.img = imb.img[y:y+h, x:x+w]
            cimb.convert("grey")
            #szdf = (abs(prod(ima.size) - (w * h)) / 1000)
            df = mse(ima.resize((w, h)), cimb)
            if df < mdf:
                return (x, y, w, h)
        except Exception as e:
            pass

#LVIM = read_img("src/leave/btn.png", "grey")
#LVIM_GR = read_img("src/leave/btngr.png", "grey")
LVIM = read_img("src/leave/lvbtn.png", "grey")

def end_game(img):
    """
    Image -> Point | None
    Return a point of the leave button if found
    else return None
    """
    res = recal(LVIM, ogsz=1920, wonly=True)
    return look_for(res, img, 0.9)


PLYIM = read_img("src/queue/plybtn.png",   "grey")
CMPIM = read_img("src/queue/cmptvbtn.png", "grey")
RCNIM = read_img("src/queue/recon.png",    "grey")
IDLIM = read_img("src/queue/idle.png",     "grey")
CNFIM = read_img("src/queue/okbtn.png",    "grey")
NOIM  = read_img("src/queue/nobtn.png",    "grey")
RTRIM = read_img("src/queue/rtrbtn.png",   "grey")
VCTIM = read_img("src/queue/vctbnr.png",   "grey")
DFTIM = read_img("src/queue/dftbnr.png",   "grey")
CNCIM = read_img("src/queue/cnclbtn.png",  "grey")

def cancel_btn(img):
    """
    Image -> Point | None
    Return cancel position on screen if it's found
    if not return None
    """
    rcncim = recal(CNCIM, ogsz=1920, wonly=True)
    return look_for(rcncim, img, 0.8)

def play_btn(img):
    """
    Image -> Point | None
    Return the screen position of the play button
    if it's visible on screen
    if not return None
    """
    rect = find_n_compare(PLYIM, img)
    if not (rect is None):
        return rect[:2]

def cmptv_btn(img):
    """
    Image -> Point | None
    Return the screen position of the 
    competitive button 
    if it's visible on screen
    if not return None
    """
    rect = find_n_compare(CMPIM, img)
    if not (rect is None):
        return rect[:2]

def reconnect_dialog(img):
    """
    Image -> Rect | None
    Return the box location of the reconnect
    dialog if it is visible
    otherwise return None
    """
    return find_n_compare(RCNIM, img)

def confirm_dialog(img):
    """
    Image -> Point | None
    Return the point of ok button
    if not found returns None
    """
    ncnfim = recal(CNFIM, ogsz=1920, wonly=True)
    res = look_for(ncnfim, img, 0.8, scr=True)

    if not (res is None):
        okpos, okscr = res
        nnoim = recal(NOIM, ogsz=1920, wonly=True)
        res = look_for(nnoim, img, 0.75, scr=True)
        if not (res is None):
            nopos, noscr = res
            #print(okpos, nopos, okscr, noscr)
            if noscr > okscr and dist(okpos, nopos) < 40:
                return
        return okpos

def retry_button(img):
    """
    Image -> Point | None
    Return the middle point of 'retry' button
    if not found return None
    """
    res = recal(RTRIM, ogsz=1920, wonly=True)
    return look_for(res, img, 0.95)

def victory_defeat(img):
    """
    Image -> Game Result | None
    Return 'VICTORY' if the victory banner is visible
    or 'DEFEAT' if the defeat banner is visible
    if either of which is not visible return None
    """
    rvim = recal(VCTIM, ogsz=1920, wonly=True)
    vct  = look_for(rvim, img, 0.8)
    if vct:
        return VICTORY
    else:
        rdim = recal(DFTIM, ogsz=1920, wonly=True)
        dft = look_for(rdim, img, 0.8)
        if dft:
            return DEFEAT

def locate_chicken(img):
    """
    Image -> Point | None
    Return position of large chicken on screen
    if found instead return None
    """
    dst = cv.inRange(img.img, CHKNCLR, CHKNCLR)
    dst = process(dst)

    cnts, _ = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv.contourArea(cnt) 
        if area > 500:
            x, y, w, _ = cv.boundingRect(cnt)
            pos = (x + w, y + round(recal(150)))
            return pos
    
BRLIM = read_img("src/modes/barrel.png", "grey")

def barrels(img):
    """
    Image -> list(Point)
    Return list of all barrel positions 
    on screen if found
    """
    return remove_close(find_all(recal(BRLIM, ogsz=1920, wonly=True), img))

def square(wh):
    """
    (int, int) -> bool
    Return True if given width and height
    belong to a square or almost a square
    return False if not
    """
    w, h = wh
    #print(abs(w - h), w * h * 0.004)
    return abs(w - h) < ((w * h) * 0.005)


#def calc_hist(img):
#    """
#    np img -> hist
#    Calculate given image histogram and return it
#    """
#    hb = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256], accumulate=False)
#    hb = cv.normalize(hb, hb).flatten()
#    return hb

#lochist = (calc_hist(cv.cvtColor(cv.imread("_cc/chks/chick_3.png"), cv.COLOR_BGR2HSV)),)
           #calc_hist(cv.cvtColor(cv.imread("_cc/chks/chick_5.png"), cv.COLOR_BGR2HSV)),
           #calc_hist(cv.cvtColor(cv.imread("_cc/chks/chick_6.png"), cv.COLOR_BGR2HSV)))

#lodf = (0.4,)
        #0.9, 0.8)

#def hist_check(img):
#    """
#    np img -> bool
#    Return True if given image
#    histogram matches chick histogram
#    """
#    global lochist, lodf
#
#    ghist = calc_hist(img)
#    for chist, cdf in zip(lochist, lodf):
#        df = cv.compareHist(chist, ghist, 0)
#        if df > cdf:
#            #print("DIFFERENCE:", df)
#            return True
#
#    return False
#
#def _old_locate_chicks(img):
#    hsv = cv.cvtColor(img.img, cv.COLOR_BGR2HSV)
#    lodst = (cv.inRange(hsv, (15, 50, 60), (60, 230, 150)),)
#             #cv.inRange(hsv, (13, 200, 100), (26, 226, 150)),
#             #cv.inRange(hsv, (10, 140, 125), (20, 205, 230)))
#    chks = []
#
#    for dst in lodst:
#        bdst = cv.GaussianBlur(dst, (9, 9), 1.6)
#        #cv.imshow("bdst", bdst)
#        cnts, _ = cv.findContours(bdst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#        for cnt in cnts:
#            x, y, w, h = cv.boundingRect(cnt)
#            area = w * h
#            if area > 200:
#                pimg  = hsv[y:y+h, x:x+w]
#                dpimg = dst[y:y+h, x:x+w]
#                pxls = (dpimg == 255).sum()
#                if hist_check(pimg):
#                    #print(pxls)
#                    if 1700 > pxls > 100:
#                        shpdf = dist((55, 57), (w, h))
#                        #print("pixels:", pxls, "shape:", (w, h), 
#                        #    "shape diff:", shpdf, (w, h))
#                        if shpdf < 25:
#                            #print("passed!")
#                            chks.append((x, y))
#
#    return chks

def locate_chicks(img):
    """
    Image -> list(Point)
    Look for chicks in given game image
    and return all the chicks on screen
    in a list
    """
    if not (chk_model is None):
        out = chk_model.detect(img, 0.4) # list((box, scr, name))
        lopos = [(o[0][1][0] + (o[0][1][2] - o[0][1][0]), 
                  o[0][1][1] + (o[0][1][3] - o[0][1][1])) for o in out]
        return lopos
    else:
        load_chick_model()
        return []

FLAGS = {
    "dropped":read_img("src/modes/flag_drpd.png"),
    "enemy"  :read_img("src/modes/flag_enmy.png"),
    "self"   :read_img("src/modes/flag_self.png"),
    "ally"   :read_img("src/modes/flag_ally.png"),
}

def flag_loc_tp(img):
    """
    Image -> str, Point | None
    Given game image find the flag and
    return its type and location on screen
    if not found returns None
    """
    for tp, fim in FLAGS.items():
        rfim = recal(fim, ogsz=1920, wonly=True)
        floc = look_for(rfim, img, 0.8, clr_match=True)
        if not (floc is None):
            ploc = floc[0], floc[1] + round(recal(160, ogsz=1920, wonly=True))
            return tp, floc, ploc

def init_nr():
    """
    None -> None
    Initiate the OCR for reading
    english words on scren
    """
    init_ocr(["en", "de"])

def crop_mode(img):
    """
    Image -> Image
    Crop the part where the mode is displayed
    """
    w, h = img.size
    nx = w // 4 
    nw = w - (nx * 2)
    ny, nh = (h // 15) + 31, h // 6 
    return img.crop((nx, ny, nw, nh))

def thresh_words(img):
    """
    Image -> Imaeg
    Threshold image to binary
    with white words as white
    """
    return Image(cv.inRange(img.img, (255, 255, 255, 255),
                        (255, 255, 255, 255)))

GMSR = [
    ("TEAM DEATHMATCH",  1),
    ("POULTRY PUSHER",   2),
    ("CHICKEN CHASER",   3),
    ("KING OF THE HILL", 4),
    ("HOLD THE FLAG",    5),
    ("KILL CONFIRMED",   6)
]

def gms_to_mode(gms):
    """
    str -> int[1-5] | None
    Identify mode based on guiding sentence
    """
    for sgms, md in GMSR:
        smr = fuzz.ratio(sgms, gms)
        if smr > 80:
            return md
    return None 

def main_scene(img):
    """
    Image -> Image, int, int
    Given game window image
    crop it to the middle main
    scene part and return it
    as well as the indices used for
    cropping
    """
    _, h = img.size
    sy = (h // 6) + 31
    ey = h - (h // 7)
    return img.img[sy:ey,:], 0, sy

def close_obj(bima, bimb):
    """
    Binary Numpy Image, Binary Numpy Imaege -> Point | None
    Return first objects that are close to each other
    in both images
    if nothing was found returns None
    ASSUMPTION: Both images are of the same size
    """
    loca, _ = cv.findContours(bima, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    locb, _ = cv.findContours(bimb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    def point(cnt, sz=False):
        x, y, w, h = cv.boundingRect(cnt)
        pnt = (x + (w//2), y)
        return pnt if not sz else (pnt, w)

    def close(pa, pb):
        return dist(pa, pb) < 100

    for cnta in loca:
        for cntb in locb:
            (pa, sza), pb = point(cnta, sz=True), point(cntb)
            if close(pa, pb):# and sza > recal(60):
                #bima = cv.drawMarker(bima, pa, 255, cv.MARKER_CROSS, 50)
                #bimb = cv.drawMarker(bimb, pb, 255, cv.MARKER_CROSS, 50)
                #cv.imshow("white", bima)
                #cv.imshow("grey", bimb)
                #cv.waitKey(0)
                return pa

cptim = read_img("src/modes/cptr_hill.png", "grey")
dfnim = read_img("src/modes/dfnd_hill.png", "grey")

def hill(img):
    """
    Image -> Point | None
    Find the hill in the given image
    and it's position on screen
    if not found returns None instead
    """
    global cptim, dfnim

    wtim = cv.inRange(img.img, (255, 255, 255, 255),
                               (255, 255, 255, 255))
    wtim = Image(wtim)
    wpos = look_for(recal(cptim, ogsz=1920, wonly=True), wtim, 0.8)
    if wpos is None:
        wpos = look_for(recal(dfnim, ogsz=1920, wonly=True), wtim, 0.8)

    if not (wpos is None):
        hpos = wpos[0], wpos[1] + int(recal(200))
        return hpos
    #grim = cv.inRange(mcim, (241, 241, 241, 255),
    #                        (241, 241, 241, 255))
    #wtim = process(wtim)
    #grim = process(grim)

    #hill = close_obj(wtim, grim)

    #if not (hill is None):
    #    hill = hill[0] + ax, hill[1] + ay
    #    hill = hill[0], hill[1] + int(recal(200))
        #cimg = img.img.copy()
        #cimg = cv.drawMarker(cimg, hill, (0, 255, 0), cv.MARKER_CROSS, 50)
        #cv.imshow("Marker", cimg)
        #cv.waitKey(1)
    
def __record(reg):
    sleep(3)
    i = 0 

    while True:
        img = get_region(reg)
        save_img(img, "__test/_pathing/base/{}.png".format(i))
        print(i)
        i += 1
        sleep(0.1)

def size_check(box, wh):
    ewh = recal_wh(wh)
    wh = box[2:]
    df = dist(ewh, wh)
    #print("Difference:", df)
    return df < ((ewh[0] + ewh[1]) / 4)

MNX, MNY = 20, 45
MXX, MXY = rltv_szu[0] - 5, rltv_szu[1] - 5

def cut_rec(*args, **kwargs):
    """
    str, bi np im, box, Point, bool -> bi np im
    Remove a rectangle from either:
    - top-left
    - top-right
    - bottom-right
    - bottom-left
    corner of size 'szt',
    Only removes the rectangle if its within the image,
    this is ignored if 'igm' is True
    """
    return cut_shape(*args, **kwargs, shape='rec')

def cut_cir(*args, **kwargs):
    """
    Cut a Circle
    """
    return cut_shape(*args, **kwargs, shape='cir')

def cut_tri(*args, **kwargs):
    """
    str, bi np im, box, tuple(int, int), bool, bool -> bi np im
    Remove a triangle from either:
    - top-left
    - top-right
    - bottom-right
    - bottom-left
    of the given 'vmap' binary image
    of size 'sz' from the box 'box'
    """
    return cut_shape(*args, **kwargs, shape='tri')

def cut_shape(loc, vmap, box, szt, igm=False, flipx=False, flipy=False, shape='tri'):

    if not (shape in ('tri', 'rec', 'cir')):
        print("[ERROR] SHAPE {} NOT SUPPORTED".format(shape))
        return vmap

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    if shape == 'cir':
        r, ofst = szt
        if loc == "l":
            cx = x1 + ofst
            cy = y1 + (bh // 2)
        elif loc == 't':
            cx = x1 + (bw // 2)
            cy = y1 + ofst
        elif loc == 'r':
            cx = x2 - ofst
            cy = y1 + (bh // 2)
        elif loc == 'b':
            cx = x1 + (bw // 2)
            cy = y2 - ofst

        vmap = cv.circle(vmap, (cx, cy), r, 0, -1)

    else:
        szx, szy = szt
        if szx > bw:
            szx = bw
        if szy > bh:
            szy = bh

        if loc == "tl":
            if not igm and (x1 <= MNX or y1 <= MNY):
                return vmap
            p1 = (x1, y1)
            if shape == 'tri':
                p2 = (x1 + szx, y1 if not flipy else y1 + szy)
                p3 = (x1 if not flipx else x1 + szx, y1 + szy)
            elif shape == 'rec':
                p2 = (x1 + szx, y1 + szy)
        elif loc == "tr":
            if not igm and (x2 >= MXX or y1 <= MNY):
                return vmap
            p1 = (x2, y1)
            if shape == "tri":
                p2 = (x2 - szx, y1 if not flipy else y1 + szy)
                p3 = (x2 if not flipx else x2 - szx, y1 + szy)
            elif shape == 'rec':
                p2 = (x2 - szx, y1 + szy)
        elif loc == "bl":
            if not igm and (x1 < MNX or y2 > MXY):
                return vmap
            p1 = (x1, y2)
            if shape == 'tri':
                p2 = (x1 + szx, y2 if not flipy else y2 - szy)
                p3 = (x1 if not flipx else x1 + szx, y2 - szy)
            elif shape == 'rec':
                p2 = (x1 + szx, y2 - szy)
        elif loc == "br":
            if not igm and (x2 >= MXX or y2 >= MXY):
                return vmap
            p1 = (x2, y2)
            if shape == 'tri':
                p2 = (x2 - szx, y2 if not flipy else y2 - szy)
                p3 = (x2 if not flipx else x2 - szx, y2 - szy)
            elif shape == 'rec':
                p2 = (x2 - szx, y2 - szy)
        else:
            return vmap

        if shape == 'tri':
            tri = np.array((p1, p2, p3))
            vmap = cv.drawContours(vmap, [tri], 0, 0, -1)
        elif shape == 'rec':
            vmap = cv.rectangle(vmap, p1, p2, 0, -1)

    return vmap

def open_fire_line(sp, ep):
    """
    Point, Point -> bool
    Return True if line of fire connects
    from point 'sp' to point 'ep'
    and return False if the line is blocked
    by an object
    """
    global loob
    lfmap = make_lfmap()
    lfm = cv.line(np.zeros_like(lfmap), sp, ep, 1)
    return not (255 in lfmap[lfm==1])

def lf_detail_vault(vmap):
    """
    bi np im -> bi np im
    Details for Vault Of Value
    line of fire map
    """
    global loob

    h, w = vmap.shape[:2]

    for ((_, (x1, y1, x2, y2)), _, name) in loob:
        if "Wall" in name:
            # Pre-efects
            # WALLS
            if name == "Wall-2":
                y2 += 50
                if y2 > h:
                    y2 = h
            elif name == "Wall-4":
                y1 += 50
                if y1 > h:
                    y1 -= 50 

            # Object Blocks
            vmap[y1:y2, x1:x2] = 255
            box = x1, y1, x2, y2

            # After-efects
            # WALLS
            if   name == "Wall-1":
                vmap = cut_tri("br", vmap, box, (100, 100), igm=True)
            elif name == "Wall-4":
                vmap = cut_tri("bl", vmap, box, (150, 150), igm=True)
                vmap = cut_tri("br", vmap, box, (200, 350), igm=True, flipy=True)
                vmap = cut_tri("tr", vmap, box, (250, 600), igm=True, flipy=True)
            elif name == "Wall-6":
                vmap = cut_tri("bl", vmap, box, (200, 300), igm=True)
                vmap = cut_tri("tr", vmap, box, (100, 100), igm=True)
            elif name == "Wall-7":
                vmap = cut_tri("br", vmap, box, (270, 250), igm=True, flipx=True)
                vmap = cut_tri("br", vmap, box, (500, 200), igm=True, flipx=True)
                vmap = cut_tri("bl", vmap, box, (250, 200), igm=True)
                vmap = cut_tri("tr", vmap, box, (150, 150), igm=True)
                vmap = cut_tri("tl", vmap, box, (150, 150), igm=True)
            elif name == "Wall-9":
                vmap = cut_tri("tl", vmap, box, (350, 250), igm=True, flipy=True)
                vmap = cut_tri("tl", vmap, box, (150, 350), igm=True)
                vmap = cut_tri("tr", vmap, box, ( 50,  50), igm=True)
                vmap = cut_tri("br", vmap, box, ( 50,  50), igm=True)
            elif name == "Wall-12":
                vmap = cut_tri("tl", vmap, box, (200, 200), igm=True)
            elif name == "Wall-15":
                vmap = cut_tri("bl", vmap, box, (100, 100), igm=True)
                vmap = cut_tri("br", vmap, box, ( 50,  50), igm=True)
                vmap = cut_tri("tr", vmap, box, (350, 150), igm=True, flipx=True)
                vmap = cut_tri("tr", vmap, box, (450, 150))
            elif name == "Wall-17":
                vmap = cut_tri("tr", vmap, box, (150, 200), igm=True)

    return vmap

def lf_detail_dc(lfmap):
    """
    bi np im -> bi np im
    Details for Death Canyon map
    line of fire map
    """
    global loob

    for ((_, (x1, y1, x2, y2)), _, name) in loob:
        if "Bridge" in name:
            if name == "Bridge-2":
                if y1 > 170: 
                    if x2 < (rltv_szu[0] + 16):
                        lfmap[y1-170:y1+60, x2] = 255
                    if x1 < (rltv_szu[0] - 85):
                        lfmap[y1-170:y1-105, x1+85] = 255

                if y2 < (rltv_szu[1] - 170):
                    if x2 < (rltv_szu[0] + 16):
                        lfmap[y2+105:y2+170, x2] = 255
                    lfmap[y2-60:y2+170, x1] = 255



        if "House" in name:
            lfmap[y1:y2, x1:x2] = 255
        elif "Obstacle" in name and name[-1] in ("1", "2", "5", "6"):
            lfmap[y1:y2, x1:x2] = 255

        # After-efects
        if name == "Obstacle-1":
            lfmap = cut_rec("tr", lfmap, (x1, y1, x2, y2), (80, 80), igm=True)
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (40 , 115), igm=True)
        elif name == "Obstacle-2":
            lfmap = cut_rec("bl", lfmap, (x1, y1, x2, y2), (80, 80), igm=True)
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (40 , 115), igm=True)
        elif name == "House-1":
            lfmap = cut_rec("tl", lfmap, (x1, y1, x2, y2), (400,  50) , igm=True)
        elif name == "House-2":
            lfmap[y2:, x1:x2] = 255
        elif name == "House-3":
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (50,  400) , igm=True)
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (200, 200), igm=True)
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (150, 150), igm=True)
            lfmap = cut_rec("tl", lfmap, (x1, y1, x2, y2), (500, 50), igm=True)
        elif name == "House-4":
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (300, 220), igm=True)
            lfmap = cut_tri("bl", lfmap, (x1, y1, x2, y2), (150, 150), igm=True)
            #lfmap = cut_tri("tr", lfmap, (x1, y1, x2, y2), (150, 150))
            #lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (150, 150))
            lfmap = cut_rec("tr", lfmap, (x1, y1, x2, y2), (120, 550), igm=True)
            lfmap = cut_rec("tr", lfmap, (x1, y1, x2, y2), (300, 100), igm=True)
        elif name == "House-5":
            lfmap[:y2, x1:x2] = 255
            lfmap = cut_rec("br", lfmap, (x1, y1, x2, y2), (180, 80), igm=True)
        elif name == "House-6":
            lfmap[:y2, x1:x2] = 255
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (200, 50),  igm=True)
            lfmap = cut_tri("bl", lfmap, (x1, y1, x2, y2), (100, 150), igm=True)
        elif name == "House-7":
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (200, 250), igm=True)
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (150, 150), igm=True)
            lfmap = cut_rec("tl", lfmap, (x1, y1, x2, y2), (500, 50), igm=True)
        elif name == "House-8":
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (100, 50) , igm=True) 
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (50, 50)  , igm=True)
            lfmap = cut_tri("tr", lfmap, (x1, y1, x2, y2), (400, 50)  , igm=True)
        elif name == "House-9":
            lfmap = cut_tri("br", lfmap, (x1, y1, x2, y2), (200, 200), igm=True)
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (100, 220), igm=True)
            lfmap = cut_tri("tr", lfmap, (x1, y1, x2, y2), (200, 200), igm=True)
            lfmap = cut_rec("br", lfmap, (x1, y1, x2, y2), (500, 50),  igm=True)
        elif name == "House-10":
            lfmap[y2:, x1:x2] = 255
            lfmap = cut_tri("tl", lfmap, (x1, y1, x2, y2), (300, 200), igm=True)
            lfmap = cut_rec("tl", lfmap, (x1, y1, x2, y2), (500, 50), igm=True)

    return lfmap

def lf_mask_detail(lfmap):
    """
    bi np im -> bi np im
    Build a line of fire obstacle map using
    the detected objects in the global variable 'loob'
    """
    global loob

    for ((mask, _), _, name) in loob:
        if name.startswith("w"):
            lfmap = cv.drawContours(lfmap, [mask], 0, (255, 255, 255), -1)
            #lfmap[y1:y2, x1:x2][mask] = 255

    return lfmap

def lf_detail(lfmap):
    """
    bi np im -> bi np im

    """
    global cur_map

    if cur_map == "DECA":
        lfmap = lf_detail_dc(lfmap)
    elif cur_map == "VAVA":
        lfmap = lf_detail_vault(lfmap)
    elif cur_map in supported_maps:
        lfmap = lf_mask_detail(lfmap)

    return lfmap

def make_lfmap():
    """
    None -> bi np im
    Make a black and white
    map of the fire blocking objects
    """
    global rltv_szu

    lfmap = np.zeros((rltv_szu[1] + 39, rltv_szu[0] + 16), dtype=np.uint8)
    return lf_detail(lfmap)

def detail_vault(vmap):
    """
    bi np im -> bi np im
    Details for Vault Of Value map
    """
    global loob, game_mode

    h, w = vmap.shape[:2]

    for ((_, (x1, y1, x2, y2)), _, name) in loob:
        # Pre-efects
        # WALLS
        if name == "Wall-2":
            y2 += 50
            if y2 > h:
                y2 = h
        elif name == "Wall-4":
            y1 += 50
            if y1 > h:
                y1 -= 50 
        elif name == "Wall-14" and game_mode == 3:
            x2 += 50
            if x2 > w:
                x2 = w
        elif name == "Wall-11" and game_mode == 3:
            x1 -= 50
            if x1 < 0:
                x1 = 0
        if name == "Wall-18":
            y1 -= 50
            if y1 < 0:
                y1 = 0
        # PITS
        if name == "Pit-1":
            x2 += 100
            if x2 > w:
                x2 = w
        elif name in ("Pit-2", "Pit-3"):
            y1 = 0
        elif name == "Pit-4":
            y2 += 100
            if y2 > h:
                y2 = h
        elif name == "Pit-5":
            x1 -= 50
            x2 += 100
            if x1 < 0:
                x1 = 0
            if x2 > h:
                x2 = h
        elif name == "Pit-6":
            y2 = h
        elif name == "Pit-7":
            y1 -= 20
            y2 += 50
            if y2 > h:
                y2 = h
            if y1 < 0:
                y1 = 0
        elif name == "Pit-8":
            x1 = 0

        # Object Blocks
        vmap[y1:y2, x1:x2] = 255
        box = x1, y1, x2, y2

        # After-efects
        # WALLS
        if   name == "Wall-1":
            vmap = cut_tri("br", vmap, box, (100, 100), igm=True)
        elif name == "Wall-4":
            vmap = cut_tri("bl", vmap, box, (150, 150), igm=True)
            #vmap = cut_tri("br", vmap, box, (200, 380), igm=True, flipy=True)
            #vmap = cut_tri("tr", vmap, box, (250, 600), igm=True, flipy=True)
            vmap = cut_cir("r",  vmap, box, (150,  50), igm=True)
            vmap = cut_rec("br", vmap, box, (120, 150), igm=True)
        elif name == "Wall-6":
            vmap = cut_tri("bl", vmap, box, (200, 300), igm=True)
            vmap = cut_tri("tr", vmap, box, (100, 100), igm=True)
        elif name == "Wall-7":
            vmap = cut_cir("b",  vmap, box, (150,  50), igm=True)
            vmap = cut_rec("bl", vmap, box, (150, 130), igm=True)
            #vmap = cut_tri("br", vmap, box, (270, 250), igm=True, flipx=True)
            #vmap = cut_tri("br", vmap, box, (500, 150), igm=True, flipx=True)
            #vmap = cut_tri("bl", vmap, box, (250, 200), igm=True)
            vmap = cut_tri("tr", vmap, box, (150, 150), igm=True)
            vmap = cut_tri("tl", vmap, box, (150, 150), igm=True)
        elif name == "Wall-9":
            #vmap = cut_tri("tl", vmap, box, (150, 300), igm=True, flipy=True)
            #vmap = cut_tri("tl", vmap, box, (150, 350), igm=True)
            vmap = cut_cir("l",  vmap, box, (120, 100), igm=True)
            vmap = cut_rec("tl", vmap, box, (110, 200), igm=True)
            vmap = cut_tri("tr", vmap, box, ( 50,  50), igm=True)
            vmap = cut_tri("br", vmap, box, ( 50,  50), igm=True)
        elif name == "Wall-12":
            vmap = cut_tri("tl", vmap, box, (200, 200), igm=True)
        elif name == "Wall-15":
            vmap = cut_tri("bl", vmap, box, (100, 100), igm=True)
            vmap = cut_tri("br", vmap, box, ( 50,  50), igm=True)
            vmap = cut_cir("t",  vmap, box, (150,  50), igm=True)
            vmap = cut_rec("tr", vmap, box, (150, 110), igm=True)
            #vmap = cut_tri("tr", vmap, box, (350, 150), igm=True, flipx=True)
            #vmap = cut_tri("tr", vmap, box, (450, 100))
        elif name == "Wall-17":
            vmap = cut_tri("tr", vmap, box, (150, 200), igm=True)

        # PITS
        elif name == "Pit-1":
            vmap = cut_tri("tl", vmap, box, (200, 150), igm=True)
            vmap = cut_tri("tr", vmap, box, (150, 150), igm=True)
            vmap = cut_tri("bl", vmap, box, (500, 20) , igm=True, flipx=True)
        elif name == "Pit-2":
            vmap = cut_tri("br", vmap, box, (50, 300) , igm=True, flipy=True)
            vmap = cut_tri("bl", vmap, box, (50, 300) , igm=True)
        elif name == "Pit-3":
            vmap = cut_tri("bl", vmap, box, (50, 300) , igm=True, flipy=True) 
            vmap = cut_tri("br", vmap, box, (300, 50) , igm=True) 
        elif name == "Pit-4":
            vmap = cut_tri("tr", vmap, box, (150, 150), igm=True)
            vmap = cut_tri("tl", vmap, box, (50, 500) , igm=True)
        elif name == "Pit-5":
            vmap = cut_tri("tr", vmap, box, (50, 500) , igm=True)
            vmap = cut_tri("br", vmap, box, (150, 150), igm=True)
            vmap = cut_tri("bl", vmap, box, (200, 200), igm=True)
        elif name == "Pit-6":
            vmap = cut_tri("tr", vmap, box, (250, 100), igm=True)
            vmap = cut_tri("tl", vmap, box, (20, 250) , igm=True)
        elif name == "Pit-7":
            vmap = cut_tri("tl", vmap, box, (100, 100), igm=True)
            vmap = cut_tri("bl", vmap, box, (150, 150), igm=True)
            vmap = cut_tri("br", vmap, box, (100, 400), igm=True, flipy=True)
            vmap = cut_tri("tr", vmap, box, (100, 300), igm=True, flipy=True)
        elif name == "Pit-8":
            vmap = cut_tri("tr", vmap, box, (300, 20) , igm=True)
            vmap = cut_tri("br", vmap, box, (100, 300), igm=True)
    
    kernel = np.ones((5, 5), np.uint8)
    return cv.dilate(vmap, kernel, iterations=3) 

def detail_dc(vmap):
    """
    bi np im -> bi np im
    Details for Death Canyon map
    """
    global loob

    bsx, bex = None, None
    h, w = vmap.shape[:2]
    lobyr = []
    for ((_, (x1, y1, x2, y2)), _, name) in loob:
        if "House" in name or "Obstacle" in name:
            # Pre-efects
            if name == "Obstacle-2":
                x2 += 50
                if x2 > w:
                    x2 = w
            elif name == "Obstacle-4":
                x1 -= 300
                x2 += 20
                y1 -= 20
                if x1 < 0:
                    x1 = 0
                if x2 > w:
                    x2 = w
                if y1 < 0:
                    y1 = 0
                if y2 > h:
                    y2 = h
            elif name == "Obstacle-8":
                y2 += 20
                x1 -= 20
                x2 += 300
                if y2 > h:
                    y2 = h
                if x1 < 0:
                    x1 = 0
                if x2 > w:
                    x2 = w
            elif name == "House-5":
                y1 = 0
            elif name == "House-6":
                y1 = 0
                y2 += 50
                if y2 > h:
                    y2 = h
            elif name == "House-7":
                y2 += 50
                if y2 > h:
                    y2 = h

            vmap[y1:y2, x1:x2] = 255

            # After-efects
            if name == "Obstacle-1":
                vmap = cut_rec("tr", vmap, (x1, y1, x2, y2), (80, 80), igm=True)
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (40 , 115), igm=True)
            elif name == "Obstacle-2":
                vmap = cut_rec("bl", vmap, (x1, y1, x2, y2), (80, 80), igm=True)
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (40 , 115), igm=True)
            elif name == "House-1":
                vmap = cut_rec("tl", vmap, (x1, y1, x2, y2), (400,  50) , igm=True)
            elif name == "House-2":
                vmap[y2:, x1:x2] = 255
            elif name == "House-3":
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (50,  400) , igm=True)
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (200, 200), igm=True)
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (150, 150), igm=True)
                vmap = cut_rec("tl", vmap, (x1, y1, x2, y2), (500, 50), igm=True)
            elif name == "House-4":
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (300, 220), igm=True)
                vmap = cut_tri("bl", vmap, (x1, y1, x2, y2), (150, 150), igm=True)
                #vmap = cut_tri("tr", vmap, (x1, y1, x2, y2), (150, 150))
                #vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (150, 150))
                vmap = cut_rec("tr", vmap, (x1, y1, x2, y2), (120, 550), igm=True)
                vmap = cut_rec("tr", vmap, (x1, y1, x2, y2), (300, 100), igm=True)
            elif name == "House-5":
                vmap = cut_rec("br", vmap, (x1, y1, x2, y2), (180, 80), igm=True)
            elif name == "House-6":
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (200, 50),  igm=True)
                vmap = cut_tri("bl", vmap, (x1, y1, x2, y2), (100, 150), igm=True)
            elif name == "House-7":
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (200, 250), igm=True)
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (150, 150), igm=True)
                vmap = cut_rec("tl", vmap, (x1, y1, x2, y2), (500, 50), igm=True)
            elif name == "House-8":
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (100, 50) , igm=True)
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (50, 50)  , igm=True)
                vmap = cut_tri("tr", vmap, (x1, y1, x2, y2), (400, 50)  , igm=True)
            elif name == "House-9":
                vmap = cut_tri("br", vmap, (x1, y1, x2, y2), (200, 200), igm=True)
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (100, 220), igm=True)
                vmap = cut_tri("tr", vmap, (x1, y1, x2, y2), (200, 200), igm=True)
                vmap = cut_rec("br", vmap, (x1, y1, x2, y2), (500, 50),  igm=True)
            elif name == "House-10":
                vmap[y2:, x1:x2] = 255
                vmap = cut_tri("tl", vmap, (x1, y1, x2, y2), (300, 200), igm=True)
                vmap = cut_rec("tl", vmap, (x1, y1, x2, y2), (500, 50), igm=True)

        elif "Bridge" in name:
            lobyr.append((y1, y2))
            if bsx is None:
                bsx = x1
                bex = x2
            else:
                if x1 < bsx:
                    bsx = x1
                if x2 > bex:
                    bex = x2

    if not (bsx is None):
        lobyr.sort(key=lambda k:k[0])
        ly = 0
        for y1, y2 in lobyr:
            vmap[ly:y1, bsx:bex] = 255
            ly = y2
        vmap[ly:, bsx:bex] = 255

    kernel = np.ones((5, 5), np.uint8)
    return cv.dilate(vmap, kernel, iterations=3) 

def mask_detail(vmap):
    global loob

    for ((mask, _), _, _) in loob:
        vmap = cv.drawContours(vmap, [mask], 0, (255, 255, 255), 20)
        vmap = cv.drawContours(vmap, [mask], 0, (255, 255, 255), -1)
        #vmap[y1:y2, x1:x2][mask] = 255

    return vmap

def detail(vmap):
    """
    bi np im -> bi np im
    Set accurate map detail and 
    boundaries for each detected boundary
    """
    global cur_map

    if cur_map == "DECA":
        return detail_dc(vmap)
    elif cur_map == "VAVA":
        return detail_vault(vmap)
    elif cur_map in supported_maps:
        return mask_detail(vmap)
    
    return vmap

def make_vmap(img):
    """
    Image -> numpy array
    Setup a black and white map where blocking
    objects are white and other objects are black
    """
    vmap = np.zeros(img.size[::-1], dtype=np.uint8)
    return detail(vmap)

def make_vpoints(sp):
    """
    Point -> list((Point, (int, int)))[8]
    Return all possible movement directions
    as 8 new points
    """
    global rltv_szu

    lop = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            if (x ==  0 and y ==  0):
                continue
            else:
                if x == 0 or y == 0:
                    fv = 100
                else:
                    fv = 70
                nxp = [sp[0] + (x * fv), sp[1] + (y * fv)]
                if nxp[0] < 0:
                    nxp[0] = 0
                elif nxp[0] > rltv_szu[0]:
                    nxp[0] = rltv_szu[0]
                if nxp[1] < 0:
                    nxp[1] = 0
                elif nxp[1] > rltv_szu[1]:
                    nxp[1] = rltv_szu[1]
                lop.append((nxp, (x, y)))
    return lop

def terrain_resistance(vmap, sp, npd):
    """
    numpy array, Point, (Point, (int, int))-> float[0.1:1]
    Return the resistance of terrain in 'vmap'
    with 1.0 meaning high resistance,
    and  0.1 meaning easy to go through
    """
    nxp, dr = npd
    x1, y1, x2, y2 = ((sp[0] if dr[0] ==  1 else nxp[0]),
                      (sp[1] if dr[1] ==  1 else nxp[1]),
                      (sp[0] if dr[0] == -1 else nxp[0]),
                      (sp[1] if dr[1] == -1 else nxp[1]))
    abdr = abs(dr[0]), abs(dr[1])
    if abdr == (1, 1):
        dslice = vmap[y1:y2, x1:x2].copy()
        if dr[0] == 1:
            dslice = dslice.diagonal() 
        else:
            dslice = np.fliplr(dslice).diagonal()
    else:
        if abdr[0] == 0:
            dslice = vmap[y1:y2, x1-1:x2].flatten()
        elif abdr[1] == 0:
            dslice = vmap[y1-1:y2, x1:x2].flatten()

    dslice = dslice / 255
    sm = dslice.sum()
    dsl = len(dslice)
    if dsl < 100:
        sm = (100 / dsl) * sm
    res = (sm / 100) + 0.1
    return res

def difficulty(vmap, lonpd, sp, ep):
    """
    numpy array, list((Point, (int, int)))[8], Point, Point -> list(float)[8]
    For each point in 'lonp' calculate the movement difficulty
    in terrain visible on 'vmap' from 'sp' to 'ep'
    """
    lod = []
    for npd in lonpd:
        nxp, dr = npd
        dst  = dist(nxp, ep)
        tres = terrain_resistance(vmap, sp, npd)
        #print("Distance", dst, "Terrain", tres, "Direction", dr)
        df = tres * dst
        lod.append((df, dr))
    return lod

lpt = time()
loob = [] 
cur_map = "" 

def process_mn(img):
    """
    Image -> Image
    Crop the given image
    to where game map name appears
    """
    w, h = img.size
    ny2 =  round(h * 0.20)
    nx1 =  round(w * 0.25)
    nx2 = -round(w * 0.25)
    return Image(img.img[:ny2, nx1:nx2])

LOMN = {
    "SAFE HAVEN",
    "JUNGLE TEMPLE",
    "DRAGON CAVE",
    "ARCTIC BASE",
    "DEATH CANYON",
    "FROZEN RESEARCH",
    "SHROUDED SHRINE",
    "VAULT OF VALUE",
    "STEAMING STRONGHOLD",
    "FORSAKEN CITY"
}


def mn_to_mp(gmn):
    """
    str -> str[2]
    Return the map phrase if the
    given string is a map name
    """
    br = 0
    bm = cur_map
    for mn in LOMN:
        smr = fuzz.ratio(mn, gmn)
        if smr > 70:
            mnp = mn.split(" ")
            if len(mnp) >= 2 and (len(mnp[0]) > 2 and len(mnp[1])):
                if smr > br:
                    bm = "{}{}".format(mnp[0][:2], mnp[-1][:2])
                    br = smr

    return bm 

def detect_mode_map(img):
    """
    Image -> str
    Recognize the map in the given image
    and return its string code or return empty
    string if not known
    """
    mnim = process_mn(img) 
    #result = read(mnim, allowlist="'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '").upper().replace("\n", "")
    result = read(mnim, allowlist=" ABCDEFGHIJKLMNOPQRSTUVWXYZ").upper()
    logger.info("Map Mode OCR parsing: {}".format(result))

    if " ON " in result:
        result = result.split(" ON ")
    elif " OM " in result:
        result = result.split(" OM ")
    elif " O " in result:
        result = result.split(" O ")
    elif " ON" in result:
        result = result.split(" ON")
    else:
        result = []
        
    if len(result) == 2:
        mds, mps = result
    else:
        mds, mps = "", ""
    
    if mds != "" and mps != "":
        global cur_map, game_mode, lvaoi
        prsd_map = mn_to_mp(mps)
        prsd_mode = gms_to_mode(mds)
        game_mode = prsd_mode
        if cur_map != prsd_map:
            print("\n Loading map model... \n")
            load_map_model(prsd_map)
            lvaoi = None, time() 
        cur_map = prsd_map
        if game_mode == 3:
            load_chick_model()
        elif added_brls[0] and not (game_mode is None) and cur_map == added_brls[1]:
            remove_barrels()
    else:
        prsd_mode = None
    
    return prsd_mode

added_brls = False, None

def load_chick_model():
    """
    None -> None
    Load the chick model to the global variable
    """
    global chk_model, cur_map, map_img, static_maps, added_brls
    if chk_model is None:
        print("\n Loading chicks model... \n")
        chk_model = Model("src/models/chick.onnx", ["chick",])
        if cur_map in static_maps:
            add_barrels(cur_map)
            added_brls = True, cur_map

def add_barrels(cmp, val=255):
    """
    str -> None
    Add barrels to the current map image
    """
    for i, brl in enumerate(static_objs[cmp]['barrels']):
        for l in range(1, static_objs[cmp]['brls_mp'][i][0] + 1):
            map_img[brl[1], brl[0]-l] = val
        for r in range(1, static_objs[cmp]['brls_mp'][i][1] + 1):
            map_img[brl[1], brl[0]+r] = val
        for u in range(1, static_objs[cmp]['brls_mp'][i][2] + 1):
            map_img[brl[1]-u, brl[0]] = val
        for d in range(1, static_objs[cmp]['brls_mp'][i][3] + 1):
            map_img[brl[1]+d, brl[0]] = val

        map_img[brl[1], brl[0]] = val

def remove_barrels():
    """
    None -> None
    Remove barrels from the current map image
    """
    global added_brls
    add_barrels(added_brls[1], 0)
    added_brls = False, None

def load_map_model(pm):
    """
    Map Code -> None
    Load the map objects detection model
    for the current map
    """
    global map_model, map_img

    if pm == "DECA":
        map_img   = cv.imread("src/maps/desert.png", 0)
        map_model = Model("src/models/desert.onnx", ["House-1",
                                            "House-2",
                                            "House-3",
                                            "House-4",
                                            "Bridge-1",
                                            "Bridge-2",
                                            "Obstacle-1",
                                            "Obstacle-2",
                                            "Obstacle-3",
                                            "House-5",
                                            "Bridge-3",
                                            "House-6",
                                            "House-7",
                                            "House-8",
                                            "House-9",
                                            "House-10",
                                            "Obstacle-4",
                                            "Obstacle-5",
                                            "Obstacle-6",
                                            "Obstacle-7",
                                            "Obstacle-8"])
    elif pm == "VAVA":
        map_model = Model("src/models/vault.onnx", ["Wall-1",
                                                    "Wall-2",
                                                    "Wall-3",
                                                    "Wall-4",
                                                    "Pit-1",
                                                    "Pit-2",
                                                    "Wall-5",
                                                    "Pit-3",
                                                    "Wall-6",
                                                    "Wall-7",
                                                    "Pit-4",
                                                    "Wall-8",
                                                    "Wall-9",
                                                    "Wall-10",
                                                    "Wall-11",
                                                    "Pit-5",
                                                    "Wall-12",
                                                    "Wall-13",
                                                    "Pit-6",
                                                    "Pit-7",
                                                    "Wall-14",
                                                    "Wall-15",
                                                    "Wall-16",
                                                    "Wall-17",
                                                    "Wall-18",
                                                    "Pit-8",
                                                    "Wall-19",
                                                    ])
        map_img = cv.imread("src/maps/vault.png", 0)
    elif pm == "SAHA":
        map_model = Model("src/models/haven.pt")#, "mask", ['p1', 'p2', 'p3',
                                                #            'p4', 'w1', 'w2',
                                                #            'w3', 'w4', 'w5',
                                                #            'w6', 'w7', 'w8',
                                                #            'w9'])
        map_img = cv.imread("src/maps/haven.png", 0)
    elif pm == "SHSH":
        map_model = Model("src/models/shrine.pt")
        map_img   = cv.imread("src/maps/shrine.png", 0)
    elif pm == "DRCA":
        map_model = Model("src/models/dragon.pt")
        map_img   = cv.imread("src/maps/dragon.png", 0)
    elif pm == "JUTE":
        map_model = Model("src/models/temple.pt")
        map_img   = cv.imread("src/maps/temple.png", 0)
    elif pm == "FRRE":
        map_model = Model("src/models/research.pt")
        map_img   = cv.imread("src/maps/research.png", 0)
    elif pm == "FOCI":
        map_model = Model("src/models/forsaken.pt")
        map_img   = cv.imread("src/maps/forsaken.png", 0)
    elif pm == "STST":
        map_model = Model("src/models/stronghold.pt")
        map_img   = cv.imread("src/maps/stronghold.png", 0)
    elif pm == "ARBA":
        map_model = Model("src/models/base.pt")
        map_img   = cv.imread("src/maps/base.png", 0)

def map_objs(img):
    """
    Image -> list(box, str, float)
    Return objects in the given map image 
    """
    global map_model

    if map_model is None:
        return []
    elif cur_map in ("DECA", "DRCA"):
        return map_model.detect(img, 0.4)
    else:
        return map_model.detect(img, 0.5)

def inverse_d(da, db):
    """
    Direction, Direction -> bool
    Return True if given direction are against
    each other, False if not
    """
    if (0, 0) in (da, db):
        return False
    elif abs(da[0]) == 1:
        if da[0] == -db[0] and (da[1] != db[1] or da[1] == 0):
            return True 
    if abs(da[1]) == 1:
        if da[1] == -db[1] and (da[0] != db[0] or da[0] == 0):
            return True 

    return False 

def ind_of_ld(lod, d):
    """
    List(Direction), Direction -> int
    Return index of the last direction that
    matches the given direction 'd' in 'lod'
    """
    i = len(lod) - 1
    while True:
        ld = lod[i]
        if not inverse_d(ld, d):
            i -= 1
            if i == 0:
                return 0
        else:
            return i

def dom_d(lod):
    """
    List(Direction) -> Direction
    Return the most dominant direction in the
    given list of direction in the last
    10 directions if the list is larger
    than 10 else 5, 2, 1
    """
    ll = len(lod)

    if ll > 5:
        lod = lod[-5:]
    else:
        return lod[-1]

    solod = tuple(set(lod))
    dcs = []
    for ud in solod:
        dcs.append(lod.count(ud))

    return solod[dcs.index(max(dcs))]

def build_lop(ryx, lop):
    """
    y x np array, tuple(Point, Point, Point) -> list(Point)
    Tracedown the points from the end point to the start point
    and add them to a list and return it
    """
    nxp = lop[-1]
    while not np.array_equal(ryx[nxp[1], nxp[0]], (0, 0)):
        nxp = ryx[nxp[1], nxp[0]]
        nxp = int(nxp[1]), int(nxp[0])
        lop.append(nxp)

    return lop[::-1]

def nearest_b(p, vmap):
    """
    Point, bi np im, bool -> Point
    Return the nearest point to 'p'
    that serves as a 'good point' to
    start from or head to
    'good point' means black pixel
    surrounded by all black pixels
    """
    h, w = vmap.shape

    if p[1] < 0:
        p = (p[0], 0)
    if p[1] >= h:
        p = (p[0], h-1) 
    if p[0] < 0:
        p = (0, p[1])
    if p[0] >= w:
        p = (w-1, p[1])

    if vmap[p[1], p[0]] == 0:
        return p

    fctr = 1

    while fctr < w:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue

                tp = p[0] + (fctr * x), p[1] + (fctr * y)

                if (tp[0] < 0) or (tp[1] < 0) or (tp[0] >= w) or (tp[1] >= h):
                    continue

                if vmap[tp[1], tp[0]] == 0:
                    return tp
        
        fctr += 2

    return p

def heap_best_path(vmap, sp, ep, gap=5):
    """
    bi np im, Point, Point -> list(Point)
    Determine the shortest path from 'sp' to 'ep'
    based on the given map 'vmap'
    """
    h, w = vmap.shape
    # vmap is of size CSWxCSH, shape CSHxCSW
    sp, ep = nearest_b(sp, vmap), nearest_b(ep, vmap)
    # Point reference book keeping 
    ryx = np.zeros(vmap.shape[:2] + (2,), dtype=np.uint16)
    # Tunring vmap into bool array
    iswall = vmap > 0 
    # Keeping track of which points we've visited
    visited = np.zeros(vmap.shape[:2], dtype='bool')
    visited[:, :] = False
    visited[sp[1], sp[0]] = True
    visited[ep[1], ep[0]] = False
    # Costs heap, data structure for extracting minimum cost fast
    costs = []
    heapq.heapify(costs)
    costs.sort(key=lambda x: x[0])
    # Keeping track of the closest point so far, in case we can't reach ep
    cp = sp
    bpsf = cp
    bcsf = None
    # Testing
    #svmap = vmap.copy()
    
    while True:
        for x in range(-1, 2):
            for y in range(-1, 2):
                tp = cp[0] + x, cp[1] + y
                if (tp[0] < 0    or tp[1] < 0    or
                    tp[0] >= w or tp[1] >= h or
                    (x, y) == (0, 0)):
                    continue
                if tp == ep:
                    return build_lop(ryx, [ep, tp, cp])
                elif not visited[tp[1], tp[0]] and not iswall[tp[1], tp[0]]:
                    visited[tp[1], tp[0]] = True
                    pted = dist(tp, ep)
                    if gap > 0:
                        wly  = tp[1] - gap if tp[1] >= gap else 0
                        wlx  = tp[0] - gap if tp[0] >= gap else 0
                        wts  = iswall[wly:tp[1]+gap, wlx:tp[0]+gap].sum() / gap
                        cost = wts + pted
                    else:
                        cost = pted
                    ryx[tp[1], tp[0]] = (cp[1], cp[0])
                    heapq.heappush(costs, (cost, tp))

        if len(costs) > 0:
            min_cost, min_loc = heapq.heappop(costs)
            mloc = min_loc[1], min_loc[0]
            cp = mloc[1], mloc[0]

            if (bcsf is None) or (min_cost < bcsf):
                bcsf = min_cost
                bpsf = cp

            # Testing
            #svmap[mloc] = 125
            #cv.imshow("Pathing", cv.resize(svmap, (800, 600)))
            #cv.waitKey(0)
        else:
            if bpsf == sp:
                return [sp, ep]
            else:
                return build_lop(ryx, [bpsf,])


#def best_path(vmap, sp, ep):
#    """
#    bi np im, Point, Point -> list(Point)
#    Determine the shortest path from 'sp' to 'ep'
#    based on the given map 'vmap'
#    """
#    # vmap is of size CSWxCSH, shape CSHxCSW
#    h, w = vmap.shape
#    sp, ep = nearest_b(sp, vmap), nearest_b(ep, vmap)
#    ryx = np.zeros(vmap.shape[:2] + (2,), dtype=np.uint16)
#    visited = np.zeros(vmap.shape[:2], dtype='bool')
#    visited[:, :] = False
#    visited[sp[1], sp[0]] = True
#    visited[ep[1], ep[0]] = False
#    cost_loc = {}
#    cp = sp
#    bpsf = cp
#    bcsf = None
#    # Testing
#    #svmap = vmap.copy()
#    
#    while True:
#        for x in range(-1, 2):
#            for y in range(-1, 2):
#                tp = cp[0] + x, cp[1] + y
#                if (tp[0] < 0 or tp[1] < 0 or
#                    tp[0] >= h or tp[1] >= w or
#                    (x, y) == (0, 0)):
#                    continue
#                if tp == ep:
#                    print("Successfully found a route!")
#                    return build_lop(ryx, [ep, tp, cp])
#                elif not visited[tp[1], tp[0]]:
#                    visited[tp[1], tp[0]] = True
#
#                    if vmap[tp[1], tp[0]] == 0:
#                        cost = dist(tp, ep)# + dist(tp, sp))
#                        ryx[tp[1], tp[0]] = (cp[1], cp[0])
#
#                        if cost_loc.get(cost) is None:
#                            cost_loc[cost] = [tp,]
#                        else:
#                            cost_loc[cost].append(tp)
#
#        costs = cost_loc.keys()
#        if len(costs) > 0:
#            min_cost = min(costs)
#            loc = cost_loc[min_cost][0]
#            mloc = loc[1], loc[0]
#
#            cp = mloc[1], mloc[0]
#            if (bcsf is None) or (min_cost < bcsf):
#                bcsf = min_cost
#                bpsf = cp
#
#            cost_loc[min_cost].remove(loc)
#            if len(cost_loc[min_cost]) == 0:
#                del cost_loc[min_cost]
#            # Testing
#            #svmap[mloc] = 255
#            #cv.imshow("Pathing", cv.resize(svmap, (800, 600)))
#            #cv.waitKey(1)
#        else:
#            if bpsf == sp:
#                return [sp, ep]
#            else:
#                return build_lop(ryx, [bpsf,])

def r_to_d(r): 
    return r * (180/pi)

def ang_round(atheta):
    return 45 * round(atheta/45)

a_to_d = {
    0  :( 1,  0),
    45 :( 1, -1),
    90 :( 0, -1),
    135:(-1, -1),
    180:(-1,  0),
    225:(-1,  1),
    270:( 0,  1),
    315:( 1,  1),
    360:( 1,  0)
}

def path_to_dr(lop, max_d=5, custom_steps=()):
    """
    list(Point) -> Direction
    Based on given points decides
    the next direction to move toward
    to follow the path
    """
    if custom_steps:
        tpi = choice(custom_steps)
    else:
        tpi = choice((1, 3, 4, 5, 6, 7, 8, 9, 10, 20))
    
    mnp = tpi if len(lop) >= tpi else len(lop)

    sp = lop[0]
    if len(lop) <= 1:
        ep = lop[-1]
    elif dist(sp, lop[1]) < max_d:
        ep = lop[mnp-1]
    else:
        ep = lop[1]

    if dist(sp, ep) < 1:
        return (0, 0), ep

    a, o = ep[0] - sp[0], -(ep[1] - sp[1])

    #print(a, o)
    if a == 0:
        rtheta = pi/2 if o > 0 else pi*3/4
    elif o == 0:
        rtheta = 0 if a > 0 else pi
    else:
        rtheta = atan(abs(o)/abs(a))

    if a < 0 and o > 0:
        rtheta = pi - rtheta
    elif a < 0 and o < 0:
        rtheta += pi
    elif a > 0 and o < 0:
        rtheta = (2*pi) - rtheta

    #print(rtheta)
    atheta = r_to_d(rtheta)
    #print(atheta)
    fatheta = ang_round(atheta)
    #print(fatheta)

    return a_to_d[fatheta], ep

def draw_path(lop, vmap):
    """
    list(Point), bi np img -> bi np img
    Draw the given points on the given image
    and return it
    """
    for p in lop:
        vmap = cv.drawMarker(vmap, p, 125, cv.MARKER_DIAMOND, 1, 2)

    return vmap

def untangle(path, vmap):
    """
    list(Point), bi np im -> list(Point)
    Remove redundant points in given path
    and return it
    """
    cp = path[0]
    for i in range(len(path)-1, -1, -1):
        p = path[i]
        if not (255 in vmap[cv.line(vmap.copy(), cp, p, 1, thickness=2) == 1]):
            return [cp,] + path[i:]
            #if dist(p, path[n]) < 40:

    return path

CSW, CSH = 114, 64

def shrink_vmap(vmap, sp, ep):
    """
    bi np im, Point, Point -> bi np im, Point, Point
    Compress vmap to size 64x64
    and return the corresponding points of sp and ep
    in the new grid
    """
    xfc, yfc = vmap.shape[1] / CSW, vmap.shape[0] / CSH
    vmap = cv.resize(vmap, (CSW, CSH))
    nsp = (round(sp[0]/xfc), round(sp[1]/yfc))
    nep = (round(ep[0]/xfc), round(ep[1]/yfc))
    if nsp[0] > CSW - 1:
        nsp = CSW - 1, nsp[1]
    if nsp[1] > CSH - 1:
        nsp = nsp[0], CSH - 1 
    if nep[0] > CSW - 1:
        nep = CSW - 1, nep[1]
    if nep[1] > CSH - 1:
        nep = nep[0], CSH - 1 
    
    return vmap, nsp, nep

DIRECTS = (
    ( 0,  0),
    ( 1,  0),
    ( 1,  1),
    ( 0,  1),
    (-1,  0),
    (-1, -1),
    ( 0, -1),
    ( 1, -1),
    (-1,  1)
)

def random_dr():
    """
    None -> Direction
    Return a random direction vector
    """
    return choice(DIRECTS)

lpmp = 20, 10 

def direction(img, sp, ep, mp):
    """
    Image, Point, Point | str, bool -> Direction
    Call the appropriate direction finding function
    based on map support and ronvert points 'sp' and 'ep' 
    to map points
    End point can be string of the end location e.g. 'hill'
    """
    global cur_map, loob, map_img, lpmp

    loob = map_objs(img)

    if sp == ep:
        return 0, 0

    if cur_map in static_maps:
        msp = map_point(sp)
        if msp is None:
            if not mp:
                return rt_direction(img, sp, ep)
            else:
                return random_dr()
        msp = nearest_b(msp, map_img)
        lpmp = msp
        if type(ep) is str:
            mep = get_obj_loc(ep, msp, cur_map)
        else:
            if mp:
                mep = ep
            else:
                mep = map_point(ep)
            if mep is None:
                if not mp:
                    return rt_direction(img, sp, ep)
                else:
                    return random_dr()
            mep = nearest_b(mep, map_img)
        if msp == mep:
            return random_dr()
        return mpd_direction(msp, mep)
    else:
        return rt_direction(img, sp, ep)

objects_mps = {
    "JUTE":{
        "pit-1" :(44,  11),
        "pit-5" :(13,  11),
        "pit-4" :(42,   1),
        "pit-7" :(13,   1),
        "pit-3" :(44 , 21),
        "pit-6" :(11,  21),
        "pit-2" :(29,  15),
        "wall-2":(27,   6)
    },
    "DECA":{
        'House-8'   :(14, 11),
        'House-6'   :(25,  7),
        'House-7'   :(20, 15),
        'House-9'   :( 9, 18),
        'House-10'  :(13, 22),
        'Obstacle-6':(22, 21),
        'Obstacle-8':(30, 16),
        'Bridge-2'  :(34, 13),
        'Bridge-1'  :(32, 23),
        'House-2'   :(43, 21),
        'House-1'   :(54, 16),
        'House-3'   :(47, 10),
        'Bridge-3'  :(36,  4),
        'Obstacle-5':(45,  5),
        'House-5'   :(55,  2),
        'House-4'   :(59,  8)
    },
    "FRRE":{
        'wall-1' :(18, 18),
        'wall-2' :(28, 13),
        'wall-3' :(24, 23),
        'wall-4' :(35, 35),
        'wall-5' :(33, 31),
        'wall-6' :(44, 30),
        'wall-7' :(28, 27),
        'wall-8' :(15, 26),
        'wall-9' :(36, 24),
        'wall-10':(35, 18),
        'wall-11':(46, 20),
        'pit-1'  :(16, 23),
        'pit-2'  :(25, 29),
        'pit-3'  :(38, 16),
        'pit-4'  :(46, 23)
    },
    "FOCI":{
        'wall-1' :(46, 23), 
        'wall-3' :(38, 24), 
        'wall-4' :(34, 29),
        'wall-5' :(43, 12),
        #'wall-6' :(44,  4),
        'wall-7' :(35, 13),
        'wall-8' :(28, 12),
        'wall-9' :(14, 10),
        'wall-10':(10, 18),
        'wall-11':(16, 26)
    },
    "STST":{
        'wall-1' :(25,  5),
        'wall-2' :(18,  8),
        'wall-3' :(32,  8),
        'wall-12':(30, 13),
        'wall-11':(25, 16),
        'wall-9' :(25, 21),
        'wall-8' :(25, 25),
        'wall-7' :(22, 27),
        'wall-5' :(19, 32),
        'wall-6' :(31, 32),
        'wall-4' :(25, 35),
        'wall-13':( 3, 20),
        'wall-10':(47, 20)
    },
    "ARBA":{
        'wall-1':(20, 15),
        'wall-2':(23, 21),
        'wall-3':(28, 17),
        'wall-4':(43, 13),
        'wall-5':(49, 17)
    },
    "SAHA":{
        'w1':(26, 35),
        'w2':(35, 33),
        'w3':(41, 36),
        'w4':(41, 29),
        'w5':(17, 33),
        'w6':(37, 16),
        'w7':(31, 23),
        'w8':(21, 27),
        'w9':(11, 15)
    },
    "SHSH":{
        'w1':(27, 26),
        'w2':(16, 25),
        'w3':(36, 23),
        'p2':(15, 15),
        'p3':(34, 15)
    },
    "VAVA":{
        'Wall-2' :(25, 10),
        'Wall-3' :(31,  8),
        #'Wall-5' :(32, 13),
        'Wall-8' :(40, 21),
        'Wall-10':(42, 26),
        'Wall-11':(36, 27),
        'Wall-13':(10, 18),
        #'Wall-14':(16, 17),
        'Wall-16':(13, 22),
        'Wall-18':(20, 30),
        'Wall-19':(22, 35),
    },
    "DRCA":{
        'w2':(26, 20),
        'w3':(33, 20),
        'p3':(38, 22),
        'w4':(40, 25),
        'w5':(41, 30),
        'p4':(37, 34),
        'w6':(33, 36),
        'w7':(26, 36),
        'p5':(22, 34),
        'w8':(19, 30),
        'w9':(19, 25),
        'p2':(22, 22)
    }
}

def pred_pos(osp, name, psp):
    dapos = objects_mps[cur_map]
    oap = dapos[name] 
    pax = (oap[0] - round((osp[0] - psp[0]) / 50) if osp[0] > psp[0] else
           oap[0] + round((psp[0] - osp[0]) / 50))
    pay = (oap[1] - round((osp[1] - psp[1]) / 50) if osp[1] > psp[1] else
           oap[1] + round((psp[1] - osp[1]) / 50))
    return pax, pay

#def jute_mp(p):
#    """
#    Point -> Map Point | None
#    Return the map point for the jungle temple map
#    TODO: extend and generalize to work for all maps
#            as much as possible
#    """
#    global loob, center
#
#    lppos = []
#    p1c, p3c, p4c = (0,)    * 3
#    fp1, fp3, fp4 = (None,) * 3
#    side = None
#    gc   = center 
#    mxd  = dist(gc, (0, 0))
#
#    for (_, box), _, name in loob:
#        ow, oh = box[2] - box[0], box[3] - box[1]
#        spos = box[0] + (ow // 2), box[1] + (oh // 2)
#        pcd    = dist(spos, gc)
#        weight = round(mxd / (pcd + 1))
#
#        if name == "pit-1":
#            if p1c == 0:
#                fp1 = (spos, weight)
#            p1c += 1
#        elif name == "pit-4":
#            if p4c == 0:
#                fp4 = (spos, weight)
#            p4c += 1
#        elif name == "pit-3":
#            if p3c == 0:
#                fp3 = (spos, weight)
#            p3c += 1
#        elif name in ("pit-2", "wall-2"):
#            if side is None:
#                side = 'l' if spos[0] > p[0] else 'r'
#            pap = pred_pos(spos, name, p)
#            lppos.append((pap, name, weight))
#        else:
#            continue
#    else:
#        # Add pit 1, 3, 4
#        if p1c == 1:
#            if side is None:
#                side = 'l' if fp1[0][0] > p[0] else 'r'
#            pap = pred_pos(fp1[0], "pit-1", p, side)
#            lppos.append((pap, "pit-1", fp1[1]))
#        if p3c == 1:
#            if side is None:
#                side = 'l' if fp3[0][0] > p[0] else 'r'
#            pap = pred_pos(fp3[0], "pit-3", p, side)
#            lppos.append((pap, "pit-3", fp3[1]))
#        if p4c == 1:
#            if side is None:
#                side = 'l' if fp4[0][0] > p[0] else 'r'
#            pap = pred_pos(fp4[0], "pit-4", p, side)
#            lppos.append((pap,  "pit-4", fp4[1]))
#
#    tp = 0
#    xsm, ysm = 0, 0
#    for pap, name, weight in lppos:
#        xsm += (pap[0] * weight)
#        ysm += (pap[1] * weight)
#        tp  += weight 
#
#    if tp != 0:
#        pap = round(xsm / tp), round(ysm / tp)
#        return pap
def update_ploc(img, ploc):
    global lpmp, loob
    loob = map_objs(img) 
    mp = map_point(ploc)
    lpmp = lpmp if mp is None else mp

doobig = {
    '':[],
    'JUTE':['wall-1', 'wall-3'],
    'DECA':["Obstacle-1",
            "Obstacle-2",
            "Obstacle-3",
            "Obstacle-4",
            "Obstacle-7"],
    'FRRE':[],
    'FOCI':[
            'wall-2',
            'wall-6'
            ],
    'STST':[
        'pit-1',
        'pit-2',
        'pit-3',
        'pit-4',
    ],
    'ARBA':[
        'pit-1',
    ],
    'SAHA':[
        'p1',
        'p2',
        'p3',
        'p4',
    ],
    'SHSH':[
        'p1',
    ],
    'VAVA':[
            'Wall-5',
            'Wall-14',
            'Wall-12',
            'Wall-17',
            "Wall-1",
            "Wall-4",
            "Pit-1",
            "Pit-2",
            "Pit-3",
            "Wall-6",
            "Wall-7",
            "Pit-4",
            "Wall-9",
            "Pit-5",
            "Pit-6",
            "Pit-7",
            "Wall-15",
            "Pit-8",
    ],
    'DRCA':[
        'w1',
        'p1'
    ]
}

def map_point(p):
    """
    Point -> Map Point | None
    Return the map point for the jungle temple map
    """
    global loob, center, doobig, cur_map

    lppos  = []
    gc     = center 
    mxd    = dist(gc, (0, 0)) ** 2
    loobig = doobig[cur_map]

    for (_, box), _, name in loob:
        if name in loobig:
            continue 
        ow, oh = box[2] - box[0], box[3] - box[1]
        spos   = box[0] + (ow // 2), box[1] + (oh // 2)
        pcd    = dist(spos, gc) ** 2
        weight = round(mxd / (pcd + 1))
        pap    = pred_pos(spos, name, p)
        lppos.append((pap, weight))

    tp = 0
    xsm, ysm = 0, 0
    for pap, weight in lppos:
        xsm += (pap[0] * weight)
        ysm += (pap[1] * weight)
        tp  += weight 

    if tp != 0:
        pap = round(xsm / tp), round(ysm / tp)
        return pap
    
def aoi_closest_to(mp):
    """
    Map Point -> Map Point
    Find the area of interest closest to the given
    map point
    """
    global cur_map

    loaoi = static_objs[cur_map]['aoi']
    baoi, bd = None, None
    for aoi in loaoi:
        dst = dist(mp, aoi)
        if baoi is None:
            baoi = aoi
            bd   = dst
        elif dst < bd:
            baoi = aoi
            bd   = dst

    return baoi

lvaoi = None, time()

def aoi_farthest_from(mp):
    """
    Map Point -> Map Point
    Find the area of interest farthest from the
    given map point
    """
    global cur_map, lvaoi

    if not (lvaoi[0] is None) and (time() - lvaoi[1]) < 10:
        return lvaoi[0]

    loaoi = static_objs[cur_map]['aoi']
    baoi, bd = None, None
    for aoi in loaoi:
        dst = dist(mp, aoi)
        if baoi is None:
            baoi = aoi
            bd   = dst
        elif dst > bd:
            baoi = aoi
            bd   = dst

    lvaoi = baoi, time()
    return baoi

def aoi_random():
    """
    Map Point -> Map Point
    Return a random area of interest
    from the list of aoi(s) based on
    current map
    """
    global cur_map, lvaoi

    if not (lvaoi[0] is None) and (time() - lvaoi[1]) < 10:
        return lvaoi[0]

    lopp  = static_objs[cur_map]['aoi']
    if not (lvaoi[0] is None):
        li    = lopp.index(lvaoi[0])
        lopp  = lopp[:li] + lopp[li+1:]

    aoi   = choice(lopp)
    lvaoi = aoi, time()

    return aoi

static_objs = {
    "JUTE":{
        "hill"   :((28,  9), (27, 9), (29, 9)),
        "barrels":[(16, 11), (40, 10)],
        "brls_mp":((0, 1, 0, 0), (1, 0, 0, 0)),
        "aoi"    :((7, 10), (20, 12), (36, 12), (50, 10))
        },
    "DECA":{
        "hill":((35, 12), (35, 13)),
        "barrels":[(47, 9), (21, 17)],
        "brls_mp":((0, 0, 0, 0), (0, 0, 0, 0)),
        "aoi"    :((13, 15), (19, 8), (24, 19), (27, 12),
                   (43, 8), (41, 15), (49, 17), (54, 10))
    },
    "FRRE":{
        'hill'   :((32, 14), (32, 15), (32, 16)),
        'barrels':[(27, 19), (38, 29)],
        'brls_mp':((0, 0, 0, 0), (0, 0, 0, 0)),
        'aoi'    :((21, 13), (30, 21), (18, 28), (36, 10),
                   (41, 21), (31, 28), (30, 35), (39, 28),
                   (43, 35), (50, 29), (50, 21))
    },
    "FOCI":{
        'hill'   :((30, 15), (29, 16), (28, 15)),
        'barrels':[(41, 15), (20, 17)],
        'brls_mp':((0, 0, 0, 0), (1, 0, 1, 0)),
        'aoi'    :((16, 19), (22, 17), (23, 25), (41, 26), 
                   (39, 18), (48, 15), (32, 10), (23, 11))
    },
    "STST":{
        'hill'   :((8, 19), (9, 19), (10, 19)),
        'barrels':[(47, 14), (3, 25)],
        'brls_mp':[(0, 0, 0, 0), (0, 0, 0, 0)],
        'aoi'    :((9, 19), (25, 19), (41, 19), (25, 8),
                   (25, 32))
    },
    "ARBA":{
        'hill'   :((32, 21), (33, 21), (34, 21)),
        'barrels':[(23, 15), (45, 17)],
        'brls_mp':[(0, 1, 0, 0), (0, 0, 0, 0)],
        'aoi'    :((20, 18), (26, 11), (42, 20), (43, 10))
    },
    "SAHA":{
        'hill'   :((25, 24), (26, 24), (27, 24)),
        'barrels':[(22, 22), (37, 33)],
        'brls_mp':[(0, 0, 1, 0), (0, 1, 0, 0)],
        'aoi'    :((8, 20), (24, 24), (20, 35), (38, 35),
                   (40, 24), (31, 13))
    },
    "SHSH":{
        'hill'   :((25, 18), (26, 18), (27, 18)),
        'barrels':[(26, 20), (27, 30)],
        'brls_mp':[(0, 0, 0, 0), (0, 0, 0, 0)],
        'aoi'    :((9, 24), (25, 18), (27, 31), (43, 23))
    },
    "VAVA":{
        'hill'   :((25, 21), (26, 21), (27, 21)),
        'barrels':[(19, 16), (34, 26)],
        'brls_mp':[(0, 0, 0, 0), (0, 0, 0, 0)],
        'aoi'    :((18, 20), (26,  8), (34, 22), (27, 34)) 
    },
    "DRCA":{
        'hill'   :((29, 19), (30, 19), (31, 19)),
        'barrels':[(40, 27), (20, 27)],
        'brls_mp':[(0, 0, 0, 0), (0, 0, 0, 0)],
        'aoi'    :[(16, 18), (20, 28), (17, 37), (30, 35),
                   (43, 37), (40, 27), (44, 17), (30, 18)]
    }
}

map_points = {
    "JUTE":(
        (10, 16),
        ( 9, 16),
        (17,  5),
        (10,  5),
        (18, 16),
        (13, 16),
        (23,  3),
        (21, 18),
        (28,  1),
        (28, 20),
        (33,  3),
        (23, 10),
        (32, 10),
        (35, 18),
        (40,  5),
        (40, 16),
        (46,  5),
        (47, 16)
    ),
    "DECA":(
        ( 5, 19),
        ( 6, 14),
        ( 6, 22),
        (11,  9),
        (11, 14),
        (10, 22),
        (19,  8),
        (16, 17),
        (19, 21),
        (24, 11),
        (23, 17),
        (22, 23),
        (29,  3),
        (29, 10),
        (29, 14),
        (28, 23),
        (35,  3),
        (35, 11),
        (35, 14),
        (36, 23),
        (42,  3),
        (41, 12),
        (39, 19),
        (47,  3),
        (45,  8),
        (44, 16),
        (49,  5),
        (48, 18),
        (50, 14),
        (52,  8),
        (56,  6),
        (55, 12),
        (54, 19),
        (60,  4),
        (60, 13),
        (63,  6),
        (62, 11)
    ),
    'FRRE':(
        (12, 25),
        (12, 20),
        (17, 15),
        (15, 28),
        (22, 15),
        (21, 20),
        (21, 26),
        (21, 31),
        (24, 11),
        (25, 18),
        (25, 26),
        (27, 32),
        (28, 23),
        (30, 19),
        (32, 15),
        (28,  8),
        (34,  8),
        (37, 11),
        (35, 20),
        (31, 27),
        (32, 37),
        (37, 37),
        (40, 35),
        (39, 31),
        (38, 26),
        (40, 23),
        (42, 17),
        (42, 11),
        (47, 15),
        (44, 26),
        (46, 34),
        (50, 32),
        (50, 26),
        (50, 19) 
    ),
    "FOCI":(
        ( 4, 14),
        ( 4, 21),
        ( 7, 23),
        (10, 23),
        (10, 28),
        (13, 31),
        (14, 20),
        (17, 14),
        (23,  8),
        (23, 13),
        (20, 17),
        (24, 18),
        (21, 23),
        (24, 26),
        (21, 31),
        (30, 26),
        (32, 15),
        (32,  8),
        (35,  3),
        (38, 10),
        (39, 15),
        (40, 20),
        (40, 24),
        (38, 26),
        (43, 27),
        (43, 17),
        (47, 10),
        (47, 15),
        (50, 18),
        (53, 18),
        (53, 25),
        (51, 27)
    ),
    "STST":(
        (10,  4),
        ( 6,  8),
        ( 6, 16),
        ( 7, 24),
        ( 8, 31),
        (13, 34),
        (12, 19),
        (20,  4),
        (19, 14),
        (17, 19),
        (17, 25),
        (18, 28),
        (20, 36),
        (25, 31),
        (26, 28),
        (20, 23),
        (21, 18),
        (26, 14),
        (27, 11),
        (25,  9),
        (30,  4),
        (33, 11),
        (33, 14),
        (30, 14),
        (30, 18),
        (29, 23),
        (31, 26),
        (30, 36),
        (37, 34),
        (44, 32),
        (45, 26),
        (40, 20),
        (42,  8),
        (36,  4),
        (35, 19)
    ),
    "ARBA":(
        (15, 10),
        (15, 17),
        (18, 24),
        (25, 24),
        (26, 20),
        (26, 17),
        (26, 10),
        (30, 21),
        (39, 10),
        (39, 16),
        (39, 21),
        (44, 17),
        (46, 21),
        (47, 14),
        (50, 10),
        (54, 14),
        (54, 21),
        (23, 11)
    ),
    "SAHA":(
        ( 6, 12),
        ( 6, 18),
        (11, 23),
        (12, 29),
        (14, 35),
        (14,  9),
        (17, 24),
        (18, 29),
        (21, 32),
        (20, 36),
        (23, 37),
        (27, 26),
        (24, 24),
        (23,  9),
        (27, 11),
        (32, 14),
        (32, 19),
        (31, 24),
        (36, 37),
        (42, 31),
        (42, 24),
        (42, 19),
        (41, 16),
        (41, 12),
        (25, 19)
    ),
    "SHSH":(
        ( 7, 10),
        ( 7, 20),
        (10, 29),
        (18, 29),
        (21, 29),
        (21, 20),
        (25, 15),
        (25, 10),
        (32, 19),
        (32, 28),
        (29, 32),
        (41, 10),
        (41, 19),
        (41, 28)
    ),
    "VAVA":(
        (16,  7),
        ( 8, 13),
        (16, 13),
        (12, 16),
        ( 8, 20),
        (12, 20),
        (11, 23),
        ( 9, 27),
        (14, 26),
        (10, 30),
        (16, 30),
        (19, 33),
        (19, 37),
        (18, 26),
        (18, 21),
        (20, 14),
        (21, 22),
        (21, 18),
        (20, 11),
        (24,  8),
        (28,  6),
        (29,  9),
        (29, 14),
        (26, 17),
        (26, 21),
        (27, 24),
        (25, 26),
        (25, 28),
        (28, 37),
        (32, 33),
        (33, 28),
        (31, 24),
        (31, 21),
        (33, 16),
        (35, 14),
        (35,  9),
        (35,  6),
        (38, 13),
        (45, 13),
        (38, 16),
        (41, 19),
        (41, 22),
        (45, 22),
        (39, 24),
        (45, 28),
        (39, 30),
        (36, 30),
        (36, 37)
    ),
    "DRCA":(
        ( 9, 14),
        (15, 16),
        (15, 28),
        (16, 38), 
        (30, 38),
        (30, 34),
        (23, 28),
        (29, 21),
        (30, 16),
        (44, 16),
        (48, 16),
        (37, 27),
        (44, 27),
        (44, 38)
    )
}

def closest_to(p, lop):
    """
    Point, List of Point -> Point
    Return the closest point in 'lop'
    to 'p'
    """
    bp, bd = None, None
    for lp in lop:
        pd = dist(lp, p)
        if bd is None:
            bp = lp
            bd = pd
        elif pd < bd:
            bp = lp
            bd = pd
    return bp

def get_obj_loc(name, msp, cm):
    """
    str, Map Point, str -> Map Point
    Given name of game object return
    it's location based on given map name 'cm'
    'msp' starting point is used to determine
    best point if many are available
    """
    if name == "hill":
        return static_objs[cm][name]
    elif name == "barrels":
        return closest_to(msp, static_objs[cm][name])

def reachable(pa, pb):
    """
    Map Point, Map Point -> bool
    Return True if a it's possible to go from
    point 'pa' to 'pb' in the current map
    """
    global map_img
    elm = cv.line(np.zeros_like(map_img), pa, pb, 1)
    return not (255 in map_img[elm==1])

def last_reachable(mp, lomp):
    """
    Map Point, List(Map Point) -> Map Point
    Given a source map point return the farthest
    reachable map point given a sorted list of map point
    based on distance
    """
    for imp in lomp[::-1]:
        if reachable(mp, imp):
            return imp

def next_point(msp, mep):
    """
    Map Point, Map Point -> Map Point | None
    Return a point reachable from 'msp'
    that gets player closer  to   'mep'
    """
    global map_img, cur_map

    lomp  = map_points[cur_map] + (mep,)

    if reachable(msp, mep):
        return (mep,)
    else:
        cp   = msp
        lovp = {
            cp:((), 0),
            } 
        lorp = []
        heapq.heapify(lorp)

        while cp != mep:
            # Add all reachable points
            for mp in filter(lambda mp: reachable(cp, mp), lomp):
                heapq.heappush(lorp, (lovp[cp][1] + dist(mp, cp), mp, cp))

            # Select and visit next closest point
            for _ in range(len(lorp)):
                spd, sp, lp = heapq.heappop(lorp)
                if (not (sp in lovp) or
                    lovp[sp][1] > spd):
                    lovp[sp] = (lovp[lp][0] + (sp,), spd)
                    cp = sp
                    break
            else:
                return (mep,)

        return lovp[mep][0]

# Direction based on a pre-built map
def mpd_direction(msp, mep):
    """
    Map Point, Map Point -> Direction
    Return the direction to follow to get
    to point 'mep'
    """
    mnp = next_point(msp,  mep)[0]
    #path     = heap_best_path(map_img, msp, mep, gap=1)
    #path     = untangle(  path, map_img)
    #dr, _    = path_to_dr(path, max_d=2, custom_steps=range(1, 11))
    dr, _     = path_to_dr([msp, mnp])
    #drwmap = map_img.copy()
    #drwmap[msp[1], msp[0]] = 125 
    #drwmap[mnp[1], mnp[0]] = 125 
    #cv.imshow("map", cv.resize(drwmap, (1066, 600)))
    #cv.waitKey(1)
    return dr

CHK_CLCT_IM = read_img('src/modes/chk_clctd.png')

def chck_collected(img):
    """
    Image -> bool
    Return True if tank is currently holding a chick
    or more return False if no chick was collected
    """
    pos = look_for(recal(CHK_CLCT_IM ,ogsz=1920, wonly=True), img,
                   0.8, clr_match=True)
    return not (pos is None)

# Real Time live direction calculation
def rt_direction(img, sp, ep):
    """
    Image, Point, Point -> (int[-1:1], int[-1:1])
    Determine the direction to go based on given list of objects,
    this returns least blocking direction toward ep from sp
    serving as a pathing solution
    Return one of the 8 possible move directions (left-right, up-down)
    """
    global lpt

    vmap = make_vmap(img)
    # Testing, TODO: Remove or Comment
    #vmap = cv.drawMarker(vmap, sp, 255, cv.MARKER_CROSS, 50, 5)
    #cv.imshow("vmap", cv.resize(vmap, (1066, 600)))
    #cv.waitKey(1)
    #return
    # ---------------------------------
    vmap, sp, ep = shrink_vmap(vmap, sp, ep)
    # Testing, TODO: Remove
    #img.img = np.pad(img.img, ((0, 0), (0, 0), (0, 1)))
    #img.img[:, :, 3] = 255
    #img.img[vmap==255] = np.array((0, 255, 0, 50), dtype=img.img.dtype)
    #cv.imshow("Path Image", cv.resize(img.img, (800, 600)))
    #cv.waitKey(0)
    #return
    #
    #pst = time()
    #bpath = best_path(vmap, sp, ep)
    #pt = time() - pst

    # Testing
    #hpst = time()
    bpath = heap_best_path(vmap, sp, ep)
    #hpt = time() - hpst
    bpath = untangle(bpath, vmap)
    #tvmap = draw_path(bpath, vmap.copy())  # Testing
    dr, pe = path_to_dr(bpath)
    #lonpd = make_vpoints(sp)
    #lod = difficulty(vmap, lonpd, sp, ep)
    #lod.sort(key=lambda d: d[0])
    #for npd in lonpd:
    #    p, d = npd
    #    vmap = cv.drawMarker(vmap, p, 255, cv.MARKER_CROSS, 20, 5)
    #    #vmap = cv.putText(vmap, str(d), (p[0], p[1] - 50), 
    #    #                  cv.FONT_HERSHEY_COMPLEX, 1, 255, 2)
    #print(lod)
    #print(dr)
    #vmap = draw_path(bpath, vmap)  # Testing
    #cv.imshow("oimg", cv.resize(
    #    np.where(np.pad(np.reshape(vmap, (vmap.shape[0], vmap.shape[1], 1)), ((0, 0), (0, 0), (0, 2)), constant_values=255)==(255, 255, 255), 
    #             img.img, np.array((255, 255, 255), 
    #                                          dtype=img.img.dtype)), 
    #    (800, 600)))
    #print("Detect T: {} | Path T: {} | Heap T: {}"
    #      .format(dt, pt, hpt))
    #vmap = cv.drawMarker(vmap, pe, 255, cv.MARKER_DIAMOND, 2, 1)
    #vmap = cv.drawMarker(vmap, sp, 255, cv.MARKER_DIAMOND, 2, 1)
    #vmap = cv.drawMarker(vmap, ep, 255, cv.MARKER_DIAMOND, 2, 1)
    #cv.imshow("vmap",  cv.resize( vmap, (1066, 600)))
    #cv.waitKey(1)
    return dr

grim  = read_img("src/contracts/garage.png",   "grey")
cntim = read_img("src/contracts/contract.png", "grey")

def contracts(img):
    """
    Image -> List of Point
    Return a list of all contracts positions visible on screen
    """
    global cntim

    rcntim = recal(cntim, ogsz=1920, wonly=True)
    return remove_close(find_all(rcntim, img, 0.8))

def contract_slctd(img, cp):
    """
    Image, Point -> bool
    Return True if the contract in the given position is selected
    False if not
    """
    for x in range(300):
        clr = img.img[cp[1], cp[0]+x]
        if clr[0] < 100 and clr[1] > 200:
            return True
    return False

piltim   = read_img("src/contracts/pilot.png",         "grey")
grgbtnim = read_img("src/contracts/garage.png",        "grey")
pltofim  = read_img("src/contracts/plt_off.png",       "grey")
cntrim   = read_img("src/contracts/contracts_btn.png", "grey")
bkbtnim  = read_img("src/contracts/back.png",          "grey")

def in_garage(img):
    """
    Image -> bool
    return True if currently in the garage interface
    True if not
    """
    rpiltim = recal(piltim, ogsz=1920, wonly=True)
    return not look_for(rpiltim, img, 0.8) is None

def garage_btn(img):
    """
    Image -> Point | None
    Return None if the garage button was not found
    else return the position on screen of the button
    """
    rgrgbtnim = recal(grgbtnim, ogsz=1920, wonly=True)
    return look_for(rgrgbtnim, img, 0.8)

def pilot_uncheck(img):
    """
    Image -> Point | None
    Return None if the pilot uncheck mark is not visible
    return its position on screen if its visible
    """
    rpltofim = recal(pltofim, ogsz=1920, wonly=True)
    return look_for(rpltofim, img, 0.9)

def contracts_btn(img):
    """
    Image -> Point | None
    Return None if the 'change contracts' button is not visible
    or return the button position on screen if it's visible
    """
    rcntrim = recal(cntrim, ogsz=1920, wonly=True)
    return look_for(rcntrim, img, 0.8)

def back_btn(img):
    """
    Image -> Point | None
    Return None if the 'back' button is not visible on screen
    if it's visible return its location on screen
    """
    rbkbtnim = recal(bkbtnim, ogsz=1920, wonly=True)
    return look_for(rbkbtnim, img, 0.8)

#def _map_sift():
#    global rltv_sz
#    FOLDER = "src/maps/desert/"
#    loip = listdir(FOLDER)
#    loim = [read_img(join(FOLDER, ip)) for ip in loip]
#
#    for i in range(58):
#        tstim = read_img("_pathing/{}.png".format(i))
#        rltv_sz = prod(tstim.size)
#        for oim in loim:
#            wh = oim.size
#            print(wh)
#            exsz = recal_wh(wh)
#            box = sift_find(oim, tstim, 10)
#            if box:
#                x, y, w, h = box
#                x1, y1, x2, y2 = x, y, x+w, y+h
#                print("Expected size {}, Actual Size {}".format(exsz, (w, h)))
#                if size_check(box, oim.size):
#                    print("Passed size check!\n")
#                    clr = (0, 255, 0)
#                else:
#                    clr = (0, 0, 255)
#                tstim.img = cv.rectangle(tstim.img, (x1, y1), (x2, y2),
#                            clr, 5)
#        tstim = tstim.resize((500, 350))
#        tstim.show()
        
#def clr_map(oim):
#    #oim = cv.resize(oim, (500, 281))
#    im = cv.cvtColor(oim, cv.COLOR_BGR2HSV)
#    r, g, b = cv.split(im)
#    fig = plt.figure()
#    axis = fig.add_subplot(1, 1, 1, projection="3d")
#    pixel_colors = oim.reshape((np.shape(im)[0]*np.shape(im)[1], 3))
#    norm = colors.Normalize(vmin=-1.,vmax=1.)
#    norm.autoscale(pixel_colors)
#    pixel_colors = norm(pixel_colors).tolist()
#    axis.scatter(r.flatten(), g.flatten(), b.flatten(), 
#                 facecolors=pixel_colors, marker='.')
#    axis.set_xlabel("Hue")
#    axis.set_ylabel("Saturation")
#    axis.set_zlabel("Value")
#    plt.show()
#
#def clr_pathing():
#    for i in range(58):
#        oim = cv.imread("_pathing/{}.png".format(i))
#        #clr_map(oim)
#        #break
#        im = cv.cvtColor(oim, cv.COLOR_BGR2HSV)
#        dst = cv.inRange(im, (30, 0, 0), (255, 255, 255))
#        dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
#        #dst = np.expand_dims(dst, axis=2)
#        #dst = np.pad(dst, ((0, 0), (0, 0), (0, 2)))
#        #dst[dst==(255, 0, 0)] = np.array((255, 255, 255))
#        oim = np.where(dst == (255, 255, 255), np.array((255, 255, 255), dtype=oim.dtype), 
#                                   oim)
#
#        oim = cv.resize(oim, (500, 350))
#        dst = cv.resize(dst, (500, 350))
#        print(i)
#        cv.imshow("Image", oim)
#        cv.imshow("Threshold", dst)
#        cv.waitKey(0)
#
#def depth_pathing(img):
#    midas = torch.hub.load("intel-isl/MiDaS", model_type)
#    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#    midas.to(device)
#    midas.eval()
#
#    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#
#    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#        transform = midas_transforms.dpt_transform
#    else:
#        transform = midas_transforms.small_transform
#
#    for i in range(1, 58):
#        oimg = cv.imread("_pathing/{}.png".format(i))
#        oimg = oimg[148:948, :]
#        oimg = cv.resize(oimg, (500, 350))
#        img = cv.cvtColor(oimg, cv.COLOR_BGR2RGB)
#        #noutput = cv.cvtColor(oimg, cv.COLOR_BGR2GRAY)
#
#        input_batch = transform(img).to(device)
#
#        with torch.no_grad():
#            prediction = midas(input_batch)
#
#            prediction = torch.nn.functional.interpolate(
#                prediction.unsqueeze(1),
#                size=img.shape[:2],
#                mode="bicubic",
#                align_corners=False,
#            ).squeeze()
#
#        output = prediction.cpu().numpy()
#
#        noutput = output / np.max(output)
#        noutput *= 255
#        noutput = noutput.astype(np.uint8)
#
#        cv.imshow("Test", noutput)
#        cv.imshow("Image", oimg)
#        #cv.imshow("Edge", edgim)
#        def aux(*args):
#            e, x, y, _, _ = args
#            print((x, y), "Depth:", noutput[y, x])#, output[y, x])
#        cv.setMouseCallback("Image", aux)
#        if cv.waitKey(0) == ord("q"):
#            cv.destroyAllWindows()
#            break
#
#def build_histogram(image, bins=256):
#    # convert from BGR to RGB
#    #rgb_image = np.flip(image, 2)
#    rgb_image = image
#    # show the image
#    #plt.imshow(rgb_image)
#    # convert to a vector
#    image_vector = rgb_image.reshape(1, -1, 3)
#    # break into given number of bins
#    div = 256 / bins
#    bins_vector = (image_vector / div).astype(int)
#    # get the red, green, and blue channels
#    red = bins_vector[0, :, 0]
#    green = bins_vector[0, :, 1]
#    blue = bins_vector[0, :, 2]
#    # build the histograms and display
#    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
#    axs[0].hist(red, bins=bins, color='r')
#    axs[1].hist(green, bins=bins, color='g')
#    axs[2].hist(blue, bins=bins, color='b')
#    #plt.show()
def build_map_memo(pm):
    global cur_map, center, map_img

    #hwnd = find_window("Spider Tanks", exact=True)
    #win = Window(hwnd)
    #win.repos(0, 0)
    #new_win(win.size)
    #reg = 0, 0, win.size[0], win.size[1]

    cur_map = pm
    #load_map_model(pm)
    #load_chick_model()
    map_img = cv.imread("src/maps/temple.png", 0)

    #center = round(reg[2] / 2), round(reg[3] / 2)
    h, w   = map_img.shape
    #file = open('JUTE.memo', 'wb')
    #memo = pickle.load(file)
    memo   = {}

    st = time()
    for sx in range(w):
        for sy in range(h):
            for ex in range(w):
                for ey in range(h):
                    msp = (sx, sy)
                    mep = (ex, ey)
                    path     = heap_best_path(map_img, msp, mep, no_gap=True)
                    path     = untangle(  path, map_img)
                    memo[(msp, mep)] = path[1:6]
        print((sx, sy), (ex, ey))
        print("Time elapsed so far: {}".format(time() - st))
    
    file = open('JUTE.memo', 'wb')
    pickle.dump(memo, file)
    file.close()


if __name__ == "__main__":
    import pickle
    from cvbot.capture import screenshot , get_region
    from cvbot._screen import MON
    from cvbot.io import read_img
    from cvbot.windows import get_window
    from cvbot import mouse 
    from time import sleep
    from sys import argv
    from os import listdir
    from os.path import join
    from cvbot.windows import Window, find_window
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

    #hwnd = find_window("Spider Tanks", exact=True)
    #win = Window(hwnd)
    #win.repos(0, 0)
    #new_win(win.size)
    #reg = 0, 0, win.size[0], win.size[1]
    #__record(reg)
    #quit()

    #img = get_region(reg)
    #loc = contracts(img)
    #for con in loc:
    #    if contract_slctd(img, con):
    #        print("Contract {} is selected!".format(con))
    #quit()
    #build_map_memo("JUTE")
    #quit()
    #print("\nLoading paths")
    #st = time()
    #with open("JUTE.memo", "rb") as mf:
    #    memo = pickle.load(mf)
    #    with lzma.open("JUTE.xz", "wb") as lf:
    #        pickle.dump(memo, lf)
    #    lt = time() - st
    #    print(memo[(7, 10), (28, 9)])
    #print("Took {} to load".format(lt))
    #quit()
    #st = time()
    #with lzma.open("JUTE.xz", "rb") as pf:
    #    memo = pickle.load(pf)
    #    #data = pf.read()
    #    lt = time() - st
    #    print(memo[(7, 10), (28, 9)])
    #print("Took {} to load".format(lt))
    #quit()
    hwnd = find_window("Spider Tanks", exact=True)
    win = Window(hwnd)
    win.repos(0, 0)
    new_win(win.size)
    reg = 0, 0, win.size[0], win.size[1]

    img = get_region(reg)
    #p = retry_button(img)
    #print(p)
    #if not (p is None):
    #    mouse.click(p)
    #quit()
    #__record(reg)
    #quit()

    #wnsz = 1616, 939
    #new_win(wnsz)
    #reg = 0, 0, wnsz[0], wnsz[1]
    #get_region = lambda reg: read_img("D:/dev/spdrtank/__test/_cc/7878.png")

    pm = "DRCA"
    cur_map = pm
    load_map_model(pm)
    #load_chick_model()

    # Width x Height: 50 x 20
    # Pit-1  -> 16, 10
    # Pit-4  -> 14, -2

    center = round(reg[2] / 2), round(reg[3] / 2)


    while True:
        img = get_region(reg)
        #objs = []
        #for (mask, box), scr, name in map_objs(img):
        #    obc = box[2] - box[0], box[3] - box[1]
        #    objs.append((obc, name))
        ppos = player(img)
        if not ppos:
            print("Player not found")
            sleep(1)
            continue
        else:
            ppos = ppos[0]

        loob = map_objs(img)
        mp   = map_point(ppos)

        for (_, box), _, name in loob:
            ow, oh = box[2] - box[0], box[3] - box[1]
            spos   = box[0] + (ow // 2), box[1] + (oh // 2)
            img.img = cv.drawMarker(img.img, spos, (0, 255, 0), 
                                    cv.MARKER_DIAMOND, 50, 5)
            img.img = cv.putText(img.img, name, (spos[0], spos[1] - 10),
                                 cv.FONT_HERSHEY_COMPLEX, 
                                 1, (0, 255, 0), 3)
        #chks = locate_chicks(img)
        tmim = cv.imread("src/maps/dragon.png", 0)
        #for chk in chks:
        #    cmp = map_point(chk)
        #    tmim[cmp[1], cmp[0]] = 125

        if mp is None:
            print("Map point is none for ppos: " + str(ppos))
        else:
            #print("In next point")
            #mp = nearest_b(mp, tmim)
            #path = next_point(mp, choice(((25, 21), (26, 21), (27, 21))))
            #print(path)
            #print("Out next point")
            #mnp = path[0]
            #tmim[mnp[1], mnp[0]] = 125
            tmim[mp[1], mp[0]] = 125
            print("Map loc:" + str(mp))
        #objs.append((ppos, "player"))
        #result = ""
        #for c, n in objs:
        #    result += "{}: {} ".format(n, dist(c, ppos))
        #print("")
        #print(result)
        #print("")
        cv.imshow("test", cv.resize(img.img, (1066, 600)))
        cv.imshow("map",  cv.resize(tmim,    (1066, 600)))
        cv.waitKey(1)
        sleep(1)
        print("\n")

    quit()
    #new_win((1920, 1080))
    #hlim = read_img("htest.png")
    #pos = hill(hlim)
    #print(pos)
    #quit()
    #cur_map = "SAHA"
    #load_map_model(cur_map)
    #while True:
    #    img = get_region(reg)
    #    loob = map_objs(img)
    #    vmap = make_vmap(img)
    #    cv.imshow("test", cv.resize(vmap, (1066, 600)))
    #    cv.waitKey(1)
    #rect = confirm_dialog(img)
    #print("CONFIRM DIALOG RETURNED: {}".format(rect))
    #rect = retry_button(img)
    #print("CONFIRM DIALOG RETURNED: {}".format(rect))
    #if not (rect is None):
    #    pos = rect[0] + 5, rect[1] + 5
    #    print(pos)
    #    #mouse.click(pos)
    #quit()
    #center = reg[2] // 2, reg[3] // 2
    #p1 = 1130, 56
    #p2 = 94, 756
    #cur_map = "DRCA"
    #load_map_model(cur_map)
    ##img = get_region(reg)
    #img = read_img("__test/_cc/{}.png".format(argv[1]))
    #direction(img, p1, p2)
    #quit()
    #init_ocr(['en'])
    #img = read_img("test.png")
    #recognize_map(img)
    #img = read_img("test.png")
    #direction(img, (0, 0), (0, 0))
    #quit()
    sp = (71, 64)
    ep = (45, 62)
    cur_map = "VAVA"
    load_map_model("VAVA")
    for i in range(int(argv[1]), 700):
        img = read_img("_test/_pathing/vault/{}.png".format(i))
        print("Image:", i)
        direction(img, sp, ep)
    quit()
    if argv[1] == "test":
        while True:
            img = read_img("test.png")
            #ep = barrels(img)[int(argv[2])]
            ep = locate_chicks(img)[int(argv[2])]
            sp = player(img)[0]
            cur_map = "VAVA"
            load_map_model("VAVA")
            print(sp, ep)
            direction(img, sp, ep)
    else:
        for i in range(int(argv[1]), 8000):
            print("\n", i, "\n")
            img = read_img("_test/_cc/{}.png".format(i))
            bi = int(argv[2])
            lob = barrels(img)
            if bi >= len(lob):
                bi = 0
            ep = barrels(img)[bi]
            sp = player(img)[0]
            direction(img, sp, ep)
            print("Open Fire:", open_fire_line(sp, ep))
    #ngim = img.grey()
    #edgim = cv.Canny(ngim, 150, 200)
    #cv.imshow("edges", edgim)
    #cv.waitKey(0)
    quit()
    #img = get_region(reg)
    #out = dsr_model.detect(img, 0.5)
        #for ((x1, y1, x2, y2), scr, name) in out:
        #    img.img = cv.rectangle(img.img, (x1, y1), (x2, y2), 
        #                           (0, 255, 0), 5)
        #    img.img = cv.putText(img.img, name, (x1, y1-50),
        #                         cv.FONT_HERSHEY_COMPLEX, 2, 
        #                         (0, 255, 0), 5)
        #img = img.resize((800, 600))
        #cv.imshow("image", img.img)
        #cv.waitKey(1)
    gsp = (620, 568)
    gep = (1504, 571)
    direction(out, gsp, gep)
    quit()
    #__record(reg)
    quit()
    #__record(reg)
    #oim = cv.resize(oim, (500, 281))
    #oim = cv.imread("chktst.png".format(argv[1]))
    #oim = cv.cvtColor(oim, cv.COLOR_BGR2HSV)
    #cv.imwrite("hsv.png", oim)
    #hist_check(oim)
    #quit()
    #for i in range(5):
    #    oim = cv.imread("_cc/chks/chick_{}.png".format(i))
    #    oim = cv.cvtColor(oim, cv.COLOR_BGR2HSV)
    #build_histogram(oim)
    #plt.show()
    #clr_map(oim)
    #quit()
    #quit()

    #for i in range(1000):
    while True:
        #img = read_img("_cc/{}.png".format(i))
        img = get_region(reg)
        locate_chicks(img)

    #    for chk in loc:
    #        img.img = cv.drawMarker(img.img, chk, (0, 255, 0), cv.MARKER_CROSS, 50, 5)
    #    img = img.resize((1280, 720))
    #    print("i:", i, "Chicks:", len(loc))
        cv.imshow("test", img.img)
        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break
    #wp = Weapon(None)
    #wp.update(img)
    #print(wp.type)
    #init_nr()
    quit()
    gndclr = (50, 80, 120), (100, 170, 220)

    for i in range(1, 58):
        oimg = cv.imread("_pathing/{}.png".format(i))
        oimg = oimg[148:948, :]

        edgim = cv.cvtColor(oimg, cv.COLOR_BGR2GRAY)
        edgim = cv.GaussianBlur(edgim, (9, 9), 0.6)
        edgim = cv.Canny(edgim, 90, 255)

        cnts, _ = cv.findContours(edgim, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #oimg = cv.drawContours(oimg, cnts, -1, (255, 255, 255), cv.FILLED)
        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            pimg = oimg[y:y+h, x:x+w].copy()
            gcc = (cv.inRange(pimg, gndclr[0], gndclr[1]) == 255).sum()
            #wpxls = (pimg > 125).sum()
            #rect = cv.minAreaRect(cnt)
            #_, (w, h), _ = rect
            #area = w * h
            #box = cv.boxPoints(rect)
            #box = np.int0(box)
            #if w == 0 or h == 0:
            #    whr = 0
            #else:
            #    whr = w / h if w < h else h / w
            #bpratio = bpxls / area
            if area > 2500:
                print(area, gcc)
                if gcc < 5000:
                    oimg = cv.rectangle(oimg, (x, y), (x+w, y+h), (0, 255, 0), 5)
                #print(area, wpxls)
                #print(whr)
                #oimg = cv.drawContours(oimg, (cnt,), 0, (0, 255, 0), -1)
                #if whr > 0.2:
                #    oimg = cv.drawContours(oimg, [box], 0, (0, 255, 0), 2)
                #if bpratio < 0.90:
        #        if wpxls > 1500:
                #oimg = cv.rectangle(oimg, (x, y), (x+w, y+h), (0, 255, 0), 5)

        edgim = cv.resize(edgim, (500, 350))
        oimg = cv.resize(oimg, (500, 350))

        cv.imshow("Image", oimg)
        cv.imshow("Edge", edgim)
        if cv.waitKey(0) == ord("q"):
            cv.destroyAllWindows()
            break
        #plt.imshow(output)

