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
game_mode = None
supported_maps = ("SAHA", "SHSH", "DRCA")#, "JUTE")
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
    global rltv_sz, rltv_szu, fullscreen
    fullscreen = wsz[0] == MON['width']
    if not fullscreen:
        w, h = (wsz[0] - 16), (wsz[1] - 39)
    else:
        w, h = wsz
    rltv_sz = w * h
    rltv_szu = w, h

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

LVIM = read_img("src/leave/btn.png", "grey")
LVIM_GR = read_img("src/leave/btngr.png", "grey")

def end_game(img):
    """
    Image -> rect | None
    Return a point of the leave button if found
    else return None
    """
    rect = find_n_compare(LVIM, img)
    if rect is None:
        return find_n_compare(LVIM_GR, img)
    else:
        return rect 

PLYIM = read_img("src/queue/plybtn.png",   "grey")
CMPIM = read_img("src/queue/cmptvbtn.png", "grey")
RCNIM = read_img("src/queue/recon.png",    "grey")
IDLIM = read_img("src/queue/idle.png",     "grey")
CNFIM = read_img("src/queue/okbtn.png",    "grey")
NOIM  = read_img("src/queue/nobtn.png",    "grey")
RTRIM = read_img("src/queue/rtrbtn.png",   "grey")
VCTIM = read_img("src/queue/vctbnr.png",   "grey")
DFTIM = read_img("src/queue/dftbnr.png",   "grey")

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
    return look_for(res, img, 0.9)

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
        lopos = [(o[0][0] + (o[0][2] - o[0][0]), 
                o[0][1] + (o[0][3] - o[0][1])) for o in out]
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
]

def gms_to_mode(gms):
    """
    str -> int[1-5] | None
    Identify mode based on guiding sentence
    """
    for sgms, md in GMSR:
        smr = fuzz.ratio(sgms, gms)
        if smr > 90:
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
        save_img(img, "__test/_pathing/temple/{}.png".format(i))
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

    for ((x1, y1, x2, y2), _, name) in loob:
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

    for ((x1, y1, x2, y2), _, name) in loob:
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

    for (mask, _, name) in loob:
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

    for ((x1, y1, x2, y2), _, name) in loob:
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
    for ((x1, y1, x2, y2), _, name) in loob:
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

    for (mask, _, _) in loob:
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
    ny2 =  round(h * (0.05 if fullscreen else 0.1))
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
    for mn in LOMN:
        smr = fuzz.ratio(mn, gmn)
        if smr > 90:
            mnp = mn.split(" ")
            if len(mnp) >= 2 and (len(mnp[0]) > 2 and len(mnp[1])):
                return "{}{}".format(mnp[0][:2], mnp[-1][:2])

    return cur_map 

def detect_mode_map(img):
    """
    Image -> str
    Recognize the map in the given image
    and return its string code or return empty
    string if not known
    """
    mnim = process_mn(img) 
    result = read(mnim, allowlist="'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '").upper().replace("\n", "")
    logger.info("Map Mode OCR parsing: {}".format(result))

    if " ON " in result:
        mds, mps = result.split(" ON ")
    elif " OM " in result:
        mds, mps = result.split(" OM ")
    elif " O " in result:
        mds, mps = result.split(" O ")
    elif " ON" in result:
        mds, mps = result.split(" ON")
    else:
        mds, mps = "", ""
    
    if mds != "" and mps != "":
        global cur_map, game_mode
        prsd_map = mn_to_mp(mps)
        prsd_mode = gms_to_mode(mds)
        game_mode = prsd_mode
        if cur_map != prsd_map:
            print("\n Loading map model... \n")
            load_map_model(prsd_map)
        if game_mode == 3:
            print("\n Loading chicks model... \n")
            load_chick_model()
        cur_map = prsd_map
    else:
        prsd_mode = None
    
    return prsd_mode

def load_chick_model():
    """
    None -> None
    Load the chick model to the global variable
    """
    global chk_model
    if chk_model is None:
        chk_model = Model("src/models/chick.onnx", ["chick",])

def load_map_model(pm):
    """
    Map Code -> None
    Load the map objects detection model
    for the current map
    """
    global map_model

    if pm == "DECA":
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
    elif pm == "SAHA":
        map_model = Model("src/models/haven.pt")#, "mask", ['p1', 'p2', 'p3',
                                                #            'p4', 'w1', 'w2',
                                                #            'w3', 'w4', 'w5',
                                                #            'w6', 'w7', 'w8',
                                                #            'w9'])
    elif pm == "SHSH":
        map_model = Model("src/models/shrine.pt")
    elif pm == "DRCA":
        map_model = Model("src/models/dragon.pt", ['p2', 'w1', 'w2', 'p1'])
    #elif pm == "JUTE":
    #    map_model = Model("src/models/temple.pt")

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
    Point, bi np im -> Point
    Return the nearest point to 'p'
    that serves as a 'good point' to
    start from or head to
    'good point' means black pixel
    surrounded by all black pixels
    """
    if vmap[p[1], p[0]] == 0:
        return p

    fctr = 2

    while fctr < CSW:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue

                tp = p[0] + (fctr * x), p[1] + (fctr * y)

                if (tp[0] < 0) or (tp[1] < 0) or (tp[0] >= CSW) or (tp[1] >= CSH):
                    continue

                if vmap[tp[1], tp[0]] == 0:
                    return tp
        
        fctr += 2

    return p

def heap_best_path(vmap, sp, ep):
    """
    bi np im, Point, Point -> list(Point)
    Determine the shortest path from 'sp' to 'ep'
    based on the given map 'vmap'
    """
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
                    tp[0] >= CSW or tp[1] >= CSH or
                    (x, y) == (0, 0)):
                    continue
                if tp == ep:
                    return build_lop(ryx, [ep, tp, cp])
                elif not visited[tp[1], tp[0]] and not iswall[tp[1], tp[0]]:
                    visited[tp[1], tp[0]] = True
                    wly  = tp[1] - 5 if tp[1] >= 5 else 0
                    wlx  = tp[0] - 5 if tp[0] >= 5 else 0
                    wts  = iswall[wly:tp[1]+5, wlx:tp[0]+5].sum() / 5
                    pted = dist(tp, ep)
                    cost = wts + pted
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



def best_path(vmap, sp, ep):
    """
    bi np im, Point, Point -> list(Point)
    Determine the shortest path from 'sp' to 'ep'
    based on the given map 'vmap'
    """
    # vmap is of size CSWxCSH, shape CSHxCSW
    sp, ep = nearest_b(sp, vmap), nearest_b(ep, vmap)
    ryx = np.zeros(vmap.shape[:2] + (2,), dtype=np.uint16)
    visited = np.zeros(vmap.shape[:2], dtype='bool')
    visited[:, :] = False
    visited[sp[1], sp[0]] = True
    visited[ep[1], ep[0]] = False
    cost_loc = {}
    cp = sp
    bpsf = cp
    bcsf = None
    # Testing
    #svmap = vmap.copy()
    
    while True:
        for x in range(-1, 2):
            for y in range(-1, 2):
                tp = cp[0] + x, cp[1] + y
                if (tp[0] < 0 or tp[1] < 0 or
                    tp[0] >= CSW or tp[1] >= CSH or
                    (x, y) == (0, 0)):
                    continue
                if tp == ep:
                    print("Successfully found a route!")
                    return build_lop(ryx, [ep, tp, cp])
                elif not visited[tp[1], tp[0]]:
                    visited[tp[1], tp[0]] = True

                    if vmap[tp[1], tp[0]] == 0:
                        cost = dist(tp, ep)# + dist(tp, sp))
                        ryx[tp[1], tp[0]] = (cp[1], cp[0])

                        if cost_loc.get(cost) is None:
                            cost_loc[cost] = [tp,]
                        else:
                            cost_loc[cost].append(tp)

        costs = cost_loc.keys()
        if len(costs) > 0:
            min_cost = min(costs)
            loc = cost_loc[min_cost][0]
            mloc = loc[1], loc[0]

            cp = mloc[1], mloc[0]
            if (bcsf is None) or (min_cost < bcsf):
                bcsf = min_cost
                bpsf = cp

            cost_loc[min_cost].remove(loc)
            if len(cost_loc[min_cost]) == 0:
                del cost_loc[min_cost]
            # Testing
            #svmap[mloc] = 255
            #cv.imshow("Pathing", cv.resize(svmap, (800, 600)))
            #cv.waitKey(1)
        else:
            if bpsf == sp:
                return [sp, ep]
            else:
                return build_lop(ryx, [bpsf,])

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

def path_to_dr(lop):
    """
    list(Point) -> Direction
    Based on given points decides
    the next direction to move toward
    to follow the path
    """
    tpi = choice((1, 3, 4, 5, 6, 7, 8, 9, 10, 20))
    mnp = tpi if len(lop) >= tpi else len(lop)

    sp = lop[0]
    if len(lop) <= 1:
        ep = lop[-1]
    elif dist(sp, lop[1]) < 5:
        ep = lop[mnp-1]
    else:
        ep = lop[1]

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

def direction(img, sp, ep):
    """
    Image, Point, Point -> (int[-1:1], int[-1:1])
    Determine the direction to go based on given list of objects,
    this returns least blocking direction toward ep from sp
    serving as a pathing solution
    Return one of the 8 possible move directions (left-right, up-down)
    """
    global loob, lpt

    loob = map_objs(img) 

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

if __name__ == "__main__":
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


    hwnd = find_window("Spider Tanks", exact=True)
    win = Window(hwnd)
    win.repos(0, 0)
    new_win(win.size)
    reg = 0, 0, win.size[0], win.size[1]
    #__record(reg)
    #quit()

    while True:
        img = get_region(reg)
        img = process_mn(img)
        result = read(img, allowlist="'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '").upper().replace("\n", "")
        print(result)
        cv.imshow("test", cv.resize(img.img, (1066, 600)))
        cv.waitKey(1)
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

