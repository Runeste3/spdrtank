from math import dist
from socket import MsgFlag
from time import sleep, time
from cvbot.capture import get_region
from cvbot.keyboard import listener, combo_press, kbd, press
from cvbot.mouse import click, ms, msbtn, move
from threading import Thread, Timer
from random import randrange, uniform
from collections import deque
from datetime import datetime
import detect
# Testing
import cv2 as cv
import traceback


DQSZ = 20
mv_far = False
boton = True
running = True
gmsg = ""
cmptv = False
img = None
hp, energy = 0, 0 
xyd = [0, 0]
atkmode = 1
loals = []
loenm = []
loob = []
laht = None
gmp_to_gmn = {
    "":"Unknown",
    "SAHA":"Safe Haven",
    "JUTE":"Jungle Temple",
    "DRCA":"Dragon Cave",
    "ARBA":"Arctic Base",
    "DECA":"Death Canyon",
    "FRRE":"Frozen Research",
    "SHSH":"Shrouded Shrine",
    "VAVA":"Vault of Value",
    "STST":"Steaming Stronghold",
    "FOCI":"Forsaken City"
}

gmode_to_smode = {
    1:"Team Deathmatch",
    2:"Poultry Pusher",
    3:"Chicken Chaser",
    4:"King of the hill"
}

def log(m):
    global gmsg
    gmsg = "[{}] {}...".format(datetime.now().strftime("%M:%S"), 
                               m)

def release_all():
    kbd.release("a")
    kbd.release("w")
    kbd.release("s")
    kbd.release("d")

def botoff():
    global boton
    boton = False

def pause_run():
    global running
    running = not running
    detect.weapon.type = detect.Weapon.NORMAL
    print("")
    print("Bot {}!".format("Paused" if not running else "Started"))
    print("")

def sr_cmptv():
    global cmptv
    cmptv = not cmptv 
    print("")
    print("Auto-Queue {}!".format("Off" if not cmptv else "On"))
    print("")

def selfloc(img):
    """
    Image -> Point
    Return the location of user tank
    """
    lopps = detect.player(img)
    npps = len(lopps)
    if npps == 0:
        return CENTER
    else:
        pps = nearest(slfloc, lopps)

    return pps 

def vec_to_move(vec):
    """
    Vector -> None
    Use keyboard to move like the given vector
    """
    global xyd

    xm, ym = vec

    if xm < 0:
        kbd.release("d")
        kbd.press("a")
        xyd[0] = -1
    elif xm == 0:
        kbd.release("d")
        kbd.release("a")
        xyd[0] = 0
    else:
        kbd.release("a")
        kbd.press("d")
        xyd[0] = 1

    if ym < 0:
        kbd.release("s")
        kbd.press("w")
        xyd[1] = -1
    elif ym == 0:
        kbd.release("w")
        kbd.release("s")
        xyd[1] = 0
    else:
        kbd.release("w")
        kbd.press("s")
        xyd[1] = 1

    #combo_press("{}+{}".format(xk, yk), 0.5)
    #print("Pressing {}+{}".format(xk, yk))


def nearest(loc, loloc):
    """
    Point, list(Point) -> Point
    Return the nearest point in 'loloc'
    to the 'loc'
    """
    if len(loloc) > 1:
        lod = [dist(loc, lc) for lc in loloc]
        ni = lod.index(min(lod))
        nloc = loloc[ni]
        return nloc
    else:
        return loloc[0]

def farthest(loc, loloc):
    """
    Point, list(Point) -> Point, None
    Return the farthest point in 'loloc'
    to the 'loc'
    """
    if len(loloc) > 1:
        lod = [dist(loc, lc) for lc in loloc]
        ni = lod.index(max(lod))
        nloc = loloc[ni]
        return nloc
    else:
        return loloc[0] if len(loloc) == 1 else None

def nearest_ally():
    """
    None -> Point
    Return the nearest ally
    """
    global slfloc, loals
    naly = nearest(slfloc, loals)
    return naly

def nearest_enm():
    """
    None -> Point
    Return the nearest enemy
    """
    global slfloc, loenm 
    nenm = nearest(slfloc, loenm)
    return nenm

def farthest_ally():
    """
    None -> Point, None
    Return the farthest ally
    """
    global slfloc, loals
    faly = farthest(slfloc, loals)
    return faly

def farthest_enm():
    """
    None -> Point
    Placeholder for future use
    Return the farthest enemy
    """
    raise Exception("Should not be called!")

def move_toward(pos):
    """
    Point -> None
    Move toward a point on screen
    """
    global img, loob, slfloc
    #vect = pos[0] - slfloc[0], pos[1] - slfloc[1]
    #print("\n", "Move to:", pos, "\n")
    try:
        vect = detect.direction(img, selfloc(img), pos)
        #cv.imshow("vmap", cv.resize(vmap, (800, 600)))
        #cv.waitKey(1)
    except Exception as e:
        print(traceback.format_exc())
        vect = (0, 0)

    vec_to_move(vect)

def mvto_nrst(target):
    """
    str -> Point 
    Move toward the nearest ally/enemy
    and Return nearest
    """
    global loals, loenm
    nobj = nearest_ally() if target == "ally" else nearest_enm()
    log("Moving toward {}".format(target))
    move_toward(nobj)
    return nobj

def mvto_frst(target):
    """
    str -> Point, None
    Move toward the farthest ally/enemy
    and Return farthest 
    """
    global loals, loenm
    nobj = farthest_ally() if target == "ally" else farthest_enm()
    if not (nobj is None):
        log("Moving toward {}".format(target))
        move_toward(nobj)
    return nobj

def stuck(lmpq):
    """
    Deque -> bool
    Return True if the bot is stuck
    """
    global DQSZ

    if len(lmpq) == DQSZ:
        avgds = 0
        for _ in range(0, DQSZ - 1):
            cpos = lmpq.popleft()
            ds = dist(cpos, slfloc)
            avgds += ds

        avgds /= (DQSZ - 1) 
        return avgds > 600

    return False

def random_move():
    """
    None -> None
    Do a random movement to avoid enemy attacks
    """
    log("Moving randomly")
    vec_to_move((randrange(-10, 10), randrange(-10, 10)))
    
def tdm_move():
    """
    None -> None
    Do movement in team deathmatch game
    """
    global img, slfloc, atkmode, loenm, loals

    #if stuck(lmpq):
    #    print("\nStuck moving away\n")
    #    lmp = lmpq.pop()
    #    vect = slfloc[0] - lmp[0], slfloc[1] - lmp[1]
    #    vec_to_move(vect)
    #    sleep(2)
    if atkmode == 1:
        if len(loals) > 0:
            if mv_far:
                mvto_frst("ally")
            else:
                mvto_nrst("ally")
        else:
            loai = detect.aly_inds(img)
            if loai:
                naly = loai[0]
                log("Moving toward ally arrow")
                move_toward(naly)
            elif loenm:
                random_move()
                sleep(uniform(0.3, 1.0))
    else:
        if loenm:
            mvto_nrst("enemy")
        elif loals:
            if mv_far:
                mvto_frst("ally")
            else:
                mvto_nrst("ally")
        else:
            loai = detect.aly_inds(img)
            if loai:
                naly = loai[0]
                log("Moving toward ally arrow")
                move_toward(naly)

def basic_move():
    """
    None -> None
    Do basic movement to allies
    """
    global loals, mv_far, loenm
    if loals:
        if not mv_far:
            mvto_nrst("ally")
        else:
            mvto_frst("ally")
    else:
        loai = detect.aly_inds(img)
        if loai:
            naly = loai[0]
            log("Moving toward ally arrow")
            move_toward(naly)
        elif loenm:
            random_move()

def pltry_move():
    """
    None -> None
    Do movement in poultry pusher mode
    """
    global img, slfloc

    cpos = detect.locate_chicken(img)
    if cpos is None:
        basic_move()
    else:
        if dist(cpos, slfloc) > detect.recal(100):
            log("Moving to the poultry")
            move_toward(cpos)
        else:
            release_all()

def chkchs_move():
    """
    None -> None
    Do movement for chicken chaser mode
    """
    global img, slfloc

    lobrl = detect.barrels(img)
    if lobrl:
        lobrl.sort(key=lambda brl: dist(brl, slfloc))
        brl = lobrl[0][0], int(lobrl[0][1] + detect.recal(70))
        if dist(brl, slfloc) > detect.recal(100):
            log("Unloading collected chicks")
            move_toward(brl)
        else:
            release_all()
    else:
        lochk = detect.locate_chicks(img)
        if lochk:
            lochk.sort(key=lambda chk: dist(chk, slfloc))
            chk = lochk[0]
            log("Collecting chicks")
            move_toward(chk)
        else:
            basic_move()

def koh_move():
    """
    None -> None
    Do movement for king of the hill mode
    """
    global img, slfloc

    hill = detect.hill(img)
    if not (hill is None):
        log("Moving to the hill")
        move_toward(hill)
    else:
        basic_move()

def moving():
    """
    None -> None
    Move to the nearest ally if found
    """
    global img, boton, slfloc, atkmode, loenm, loals
    global running, gmode

    while boton:
        if img is None:
            sleep(1)
            continue
        ntrlsd = False
        while running and boton:
            ntrlsd = True
            sleep(0.3)
            if gmode == 2:
                pltry_move()
            elif gmode == 3:
                chkchs_move()
            elif gmode == 4:
                koh_move()
            else:
                tdm_move()

        if ntrlsd:
            release_all()
        sleep(1)

def sort_near(p, lop, dsdfp=None):
    """
    Point, list(Point) -> list((Point, float))
    Return a sorted list based on distance
    from each point on the list 'lop'
    to the point 'p' and return a tuple
    with the distance as the second element
    """
    return list(sorted(map(lambda x: (x, dist(x, p if dsdfp is None else dsdfp)), lop), key=lambda x: x[1]))

def attacking():
    """
    None -> None
    Shoot the nearest enemy
    """
    global img, boton, slfloc, loals, loenm, atkmode
    global running, hp, laht

    lmmvt = time()
    lclkt = time()
    MXRNG = 1000

    while boton:
        while running and boton:
            sleep(0.1)
            if img is None or (slfloc == CENTER and 
                               len(loenm) == 0 and
                               hp < 1):
                continue
            if loenm:
                if atkmode == 1:
                    sloenm = sort_near(ms.position, loenm, dsdfp=slfloc)
                else:
                    sloenm = sort_near(slfloc, loenm)

                for nenm, dste in sloenm:
                    if dste < MXRNG and detect.open_fire_line(slfloc, nenm):
                        attack(nenm, dste)
                        break
                #print("Shooting at {}".format(nenm))
            else:
                ms.release(msbtn.left)
                laht = None
                if not (hp is None) and hp > 0:
                    rstloc = (slfloc[0] + (randrange(50, CENTER[0] // 2) * xyd[0]),
                              slfloc[1] + (randrange(50, CENTER[1] // 2) * xyd[1]))

                    if rstloc[1] > (CENTER[1] * 1.5):
                        continue

                    #if (time() - lclkt) > 10:
                    #    click(rstloc)
                    #    lclkt = time()
                    if (time() - lmmvt) > 1:
                        move(rstloc)
                        lmmvt = time()
        sleep(1)

def attack(nenm, dste):
    """
    Point, int -> None
    Attack the enemy at position 'nenm' on screen
    and distance 'dste' away from character
    """
    global atkmode, laht, hp

    wt = detect.weapon.type
    dste = detect.recal(dste, reverse=True)

    if dste > detect.Weapon.TPTR[wt]:
        return
    
    ht = detect.Weapon.hold_time(wt, dste)

    if ht < 0.5:
        click(nenm)
    else:
        move(nenm)
        if laht is None:
            if detect.weapon.has_rt():
                sleep(1)
            ms.press(msbtn.left)
            laht = time()
        elif (time() - laht) > ht:
            ms.release(msbtn.left)
            laht = None

def hpen():
    """
    None -> None
    Manage HP & Energy
    """
    global boton, hp, energy, HPTHS, ENTHS
    global running

    while hp is None or energy is None:
        pass

    while boton:
        while running and boton:
            sleep(0.1)
            if hp < 1:
                sleep(0.5)
                continue
            if hp < HPTHS:
                press("2")
                sleep(2)
            if energy > ENTHS: 
                press("3")
                sleep(0.5)
        sleep(1)

def leaver():
    """
    None -> None
    Leave the match when leave button is visible
    """
    global img, boton, running

    while boton:
        while running and boton:
            if img is None:
                sleep(1)
                continue
            pos = detect.end_game(img)
            if pos is None:
                pass
            else:
                click((pos[0] + 80, pos[1] + 25), hover=0.3)
            sleep(5)
        sleep(1)

def queuer():
    """
    None -> None
    Join competitive queue automatically
    """
    global boton, running, cmptv, img

    while boton:
        while running and boton and cmptv:
            pbtn = detect.play_btn(img)
            if not (pbtn is None):
                click(pbtn)
            else:
                cbtn = detect.cmptv_btn(img)
                if not (cbtn is None):
                    click(cbtn)
            sleep(6) 
        sleep(1)

def dialoger():
    """
    None -> None
    Handle dialogs
    """
    global img, boton, running, cmptv

    while boton:
        while running and boton:
            if img is None:
                sleep(1)
                continue
            rcndg = detect.reconnect_dialog(img)
            if not (rcndg is None):
                x, y, w, h = rcndg
                ysbtn = x + (w // 3), y + h - (h // 5)
                click(ysbtn)
            else:
                okdg = detect.confirm_dialog(img)
                if not (okdg is None):
                    x, y, w, h = okdg
                    okbtn = x + (w // 2), y + h - (h // 4)
                    click(okbtn)

            sleep(7)
        sleep(1)

def mode_detective():
    """
    None -> None
    Know the current mode in play
    and change global variable 'gmode'
    """
    global img, gmode

    while boton:
        while running and boton:
            if not (img is None):
                gmode = detect.game_mode(img)
                sleep(2)
                detect.recognize_map(img)
            sleep(3)
        sleep(1)

def run():
    """
    None -> None
    Main bot loop
    """
    global boton, slfloc, img, loals, loenm, hp, energy
    global running, atkmode, gmode, gmode_to_smode, gmsg

    Thread(target=attacking).start()
    Thread(target=moving).start()
    Thread(target=hpen).start()
    Thread(target=leaver).start()
    Thread(target=queuer).start()
    Thread(target=dialoger).start()
    Thread(target=mode_detective).start()

    while boton:
        while running and boton:
            img = get_region(GMREG)
            slfloc = selfloc(img)
            loals  = detect.allies(img)
            loenm  = detect.enemies(img)
            hp     = detect.hp(img)
            energy = detect.energy(img)
            if hp > 0:
                detect.weapon.update(img)
                atkmode = (0 if detect.weapon.type in (detect.Weapon.CARVER,
                                                       detect.Weapon.FLAME,
                                                       detect.Weapon.SANTA)
                        else 1)

            if running:
                print(gmsg,
                      "|",
                      "HP:", hp, "ENR:", energy, 
                      "|",
                      detect.Weapon.tp_to_name(detect.weapon.type),
                      "|",
                      gmode_to_smode[gmode],
                      "|",
                      gmp_to_gmn[detect.cur_map],
                      " " * 10, end="\r")
        sleep(1)

if __name__ == "__main__":
    # Testing
    #import cv2 as cv
    #from cvbot.io import read_img
    #from cvbot.images import Image

    #i = 0
    #def get_region(reg):
    #    global i
    #    img = read_img("test/{}.png".format(i))
    #    i += 1
    #    return img
    #-------------------
    from cvbot.windows import find_window, Window
    hwnd = find_window("Spider Tanks", exact=True)
    win = Window(hwnd)
    detect.new_win(win.size)
    win.repos(0, 0)
    GMREG  = 0, 0, win.size[0], win.size[1]
    CENTER = (GMREG[2] // 2,
            GMREG[3] // 2)
    slfloc = CENTER
    HPTHS, ENTHS = "", ""
    gmode = 1

    print("\n Initiating OCR... \n")
    detect.init_nr()

    while not (HPTHS.isnumeric() and ENTHS.isnumeric()):
        HPTHS, ENTHS = (input("HP threshold(0-100): "), 
                        input("Energy threshold(0-100): "))
    HPTHS, ENTHS = int(HPTHS), int(ENTHS)
    mv_far = input("Follow the closest teammate? Y/N: ").lower() == "n"

    if 100 > HPTHS > 0 and 100 > ENTHS > 0:
        listener("on/off", "q", botoff)
        listener("pause/run", "p", pause_run)
        listener("competitive", "m", sr_cmptv)

        print("")
        run()
