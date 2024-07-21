from math import dist
from time import sleep, time
from cvbot.capture import get_region
from cvbot.keyboard import listener, kbd, press, add_kf
from cvbot.mouse import click, ms, msbtn, move, scroll
from threading import Thread, Timer
from random import randrange, uniform, choice
from collections import deque
from datetime import datetime
from cvbot.windows import find_window, Window
from os import get_terminal_size as gts
from os import listdir
import detect
import json
import subprocess
import logging
# Testing
import traceback


# Logging
logger = logging.getLogger(__name__)

DQSZ = 20
strttm = time()
rntm   = 0
tmtoran = 3600 * 8 # 8 Hours
bad_play = False
mv_far = False
boton = True
running = True
gmsg = ""
cmptv = True 
img = None
switcher = False
hp, energy = 0, 0 
xyd = [0, 0]
atkmode = 1
loals = []
loenm = []
loob = []
laht = None
atarget = None
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
    4:"King of the hill",
    5:"Hold the flag",
    6:"Kill Confirmed"
}

def st_mp():
    """
    None -> bool
    Return True if the current map is updated
    to recognition based map
    False otherwise
    """
    return detect.cur_map in detect.static_maps

def log(m):
    global gmsg
    logger.info(m)
    scnds = (time() - strttm) + rntm
    hrs,  rmscnds = divmod(scnds, 3600)
    mnts, rmscnds = divmod(rmscnds, 60)
    gmsg = "[{}] {}...".format("{:02}:{:02}:{:02}".format(round(hrs), 
                                                 round(mnts),
                                                 round(rmscnds)), 
                               m)
    # datetime.now().strftime("%M:%S")

def release_all():
    kbd.release("a")
    kbd.release("w")
    kbd.release("s")
    kbd.release("d")

def botoff():
    global boton
    if boton:
        boton = False
        print("\n")
        print("Exitting...")
        print("\n")
    else:
        print("\n")
        print("Waiting for bot loops to exit...")
        print("\n")

def switch_switcher():
    global switcher
    switcher = not switcher
    print("\n")
    print("Tank Switcher is {}".format("ON" if switcher else "OFF"))
    print("\n")
    save_conf(no_prompt=True, save_rt=False)

def reset():
    botoff()
    save_conf(no_prompt=True, save_rt=True)
    sleep(5)
    subprocess.call(['python\python', 'bot.py'], shell=True)

def win_trade():
    global bad_play
    bad_play = not bad_play
    print("\n")
    print("Trying to {} next match!".format("Win" if not bad_play else "Lose"))
    print("\n")

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

last_ppos = None
def selfloc(img):
    """
    Image -> Point
    Return the location of user tank
    """
    global last_ppos
    lopps = detect.player(img)
    npps = len(lopps)
    if npps == 0:
        if last_ppos is None:
            return CENTER
        else:
            return last_ppos
    else:
        pps = nearest(slfloc, lopps)
        last_ppos = pps

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

def move_toward(pos, mp=False):
    """
    Point -> None
    Move toward a point on screen
    """
    global img, loob, slfloc
    #vect = pos[0] - slfloc[0], pos[1] - slfloc[1]
    #print("\n", "Move to:", pos, "\n")
    try:
        vect = detect.direction(img, selfloc(img), pos, mp=mp)
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
    if not (nobj is None):
        log("Moving toward {} at {}".format(target, nobj))
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
        log("Moving toward {} at {}".format(target, nobj))
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

def form_vector(lop, p, normalize=1):
    """
    List of Point, Point, int -> Vector
    Calculate the vector formed from all the vectors
    going from all points 'lop' to 'p'
    """
    vec = [0, 0]
    nop = len(lop)
    for lp in lop:
        vec[0] += p[0] - lp[0]
        vec[1] += p[1] - lp[1]

    vec[0] = round(vec[0] / (normalize * nop))
    vec[1] = round(vec[1] / (normalize * nop))

    return vec

def best_aoi(loenm):
    """
    List of Point -> Map Point
    Based on enemy locations and self location
    determine where in the map is considered a
    safe area of interest
    """
    vec = form_vector(loenm, slfloc, normalize=50)
    mp = vec[0] + detect.lpmp[0], vec[1] + detect.lpmp[1]
    return detect.aoi_closest_to(mp)
    
def next_aoi():
    """
    None -> None
    Return the next best aoi to visit
    """
    return detect.aoi_random()

def roam():
    """
    None -> None
    Move to an area of interest
    and away from enemies
    """
    global loenm, slfloc, atkmode

    if atkmode == 1 and loenm and sort_near(slfloc, loenm)[0][1] < 400:
        mp = best_aoi(loenm)
    else:
        mp = next_aoi()

    log("Roaming to {}".format(mp))
    move_toward(mp, mp=True)

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
                log("Moving to arrows {} {}".format(len(loai), naly))
                move_toward(naly)
            elif loenm:
                if st_mp():
                    roam()
                else:
                    random_move()
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
                log("Moving to arrows {} {}".format(len(loai), naly))
                move_toward(naly)
            elif st_mp():
                roam()
            else:
                random_move()

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
            log("Moving to arrows: {} {}".format(len(loai), naly))
            move_toward(naly)
        elif loenm:
            if st_mp():
                roam()
            else:
                random_move()

def pltry_move():
    """
    None -> None
    Do movement in poultry pusher mode
    """
    global img, slfloc

    cpos = detect.locate_chicken(img)
    if cpos is None:
        if st_mp():
            roam()
        else:
            basic_move()
    else:
        if dist(cpos, slfloc) > detect.recal(100):
            log("Moving to poultry: {}".format(cpos))
            move_toward(cpos)
        else:
            release_all()

def chkchs_move():
    """
    None -> None
    Do movement for chicken chaser mode
    """
    global img, slfloc

    cm = detect.cur_map

    stmp = st_mp()

    if stmp:
        lobrl = detect.static_objs[cm]['barrels']
    else:
        lobrl = detect.barrels(img)

    if (lobrl and not stmp) or (stmp and detect.chck_collected(img)):
        if stmp and not (detect.lpmp is None):
            lobrl = list(sorted(lobrl, key=lambda brl: dist(brl, detect.lpmp)))
        else:
            lobrl.sort(key=lambda brl: dist(brl, slfloc))

        brl = lobrl[0]
        if not stmp:
            brl = brl[0], int(brl[1] + detect.recal(70))

        if stmp:
            log("Unloading chicks: {} {}".format(len(lobrl), brl))
            detect.update_ploc(img, slfloc)
            if dist(brl, detect.lpmp) < 3:
                if uniform(0.0, 1.0) > 0.7:
                    move_toward(brl, mp=True)
                else:
                    release_all()
            else:
                move_toward(brl, mp=True)
        elif dist(brl, slfloc) > (detect.recal(100)):
            log("Unloading chicks: {} {}".format(len(lobrl), brl))
            move_toward(brl)
        else:
            release_all()
    else:
        lochk = detect.locate_chicks(img)
        if lochk:
            lochk.sort(key=lambda chk: dist(chk, slfloc))
            chk = lochk[0]
            log("Collecting chicks: {} {}".format(len(lochk), chk))
            move_toward(chk)
        elif st_mp():
            roam()
        else:
            basic_move()

def flag_move():
    """
    None -> None
    Do movement for hold the flag mode
    """
    global img, slfloc, atarget

    res = detect.flag_loc_tp(img)

    if not (res is None):
        ftp, floc, ploc = res

        if ftp == "enemy":
            sfd = dist(slfloc, floc)
            atarget = ploc, sfd
            
            if sfd > 400:
                log("Moving toward flag: {}".format(floc))
                move_toward(floc)
            elif atkmode == 0:
                log("Moving toward flag: {}".format(floc))
                move_toward(ploc)
            else:
                random_move()
        elif ftp == "self":
            basic_move()
        elif ftp == "ally":
            sfd = dist(slfloc, floc)
            if sfd < 800:
                log("Moving toward flag: {}".format(floc))
                move_toward(ploc)
            else:
                log("Moving toward flag: {}".format(floc))
                move_toward(floc)
        else:
            log("Moving toward flag: {}".format(floc))
            move_toward(floc)
    elif st_mp():
        roam()
    else:
        basic_move()

def koh_move():
    """
    None -> None
    Do movement for king of the hill mode
    """
    global img, slfloc

    cm = detect.cur_map
    if st_mp():
        hill = choice(detect.static_objs[cm]['hill'])
        #if not (detect.lpmp is None) and dist(detect.lpmp, hill) < 3:
        #    hill = detect.lpmp
    else:
        hill = detect.hill(img)
        #if not (hill is None) and dist(slfloc, hill) < 150:
        #    return

    if not (hill is None):
        log("Moving to hill: {}".format(hill))
        move_toward(hill, mp=True)
    else:
        basic_move()

def moving():
    """
    None -> None
    Move to the nearest ally if found
    """
    global img, boton, slfloc, atkmode, loenm, loals
    global running, gmode, bad_play

    while boton:
        if img is None:
            sleep(1)
            continue
        ntrlsd = False
        while running and boton:
            ntrlsd = True
            sleep(0.3)
            if bad_play:
                if st_mp():
                    roam()
                else:
                    basic_move()
            elif gmode == 2:
                pltry_move()
            elif gmode == 3:
                chkchs_move()
            elif gmode == 4:
                koh_move()
            elif gmode == 5:
                flag_move()
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

def release_mouse():
    """
    None -> None
    Release mouse click/control
    this function is mainly used to stop attacking
    """
    global laht
    ms.release(msbtn.left)
    laht = None

def attacking():
    """
    None -> None
    Shoot enemies
    """
    global img, boton, slfloc, loals, loenm, atkmode
    global running, hp, laht, xyd, atarget, bad_play

    lmmvt = time()
    wt = 5

    while boton:
        while running and boton:
            sleep(0.1)
            if img is None or (slfloc == CENTER and 
                               len(loenm) == 0 and
                               hp < 1):
                continue
            if bad_play:
                chnc = uniform(0.0, 1.0)
                if chnc > 0.99 and loenm:
                    click(loenm[0])
                    sleep(1)
            elif loenm:
                if atkmode == 1:
                    sloenm = sort_near(ms.position, loenm, dsdfp=slfloc)
                else:
                    sloenm = sort_near(slfloc, loenm)

                if not (atarget is None):
                    sloenm.insert(0, atarget)
                    atarget = None

                wt = detect.weapon.type

                for nenm, dste in sloenm:
                    dste = detect.recal(dste, reverse=True)
                    if (dste < detect.Weapon.TPTR[wt] and 
                        (atkmode == 0 or detect.open_fire_line(slfloc, nenm))):
                        attack(nenm, dste, wt)
                        break
                else:
                    release_mouse()
            else:
                release_mouse()
                if (time() - lmmvt) > wt:
                    rstloc = (slfloc[0] + (randrange(50, CENTER[0] // 2) * xyd[0]),
                            slfloc[1] + (randrange(50, CENTER[1] // 2) * xyd[1]))

                    if rstloc[1] > (CENTER[1] * 1.5):
                        continue

                    move(rstloc)
                    lmmvt = time()
                    wt = randrange(1, 5)
        sleep(1)

def attack(nenm, dste, wt):
    """
    Point, int, Weapon Type -> None
    Attack the enemy at position 'nenm' on screen
    and distance 'dste' away from character
    """
    global atkmode, laht, hp

    logger.info("Attacking {} of distance {} away".format(nenm, dste))

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
            release_mouse()

def hpen():
    """
    None -> None
    Manage HP & Energy
    """
    global boton, hp, energy, HPTHS, ENTHS
    global running

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

def queuer():
    """
    None -> None
    Join competitive queue automatically
    """
    global boton, running, cmptv, img, hp, switcher

    cmp_prsd = False
    in_game = False

    while boton:
        while running and boton:
            if cmptv and cmp_prsd and hp > 0:
                in_game = True
                cmp_prsd = False
                logger.info("In game")

            if not (img is None):
                pbtn = detect.play_btn(img)
                if not (pbtn is None):
                    if switcher and bad_play and game_played:
                        switch_contract()
                    if cmptv:
                        click(pbtn)
                        sleep(0.5)
                        move(CENTER)
                        sleep(0.5)
                else:
                    cbtn = detect.cmptv_btn(img)
                    if not (cbtn is None):
                        if switcher and bad_play and game_played:
                            click_back()
                            sleep(5)
                            switch_contract()
                        if cmptv and in_game:
                            running = False
                            for _ in range(5):
                                if boton:
                                    sleep(1)
                            print("\n")
                            for i in range(30):
                                print("Resetting after {} seconds...".format(50 - i), end="\r")
                                if boton:
                                    sleep(1)
                                else:
                                    return
                            print("\nRESETTING\n")
                            reset()
                            logger.info("Resetting")
                            return


                        if cmptv:
                            click(cbtn)
                            sleep(0.5)
                            move(CENTER)
                            sleep(0.5)
                            cmp_prsd = True
            for _ in range(6):
                if boton:
                    sleep(1) 
        sleep(1)

def button_handler():
    """
    None -> None
    Handle buttons clicking 
    """
    global img, boton, running, cmptv

    def leave_btn():
        pos = detect.end_game(img)
        lvfnd = not (pos is None)
        if lvfnd:
            logger.info("Leave button detected: {}".format(pos))
            click((pos[0] + 80, pos[1] + 25), hover=0.3)
        return lvfnd

    def recon_dial():
        rcndg = detect.reconnect_dialog(img)
        rcnfnd = not (rcndg is None)
        if rcnfnd:
            logger.info("Clicking reconnect button {}".format(rcndg))
            x, y, w, h = rcndg
            ysbtn = x + (w // 3), y + h - (h // 5)
            click(ysbtn)
        return rcnfnd

    def ok_btn():
        okbtn = detect.confirm_dialog(img)
        okfnd = not (okbtn is None)
        if okfnd:
            logger.info("Clicking ok button {}".format(okbtn))
            click(okbtn)
        return okfnd

    def rtr_btn():
        rtrbtn = detect.retry_button(img)
        rtrfnd = not (rtrbtn is None)
        if rtrfnd:
            logger.info("Clicking retry button {}".format(rtrbtn))
            click(rtrbtn)
        return rtrfnd

    rec_ans, ok_ans, rtr_ans, lv_ans = (False,) * 4

    while boton:
        while running and boton:
            if img is None:
                sleep(1)
                continue

            rec_ans = recon_dial()
            if not rec_ans:
                ok_ans = ok_btn()
                if not ok_ans:
                    rtr_ans = rtr_btn()
                    if not rtr_ans:
                        lv_ans = leave_btn()

            if rec_ans or ok_ans or rtr_ans or lv_ans:
                sleep(0.5)
                move((500, 100))

            for _ in range(5):
                if boton:
                    sleep(1)
        sleep(1)

def bad_play_switch(on):
    """
    bool -> None
    Turn 'bad_play' global variable
    on or off (True of False)
    and also write to the local config file
    the bad_play status
    """
    global bad_play
    bad_play = on
    print("\n Trying to {} next match! \n".format("Win" if not bad_play else "Lose"))
    save_conf(no_prompt=True)

def inspector():
    """
    None -> None
    Get information about the current
    game mode, map and result
    """
    global img, gmode, hp

    lgrt = None

    def detect_result():
        nonlocal lgrt
        global game_played
        vdn = detect.victory_defeat(img)
        if vdn == detect.VICTORY:
            log("Victory detected!")
            bad_play_switch(True)
            game_played = True
            lgrt = time()
        elif vdn == detect.DEFEAT:
            log("Defeat detected!")
            bad_play_switch(False)
            game_played = True
            lgrt = time()

    while boton:
        while running and boton:
            if img is None:
                sleep(1)
                continue

            gmr   = detect.detect_mode_map(img)
            logger.info("Detected game mode/map {} {}".format(gmr, detect.cur_map))
            if not (gmr is None):
                gmode = gmr
            
            if hp == 0:
                for _ in range(5):
                    if boton:
                        if lgrt is None or ((time() - lgrt) > 210):
                            detect_result()
                        sleep(1)
            else:
                for _ in range(30):
                    if boton:
                        if lgrt is None or ((time() - lgrt) > 210):
                            detect_result()
                        sleep(1)
        sleep(1)

def open_garage():
    """
    None -> bool
    Return True if the garage was opened
    False if not
    """
    for _ in range(10):
        if not detect.in_garage(img):
            gbtn = detect.garage_btn(img)
            if gbtn is None:
                sleep(1)
            else:
                click(gbtn)
                sleep(0.5)
                move(CENTER)
                sleep(0.5)
        else:
            return True
    return False

def open_contracts():
    """
    None -> bool
    Return True if the contacts menu was opened
    False if not
    """
    for _ in range(5):
        pchk = detect.pilot_uncheck(img)
        if not (pchk is None):
            click(pchk)
            sleep(0.5)
            move(CENTER)
        else:
            break
        sleep(1)
    for _ in range(5):
        cbtn = detect.contracts_btn(img)
        if not (cbtn is None):
            click(cbtn)
            sleep(0.5)
            move(CENTER)
            locp = detect.contracts(img)
            if locp:
                return True
        else:
            break
        sleep(1)
    return False

def click_back():
    """
    None -> None
    Click the back button if it's visible
    """
    for _ in range(3):
        pos = detect.back_btn(img)
        if not (pos is None):
            click(pos)
            sleep(0.5)
            move(CENTER)
            sleep(0.5)

def exit_garage():
    """
    None -> None
    Exit the garage menu
    """
    global img

    while detect.in_garage(img):
        click((detect.rltv_szu[0]//3, 60))
        sleep(0.5)
        bbtn = detect.back_btn(img)
        if not (bbtn is None):
            click(bbtn)
            sleep(1)

def switch_cont_random():
    """
    None -> None
    ASSUMES CONTRACTS MENU IS OPEN
    Switch to a random non-selected contract
    """
    global img

    pos = detect.contracts_btn(img)
    rcbtn = pos[0], pos[1] + detect.recal(120, ogsz=1600, wonly=True)
    click(rcbtn)
    sleep(0.5)
    # ----- Random contract Selection ----------------------------------
    #
    #locp = detect.contracts(img)
    #if locp:
    #    if len(locp) >= 3:
    #        # random scroll
    #        move(CENTER)
    #        sleep(0.5)
    #        scramt = choice((5, 10, 15, 20, 25, 30, 35, 40, 45, 50))
    #        updn   = choice((True, False))
    #        print(scramt, updn)
    #        for _ in range(scramt):
    #            scroll(100, updn, True)
    #            sleep(0.1)
    #        sleep(1)

    #    # random select of the availabl contracts
    #    locp = detect.contracts(img)
    #    for _ in range(5):
    #        cp = choice(locp)
    #        if select_contract(cp):
    #            return
    #------------------------------------------------------------------

def switch_cont_top():
    """
    None -> None
    ASSUMES CONTRACTS MENU IS OPEN
    Switch to the top-most non-selected contract
    """
    global img

    move(CENTER)
    sleep(0.5)
    for _ in range(100):
        scroll(100, False, True)
        sleep(0.1)
    sleep(0.5)
    locp = detect.contracts(img)
    if locp:
        for cp in locp:
            if select_contract(cp):
                return

def select_contract(cp):
    """
    Point -> bool
    Return True if the contract with given position 'cp'
    is currently select, False otherwise
    """
    global img

    if not detect.contract_slctd(img, cp):
        for _ in range(5):
            click(cp)
            sleep(0.5)
            move((0, 0))
            sleep(0.5)
            if len(detect.contracts(img)) == 0:
                return True

    return False

game_played = False

def switch_contract():
    """
    None -> None
    Switch the current tank contract either randomly
    or to the top-most contact
    """
    global bad_play, img, rntm, tmtoran, game_played
    global strttm

    if bad_play and game_played:
        log("Switcher engaged, trying to open garage!")
        if open_garage():
            log("Garage opened, opening contracts!")
            if open_contracts():
                noc = len(detect.contracts(img))
                log("Contracts visible, {} tanks".format(noc))
                if noc <= 0:
                    pass
                elif ((time() - strttm) + rntm) > tmtoran:
                    # Switch Randomly
                    switch_cont_random()
                else:
                    # Switch to top-most
                    switch_cont_top()
                game_played = False
            exit_garage()
        log("Exitting switcher!")

def printer():
    """
    None -> None
    Print the current bot actions
    """
    global gmsg, hp, energy, gmode

    tl = gts()[0] 

    while boton:
        while running and boton:
            if not (img is None):
                wnm = detect.Weapon.tp_to_name(detect.weapon.type)
                gmds = gmode_to_smode[gmode]
                gmps = gmp_to_gmn[detect.cur_map]

                msg = f"{gmsg} | HP: {hp} | ENR: {energy} | {wnm} | {gmds} | {gmps}"

                if len(msg) > tl:
                    msg = msg[:(tl-3)] + "..."
                else:
                    msg = f"{msg:{tl}}"

                print(msg, end="\r")

            sleep(0.5)
        sleep(1)

def run():
    """
    None -> None
    Main bot loop
    """
    global boton, slfloc, img, loals, loenm, hp, energy
    global running, atkmode, gmode, gmode_to_smode, gmsg

    logger.info("Starting threads")

    # ------------------- BOT LOOPS ------------------------------------
    # (2) Attacking/Mouse
    Thread(target=attacking).start()
    # (3) Movement/Keyboard
    Thread(target=moving).start()
    # (4) Using skills
    Thread(target=hpen).start()
    # (5) Join competetive queue 
    Thread(target=queuer).start()
    # (6) Click on game UI buttons
    Thread(target=button_handler).start()
    # (7) Get pre/post-game information(e.g. game result, mode, map)
    Thread(target=inspector).start()
    # (8) Print the current bot status
    Thread(target=printer).start()
    # 
    logger.info("Starting main loop")
    logger.info("Game region: {}".format(GMREG))
    #
    # (1) Main loop, getting images of the game, parse HP, energy,
    #     find enemies, allies and player location.
    # ------------------------------------------------------------------
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
                                                       detect.Weapon.SANTA,
                                                       detect.Weapon.SHOTGUN)
                           else 1)
        sleep(1)

def read_conf():
    global HPTHS, ENTHS, mv_far, bad_play, rntm, tmtoran 
    global switcher

    try:
        with open('config.json', 'r') as f:
            conf = json.load(f)

        HPTHS    = int(conf['hpt'])
        ENTHS    = int(conf['ent'])
        mv_far   = conf['mf'] == "1"
        bad_play = conf['bp'] == "1"
        rntm     = float(conf['rntm'])
        tmtoran  = int(conf['tmtoran'])
        switcher = conf['swtch'] == "1"
    except:
        print("COULDN'T FIND THE BOT PREFERENCES, PLEASE SET IT AGAIN")
        save_conf()
        read_conf()

def save_conf(no_prompt=False, save_rt=False):
    global mv_far, HPTHS, ENTHS, running, bad_play, rntm
    global tmtoran, switcher, strttm

    if not no_prompt:
        running = False

        sleep(0.5)

        print("\n")
        HPTHS, ENTHS, tmtoran = "", "", ""

        while not (HPTHS.isnumeric() and ENTHS.isnumeric()):
            HPTHS, ENTHS, tmtoran = (input("HP threshold(0-100): "), 
                                     input("Energy threshold(0-100): "),
                                     input("Hours to select random contract: "))
        HPTHS, ENTHS, tmtoran = int(HPTHS), int(ENTHS), int(float(tmtoran) * 3600)
        mv_far   = input("Follow the closest teammate? Y/N: ").lower() == "n"
        switcher = input("Switch tank/contracts? Y/N: ").lower() == "y"

        print("\n")

        running = True

    conf = {}
    conf['hpt']     = str(HPTHS)
    conf['ent']     = str(ENTHS)
    conf['tmtoran'] = str(tmtoran)
    conf['mf']      = "1" if mv_far else "0"
    conf['bp']      = "1" if bad_play else "0"
    conf['rntm']    = "0" if not save_rt else str((time() - strttm) + rntm)
    conf['swtch']   = "1" if switcher else "0"

    with open('config.json', 'w') as f:
        json.dump(conf, f)

def init():
    global GMREG, CENTER, slfloc, gmode, HPTHS, ENTHS
    global hwnd, win, game_played, rntm

    logfn = len(listdir("logs"))
    logging.basicConfig(filename="logs/prog_{}.log".format(logfn+1), 
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)
    hwnd = find_window("Spider Tanks", exact=True)
    win = Window(hwnd)
    detect.new_win(win.size)
    win.repos(0, 0)
    GMREG  = (
        0, 0, 
        win.size[0] if win.size[0] < detect.MON['width']  else detect.MON['width'], 
        win.size[1] if win.size[1] < detect.MON['height'] else detect.MON['height']
    )
    CENTER = (GMREG[2] // 2,
              GMREG[3] // 2)
    slfloc = CENTER
    gmode = 1

    logger.info("Starting bot")
    print("\n Initiating OCR... \n")
    detect.init_nr()

    read_conf()
    game_played = rntm != 0
    print("\n Trying to {} next match! \n".format("Win" if not bad_play else "Lose"))

    if 100 > HPTHS > 0 and 100 > ENTHS > 0:
        # -------- Shortcut Listeners -----------------
        # Turn the bot off
        add_kf("q", botoff)
        # Pause/Un-pause the bot
        add_kf("p", pause_run)
        # Turn automatic competitive queue joining on/off
        add_kf("m", sr_cmptv)
        # Edit user preferences
        add_kf("y", save_conf)
        # Switch win trade target for the next game win/lose
        add_kf("x", win_trade)
        # Switch tank switcher on or off
        add_kf("l", switch_switcher)

        listener()

        print("")

        run()

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
    #hwnd = find_window("Spider Tanks", exact=True)
    #win = Window(hwnd)
    #win.repos(0, 0)
    #detect.new_win(win.size)
    #reg = 0, 0, win.size[0], win.size[1]
    #GMREG  = (
    #    0, 0, 
    #    win.size[0] if win.size[0] < detect.MON['width']  else detect.MON['width'], 
    #    win.size[1] if win.size[1] < detect.MON['height'] else detect.MON['height']
    #)
    #CENTER = (GMREG[2] // 2,
    #          GMREG[3] // 2)
    #slfloc = CENTER
    #bad_play = True
    #game_played = True
    #img = get_region(GMREG)
    #rntm = 0
    #tmtoran = 1000000
    #Thread(target=switch_contract).start()
    #while True:
    #    img = get_region(GMREG)
    #    sleep(0.3)
    init()
