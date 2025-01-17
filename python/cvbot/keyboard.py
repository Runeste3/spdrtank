from pynput.keyboard import Controller, Key, Listener
from time import sleep


kbd = Controller()
lstnr = None
kfd = {}


def add_kf(key, func):
    """
    str, func -> None
    Add key:function pair to 'kfd' for main
    listener callback
    """
    global kfd
    key = _parse_key(key, listen=True)
    kfd[key] = func

def listener():
    """
    None -> None
    Initiate a callback function when a key is pressed
    """
    global lstnr

    if not (lstnr is None):
        lstnr.stop()

    def aux(k):
        for key in kfd.keys():
            if str(k) == str(key):
                kfd[key]()

    lstnr = Listener(aux)
    lstnr.start()

def ktype(wrd):
    """
    str -> None
    Press all characters in 'wrd' on the keyboard
    """
    kbd.type(wrd)

def _parse_key(key, listen=False):
    """
    str -> pynput.Key
    Find the key that represents given key as string in pynput keys
    """
    if len(key) == 0:
        return ""
    elif len(key) == 1:
        if listen:
            return "'{}'".format(key)
        else:
            return key
    else:
        key = key.lower()

        return eval("Key.{}".format(key))

def combo_press(keys, hold=0.1):
    """
    str -> None
    Press multiple keys at the same type
    """
    keys = [_parse_key(k) for k in keys.split("+")]

    for k in keys:
        if k == "":
            continue
        kbd.press(k)

    sleep(hold)

    for k in keys:
        if k == "":
            continue
        kbd.release(k)
            
def press(key):
    """
    str -> None
    Press the given key as a str variable
    """
    key = _parse_key(key)
    kbd.tap(key)

def hold_for(key, tm):
    """
    str, float -> None
    Press and hold a key for duration 'tm'
    """
    key = _parse_key(key)
    kbd.press(key)
    sleep(tm)
    kbd.release(key)

def del_all():
    """
    None -> None
    Press "Ctrl-A" followed by "Delete"
    Used to remove all the characters in a text box
    """
    kbd.press(Key.ctrl_l)
    kbd.press("a")
    sleep(0.1)
    kbd.release(Key.ctrl_l)
    kbd.release("a")

    sleep(0.1)
    kbd.press(Key.backspace)

def altab():
    """
    None -> None
    Press "Alt-Tab"
    """
    kbd.press(Key.alt_l)
    kbd.press(Key.tab)
    sleep(0.1)
    kbd.release(Key.alt_l)
    kbd.release(Key.tab)
