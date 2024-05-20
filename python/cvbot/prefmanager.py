from os.path import join
from configparser import ConfigParser


FOLDER = ""

def read(name):
    """
    str, str -> dict(str:any)
    Read the 'name'.init file
    insdie 'FOLDER'
    """
    cp = ConfigParser()
    cp.read(join(FOLDER, name) + ".init")

    opts = dict(cp['pref'])

    for key, value in opts.items():
        if value in ("True", "False"):
            opts[key] = True if value == "True" else False
        elif value.isnumeric():
            opts[key] = int(value)

    return opts

def save(name, dct):
    """
    str, str, dict(str:any) -> None
    Write given dictionary to 'name'.init file
    in the folder 'folder'
    """
    cp = ConfigParser()
    cp.read_dict({"pref":dct})

    with open(join(FOLDER, name) + ".init", 'w') as f:
        cp.write(f)    
