#from easyocr import Reader
import pytesseract
from cv2 import cvtColor, COLOR_BGR2RGB
from os import getcwd 
from os.path import join

pytesseract.pytesseract.tesseract_cmd = join(getcwd(), 'tesseract/tesseract.exe')
reader = None

def init_ocr(langs, model_path=None):
    """
    list(str) -> None
    Initiate easyocr NN model for future detection
    """
    global reader

    #if reader is None:
    #    reader = Reader(langs, gpu=True, verbose=False, 
    #                    model_storage_directory=model_path)
    #else:
    print("\nOCR model is already initiated!")

def read(img, allowlist="", join_wrd=True):
    """
    Image, str | None -> str | list(str)
    Use easyocr to read string in given image
    and return it
    """
    global reader

    res = pytesseract.image_to_string(cvtColor(img.img, COLOR_BGR2RGB),
                                      lang='eng',
                                      config='-c tessedit_char_whitelist=' + allowlist)
    return res
    #if reader is None:
    #    print("\nOCR model was not initiated, please call 'init_ocr' before calling 'read' function")
    #else:
    #    if join_wrd:
    #        return "".join(reader.readtext(img.img, detail=0, allowlist=allowlist))
    #    else:
    #        return reader.readtext(img.img, detail=0, allowlist=allowlist)
