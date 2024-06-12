import cv2
import numpy as np
import onnxruntime as ort
from cvbot.images import Image
from ultralytics import YOLO


classifiers = {}

class Model:
    def __init__(self, path, classes=[]) -> None:
        if path.endswith("onnx"):
            self.session = ort.InferenceSession(path, 
                        providers=['CUDAExecutionProvider', 
                                    'CPUExecutionProvider'])
            self.model_type = "onnx"
        elif path.endswith("pt"):
            self.session = YOLO(path, verbose=False)
            self.model_type = "pytorch"
        self.classes = classes
        self.detect(Image(np.zeros((640, 640, 3), np.uint8)), 0.6)

    def detect(self, img, thresh):
        if self.model_type == "onnx":
            return self.detect_onnx(img, thresh)
        elif self.model_type == "pytorch":
            return self.detect_pt(img, thresh)
        else:
            print("[ERROR] MODEL TYPE NOT SUPPORTED")
            return ()
    
    def detect_pt(self, img, thresh):
        results = self.session.predict(img.img, verbose=False)
        locms = [] 

        for result in results:
            for i in range(len(result.boxes)):
                conf = result.boxes[i].conf
                if conf > thresh:
                    name = result.boxes[i].cls
                    mask = result.masks[i]
                    mask = np.array(mask.xy, np.int32)[0]
                    mask = mask.reshape((-1, 1, 2))
                    locms.append((mask, conf, name))
        
        return locms

    def detect_onnx(self, img, thresh):
        if self.session is None:
            print("[ERROR] MODEL NOT INITIATED PROPERLY!")
            return ()

        outname = [i.name for i in self.session.get_outputs()]
        inname  = [i.name for i in self.session.get_inputs()]

        img = img.img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        inp = {inname[0]:im}
        #st = time()
        outputs = self.session.run(outname, inp)[0]
        #et = time()
        #print("Time ->", round(et - st, 3))

        noutputs = []

        if len(outputs):
            for output in outputs:
                _, x0, y0, x1, y1, id, scr = output
                if scr < thresh:
                    continue
                box = np.array([x0,y0,x1,y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()

                name = self.classes[int(id)]

                toadd = (box, scr, name)

                noutputs.append(toadd)

        return noutputs 


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
#