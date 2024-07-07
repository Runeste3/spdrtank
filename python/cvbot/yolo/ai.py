import cv2 as cv
import numpy as np
import onnxruntime as ort
import math
from cvbot.images import Image
from ultralytics import YOLO


classifiers = {}

class Model:
    def __init__(self, path, classes=[], pred_type="box") -> None:
        if path.endswith("onnx"):
            self.pred_type = pred_type
            self.session = ort.InferenceSession(path, 
                        providers=['CUDAExecutionProvider', 
                                    'CPUExecutionProvider'])
            self.model_type = "onnx"
            # List
            self.classes = classes
            self.nc = len(self.classes)
        elif path.endswith("pt"):
            self.session = YOLO(path, verbose=False)
            self.model_type = "pytorch"
            # Dict
            self.classes = self.session.names if classes == [] else classes
        self.detect(Image(np.zeros((640, 640, 3), np.uint8)), 0.6)

    def detect(self, img, thresh):
        if self.model_type == "onnx" and self.pred_type == "box":
            return self.detect_onnx_box(img, thresh)
        elif self.model_type == "onnx" and self.pred_type == "mask":
            return self.detect_onnx_mask(img, thresh)
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
                box = result.boxes[i]
                conf = box.conf
                if conf > thresh:
                    clsid = int(box.cls)
                    name = self.classes[clsid]

                    mask = result.masks[i]
                    mask = np.array(mask.xy, np.int32)[0]
                    mask = mask.reshape((-1, 1, 2))

                    box = np.array(box.xyxy[0].cpu(), np.int32)

                    locms.append(((mask, box), float(conf), name))
        
        return locms

    def detect_onnx_box(self, img, thresh):
        if self.session is None:
            print("[ERROR] MODEL NOT INITIATED PROPERLY!")
            return ()

        outname = [i.name for i in self.session.get_outputs()]
        inname  = [i.name for i in self.session.get_inputs()]

        img = img.img
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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

    def detect_onnx_mask(self, oimg, thresh):
        iw, ih = oimg.size
        img = process(oimg.img)
        onms = [i.name for i in self.session.get_outputs()]
        inm = self.session.get_inputs()[0].name
        outputs = self.session.run(onms, {inm:img})

        mask_out = outputs[1]
        outputs = outputs[0]

        outputs = np.squeeze(outputs[0]).T
        scores = np.max(outputs[:, 4:4+self.nc], axis=1)
        outputs = outputs[scores > 0.6, :]

        scores  = scores[scores > thresh]
        boxes   = outputs[:, :self.nc+4]
        outputs = filter_output(boxes, scores, outputs)
        boxes   = outputs[:, :self.nc+4]
        ids     = np.argmax(boxes[:, 4:], axis=1)
        boxes   = extract_boxes(boxes, 640, 640, ih, iw)

        masks_pred = outputs[:, self.nc+4:]
        masks_sqz = np.squeeze(mask_out)
        num_mask, mask_h, mask_w = masks_sqz.shape
        masks = sigmoid(masks_pred @ masks_sqz.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_h, mask_w))

        rscld_boxes = rescale_boxes(boxes, (ih, iw), (mask_h, mask_w))
        results = []
        for i in range(len(boxes)):
            sx1, sx2, sy1, sy2 = box_to_xxyy(rscld_boxes[i])
            x1, x2, y1, y2 = box_to_xxyy(boxes[i])
            mask = masks[i][sy1:sy2, sx1:sx2]
            mask = cv.resize(mask, (x2 - x1, y2 - y1), interpolation=cv.INTER_CUBIC)
            mask = (mask > 0.1)

            scr = scores[i]
            name = self.classes[ids[i]]
            result = (mask, (x1, x2, y1, y2), scr, name)
            results.append(result)

        return results



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
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
#
def process(im):
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = cv.resize(im, (640, 640))
    im = im / 255.0
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, :, :, :].astype(np.float32)
    return im

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_to_xxyy(box):
    x1 = int(math.floor(box[0]))
    y1 = int(math.floor(box[1]))
    x2 = int(math.ceil (box[2]))
    y2 = int(math.ceil (box[3]))        
    return x1, x2, y1, y2

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def box_center(box):
    x, y, w, h = box[:4]
    return x + (w/2), y + (h/2)

def extract_boxes(box_predictions, inh, inw, ih, iw):
    # Extract boxes from predictions
    boxes = box_predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes,
                          (inh, inw),
                          (ih, iw))

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    # Check the boxes are within the image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, iw)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, ih)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, iw)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, ih)

    return boxes

def filter_output(boxes, scores, outputs):
    centers = {}
    ind_to_del = []
    
    for i in range(len(boxes)):
        cur_cent = box_center(boxes[i])
        cur_score = scores[i]

        for j in centers.keys():
            rec_cent, rec_score = centers[j]
            if math.dist(cur_cent, rec_cent) < 20:
                if cur_score > rec_score:
                    ind_to_del.append(j)
                    del centers[j]
                    centers[i] = cur_cent, cur_score
                else:
                    ind_to_del.append(i)
                break
        else:
            centers[i] = cur_cent, cur_score
            continue

    return np.delete(outputs, ind_to_del, axis=0)

def rescale_boxes(boxes, input_shape, image_shape):
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

    return boxes
