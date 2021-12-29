import argparse
import cv2
import numpy as np
from config import IMAGE_SHAPE
from segmentation_technique import SegmentationTechnique
from unet_segmentation import UNetSegmentation


class InteractiveApp:
    def __init__(self, path, segmentation_technique: SegmentationTechnique, debug_mode=False):
        self.segmentation_technique = segmentation_technique

        self.win_name = "main"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        cv2.setWindowTitle(self.win_name, "Interactive RGB-D segmentation")

        self.img = None
        self.mask = None
        self.load_img(path)
        self.res_img = self.img.copy()
        self.need_update = False

        self.pos_interactions = ([], [])
        self.neg_interactions = ([], [])

        self.debug_mode = debug_mode

    def run(self):
        while True:
            if self.need_update:
                self.update()
                self.need_update = False
            cv2.imshow(self.win_name, self.res_img)
            key = cv2.waitKey(15)
            if key == 27:
                break
            else:
                self.key_handler(key)

    def load_img(self, path):
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.img.shape != IMAGE_SHAPE:
            self.img = cv2.resize(self.img, IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
        #self.mask = np.zeros(self.img.shape, dtype=np.uint8)
        self.mask = np.full((self.img.shape[0], self.img.shape[1]), cv2.GC_PR_BGD, dtype=np.uint8)

    def update(self):
        img = self.img.copy()
        # Mark foreground in the image.
        img[:, :, 1][self.mask == cv2.GC_FGD] = 0
        img[:, :, 1][self.mask == cv2.GC_PR_FGD] = img[:, :, 1][self.mask == cv2.GC_PR_FGD] / 2
        # Mark background in the image.
        img[:, :, 2][self.mask == cv2.GC_BGD] = 0
        img[:, :, 2][self.mask == cv2.GC_PR_BGD] = img[:, :, 2][self.mask == cv2.GC_PR_BGD] / 2

        neg = np.zeros(self.img.shape, dtype=np.uint8)
        neg[self.neg_interactions] = (0, 0, 255)
        neg = cv2.GaussianBlur(neg, (35, 35), 5)
        neg = cv2.normalize(neg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img += neg

        pos = np.zeros(self.img.shape, dtype=np.uint8)
        pos[self.pos_interactions] = (0, 255, 0)
        pos = cv2.GaussianBlur(pos, (35, 35), 5)
        pos = cv2.normalize(pos, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img += pos

        self.res_img = img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pos_interactions[0].append(y)
            self.pos_interactions[1].append(x)
            self.segment()
            self.need_update = True

        if event == cv2.EVENT_RBUTTONUP:
            self.neg_interactions[0].append(y)
            self.neg_interactions[1].append(x)
            self.segment()
            self.need_update = True

    def segment(self):
        prediction = self.segmentation_technique.segment(self.pos_interactions, self.neg_interactions)
        new_mask = np.zeros(self.mask.shape, dtype=np.uint8)
        new_mask[prediction == 0] = cv2.GC_BGD
        new_mask[prediction == 1] = cv2.GC_FGD

        if self.debug_mode:
            print("bgd_count", len(np.where(prediction == 0)[0]))
            print("fgd_count", len(np.where(prediction == 1)[0]))

        self.mask = new_mask

    def reset(self):
        self.res_img = self.img.copy()
        self.pos_interactions = ([], [])
        self.neg_interactions = ([], [])
        self.need_update = False

    def key_handler(self, key):
        if key == ord('r'):
            self.reset()
        else:
            pass


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", help="image name (without _leftImg8bit.png)", required=True)
    parser.add_argument("-m", "--model", help="stored model parameters")
    parser.add_argument("-d", "--debug", help="debug mode", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    image_file = args.img + "_leftImg8bit.png"
    disparity_file = args.img + "_disparity.png"
    if args.model:
        model_pth = args.model
    else:
        model_pth = "InteractiveModel.pth"

    debug_mode = args.debug

    unet_technique = UNetSegmentation(image_file, disparity_file, model_pth, debug_mode)
    InteractiveApp(image_file, unet_technique, debug_mode).run()
