import cv2
import numpy as np
import pickle
from paths import HISTOGRAM_PATH, ensure_directories


def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if np.any(imgCrop == None):
                imgCrop = img[y : y + h, x : x + w]
            else:
                imgCrop = np.hstack((imgCrop, img[y : y + h, x : x + w]))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


def get_hand_hist():
    cam = None
    for idx in (0, 1, 2):
        temp = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if temp.isOpened():
            temp.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = temp.read()
            if ok:
                cam = temp
                break
        temp.release()

    if cam is None:
        print("ERROR: Camera not available.")
        return

    x, y, w, h = 60, 40, 520, 400
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist = None

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c') and imgCrop is not None:
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break

        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()
    if hist is not None:
        ensure_directories()
        with open(HISTOGRAM_PATH, "wb") as f:
            pickle.dump(hist, f)
        print("Histogram saved successfully.")


if __name__ == "__main__":
    get_hand_hist()
