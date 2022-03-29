"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == LOAD_GRAY_SCALE:
        img = cv.imread(img_path, 2)
    else:  # rep = LOAD_RGB
        img = cv.imread(img_path, 1)
    cv.namedWindow('sdf')
    trackbar_name = 'gamma %d' % 200
    cv.createTrackbar(trackbar_name, 'sdf', 0, 200, on_trackbar)
    on_trackbar(0)
    cv.waitKey()


def on_trackbar(val):
    gamma = float(val / 100)
    invGamma = 1000 if gamma == 0 else 1.0 / gamma
    table = np.array([((i / 255) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    img2=cv.LUT(img, table)
    cv.imshow('gammaDisplay', img2)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
