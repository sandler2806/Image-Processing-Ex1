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
from typing import List
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 319097036


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # reading the image normalized and convert to float
    image = cv2.imread(filename)
    image = image.astype(np.float32)
    image /= 255
    # represent the image by gray scale/RGB
    if representation == 1 and len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.gray()
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imDisplay(filename: str, representation: int):
    """
      Reads an image as RGB or GRAY_SCALE and displays it
      :param filename: The path to the image
      :param representation: GRAY_SCALE or RGB
      :return: None
      """

    image = imReadAndConvert(filename, representation)
    plt.imshow(image)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    return np.dot(imgRGB, yiq_from_rgb.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    return np.dot(imgYIQ, np.linalg.inv(yiq_from_rgb).T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # if the image is RGB i take only the Y part from the TIQ image
    RGB = False
    YIQ = []
    if len(imgOrig.shape) > 2:
        RGB = True
    if RGB:
        YIQ = transformRGB2YIQ(imgOrig)
        image = YIQ[:, :, 0]
    else:
        image = imgOrig
    # transform the image to 0-255 and cast to int
    imgNormalized = (image * 255).astype(int)
    # create histogram of original and cumsum
    histOrg = calHist(imgNormalized)
    cumSum = calCumSum(histOrg)
    # normalized the cumsun and create look up table
    cumSumNormalized = cumSum / np.max(cumSum)
    lut = np.ceil(cumSumNormalized * 255)

    imgEqY = np.zeros_like(image, dtype=int)
    # create new image and change every pixel by the look up table
    for i in range(256):
        imgEqY[imgNormalized == i] = int(lut[i])
    # create histogram of the new image
    histEQ = calHist(imgEqY)
    # if the image is RGB i need to connect the new Y part to the old YIQ and transform back to RGB
    if RGB:
        YIQ[:, :, 0] = imgEqY / (np.max(imgEqY) - np.min(imgEqY))  # normalized the Y back to [0,1]
        imgEq = transformYIQ2RGB(YIQ)
    else:
        imgEq = imgEqY / (np.max(imgEqY) - np.min(imgEqY))  # normalized the image back to [0,1]
    return imgEq, histOrg, histEQ



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # if the image is RGB i take only the Y part from the TIQ image
    RGB = False
    YIQ = []
    if len(imOrig.shape) > 2:
        RGB = True
    if RGB:
        YIQ = transformRGB2YIQ(imOrig)
        imOrig = YIQ[:, :, 0]
    # transform the image to 0-255 and cast to int
    imOrig = (imOrig * 255)
    hist = calHist(imOrig.astype(int))

    borders = []
    q = np.zeros(nQuant)
    First = True
    quantizedImages = []
    mses = []

    for iterNum in range(nIter):
        # at first I find the initial borders by equal distance for each part
        if First:
            # eachPart = hist.sum() / nQuant
            # Sum = 0
            # borders.append(0)
            # for i in range(256):
            #     Sum += hist[i]
            #     if Sum >= eachPart:
            #         Sum = 0
            #         borders.append(i + 1)
            # borders.append(256)
            for k in range(nQuant + 1):
                borders.append(int(k * (256 / nQuant)))
            First = False
        # the next times I will change the borders to be the middle of 2 q(the current avg)
        else:
            for k in range(1, len(borders) - 1):
                borders[k] = int((q[k - 1] + q[k]) / 2)
        # I will find the q vector for the borders by minimizing the total intensities error
        for index in range(nQuant):
            Sum = 0
            for j in range(borders[index], borders[index + 1]):
                q[index] += hist[j] * j
                Sum += hist[j]
            q[index] /= Sum
        quantizedImage = np.zeros_like(imOrig, dtype=float)
        # I will create a new image and change all the pixels in the image to the relevant q
        for index in range(nQuant):
            quantizedImage[imOrig > borders[index]] = q[index]
        # calculate the mse
        mse = np.sqrt((imOrig - quantizedImage) ** 2).mean()
        mses.append(mse)
        quantizedImages.append(quantizedImage / 255)
        # if the mse converged I will stop the procedure
        if np.abs(mse - mses[len(mses) - 2]) < 0.001 and len(mses) > 1:
            break
    # convert back to RGB in case the image was RGB
    if RGB:
        for i in range(len(quantizedImages)):
            quantizedImages[i] = transformYIQ2RGB(np.dstack((quantizedImages[i], YIQ[:, :, 1], YIQ[:, :, 2])))

    return quantizedImages, mses


def calHist(img: np.ndarray) -> np.ndarray:
    img_flat = img.ravel()
    hist = np.zeros(256)

    for pix in img_flat:
        hist[pix] += 1

    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
    return cum_sum