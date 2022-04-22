import io
import random
from abc import abstractmethod

import pywt

from faces_repository import zig_zag_transform
from sklearn.decomposition import PCA
import cv2
import pylab
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from scipy.fft import dct
import mahotas
from mahotas.features import surf


class FeatureGetter:

    @abstractmethod
    def plot(self, *args):
        ...

    @abstractmethod
    def get_feature(self, *args):
        ...

    @abstractmethod
    def get_teach_param(self, *args):
        ...

    @abstractmethod
    def set_param(self, *args):
        ...


class Haralick(FeatureGetter):
    num_pc: int = 24

    def plot(self, image: np.ndarray) -> bytes:
        gaussian = mahotas.gaussian_filter(image, 24)
        gaussian = (gaussian > gaussian.mean())
        labelled, n = mahotas.label(gaussian)
        features = mahotas.features.haralick(labelled).mean(axis=0)

        plt.figure(figsize=(20, 10), dpi=80)
        ax = plt.gca()
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        ax.grid(linewidth=4)

        plt.plot(features)

        path = 'haralick.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        gaussian = mahotas.gaussian_filter(image, self.num_pc)
        gaussian = (gaussian > gaussian.mean())
        labelled, n = mahotas.label(gaussian)
        features = mahotas.features.haralick(labelled).mean(axis=0)
        return features

    def get_teach_param(self, image=None):
        return range(1, 120, 10)

    def set_param(self, num_pc: int):
        self.num_pc = num_pc

class PCAanalisys(FeatureGetter):
    num_pc: int = 60

    def plot(self, image: np.ndarray) -> bytes:

        pca = PCA(self.num_pc)
        converted_data = pca.fit_transform(image)

        plt.figure(figsize=(20, 10), dpi=80)
        ax = plt.gca()
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        ax.grid(linewidth=4)

        c_map = plt.cm.get_cmap('jet', 10)
        plt.scatter(converted_data[0], converted_data[1], s=130,
                    cmap=c_map)

        path = 'pca.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        pca = PCA(self.num_pc)

        converted_data = pca.fit_transform(image)

        return converted_data[0]

    def get_teach_param(self, image=None):
        return range(1, 120, 10)

    def set_param(self, num_pc: int):
        self.num_pc = num_pc


class SURF(FeatureGetter):
    param: int

    def plot(self, image: np.ndarray) -> bytes:
        min_val, max_val = image.min(), image.max()
        image = 255.0 * (image - min_val) / (max_val - min_val)
        image = image.astype(np.uint8)

        surf = cv2.SIFT_create(400)

        kp, des = surf.detectAndCompute(image, None)
        print(len(kp))
        image = cv2.drawKeypoints(image, kp, None, (255,0,0),4)


        path = 'surf.png'
        cv2.imwrite(path, image)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        min_val, max_val = image.min(), image.max()
        image = 255.0 * (image - min_val) / (max_val - min_val)
        image = image.astype(np.uint8)

        surf = cv2.SIFT_create(400)

        kp, des = surf.detectAndCompute(image, None)

        return kp

    def get_teach_param(self, image=None):
        return range(1, 10)

    def set_param(self, param: int):
        self.param = param

class WaveletTransform(FeatureGetter):
    level: int = 4

    def plot(self, image: np.ndarray) -> bytes:

        coefficients = pywt.wavedec2(image, 'db2', mode='periodization', level=self.level)

        coefficients[0] /= np.abs(coefficients[0]).max()

        for detail_level in range(self.level):
            coefficients[detail_level + 1] = [d / np.abs(d).max() for d in coefficients[detail_level + 1]]
        # show the normalized coefficients
        image, slices = pywt.coeffs_to_array(coefficients)
        # cv2.imshow( 'some picture', image)

        min_val, max_val = image.min(), image.max()
        image = 255.0 * (image - min_val) / (max_val - min_val)
        image = image.astype(np.uint8)
        path = 'wavelet_transform.png'
        cv2.imwrite(path, image)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:

        coefficients = pywt.wavedec2(image, 'db2', mode='periodization', level=self.level)

        features = zig_zag_transform(coefficients[0])

        return features

    def get_teach_param(self, image=None):
        return range(1, 10)

    def set_param(self, level: int):
        self.level = level


class Random(FeatureGetter):
    num_pixel: int = 5
    x_indexes: list = []
    y_indexes: list = []
    color: str = [0,0,255]

    def plot(self, image: np.ndarray) -> bytes:
        min_val, max_val = image.min(), image.max()
        image = 255.0 * (image - min_val) / (max_val - min_val)
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        for x, y in zip(self.x_indexes, self.y_indexes):
            cv2.circle(img=image, center=(x, y), radius=1, color=self.color, thickness=-1)

        image = image.astype(np.uint8)
        path = 'random.png'
        cv2.imwrite(path, image)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        self.x_indexes = random.sample(range(0, width), pow(self.num_pixel, 2))
        self.y_indexes = random.sample(range(0, height), pow(self.num_pixel, 2))

        features = []
        for x, y in zip(self.x_indexes, self.y_indexes):
            features.append(image[x][y])

        return features

    def get_teach_param(self, image=None):
        return range(1, 10)

    def set_param(self, num_pixel: int):
        self.num_pixel = num_pixel

class Histogram(FeatureGetter):
    num_bins: int = 30

    def plot(self, image: np.ndarray) -> bytes:

        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        hist = np.insert(hist, 0, 0.0)
        plt.figure(figsize=(20, 10), dpi=80)
        ax = plt.gca()
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        ax.grid(linewidth=3)
        plt.plot(bins, hist,  linewidth=5)
        path = 'histogram.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        return hist

    def get_teach_param(self, image=None):
        return range(1, 255, 5)

    def set_param(self, num_bins: int):
        self.num_bins = num_bins


class DFT(FeatureGetter):
    p: int = 13

    def plot(self, image: np.ndarray) -> bytes:
        ftimage = np.fft.fft2(image)
        ftimage = ftimage[0: self.p, 0: self.p]
        ftimage = np.abs(ftimage)

        pylab.imshow(np.abs(ftimage))
        path = 'dft.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):
        ftimage = np.fft.fft2(image)
        ftimage = ftimage[0: self.p, 0: self.p]

        return np.abs(ftimage)

    def get_teach_param(self, image=None):
        return range(30)

    def set_param(self, p: int):
        self.p = p


class DCT(FeatureGetter):
    p: int = 13

    def plot(self, image: np.ndarray) -> bytes:
        dct_image = dct(image, axis=1)
        dct_image = dct(dct_image, axis=0)
        dct_image = dct_image[0: self.p, 0: self.p]

        pylab.imshow(np.abs(dct_image))
        path = 'dct.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):

        c = dct(image, axis=1)
        c = dct(c, axis=0)
        c = c[0: self.p, 0: self.p]

        return c

    def get_teach_param(self, image=None):
        return range(30)

    def set_param(self, p: int):
        self.p = p


class Scale(FeatureGetter):
    scale: int = 0.3

    def plot(self, image: np.ndarray) -> bytes:
        h = image.shape[0]
        w = image.shape[1]

        new_size = (int(h * self.scale), int(w * self.scale))

        output = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        min_val, max_val = output.min(), output.max()
        img = 255.0 * (output - min_val) / (max_val - min_val)
        img = img.astype(np.uint8)

        path = 'scale.png'
        cv2.imwrite(path, img)

        return path

    def get_feature(self, image: int):
        h = image.shape[0]
        w = image.shape[1]
        new_size = (int(h * self.scale), int(w * self.scale))

        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_teach_param(self, image=None):
        return np.arange(0.1, 1.1, 0.1)

    def set_param(self, scale: int):
        self.scale = scale


class Gradient(FeatureGetter):
    window_width: int = 2

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))

    def plot(self, image: np.ndarray) -> bytes:
        height, width = image.shape

        num_steps = int(height / self.window_width)
        gradients = []

        for i in range(num_steps - 2):
            step = i * self.window_width

            start_window = image[step: step + self.window_width]
            end_window = image[step + self.window_width: step + self.window_width * 2]

            gradients.append(self._calculate_distance(start_window, end_window))

        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        ax.grid(linewidth=3)
        plt.plot(range(num_steps - 2), gradients, linewidth=4)
        path = 'scale.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: int):

        height, width = image.shape

        num_steps = height // self.window_width
        gradients = []

        for i in range(num_steps - 2):
            step = i * self.window_width

            start_window = image[step: step + self.window_width]
            end_window = image[step + self.window_width: step + self.window_width * 2]

            gradients.append(self._calculate_distance(start_window, end_window))

        return gradients

    def get_teach_param(self, image):
        height, width = image.shape

        return range(1,height//2)

    def set_param(self, window_width: int):
        self.window_width = window_width

