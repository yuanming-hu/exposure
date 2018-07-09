import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore

import cv2
import numpy as np
import math
from scipy.interpolate import interp1d


class Filter:
    def __init__(self):
        self.parameters = [int(setting[4] * 100) if len(setting) >= 5 else 50 for setting in
                           self.get_parameter_settings()]
        self.sliders = []

    def apply(self, img):
        assert False

    def reset_parameters(self):
        self.parameters = [int(setting[4] * 100) if len(setting) >= 5 else 50 for setting in
                           self.get_parameter_settings()] 
    @classmethod
    def get_num_parameters(cls):
        return len(cls.get_parameter_settings())

    @classmethod
    def get_parameter_settings(cls):
        return []

    def get_transformed_parameter(self, i):
        lower = self.get_parameter_settings()[i][1]
        upper = self.get_parameter_settings()[i][2]
        scale = self.get_parameter_settings()[i][3]
        input = self.parameters[i] / 100.0
        if scale == 'linear':
            return input * (upper - lower) + lower
        elif scale == 'log':
            return math.exp(input * math.log(1.0 * upper / lower) + math.log(lower))
        else:
            assert False, scale

    @classmethod
    def get_name(cls):
        assert False

    # Invoked when slider slides
    def load_states_from_ui(self):
        for i in range(self.get_num_parameters()):
            self.parameters[i] = self.sliders[i].value()

    # Invoked when switching tabs
    def write_states_to_ui(self):
        for i in range(self.get_num_parameters()):
            self.sliders[i].setValue(self.parameters[i])

    def get_curve(self):
        return np.ones((100, 100, 3)) * 0.851

    @classmethod
    def get_tab_widget_and_sliders(cls, window):
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        sliders = []
        for i in range(cls.get_num_parameters()):
            setting = cls.get_parameter_settings()[i]
            label = QLabel(setting[0])
            layout.addWidget(label, i, 0)
            slider = QSlider(QtCore.Qt.Horizontal)
            if len(setting) >= 5:
                slider.setValue(int(setting[4] * 100))
            else:
                slider.setValue(50)
            sliders.append(slider)
            slider.sliderMoved.connect(window.update_UI)

            def get_callback(slider):
                def callback(e):
                    QSlider.mousePressEvent(slider, e)
                    window.update_UI()

                return callback

            slider.mousePressEvent = get_callback(slider)
            layout.addWidget(slider, i, 1)

            def get_reset_slider(slider):
                def reset_slider():
                    slider.setValue(50 if len(setting) < 5 else int(setting[4] * 100))
                    window.update_UI()

                return reset_slider

            btn_reset = QPushButton("Reset", window)
            btn_reset.clicked.connect(get_reset_slider(slider))
            layout.addWidget(btn_reset, i, 2)

        return widget, sliders


class ExposureFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('Exposure', -5, 5, 'linear'),
        ]

    def apply(self, img):
        return img * math.pow(2, self.get_transformed_parameter(0))

    @classmethod
    def get_name(cls):
        return 'Exposure'


class GammaFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('Gamma', 8.0, 1.0 / 8.0, 'log'),
        ]

    def apply(self, img):
        return np.power(img, self.get_transformed_parameter(0))

    @classmethod
    def get_name(cls):
        return 'Gamma'


class WBFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        limits = 0.5
        return [
            ('Temperature', -limits, limits, 'linear'),
            ('Tint', -limits, limits, 'linear'),
        ]

    def apply(self, img):
        color_scaling = np.array((1, math.exp(-self.get_transformed_parameter(1)),
                                  math.exp(-self.get_transformed_parameter(0))), np.float32)
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling *= 1.0 / (1e-5 + 0.27 * color_scaling[0] + 0.67 * color_scaling[1] + 0.06 * color_scaling[2])
        return img * color_scaling[None, None, :]

    @classmethod
    def get_name(cls):
        return 'W.B.'


def hsv_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def rgb_to_hsv(img):
    # print(img.dtype)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


class SaturationFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('Saturation', 0, 1, 'linear', 0),
        ]

    def apply(self, img):
        hsv = rgb_to_hsv(img)
        s = hsv[:, :, 1:2]
        v = hsv[:, :, 2:3]
        enhanced_s = s + (1 - s) * (0.5 - abs(0.5 - v))
        hsv1 = np.concatenate([hsv[:, :, 0:1], enhanced_s, hsv[:, :, 2:]], axis=2)
        hsv0 = np.concatenate([hsv[:, :, 0:1], hsv[:, :, 1:2] * 0 + 0, hsv[:, :, 2:]], axis=2)
        bnw = hsv_to_rgb(hsv0)
        full_color = hsv_to_rgb(hsv1)

        param = np.array((self.get_transformed_parameter(0),), dtype=np.float32)

        param = param[:, None, None]

        bnw_param = np.maximum(0.0, -param)
        color_param = np.maximum(0.0, param)
        img_param = np.maximum(0.0, 1.0 - abs(param))

        return bnw_param * bnw + img * img_param + full_color * color_param

    @classmethod
    def get_name(cls):
        return 'Sat.'


class BNWFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('BNW', 0, -1, 'linear', 0),
        ]

    def apply(self, img):
        hsv = rgb_to_hsv(img)
        s = hsv[:, :, 1:2]
        v = hsv[:, :, 2:3]
        enhanced_s = s + (1 - s) * (0.5 - abs(0.5 - v))
        hsv1 = np.concatenate([hsv[:, :, 0:1], enhanced_s, hsv[:, :, 2:]], axis=2)
        hsv0 = np.concatenate([hsv[:, :, 0:1], hsv[:, :, 1:2] * 0 + 0, hsv[:, :, 2:]], axis=2)
        bnw = hsv_to_rgb(hsv0)
        full_color = hsv_to_rgb(hsv1)

        param = np.array((self.get_transformed_parameter(0),), dtype=np.float32)

        param = param[:, None, None]

        bnw_param = np.maximum(0.0, -param)
        color_param = np.maximum(0.0, param)
        img_param = np.maximum(0.0, 1.0 - abs(param))

        return bnw_param * bnw + img * img_param + full_color * color_param

    @classmethod
    def get_name(cls):
        return 'B&&W'


class LevelFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('Black point', 0, 1, 'linear', 0),
            ('White point', 0, 1, 'linear', 1),
        ]

    def apply(self, img):
        lower = self.get_transformed_parameter(0)
        upper = self.get_transformed_parameter(1)
        # Make sure lower < upper
        upper = lower + upper * (1 - lower)
        return np.clip((img - lower) / (upper - lower + 1e-20), 0, 1)

    @classmethod
    def get_name(cls):
        return 'Level'


def rgb2lum(image):
    return (0.27 * image[:, :, 0] + 0.67 * image[:, :, 1] + 0.06 * image[:, :, 2])[:, :, None]


def lerp(a, b, alpha):
    return (1 - alpha) * a + alpha * b


class ContrastFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        return [
            ('Contrast', -1, 1, 'linear', 0.5),
        ]

    def apply(self, img):
        contrast = np.array((self.get_transformed_parameter(0),), dtype=np.float32)
        contrast_image = -np.cos(math.pi * img) * 0.5 + 0.5
        return lerp(img, contrast_image, contrast[:, None, None])

    @classmethod
    def get_name(cls):
        return 'Contrast'


def get_spline(low, mid, high):
    x = np.array([0, 0.25, 0.5, 0.75, 1]).astype(np.float32)
    y = np.array([0, low * 0.25 + 0.25, mid * 0.25 + 0.5, high * 0.25 + 0.75, 1]).astype(np.float32)
    return interp1d(x, y, kind='cubic')


class ToneFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        limit = 1
        names = ['Shadow', 'Midtone', 'Highlight']
        return [(n, -limit, limit, 'linear') for n in names]

    def apply(self, img):
        output = get_spline(self.get_transformed_parameter(0), self.get_transformed_parameter(1),
                            self.get_transformed_parameter(2))(img)
        return output.astype(np.float32)

    @classmethod
    def get_name(cls):
        return 'Tone'

    def get_curve(self):
        L = 200
        canvas = np.ones((L, L, 3)).astype(np.float32) * 0.851
        curve_step = 16
        xs = np.linspace(0.0, 1.0, curve_step + 1).astype(np.float32)
        ys = self.apply(xs)
        ys = np.clip(ys, 0.01, 1.0)
        for i in range(curve_step):
            x, y = xs[i], ys[i]
            xx, yy = xs[i + 1], ys[i + 1]
            cv2.line(canvas, (int(L * x), int(L - L * y)), (int(L * xx), int(L - L * yy)), (0, 0, 0), 1)
        return canvas


class ColorFilter(Filter):
    @classmethod
    def get_parameter_settings(cls):
        limit = 1
        names = [
            'Red Shadow', 'Red Midtone', 'Red Highlight',
            'Green Shadow', 'Green Midtone', 'Green Highlight',
            'Blue Shadow', 'Blue Midtone', 'Blue Highlight',
        ]
        return [(n, -limit, limit, 'linear') for n in names]

    def apply(self, img):
        for i in range(3):
            output = get_spline(self.get_transformed_parameter(i * 3), self.get_transformed_parameter(i * 3 + 1),
                                self.get_transformed_parameter(i * 3 + 2))(img[:, :, i])
            img[:, :, i] = output
        return img

    @classmethod
    def get_name(cls):
        return 'Color'

    def get_curve(self):
        L = 200
        canvas = np.ones((L, L, 3)) * 0.851
        curve_step = 16
        xs = np.linspace(0.0, 1.0, curve_step + 1)[:, None, None]
        xs = np.concatenate([xs] * 3, axis=2)
        ys = self.apply(xs.copy())
        ys = np.clip(ys, 0.01, 1.0)
        for i in range(curve_step):
            for j in range(3):
                x, y = xs[i, 0, j], ys[i, 0, j]
                xx, yy = xs[i + 1, 0, j], ys[i + 1, 0, j]
                color = [0, 0, 0]
                color[j] = 0.7
                color = tuple(color)
                cv2.line(canvas, (int(L * x), int(L - L * y)), (int(L * xx), int(L - L * yy)), color, 1)
        return canvas


all_filters = [ExposureFilter, GammaFilter, SaturationFilter, WBFilter, ContrastFilter, BNWFilter, ToneFilter,
               ColorFilter]
__all__ = ['all_filters']