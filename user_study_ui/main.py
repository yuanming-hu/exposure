import sys
import os

from PyQt5.QtWidgets import (QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QLabel, QTabWidget, QApplication)
from PyQt5.QtGui import (QPixmap, QImage)

import cv2
import numpy as np
from filters import all_filters
import matplotlib.pyplot as plt

NUM_STEPS = 4


def make_empty_dir(folder):
    # This is dangerous...
    # if os.path.exists(folder):
    #     os.system('rm -rf %s' % folder)
    try:
        os.mkdir(folder)
    except:
        print('Warning: folder %s exists' % folder)
        print('         Overwriting.')


def QImageFromNumpyArray(image):
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)


def QPixmapFromNumpyArray(image):
    return QPixmap(QImageFromNumpyArray(image))


def read_image(fn):
    img = (cv2.imread(fn)[:, :, ::-1].copy() * (1 / 255.0)).astype(np.float32)
    return img


def rgb2lum(image):
    return (0.27 * image[:, :, 0] + 0.67 * image[:, :, 1] + 0.06 * image[:, :, 2])[:, :, None]


def resize_by_longside(image, size=300):
    if image.shape[0] > image.shape[1]:
        new_shape = [0, 0]
        new_shape[1] = int(image.shape[1] * 1.0 / image.shape[0] * size)
        new_shape[0] = size
    else:
        new_shape = [0, 0]
        new_shape[0] = int(image.shape[0] * 1.0 / image.shape[1] * size)
        new_shape[1] = size
    image = cv2.resize(image, (new_shape[1], new_shape[0]))
    return image


def histogram_of_image(image):
    image = np.clip(rgb2lum(image) * 255.0, 0, 255).astype(np.uint8).ravel()
    hist, _ = np.histogram(image, 255, (0, 255))
    hist = hist * 1.0 / len(image)
    hist /= hist.max() * 1.10
    H = 200
    canvas = np.ones((H, 255, 3)) * 0.851
    for i in range(255):
        h = int(hist[i] * H)
        cv2.line(canvas, (i, H - 0), (i, H - h), (155, 155, 155), 2)
    return canvas


def size_adjust(img):
    if (img.shape[0] < img.shape[1]):
        return img
    else:
        new_shape = [0, 0]
        new_shape[1] = int(img.shape[1] * img.shape[1] * 1.0 / img.shape[0])
        new_shape[0] = img.shape[1]
        canvas = np.ones((img.shape[1], img.shape[0], 3)) * 0.851
        start = int((img.shape[0] - new_shape[1]) / 2)
        canvas[:, start: start + new_shape[1], :] = cv2.resize(img, (new_shape[1], new_shape[0]))
        return canvas


def load_raw(npy):
    x = np.load(npy)
    return x.astype(np.float32)


class Retoucher(QWidget):
    def __init__(self, input_folder, username):
        super(Retoucher, self).__init__()
        self.srcdir = os.path.join('data/inputs', input_folder)
        self.imagedir = os.path.join('data/outputs', input_folder + '_' + username + '_' + 'images')
        self.metadir = os.path.join('data/outputs', input_folder + '_' + username + '_' + 'actions')
        make_empty_dir(self.imagedir)
        make_empty_dir(self.metadir)
        self.fns = [*filter(lambda x: x.endswith('.npy'), sorted(os.listdir(self.srcdir)))]
        self.input_images = [load_raw(os.path.join(self.srcdir, x)) for x in self.fns]
        self.index = 0
        self.current_stage = 0
        self.step_images = [self.input_images[self.index] for _ in range(NUM_STEPS + 1)]
        self.step_image_widgets = [None for _ in range(NUM_STEPS + 1)]
        self.actions = [[f() for f in all_filters] for _ in range(NUM_STEPS)]
        self.selected_actions = [0 for _ in range(NUM_STEPS)]
        self.label_progress = None
        self.initUI()

    def recalculate(self):
        action = self.actions[self.current_stage][self.selected_actions[self.current_stage]]
        action.load_states_from_ui()
        for i in range(NUM_STEPS):
            action = self.actions[i][self.selected_actions[i]]
            self.step_images[i + 1] = np.clip(action.apply(np.clip(self.step_images[i], 0, 1)), 0, 1)

    def switch_to_step(self, s):
        self.current_stage = s
        for a in self.actions[self.current_stage]:
            a.write_states_to_ui()

    def update_UI(self):
        self.recalculate()
        progress = "%d/%d (%.1f%%)" % (self.index, len(self.fns), 100.0 * self.index / len(self.fns))
        if self.label_progress:
            self.label_progress.setText(progress)
        self.input_image_widget.setPixmap(QPixmapFromNumpyArray(size_adjust(self.step_images[self.current_stage])))
        self.output_image_widget.setPixmap(QPixmapFromNumpyArray(size_adjust(self.step_images[self.current_stage + 1])))
        output_hist = histogram_of_image(self.step_images[self.current_stage + 1])
        current_action = self.actions[self.current_stage][self.selected_actions[self.current_stage]]
        self.filter_info_widget.setPixmap(QPixmapFromNumpyArray(current_action.get_curve()))
        self.hist_info_widget.setPixmap(QPixmapFromNumpyArray(output_hist))

        for i in range(NUM_STEPS + 1):
            scale = 1.0 / (NUM_STEPS + 1)
            img_resized = cv2.resize(self.step_images[i], (0, 0), fx=scale, fy=scale)
            img_adjust = size_adjust(img_resized)
            self.step_image_widgets[i].setPixmap(QPixmapFromNumpyArray(img_adjust))
            if i - 1 == self.current_stage:
                self.step_image_widgets[i].setStyleSheet("border: 3px solid red")
            else:
                self.step_image_widgets[i].setStyleSheet("border: 0px solid red")
            self.step_image_widgets[i].show()

            # self.hist_info_widget.setPixmap(QPixmapFromNumpyArray(np.zeros((100, 100, 3))))
            # self.filter_info_widget.setPixmap(QPixmapFromNumpyArray(np.ones((100, 100, 3))))

    def initUI(self):
        self.setWindowTitle('Photo Editor')

        outer_layout = QHBoxLayout()
        self.setLayout(outer_layout)

        images = QVBoxLayout()
        input_image = QGroupBox("Before")
        input_img_layout = QHBoxLayout()
        input_image.setLayout(input_img_layout)
        self.input_image_widget = QLabel()
        self.input_image_widget.setScaledContents(True)
        input_img_layout.addWidget(self.input_image_widget)
        images.addWidget(input_image)

        output_image = QGroupBox("After")
        output_img_layout = QHBoxLayout()
        output_image.setLayout(output_img_layout)
        self.output_image_widget = QLabel()
        self.output_image_widget.setScaledContents(True)
        output_img_layout.addWidget(self.output_image_widget)
        images.addWidget(output_image)

        outer_layout.addLayout(images)

        step_and_operation = QVBoxLayout()
        outer_layout.addLayout(step_and_operation)

        steps = QGroupBox("Steps")
        steps_layout = QHBoxLayout()

        steps.setLayout(steps_layout)
        for i in range(NUM_STEPS + 1):
            step_label = QLabel()
            steps_layout.addWidget(step_label)

            def get_callback(j):
                def callback(e):
                    self.current_stage = j
                    for k in range(len(all_filters)):
                        self.actions[j][k].write_states_to_ui()
                    self.update_UI()
                    self.operation_tabs.setCurrentIndex(self.selected_actions[j])

                return callback

            if i > 0:
                step_label.mousePressEvent = get_callback(i - 1)
            self.step_image_widgets[i] = step_label

        step_and_operation.addWidget(steps)

        infos = QHBoxLayout()
        step_and_operation.addLayout(infos)

        histogram = QGroupBox("Histogram")
        curves = QGroupBox("Curves")
        infos.addWidget(histogram)
        infos.addWidget(curves)

        hist_info = QLabel()
        filter_info = QLabel()
        self.hist_info_widget = hist_info
        self.hist_info_widget.setScaledContents(True)
        self.filter_info_widget = filter_info
        self.filter_info_widget.setScaledContents(True)

        histogram_layout = QVBoxLayout()
        histogram.setLayout(histogram_layout)
        histogram_layout.addWidget(hist_info)

        curve_layout = QVBoxLayout()
        curves.setLayout(curve_layout)
        curve_layout.addWidget(filter_info)

        operation = QGroupBox("Operation")
        step_and_operation.addWidget(operation)
        operation_layout = QVBoxLayout()
        operation.setLayout(operation_layout)
        operation_tabs = QTabWidget()
        operation_layout.addWidget(operation_tabs)

        operation_tabs.currentChanged.connect(self.filter_changed)
        self.operation_tabs = operation_tabs

        for i, f in enumerate(all_filters):
            widget, sliders = f.get_tab_widget_and_sliders(self)
            for j in range(NUM_STEPS):
                self.actions[j][i].sliders = sliders
            operation_tabs.addTab(widget, f.get_name())
            operation_tabs.mousePressEvent = lambda e: self.update_UI()

        moves_layout = QHBoxLayout()
        operation_layout.addLayout(moves_layout)
        btn_prev = QPushButton("Previous step", self)
        btn_next = QPushButton("Next step", self)
        btn_save = QPushButton("Done!", self)
        label_progress = QLabel(self)
        moves_layout.addWidget(btn_prev)
        moves_layout.addWidget(btn_next)
        moves_layout.addWidget(btn_save)
        moves_layout.addWidget(label_progress)
        self.label_progress = label_progress
        btn_prev.clicked.connect(self.move_to_prev)
        btn_next.clicked.connect(self.move_to_next)
        btn_save.clicked.connect(self.save)

        self.resize(800, 800)
        self.move(0, 0)
        self.update_UI()
        self.show()

    def filter_changed(self, i):
        self.selected_actions[self.current_stage] = i
        self.update_UI()

    def move_to_prev(self):
        if self.current_stage > 0:
            self.current_stage -= 1
            j = self.current_stage
            for k in range(len(all_filters)):
                self.actions[j][k].write_states_to_ui()
            self.update_UI()
            self.operation_tabs.setCurrentIndex(self.selected_actions[j])

    def move_to_next(self):
        if self.current_stage < NUM_STEPS - 1:
            self.current_stage += 1
            j = self.current_stage
            for k in range(len(all_filters)):
                self.actions[j][k].write_states_to_ui()
            self.update_UI()
            self.operation_tabs.setCurrentIndex(self.selected_actions[j])

    def save(self):
        fn = self.fns[self.index][:-len('.npy')]
        output = self.step_images[self.current_stage + 1]
        cv2.imwrite(os.path.join(self.imagedir, fn + '.jpg'), output[:, :, ::-1] * 255.0)

        import json
        op_seq = []
        for action, select in zip(self.actions, self.selected_actions):
            f = action[select]
            op = {'filter_id': select, 'parameters': f.parameters}
            op_seq.append(op)
        json.dump(op_seq,
                  open(os.path.join(self.metadir, fn + '.json'), "w"))

        self.goto_next_image()

    def goto_next_image(self):
        self.index += 1
        if (self.index >= len(self.input_images)):
            self.close()
        else:
            self.current_stage = 0
            self.step_images = [self.input_images[self.index] for _ in range(NUM_STEPS + 1)]
            for filters in self.actions:
                for f in filters:
                    f.reset_parameters()
                    f.write_states_to_ui()
            self.selected_actions = [0 for _ in range(NUM_STEPS)]
            self.update_UI()
            self.operation_tabs.setCurrentIndex(0)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 main.py [input folder] [user name]")
        exit(-1)
    input_folder = sys.argv[1]
    username = sys.argv[2]
    app = QApplication(sys.argv)
    _ = Retoucher(input_folder, username)
    sys.exit(app.exec_())