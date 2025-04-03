#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IFF SDK samples (https://mr-te.ch/iff-sdk) are licensed under MIT License.
#
# Copyright (c) 2022-2025 MRTech SK, s.r.o.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# std
import gc
import json
from pathlib import Path
import sys
from threading import Condition

# OpenCV
import cv2

# NumPy
import numpy

# IFF SDK
import iffsdkpy
from iffsdkpy import Chain


def load_config(filename):
    with open(filename, 'r') as cfg_file:
        config = json.load(cfg_file)

    if 'IFF' not in config:
        sys.exit("Invalid configuration provided: missing `IFF` section")

    if 'chains' not in config:
        sys.exit("Invalid configuration provided: missing `chains` section")

    if len(config['chains']) == 0:
        sys.exit("Invalid configuration provided: section `chains` must not be empty")

    if not isinstance(config['chains'], list):
        sys.exit("Invalid configuration provided: section `chains` must be an array")

    return config


def create_chains(chains_config):

    def error_handler(element_id, error_code):
        iffsdkpy.log(iffsdkpy.log_level.error, Path(__file__).stem, f"Chain element `{element_id}` reported an error: {error_code}")

    return list(map(
        lambda chain: Chain(
            json.dumps(chain),
            error_handler
        ),
        chains_config
    ))

render_image = None

def main():
    config = load_config(Path(__file__).stem + '.json')

    iff_config = json.dumps(config['IFF'])
    iffsdkpy.initialize(iff_config)

    chains = create_chains(config['chains'])

    copy_cv = Condition()

    def image_handler(image_memview, metadata):
        global render_image
        src_image = numpy.asarray(image_memview.cast('B', shape=[metadata.height, metadata.width, 4]))
        tmp_image = cv2.cvtColor(src_image, cv2.COLOR_RGBA2BGRA)
        with copy_cv:
            render_image = tmp_image
            copy_cv.notify()

    chains[0].set_export_callback('exporter', image_handler)
    chains[0].execute('{"exporter": {"command": "on"}}')

    global render_image
    size_set = False
    max_window_width  = 1280
    max_window_height = 1024
    window_name = "IFF SDK Image Broker Sample"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    iffsdkpy.log(iffsdkpy.log_level.info, Path(__file__).stem, "Press Esc to terminate the program")

    while True:
        tmp_image = None
        with copy_cv:
            if copy_cv.wait_for(lambda: render_image is not None, 1):
                if not size_set:
                    height, width, c = render_image.shape
                    if width > max_window_width:
                        height = round(max_window_width / (width / height))
                        width = max_window_width
                    if height > max_window_height:
                        width = round(max_window_height * (width / height))
                        height = max_window_height
                    cv2.resizeWindow(window_name, width, height)
                    size_set = True
                tmp_image = render_image
                render_image = None

        if tmp_image is not None:
            cv2.imshow(window_name, tmp_image)

        if cv2.pollKey() & 0xFF == 27:
            iffsdkpy.log(iffsdkpy.log_level.info, Path(__file__).stem, "Esc key was pressed, stopping the program")
            break

    cv2.destroyWindow(window_name)

    del chains
    gc.collect()

    iffsdkpy.finalize()

if __name__ == '__main__':
    main()
