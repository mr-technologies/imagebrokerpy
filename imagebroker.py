#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gc
import sys
import cv2
import json
import numpy
import iffsdkpy

from pathlib import Path
from threading import Lock

from iffsdkpy import Chain


def load_config(filename):
    with open(filename, 'r') as cfg_file:
        config = json.load(cfg_file)

    if 'IFF' not in config:
        print("Invalid configuration provided: missing `IFF` section", file=sys.stderr)
        sys.exit(1)

    if 'chains' not in config:
        print("Invalid configuration provided: missing `chains` section", file=sys.stderr)
        sys.exit(1)

    if len(config['chains']) == 0:
        print("Invalid configuration provided: section `chains` must not be empty", file=sys.stderr)
        sys.exit(1)

    if not isinstance(config['chains'], list):
        print("Invalid configuration provided: section `chains` must be an array", file=sys.stderr)
        sys.exit(1)

    return config


def create_chains(chains_config):
    def error_handler(element_id, error_code):
        pass

    return list(map(
        lambda chain: Chain(
            json.dumps(chain),
            error_handler
        ),
        chains_config))

render_image = numpy.empty(0)

def main():
    config = load_config(Path(__file__).stem + '.json')

    iff_config = json.dumps(config['IFF'])
    iffsdkpy.initialize(iff_config)

    chains = create_chains(config['chains'])

    copy_lock = Lock()

    def image_handler(image_memview, metadata):
        global render_image
        src_image = numpy.asarray(image_memview.cast('B', shape=[metadata.height, metadata.width, 4]))
        with copy_lock:
            render_image = src_image.copy()

    chains[0].set_export_callback("exporter", image_handler)
    chains[0].execute('{"exporter": {"command": "on"}}')

    global render_image
    size_set = False
    max_window_width  = 1280
    max_window_height = 1024
    window_name = "IFF SDK Image Broker Sample"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    iffsdkpy.log(iffsdkpy.log_level.info, Path(__file__).stem, "Press Esc to terminate the program")

    while True:
        with copy_lock:
            if render_image.size > 0:
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
                cv2.imshow(window_name, render_image)

        if cv2.waitKey(10) & 0xFF == 27:
            iffsdkpy.log(iffsdkpy.log_level.info, Path(__file__).stem, "Esc key was pressed, stopping the program")
            break

    chains[0].execute('{"exporter": {"command": "off"}}')

    cv2.destroyAllWindows()

    del chains
    gc.collect()

    iffsdkpy.finalize()


if __name__ == '__main__':
    main()
