# `imagebrokerpy`

`imagebroker.py` application demonstrates how to export images to the user code across IFF SDK library boundaries.
Application uses Python API provided by IFF SDK.
It comes with example configuration file (`imagebroker.json`) providing the following functionality:

* acquisition from XIMEA camera
* color pre-processing on GPU:
  * black level subtraction
  * histogram calculation
  * white balance
  * demosaicing
  * color correction
  * gamma
  * image format conversion
* automatic control of exposure time and white balance
* image export to the client code

Additionally example code renders images on the screen using [OpenCV](https://opencv.org/) library, which should be installed in the system (minimal required version is 4.5.2).
