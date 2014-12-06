################################################################################################
#
# OpenCV_modules.jl
#
# Load the OpenCV 3.0 modules
# Currently the following are enabled:
#
# core.     The Core Functionality
#            * Basic structures
#            * Operations on Arrays
#            * Clustering
#            * Utility and System Functions and Macros
#
# imgproc.  Image Processing
#            * Image Filtering
#            * Geometric Image Transformations
#            * Miscellaneous Image Transformations
#            * Drawing Functions
#            * ColorMaps
#            * Histograms
#            * Structural Analysis and Shape Descriptors
#            * Motion Analysis and Object Tracking
#            * Feature Detection
#            * Object Detection
#
# imgcodecs. Image file reading and writing
#            * Reading and Writing Images
#
# videoio.   Media I/O
#            * Reading and Writing Video
#
# highgui.   High-level GUI and Media I/O
#            * User Interface
#
# TO DO:
# video.      Video Analysis
# calib3d.    Camera Calibration and 3D Reconstruction
# features2d. 2D Features Framework
# objdetect.  Object Detection
# ml.         Machine Learning
# flann.      Clustering and Search in Multi-Dimensional Spaces
# photo.      Computational Photography
# shape.      Shape Distance and Matching
# superres.   Super Resolution
# videostab.  Video Stabilization
# viz.        3D Visualizer
#             Binary descriptors for lines extracted from an image
# optflow.    Optical Flow Algorithms
# reg.        Image Registration
#             Saliency API
# tracking.   Tracking API
# xfeatures2d.Extra 2D Features Framework
# ximgproc.   Extended Image Processing
# xobjdetect. Extended object detection.
# xphoto.     Additional photo processing algorithms
#
#
#
#
# Sources:
# http://docs.opencv.org/trunk/modules/refman.html
# http://physics.nyu.edu/grierlab/manuals/opencv/namespacecv.html#a346f563897249351a34549137c8532a0
####################################################################################################

# Core
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_core.jl"))
# Custom module for Mat arrays (e.g., get and set methods)
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_Mat.jl"))

# Image processing
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_imgproc.jl"))

# Image file reading and writing
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_imgcodecs.jl"))

# Reading and Writing Video
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_videoio.jl"))

# High-level GUI and Media I/O
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_highgui.jl"))

# Support for conversion and manipulation of images from external packages
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_ImagesSupport.jl"))




