#OpenCV.jl

The OpenCV (C++) interface for Julia.

<br>
OpenCV.jl aims to provide an interface for [OpenCV](http://opencv.org) computer vision applications (C++) directly in [Julia] (http://julia.readthedocs.org/en/latest/manual/).  It relies primarily on [Cxx.jl](https://github.com/Keno/Cxx.jl), the Julia C++ foreign function interface (FFI). OpenCV.jl comes bundled with the [Qt framework](http://qt-project.org/) - though not essential, it supports many convenient GUI functions. The package also contains thin wrappers for common C++ classes (e.g., std::vectors) to make C++/Julia interface smoother in the future.  

The OpenCV API is described [here](http://docs.opencv.org/trunk/modules/core/doc/intro.html). OpenCV.jl is organized along the following modules:

* **core:** <span> <span style="color:black"> Basic array structures (e.g., Mat), common functions (e.g, convertTo) 
* **imgproc:** <span style="color:black"> Image processing (e.g.,image filtering, transformations, color space conversion)
* **videoio:** <span style="color:black">Video capturing and video codecs.
* **highgui:** <span style="color:black"> GUI capabilities
* video: <span style="color:black">Video analysis (e.g., motion estimation, background subtraction, and object tracking)
* calib3d: <span style="color:black">Camera calibration, object pose estimation, stereo correspondence, 3D reconstruction.
* features2d: <span style="color:black">Salient feature detectors, descriptors, and descriptor matchers.
* objdetect: <span style="color:black"> detection of objects (e.g., faces)
* gpu: <span style="color:black"> GPU-accelerated algorithms 

Currently, OpenCV.jl has julia wrappers for most of the `core`, `imgproc`, `videoio` and `highgui` modules. Work is ongoing to wrap the rest of the modules including advanced object detection and tracking algorithms. (Most OpenCV C++ functions are already supported in OpenCV.jl by using `@cxx` calls directly to C++).

##Installation

Install `julia 0.4.0-dev` and `Cxx.jl` according to the following [instructions](https://github.com/Keno/Cxx.jl/blob/master/README.md). For Mac OSX, you can use the pre-compiled shared libraries (.dylib) and headers (.hpp) included in OpenCV.jl. However, you can also compile OpenCV from source with the instructions below. 

#### OSX
```julia
Pkg.clone("git://github.com/maxruby/OpenCV.jl.git")
using OpenCV
```

To compile OpenCV 3.0 on a 64-bit OSX system 

```sh
# Clone OpenCV from GitHub master branch  #v0.3
$ git clone https://github.com/Itseez/opencv.git opencv
$ git remote -v 

# Create a build directory 
$ mkdir build
$ cd build

# Install OpenCV >3.0 (master) *without CUDA*
$ cmake "Unix Makefile" -D CMAKE_OSX_ARCHITECTURES=x86_64 -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CUDA=OFF -D CMAKE_CXX_FLAGS="-std=c++11 -stdlib=libc++" -D CMAKE_EXE_LINKER_FLAGS="-std=c++11 -stdlib=libc++" ..
$ make -j4
$ make install  #prepend sudo if necessary

# Confirm installation of OpenCV shared libraries
$ pkg-config --libs opencv

# Confirm directory of OpenCV header files (.hpp)
$ cd /usr/local/include
$ ls opencv2
``` 

#### Windows and Linux
See links for info on how to install OpenCV on Debian linux [1](http://milq.github.io/install-opencv-ubuntu-debian/), Ubuntu [1](http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html) ,[2] (https://github.com/jayrambhia/Install-OpenCV) and [Windows](http://docs.opencv.org/trunk/doc/tutorials/introduction/windows_install/windows_install.html).  

##Basic interface
The OpenCV.jl API is very large, containing hundreds of functions.  Here is an illustration of a subset of commonly used functions.   

#### Basic structures
Points (Int, Float)
```julia 
cvPoint(10, 10)           # x, y
cvPoint2f(20.15, 30.55)
cvPoint2d(40.564, 12.444)
```
Size and Scalar vectors (Int, Float)
```julia 
cvSize(300, 300)          # e.g., image width, height 
cvSize2f(100.5, 110.6)
cvScalar(255,0,0)         # e.g., [B, G, R] color vector
```
Ranges 
```julia  
range = cvRange(1,100)   # e.g., row 1 to 100/Users/maximilianosuster
```
Rectangle and rotated rectangle
```julia 
cvRect(5,5,300,300)      # x, y, width, height
# 300x300 rect, centered at (10.5, 10.5) rotated by 0.5 rad
cvRotatedRect(cvPoint2f(10.5, 10.5), cvSize2f(300,300), 0.5)
```

#### Creating, copying and converting images 
cv::Mat array/image constructors -- rows (height), columns (width)
```julia 
img0 = Mat()                             # empty
img1 = Mat(600, 600, CV_8UC1)            # 600x600 Uint8 gray 

imgSize = cvSize(500, 250)     
img2 = Mat(imgSize, CV_8UC1)             # 500x250 Uint8 gray

imgColor = cvScalar(255, 0, 0)   
img3 = Mat(600, 600, CV_8UC3, imgColor)  # 600x600 Uint8 RGB (blue)
```

Create a region of interest(ROI)
```julia 
const roi = cvRect(25, 25, 100, 100)     # create a ROI
img4 = Mat(img3, roi)
```

Initialize arrays with zeros or ones
```julia 
zerosM(300,300, CV_8UC3)      # RGB filled with zeros
zerosM(imgSize, CV_8UC1)      # Gray filled with zeros  
ones(300,300, CV_8UC3)        # RGB filled with ones    
const sz = pointer([cint(5)]) # pointer to size of each dimension 
ones(2, sz, CV_8UC3)          # 2 x sz        
```

Create an identity matrix
```julia 
eye(300,300, CV_8UC3)         # 300x300 Uint8 (RGB)
```

Clone, copy, convert, basic resizing
```julia 
img2 = clone(img1) 
copy(img1, img2)
alpha=1; beta=0;  # scale and delta factors
convert(img1, img2, CV_8UC3, alpha, beta)  
resizeMat(img1, 100, cvScalar(255,0, 0)) # 100 rows, 100 x 100
```

#### Accessing basic image properties
```julia 
total(img)              # number of array elements
dims(img)               # dimensions
size(img)               # cvSize(columns, rows)
rows(img)               # rows
cols(img)               # columns
isContinuous(img)       # is stored continuously (no gaps)? 
elemSize(img)           # element size in bytes (size_t)
cvtypeval(img)          # Mat type identifier (number)
cvtypelabel(img)        # Mat type label (e.g., CV_8UC1)
depth(img)              # element depth
channels(img)           # number of matrix channels
empty(img)              # is array is empty? (true/false)
ptr(img, 10)            # uchar* or typed pointer for matrix row
```

#### Operations on image arrays
Addition and substration
```julia
img1 = Mat(300, 300, CV_8UC3, cvScalar(255, 0, 0))
img2 = Mat(300, 300, CV_8UC3, cvScalar(0, 0, 255))
img3 = imadd(img1, img2)
img4 = imsubstract(img1, img2)
```


Matrix multiplication
```julia
alpha = 1; # weight of the matrix 
beta = 0;  # weight of delta matrix (optional) 
flag = 0;  # GEMM_1_T  (transpose m1, m2 or m3)
m1 = ones(3, 3, CV_32F)    #Float32 image
m2 = ones(3, 3, CV_32F)
m3 = zerosM(3, 3, CV_32F)
gemm(m1, m2, alpha, Mat(), beta, m3, flag)
```

Array indexing
```julia
# http://docs.opencv.org/trunk/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#howtoscanimagesopencv


```

#### Opening and saving images 
Read and write with full path/name
```julia 
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
img = imread(filename)
imwrite(joinpath(homedir(), "lena_copy.png"), img)
```
Alternatively, open and save files with `Qt dialog` interface
```julia 
img = imread()
imwrite(img)
```

#### Basic image display (GUIs)
```julia 
# original highgui functions 
imshow("Lena", img)
moveWindow("Lena", 200, 200)
resizeWindow("Lena", 250, 250)
closeWindows(0,27,"Lena")   # waits for ESC to close "Lena" 
 
# custom functions
imdisplay(img, "Lena")  # optional: window resizing, key press, time
im2tile(imArray, "Tiled images")
```

#### Image processing 
Resize images
```julia
dst = clone(img)
resize(img, dst, cvSize(250,250), float(0), float(0), INTER_LINEAR)
imdisplay(img, "Lena")
imdisplay(dst, "Resized Lena")
closeWindows(0,27,"")  # waits for ESC to close all windows

interpolation options:
# INTER_NEAREST - a nearest-neighbor interpolation
# INTER_LINEAR - a bilinear interpolation (used by default)
# INTER_AREA - resampling using pixel area relation
# INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
# NTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
```

Change color format
```julia
dst = Mat()
cvtColor(img, dst, COLOR_BGR2GRAY)
```
Blur with a normalized box filter, 5x5 kernel
```julia
blurred = clone(img)
blur(img, blurred, cvSize(5,5))
imdisplay(blurred, "Box filter")
closeWindows(0,27,"")
```
Blur with a Gaussian filter, 5x5 kernel
```julia
gaussianBlur(img, dst, cvSize(5,5))
im2tile([img, dst], "Gaussian 5x5")
closeWindows(0,27,"")
````

Convolution with a 7x7 kernel
```julia
kernel = ones(5,5,CV_32F)
normkernel = normalizeKernel(ones(7,7,CV_32F), getKernelSum(kernel))
filter2D(img, dst, -1, normkernel)
im2tile([img, dst], "Convolution 7x7")
closeWindows(0,27,"")
```

Apply a laplacian filter
```julia
laplacian(img, dst, -1, 5)          # second-derivative aperture = 5
im2tile([img, dst], "laplacian")  
closeWindows(0,27,"")  
```

Apply a sobel operator (edge detection)
```julia
sobel(img, dst, -1, 1, 1, 3)        # dx = 1, dy = 1, kernel = 3x3
im2tile([img, dst], "sobel")
closeWindows(0,27,"")
```

Binary thresholding
```julia
cvtColor(img, dst, COLOR_BGR2GRAY)
src = clone(dst)
threshold(src, dst, 120, 255, THRESH_BINARY)  # thresh = 0, max = 255
 #THRESH_OTSU
 #THRESH_BINARY_INV
 #THRESH_TRUNC
 #THRESH_TOZERO
 #THRESH_TOZERO_INV
imdisplay(img, "Original")
imdisplay(dst, "Thresholded")
closeWindows(0,27, "")
```

#### Interactive image processing and display (GUIs with trackbars)
Thresholding with adjustable trackbar. Below, I created a custom C++ class `iThreshold` (src/OpenCV_util.jl) that allows the user to open a window with trackbar, set an initial value, and threshold an input image interactively. Such classes can be readily modified and extended to support real-time image processing in Julia. 
```julia
threshme = iThreshold()
setValue(threshme, 120) 
setWindow(threshme, "Threshold", "Interactive threshold")
showWindow(threshme, "Interactive threshold", img)    # img (see above)
val = getValue(threshme) 
```

#### Video acquistion, streaming and writing
Basic video stream display from default camera. All GUI classes/functions (e.g., videoCapture) can be easily called from OpenCV.jl to build new custom video acquisition functions.  
```julia
videocam()     # press ESC to stop  
```

The following identifiers can be used (depending on backend) to get/set video properties:
```
append "CAP_PROP_" to id below
POS_MSEC       Current position of the video file (msec or timestamp)  
POS_FRAMES     0-based index of the frame to be decoded/captured next
POS_AVI_RATIO  Relative position of the video file: 0 - start of the film, 1 - end of the film
FRAME_WIDTH    Width of the frames in the video stream
FRAME_HEIGHT   Height of the frames in the video stream
FPS            frame rate
FOURCC         4-character code of codec
FRAME_COUNT    Number of frames in the video file
FORMAT         Format of the Mat objects returned by retrieve()
MODE           Backend-specific value indicating the current capture mode
BRIGHTNESS     Brightness of the image (only for cameras)
CONTRAST       Contrast of the image (only for cameras)
SATURATION     Saturation of the image (only for cameras)
HUE            Hue of the image (only for cameras)
GAIN           Gain of the image (only for cameras)
EXPOSURE       Exposure (only for cameras)
CONVERT_RGB    Boolean flags indicating whether images should be converted to RGB
WHITE_BALANCE  Currently not supported
RECTIFICATION  Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
```

To get video properties, use `getVideoId` 
```julia
cam = videoCapture(CAP_ANY)   # cv::VideoCapture 
getVideoId(cam, CAP_PROP_FOURCC)   # or set to -1 (uncompressed AVI)
```
To set video properties, use `setVideoId` 
```julia
setVideoId(cam, CAP_PROP_FPS, 10.0) 
```
Close the camera input
```julia
release(cam)
```

Stream videos from the web (requires http link to source file)
```julia
vid = "http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8"
webstream(vid)
```

Write the video stream to disk
```julia
cam = videoCapture(vid)
filename = joinpath(homedir(), "myvid.avi")
fps = 25.0
nframes = 250            # default -> nframes = 0, to stop press ESC
frameSize=cvSize(0,0)    # input = output frame size
codec = -1               # fourcc(CV_FOURCC_IYUV)
isColor = true           # color
device = CAP_ANY         # default device
videoWrite (cam, filename, fps, nframes, frameSize, codec, true) 
```
  
## Advanced applications
There is a rich collection of advanced algorithms/modules for computer vision implemented in OpenCV. A number of them are found in [opencv-contrib](github.com/Itseez/opencv_contrib/tree/master/module):  

* opencv_face: Face detection
* opencv_optflow: Optical flow
* opencv_reg: Image registration
* opencv_text: Scene Text Detection and Recognition
* opencv_tracking: Long-term optical tracking API
* opencv_xfeatures2d: Extra 2D Features Framework
* opencv_ximgproc: Extended Image Processing
* opencv_xobjdetect: Integral Channel Features Detector Framework
* opencv_xphoto: Color balance/Denoising/Inpainting

## Demos
Use `run_tests()` to try one of the 10 sample demos available: basic image creation, conversion, thresholding, live video, trackbars, histograms, drawing, and object tracking.


