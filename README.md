##OpenCV.jl

The OpenCV (C++) interface for Julia.

<br>
OpenCV.jl aims to provide an interface for [OpenCV](http://opencv.org) computer vision applications (C++) directly in [Julia] (http://julia.readthedocs.org/en/latest/manual/).  It relies primarily on [Cxx.jl](https://github.com/Keno/Cxx.jl), the Julia C++ foreign function interface (FFI). OpenCV.jl comes bundled with the [Qt framework](http://qt-project.org/) - though not essential, it supports many convenient GUI functions. The package also contains thin wrappers for common C++ classes (e.g., std::vector, std::string) to make the C++/Julia interface smoother.

The OpenCV API is described [here](http://docs.opencv.org/2.4/modules/refman.html). OpenCV.jl is organized along the following modules:

* **core:** <span> <span style="color:black"> Basic array structures (e.g., Mat), common functions (e.g, convertTo)
* **imgproc:** <span style="color:black"> Image processing (e.g.,image filtering, transformations, color space conversion)
* **videoio:** <span style="color:black">Video capturing and video codecs.
* **highgui:** <span style="color:black"> GUI capabilities
* **video:** <span style="color:black">Video analysis (e.g., motion estimation, background subtraction, and object tracking)
* calib3d: <span style="color:black">Camera calibration, object pose estimation, stereo correspondence, 3D reconstruction.
* features2d: <span style="color:black">Salient feature detectors, descriptors, and descriptor matchers.
* objdetect: <span style="color:black"> detection of objects (e.g., faces)
* gpu: <span style="color:black"> GPU-accelerated algorithms

Currently, OpenCV.jl has julia wrappers for the `core`, `imgproc`, `videoio`, `highgui` and `video` modules. Work is ongoing to wrap the rest of the modules including advanced object detection and tracking algorithms. (Most OpenCV C++ functions are already supported in OpenCV.jl by using `@cxx` calls directly to C++, with some caveats).

OpenCV.jl has OpenCL support for GPU image processing.  This has been made easier recently by a smooth and transparent interface (T-API). GPU-supported code can display improvements in processing speed up to 30 fold. This is invaluable for supporting real-time applications in Julia. See section below on how to implement GPU-enabled code in OpenCV.jl.

The OpenCV API is extensively documented - rather than repeating the entire documentation here, the primary focus is on implementation of image processing and computer vision algorithms to suport Julia applications.

##Installation

Install `julia 0.6.0` and `Cxx.jl` according to the following [instructions](https://github.com/Keno/Cxx.jl/blob/master/README.md). For Mac OSX, you can use the pre-compiled shared libraries (.dylib) and headers (.hpp) included in OpenCV.jl. However, you can also compile OpenCV from source with the instructions below.  

Note that successfully building `julia 0.6.0` may require upstream updates/fixes. Currently, on MacOS Sierra 10.12.3, I had to do the following:

- Created a `Make.user` file before building with the following content:

```
override LLVM_VER=3.9.0
override BUILD_LLVM_CLANG=1
override USE_LLVM_SHLIB=1
# Optional, but recommended
override LLVM_ASSERTIONS=1
```

- Merged commit `5373cdf821cf876332d7f8f1a3a2625598a33879` [5373cdf] into the master branch:
https://github.com/JuliaLang/julia/pull/18920/commits
https://github.com/Keno/Cxx.jl/issues/300
```
$ git fetch origin pull/18920/head:origin
# merge into local master branch
```

#### OSX
To compile OpenCV 3.2.0 (beta) on a 64-bit OSX system

```sh
# Clone OpenCV from GitHub master branch  #v0.3-beta
$ git clone https://github.com/Itseez/opencv.git opencv
$ git remote -v

# Create a build directory
$ mkdir build
$ cd build

# Install OpenCV >3.0 (master) *without CUDA*
BASIC INSTALLATION
$ cmake "Unix Makefile" -D CMAKE_PREFIX_PATH="/Users/Max/Qt/5.7/clang_64" -D WITH_OPENGL=ON -D CMAKE_OSX_ARCHITECTURES=x86_64 -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CUDA=OFF -D CMAKE_CXX_FLAGS="-std=c++11 -stdlib=libc++" -D CMAKE_EXE_LINKER_FLAGS="-std=c++11 -stdlib=libc++" -D TBB_INCLUDE_DIR="/usr/local/Cellar/tbb/4.3-20141023/include/tbb" -D TBB_LIB_DIR="/usr/local/Cellar/tbb/4.3-20141023/lib" -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_QT=OFF -D WITH_OPENEXR=OFF ..

$ make -j4
$ sudo make install

# Confirm installation of OpenCV shared libraries
$ pkg-config --libs opencv

# Confirm directory of OpenCV header files (.hpp)
$ cd /usr/local/include
$ ls opencv2

```


#### Windows and Linux
See links for info on how to install OpenCV on Debian linux [1](http://milq.github.io/install-opencv-ubuntu-debian/), Ubuntu [1](http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html) ,[2] (https://github.com/jayrambhia/Install-OpenCV) and [Windows](http://docs.opencv.org/trunk/doc/tutorials/introduction/windows_install/windows_install.html).  

####Download and run OpenCV.jl
```julia
Pkg.clone("git://github.com/maxruby/OpenCV.jl.git")
using OpenCV
```

##Basic interface
OpenCV contains hundreds of algorithms and functions. Most frequently used functions for image processing are already accessible in the current version of OpenCV.jl. For simplicity, here I focus on using  functions wrapped in OpenCV.jl.   

#### <span style="color:green"> Basic structures
**Points (Int, Float)**

```julia
cvPoint(10, 10)           # x, y
cvPoint2f(20.15, 30.55)
cvPoint2d(40.564, 12.444)
```

**Size and Scalar vectors (Int, Float)**

```julia
cvSize(300, 300)          # e.g., image width, height
cvSize2f(100.5, 110.6)
cvScalar(255,0,0)         # e.g., [B, G, R] color vector
```
**Ranges**

```julia  
range = cvRange(1,100)   # e.g., row 1 to 100
```
**Rectangle and rotated rectangle**

```julia
cvRect(5,5,300,300)      # x, y, width, height
# 300x300 rect, centered at (10.5, 10.5) rotated by 0.5 rad
cvRotatedRect(cvPoint2f(10.5, 10.5), cvSize2f(300,300), 0.5)
```

#### <span style="color:green">Creating, copying and converting images
**Mat array/image constructors: rows (height), columns (width)**

```julia
img0 = Mat()                             # empty
img1 = Mat(600, 600, CV_8UC1)            # 600x600 Uint8 gray

imgSize = cvSize(500, 250)    
img2 = Mat(imgSize, CV_8UC1)             # 500x250 Uint8 gray

imgColor = cvScalar(255, 0, 0)   
img3 = Mat(600, 600, CV_8UC3, imgColor)  # 600x600 Uint8 RGB (blue)
```

**Create a region of interest (ROI)**

```julia
const roi = cvRect(25, 25, 100, 100);     # create a ROI
img4 = Mat(img3, roi)
```

**Initialize arrays with zeros or ones**

```julia
zerosM(300,300, CV_8UC3)      # RGB filled with zeros
zerosM(imgSize, CV_8UC1)      # Gray filled with zeros  
ones(300,300, CV_8UC3)        # RGB filled with ones    
const sz = pointer([cint(5)]); # pointer to size of each dimension
ones(2, sz, CV_8UC3)          # 2 x sz        
```

**Create an identity matrix**

```julia
eye(300,300, CV_8UC3)         # 300x300 Uint8 (RGB)
```

**Clone, copy, convert, basic resizing**

```julia
img2 = clone(img1);
copy(img1, img2);
alpha=1; beta=0;  # scale and delta factors
convert(img1, img2, CV_8UC3, alpha, beta)
resizeMat(img1, 100, cvScalar(255,0, 0)) # 100 rows, 100 x 100
```

#### <span style="color:green">Operations on image arrays
**Addition and substraction**

```julia
img1 = Mat(300, 300, CV_8UC3, cvScalar(255, 0, 0));
img2 = Mat(300, 300, CV_8UC3, cvScalar(0, 0, 255));
img3 = imadd(img1, img2)
img4 = imsubstract(img1, img2)
```

**Matrix multiplication**

```julia
alpha = 1; # weight of the matrix
beta = 0;  # weight of delta matrix (optional)
flag = 0;  # GEMM_1_T  (transpose m1, m2 or m3)
m1 = ones(3, 3, CV_32F);    #Float32 image
m2 = ones(3, 3, CV_32F);
m3 = zerosM(3, 3, CV_32F);
gemm(m1, m2, alpha, Mat(), beta, m3, flag)
```

**Accessing pixels and indexing Mat arrays**<br>
Image pixels in Mat containers are arranged in a row-major order.<br>
For a grayscale image, e.g., pixels are addressed by row, col

|col 0| col 1| col 2|col 3| col m|
|:----- |:--:| :--:| :--:| :--:|  :--:|
| row 0 | 0,0|  0,1|  0,2|  0,3|   0,m|
| row 1 | 1,0|  1,1|  1,2|  1,3|   1,m|
| row 2 | 2,0|  2,1|  2,2|  2,3|   2,m|
| row n | n,0|  n,1|  n,2|  n,3|   n,m|

For RGB color images, each column has 3 values (actually BGR in Mat)

|col 0| col 1| col 2| col m |
|:----- |:--:| :--:| :--:| :--:|
| row 0 |<span style="color:blue">0,0,  <span style="color:green">0,0 <span style="color:red">0,0|  <span style="color:blue">0,1 <span style="color:green">0,1 <span style="color:red">0,1| <span style="color:blue">0,2 <span style="color:green">0,2 <span style="color:red">0,2| <span style="color:blue">0,m <span style="color:green">0,m <span style="color:red">0,m
| row 1 |  <span style="color:blue">1,0 <span style="color:green">1,0 <span style="color:red">1,0|  <span style="color:blue">1,1 <span style="color:green">1,1 <span style="color:red">1,1| <span style="color:blue">1,2 <span style="color:green">1,2 <span style="color:red">1,2| <span style="color:blue">1,m <span style="color:green">1,m <span style="color:red">1,m
| row 2 | <span style="color:blue">2,0 <span style="color:green">2,0 <span style="color:red">2,0|  <span style="color:blue">2,1 <span style="color:green">2,1 <span style="color:red">2,1| <span style="color:blue">2,2 <span style="color:green">2,2 <span style="color:red">2,2|<span style="color:blue">2,m <span style="color:green">2,m <span style="color:red">2,m
| row n| <span style="color:blue">n,0 <span style="color:green">n,0 <span style="color:red">n,0|  <span style="color:blue">n,1 <span style="color:green">n,1 <span style="color:red">n,1| <span style="color:blue">n,2 <span style="color:green">n,2 <span style="color:red">n,2| <span style="color:blue">n,m <span style="color:green">n,m <span style="color:red">n,m

**Getting and setting selected pixel values** <br>
**Method 1**: Access pixel values using `pixget` and `pixset` functions. Here we use the`Mat::at`class method - slow but safe, intended only for checking and setting small numbers of pixels (not for scanning through the entire image). To illustrate we draw random red pixels on a blue image (i.e., turn them yellow).

```julia
# Creat a blue image
img = Mat(300, 300, CV_8UC3, cvScalar(255, 0, 0));  
# get value for (row1,col1)
pixget(img, 1, 1)  
# create a C++ std::vector (BGR: Red) from a Julia vector
red = tostdvec([float(0), float(0), 255.0])
# turn random pixels yellow
for i=1:1000
    pixset(img, Int(round(rand()*rows(img))), Int(round(rand()*cols(img))), red)  
end
# Display (see description for these functions below)
imdisplay(img, "Random art")
closeWindows(0,27,"") # close by pressing ESC
```
**Method 2**:  Efficient pixel scanning and manipulation using pointers in C++. Functions `setgray` and `setcolor` can be used to scan an entire image and replace pixel values. For example, scanning & exchanging the BGR values for all pixels in a 1000x1000 image took approx. 16 ms. Such functions should be modified and optimized for each operation/algorithm.

```julia
# Creat a green image
img = Mat(1000, 1000, CV_8UC3, cvScalar(0, 255, 0));
color = tostdvec([cint(255), cint(55), cint(0)]) # fuchsia
setcolor(img, color)  
imdisplay(img, "coloring the fast way")
closeWindows(0,27,"")
```

#### <span style="color:green">Opening and saving images
**Read and write with full path/name**

```julia
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png");
img = imread(filename)
imwrite(joinpath(homedir(), "lena_copy.png"), img)
```
**Alternatively, open and save files with `Qt dialog` interface**

```julia
img = imread()
imwrite(img)
```

**Open image with`Images.jl` and convert to `OpenCV` Mat**<br>
Here we convert a binary image loaded with Images to a Mat image array

```julia
using Color, FixedPointNumbers
import Images, ImageView
using OpenCV

filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.jpeg")
image = Images.imread(filename)  # load with Images.jl
converted = convertToMat(image);
ImageView.view(image)
imdisplay(converted, "converted to OpenCV Mat")
closeWindows(0,27,"")
```
#### <span style="color:green">Access image properties
```julia
printMat(img)           # crude printout of the entire Mat (uchar only)
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

#### <span style="color:green">Basic image display (GUIs)
```julia
# original highgui functions
namedWindow("Lena", WINDOW_AUTOSIZE)
imshow("Lena", img)
moveWindow("Lena", 200, 200)
resizeWindow("Lena", 250, 250)
closeWindows(0,27,"Lena")   # waits until ESC key(27) press to close "Lena"

# custom display functions
imdisplay(img, "Lena")  # optional: window resizing, key press, time
im2tile(imArray, "Tiled images")  # => closeWindows
```

#### <span style="color:green">Image processing <span>
**Resize images**

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

**Select ROI and copy to another image**<br>

```julia
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
src = imread(filename)
dst = Mat(cint(rows(src)) + 100, cint(cols(src)) + 100, CV_8UC3, cvScalar(0, 255, 255))
roi = cvRect(Int64(10),Int64(10), Int64(cols(src)), Int64(rows(src)))
final = imreplace(src, dst, roi)  
namedWindow("original", 256)
namedWindow("replace", 256)
imshow("original", src)
imshow("replace", final)
closeWindows(0,27,"")
```

**Change color format**

```julia
dst = Mat()
cvtColor(img, dst, COLOR_BGR2GRAY)
```
**Blur with a normalized box filter**

```julia
blurred = clone(img)
blur(img, blurred, cvSize(5,5))
imdisplay(blurred, "Box filter")
closeWindows(0,27,"")
```
**Blur with a Gaussian filter, 5x5 kernel**

```julia
gaussianBlur(img, dst, cvSize(5,5))
im2tile([img, dst], "Gaussian 5x5")
closeWindows(0,27,"")
```
**Binary thresholding**

```julia
cvtColor(img, dst, COLOR_BGR2GRAY)
src = clone(dst)
threshold(src, dst, 120, 255, THRESH_BINARY)  # thresh = 0, max = 255
# other methods can be invoked with e.g., #THRESH_OTSU, THRESH_BINARY_INV flags
imdisplay(img, "Original")
imdisplay(dst, "Thresholded")
closeWindows(0,27, "")
```

**Convolution**

```julia
kernel = ones(5,5,CV_32F)
normkernel = normalizeKernel(ones(7,7,CV_32F), getKernelSum(kernel))
filter2D(img, dst, -1, normkernel)
im2tile([img, dst], "Convolution 7x7")
closeWindows(0,27,"")
```

**Laplacian filter**

```julia
laplacian(img, dst, -1, 5)          # second-derivative aperture = 5
im2tile([img, dst], "laplacian")  
closeWindows(0,27,"")  
```

**Sobel operator (edge detection)**

```julia
sobel(img, dst, -1, 1, 1, 3)        # dx = 1, dy = 1, kernel = 3x3
im2tile([img, dst], "sobel")
closeWindows(0,27,"")
```

**Canny edge detection**

```julia
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
img = imread(filename)
edges = Mat()
threshold1 = 125.0; threshold2 = 350.0
apertureSize = 3; L2gradient = false
Canny(img, edges, threshold1, threshold2, apertureSize, L2gradient)
imdisplay(edges, "canny")
closeWindows(0,27,"")
```

**Image overlay (linear blending)**

```julia
filename2 = joinpath(Pkg.dir("OpenCV"), "./test/images/mandrill.jpg")
img2 = imread(filename2)
dst = Mat()
alpha = 0.5; beta = 0.2; gamma = 0.6
addWeighted(img, alpha, img2, beta, gamma, dst)
imdisplay(dst, "overlay")
closeWindows(0,27,"")
```

**Image sharpening**

```julia
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
img = imread(filename)
dst = Mat()
sharpened = Mat()
gaussianBlur(img, dst, cvSize(0, 0), 0.2)
addWeighted(img, 1.5, dst, -0.3, float(0), sharpened)
im2tile([img, dst], "sharpened")
closeWindows(0,27,"")
```

#### <span style="color:green">Video acquistion, streaming and writing
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

**To get video properties, use `getVideoId`**

```julia
cam = videoCapture(CAP_ANY)   # cv::VideoCapture
getVideoId(cam, CAP_PROP_FOURCC)   # or set to -1 (uncompressed AVI)
```
**To set video properties, use `setVideoId`**

```julia
setVideoId(cam, CAP_PROP_FPS, 10.0)
```
**Close the camera input**

```julia
release(cam)
```

**Stream videos from the web (requires http link to source file)**

```julia
vid = "http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8"
webstream(vid)
```

**Write the video stream to disk**

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

#### <span style="color:green">Interactive image processing<br>
**Videoprocessor** is a basic example of a custom C++ class I wrote to support interactive image processing and display with OpenCV In Julia. It may be useful for testing custom C++ image processing algorithms. It accepts single image or video files and video streams. The basic concept is to create a class for each image processing operation in `Videoprocessor` (e.g., Thresholding).  Currently it supports, brightness, contrast and simple thresholding filters.  You can retrieve the final values for each of the filter operations as shown below. For more details, see `src/Videoprocessor.jl`.

```julia
# single image file
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
processes = stdvec(cint(0),cint(0))
stdpush!(processes, BRIGHTNESS)# or CONTRAST/THRESHOLD
params = videoprocessor(processes, "Demo", filename,-1, 0, 30, 30, 120, 255, THRESH_BINARY, false)
at(params,0)  # BRIGHTNESS

# video stream
processes = stdvec(cint(0),cint(0))
stdpush!(processes, BRIGHTNESS)
stdpush!(processes, THRESHOLD)
params = videoprocessor(processes, "Videoprocessor")
at(params,0)  # BRIGHTNESS
at(params,1)  # THRESHOLD
```

#### <span style="color:green">Text and drawing functions<br>
Put text on image

```julia
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
img = imread(filename)
putText(img, "Hello Lena!", cvPoint(40,40), FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(255,0,0), 1, LINE_AA, false)
imdisplay(img, "Text")
closeWindows(0,27,"")
```
Draw geometric shapes (circles, rectangles, etc)

```julia
center = cvPoint(260,275)
radius = 30
color = cvScalar(0,0,255)  #red
thickness=4
lineType=LINE_AA
shift = 0
circle(img, center, radius, color, thickness,lineType, shift)
rectangle(img, cvPoint(30,30), cvPoint(150,150), cvScalar(255,0,0), thickness, lineType, shift)
imdisplay(img, "Drawing")
closeWindows(0,27,"")
```

## Advanced interfaces
#### <span style="color:green"> GPU processing with OpenCL
OpenCV.jl can be accelerated several fold by processing on the GPU with the OpenCL transparent API (T-API). The only requirement is to declare the image/array as `cv::UMat` (universal Mat) instead of `cv::Mat`. For example, a simple RGB to gray image conversion can run 10 times faster with GPU compared to CPU (here I used an NVIDIA GTX-Force 330M 512MB, CC 1.2) in OpenCV.jl:

```julia
Declare the Mat and UMat (1000x1000 RGB) source and initialize target images
julia> srcMat = Mat(1000, 1000, CV_8UC3, cvScalar(0, 255, 0));
julia> srcUMat = UMat(1000, 1000, CV_8UC3, cvScalar(0, 255, 0));
julia> dstMat = Mat()
julia> dstUMat = UMat()

CPU
julia> @time(cvtColor(srcMat, dstMat, COLOR_BGR2GRAY))
elapsed time: 0.00164426 seconds (80 bytes allocated)

GPU
julia> @time(cvtColor(srcUMat, dstUMat, COLOR_BGR2GRAY))
elapsed time: 0.000149589 seconds (80 bytes allocated)
```

## Demos
The scripts in `test/jl/tests.jl` illustrate how to use basic OpenCV functions directly in Julia. Demos in `test/cxx/demos.jl` contain both basic and advanced C++ scripts wrapped with Cxx. You can execute `run_tests()` to check these examples, including basic image creation, conversion, thresholding, live video, trackbars, histograms, drawing, and object tracking.

## Applications in computer vision
There is a rich collection of advanced algorithms/modules for computer vision implemented in OpenCV that are likely to be added in the future. A number of them are found in [opencv-contrib](github.com/Itseez/opencv_contrib/tree/master/module) e.g.,   

* opencv_face: Face detection
* opencv_optflow: Optical flow
* opencv_reg: Image registration
* opencv_text: Scene Text Detection and Recognition
* opencv_tracking: Long-term optical tracking API
* opencv_xfeatures2d: Extra 2D Features Framework
* opencv_ximgproc: Extended Image Processing
* opencv_xobjdetect: Integral Channel Features Detector Framework
* opencv_xphoto: Color balance/Denoising/Inpainting


## Extended documentation
Feel free to send questions, comments or file issues here.  Extending the documentation is planned in the context of more specialized applications.

## Known issues

- Import errors
     - flann.hpp (included in opencv2/opencv.hpp) throws an error in the Julia REPL due to `typeid` declarations which are not compatible with the `rtti` flag
