##OpenCV.jl

The OpenCV (C++) interface for Julia.

<br>
OpenCV.jl aims to provide an interface for [OpenCV](http://opencv.org) computer vision applications (C++) directly in [Julia] (http://julia.readthedocs.org/en/latest/manual/).  It relies primarily on [Cxx.jl](https://github.com/Keno/Cxx.jl), the Julia C++ foreign function interface (FFI). OpenCV.jl comes bundled with the [Qt framework](http://qt-project.org/) - though not essential, it supports many convenient GUI functions.

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

Currently, OpenCV.jl has julia wrappers for most of the `core`, `imgproc`, `videoio` and `highgui` modules. Work is ongoing to wrap the rest of the modules and advanced object detection and tracking algorithms. (Most OpenCV C++ functions can be in principle accessed already with OpenCV.jl by using `@cxx` calls directly to C++).

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
Visit [Debian linux](http://milq.github.io/install-opencv-ubuntu-debian/), [Ubuntu](http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html) and [Windows](http://docs.opencv.org/trunk/doc/tutorials/introduction/windows_install/windows_install.html) for instructions on how to install OpenCV 3.0.  

##Basic interface
Below are illustrated only a small subset of all the API functions available (there are hundreds of functions).  

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

Clone, copy, convert images
```julia 
img2 = clone(img1) 
copy(img1, img2)
alpha=1; beta=0;  # scale and delta factors
convert(img1, img2, CV_8UC3, alpha=1, beta=0)  
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
@closeWindows(0,27,"")       # waits for ESC to close all windows 
@closeWindows(0,27,"Lena")   # waits for ESC to close "Lena" 
 
# custom functions
imdisplay(img, "Lena")  # optional: window resizing, key press, time
im2tile(imArray, "Tiled images")
```

#### Image processing 
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
```
Blur with a Gaussian filter, 5x5 kernel
```julia
gaussianBlur(img, dst, cvSize(5,5))
im2tile([img, dst], "Gaussian 5x5")
````

Convolution with a 7x7 kernel
```julia
kernel = normalizeKernel(ones(7,7,CV_32F), getKernelSum(kernel))
filter2D(img, dst, -1, kernel)
im2tile([img, dst], "Convolution 7x7")
```

Apply a laplacian filter
```julia
laplacian(img, dst, -1, 5)          # second-derivative aperture = 5
im2tile([img, dst], "laplacian")    
```

Apply a sobel operator (edge detection)
```julia
sobel(img, dst, -1, 1, 1, 3)        # dx = 1, dy = 1, kernel = 3x3
im2tile([img, dst], "sobel")
```

Binary thresholding
```julia
cvtColor(img, dst, COLOR_BGR2GRAY)
src = clone(dst)
threshold(src, dst, 120, 255, THRESH_BINARY)  # thresh = 0, max = 255
imdisplay(img, "Original", "ON")
imdisplay(dst, "Thresholded", "ON")
@closeWindows(0,27, "")
```

#### Interactive image processing and display (GUIs with trackbars)
Thresholding with adjustable trackbar. Below, I created a custom C++ class `iThreshold` (src/OpenCV_util.jl) that allows the user to open a window with trackbar, set an initial value, and threshold an input image interactively. Such classes can be easily modified and extended to support GUI display for interactive image processing in Julia. 
```julia
threshme = iThreshold()
setValue(threshme, 120) 
setWindow(threshme, "Threshold", "Interactive threshold")
showWindow(threshme, "Interactive threshold", img)    # img (see above)
val = getValue(threshme) 
```

#### Video acquistion
Display video stream from default camera
```julia
videoCapture()     # press ESC to stop    
```

## Advanced interfaces
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