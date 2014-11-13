##**OpenCV.jl**


OpenCV.jl aims to provide an interface for [OpenCV](http://opencv.org) computer vision applications (C++) directly in [Julia] (http://julia.readthedocs.org/en/latest/manual/).  It relies primarily on [Cxx.jl](https://github.com/Keno/Cxx.jl), the Julia C++ foreign function interface (FFI) developed by Keno Fisher. 

A full documentation of the latest OpenCV API is available [here](http://docs.opencv.org/trunk/modules/core/doc/intro.html). The following modules are available:

* **core:** <span> <span style="color:black"> Basic array structures (e.g., Mat), common functions (e.g, convertTo) 
* **imgproc:** <span style="color:black"> Image processing (e.g.,image filtering, transformations, color space conversion)
* **video:** <span style="color:black">Video analysis (e.g., motion estimation, background subtraction, and object tracking)
* **calib3d:** <span style="color:black">Camera calibration, object pose estimation, stereo correspondence, 3D reconstruction.
* **features2d:** <span style="color:black">Salient feature detectors, descriptors, and descriptor matchers.
* **objdetect:** <span style="color:black"> detection of objects (e.g., faces)
* **highgui:** <span style="color:black"> GUI capabilities
* **videoio:** <span style="color:black">Video capturing and video codecs.
* **gpu:** <span style="color:black"> GPU-accelerated algorithms 

### Installation


Install Julia 0.4-dev and Cxx.jl according to the following [instructions](https://github.com/Keno/Cxx.jl/blob/master/README.md). OpenCV.jl comes bundled with OpenCV 3.0 (GitHub master repository) C++ header files and the libopencv shared libraries pre-compiled on Mac OSX 10.9.5. You can also compile it from source with the instructions below.  **Note:** There are many reports of build erros for OpenCV v3.0 with CUDA (e.g., linking shared libraries, etc). Here we use OpenCV built without CUDA support.

##### OSX
```c++
Pkg.clone("git://github.com/maxruby/OpenCV.jl.git")
using OpenCV
```

##### Windows and Linux

Compile OpenCV from source (e.g., below). Here we assume OpenCV headers and shared libraries are installed in `/usr/local/include/` and `/usr/local/lib/`, respectively.  Otherwise, change **cvlibdir** and **cvheaderdir** in /src/OpenCV.jl first. 

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

### **Using OpenCV.jl**

##### Creating images with cvMat
```python
# empty image
img0 = cvMat()

# Uint8 gray image, 600x600
img1 = cvMat(600, 600, CV_8UC1)

# Uint8 gray image, 500x250
imgSize = cvSize(500, 250)
img2 = cvMat(imgSize, CV_8UC1)

# Uint8 blue image, 600x600
imgColor = cvScalar(255, 0, 0)
img3 = cvMat(600, 600, CV_8UC3, imgColor)
```

##### Basic image information
```python 
# returns size_t: number of array elements
total(img)
# dimensions
dims(img)
# cvSize(cols, rows), if matrix > 2d, size = (-1,-1)
size(img)
# rows
rows(img)
# cols
cols(img)
# Check if matrix elements are stored continuously (no gaps)
isContinuous(img)
# matrix element size in bytes (size_t)
elemSize(img)
# Mat type identifier (e.g., CV_8UC1)
cvtype(img)
# identifier of the matrix element depth
depth(img)
# number of matrix channels
channels(img)
# checks if the array is empty (true/false)
empty(img)
# uchar* or typed pointer for matrix row
ptr(img, 10)

```

##### Access and manipulate cvMat arrays
```python
# Create a ROI
const roi = cvRect(25, 25, 100, 100) # create a ROI

```

##### Loading images
```python
# get filename
filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
# load file
img = imread(pointer(filename))


```
##### Graphical user interface (GUI)
```python
imdisplay(img, "Lena", WINDOW_AUTOSIZE, 0, 27)
```
##### Trackbars
##### Video acquistion
##### Image processing
##### Object detection


### **Demos**

```python
# Primarily native C++ OpenCV scripts in Julia

julia> run_tests()
Select a demo from the following options: 
	1) CreateImage
	2) ImageConversion
	3) Thresholding
	4) LiveVideo
	5) setVideoProperties
	6) LiveVideoWithTrackbars
	7) displayHistogram
	8) Drawing
	9) im2dots
	10) ObjectTracking

Select and run the demo at the julia prompt: 
e.g., run_demo(N)
```

