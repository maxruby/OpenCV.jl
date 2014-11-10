##**OpenCV.jl**




## Introduction 

OpenCV.jl aims to provide an interface for [OpenCV](http://opencv.org) computer vision applications (C++) directly in [Julia] (http://julia.readthedocs.org/en/latest/manual/).  It relies primarily on [Cxx.jl](https://github.com/Keno/Cxx.jl), the Julia C++ foreign function interface (FFI) developed by Keno Fisher. 

A full documentation of the latest OpenCV API is available [here](http://docs.opencv.org/trunk/modules/core/doc/intro.html). These are the modules:

* **core:** <span> <span style="color:black"> Basic array structures (e.g., Mat), common functions (e.g, convertTo) 
* **imgproc:** <span style="color:black"> Image processing (e.g.,image filtering, transformations, color space conversion)
* **video:** <span style="color:black">Video analysis (e.g., motion estimation, background subtraction, and object tracking)
* **calib3d:** <span style="color:black">Camera calibration, object pose estimation, stereo correspondence, 3D reconstruction.
* **features2d:** <span style="color:black">Salient feature detectors, descriptors, and descriptor matchers.
* **objdetect:** <span style="color:black"> detection of objects (e.g., faces)
* **highgui:** <span style="color:black"> GUI capabilities
* **videoio:** <span style="color:black">Video capturing and video codecs.
* **gpu:** <span style="color:black"> GPU-accelerated algorithms 

Feedback is welcome.  

## Installation

#### OpenCV

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

# Confirm installation of OpenCV shared libraries (.dylib in OSX) 
$ pkg-config --libs opencv

# Confirm directory of OpenCV header files (.hpp)
$ cd /usr/local/include
$ ls opencv2
``` 
Note: There are many reports of build erros for OpenCV v3.0 with CUDA (e.g., linking shared libraries, etc). If using v0.3, I suggest building shared libraries without CUDA support.

#### Julia
Cxx.jl requires `staged functions` only available in Julia 0.4-dev.


```sh
$ git clone git://github.com/JuliaLang/julia.git julia$ cd julia 
# build
$ make -C deps distclean-openblas distclean-arpack distclean-suitesparse && make cleanall 
$ make –j4
``` 

#### Cxx.jl 
In the Julia terminal, type
```python
Pkg.clone("https://github.com/Keno/Cxx.jl.git")
Pkg.build("Cxx")   
```

## Using OpenCV.jl

#### Creating images

```c++
using OpenCV



```

#### Creating image arrays
#### Basic image manipulation
#### Loading images
#### User interface and GUIs
#### Trackbars
#### Video acquistion
#### Image processing
#### Object detection


## Demos
```c++
julia> using OpenCV

Select a demo from the following options: 

	1: CreateImage
	2: ReadWriteImages
	3: Thresholding
	4: LiveVideo
	5: LiveVideoWithTrackbars
	6: ObjectTracking
	7: Drawing
	8: displayHistogram
	9: im2dots

Select and run the demo at the julia prompt: 
e.g., run_demo(N)
```

## To do

* Passing Julia images/image arrays from Julia to OpenCV.jl and back
* Integration of OpenCV.jl with Images.jl and VideoIO.jl packages?
* Wrapping OpenCV C++ types and functions as Julia functions

<!--Image segmentation

* [ Watershed segmentation](https://github.com/sangyoon/opencv/blob/master/watershed.cpp)-->