################################################################################################
# OpenCV.jl
#
# The OpenCV interface (C++) for Julia
#
# If OpenCV is already installed in your computer, or you wish to use different libraries,
# change the library and header paths to your local directories, e.g.,
#      /usr/local/lib/
#      /usr/local/include/
#
# To add more libraries, make sure to add the name to "opencv_libraries" in /src/OpenCV_libs.jl
#
# OpenCV lib version: OpenCV 3.2.0
#
# DOCUMENTATION
# opencv.org
# http://opencv.org
# Doxygen
# http://docs.opencv.org/master/index.html
#
#################################################################################################

# Convenient swap file extension function
swapext(f, new_ext) = "$(splitext(f)[1])$new_ext"

# Check Julia version before continuing (also checked later in Cxx)
(VERSION >= v"0.6-") ? nothing :
     throw(ErrorException("Julia $VERSION does not support C++ FFI"))

# Load Cxx
using Cxx

# Load config variables for loading OpenCV shared libraries
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_libs.jl"))
opencv_libraries = getFullLibNames();

# Autosearch for OpenCV installations using pkg-config
(so,si,pr) = readandwrite(`pkg-config --libs opencv`)
output = readstring(so)
close(so)

# Check lib installation path
@static if is_apple()
  failed_to_get_path = match(Regex(chop(cvlibdir)), output) == nothing
end

@static if is_linux()
  failed_to_get_path = match(Regex("libopencv|lopencv"), output) == nothing
end

# Check if all libraries specified in config are indeed installed
missing_libs = false
for i in libNames
    if !contains(output, i)
        missingmissing_libs = true
        println("$(i) is not found in pkg-config")
    end
end

# Load pre-built DYLD libraries (OSX ONLY), else throw an ErrorException
if failed_to_get_path || missing_libs
    cvlibdir = is_apple() ? joinpath(Pkg.dir("OpenCV"), "./deps/usr/lib/") : throw(ErrorException("No pre-installed libraries. Set path manually or install OpenCV."))
    cvheaderdir = is_apple() ? joinpath(Pkg.dir("OpenCV"), "./deps/usr/include/") : throw(ErrorException("No pre-installed headers. Set path manually or install OpenCV."))
end

addHeaderDir(cvheaderdir, kind = C_System)

# Load OpenCV shared libraries (default file extension is .dylib)
# TO DO: ensure compatible path/extension for Windows OS
for i in opencv_libraries
    if is_linux()
         i = swapext(i[1:end-6], ".so")
    end
    # Must link symbols accross libraries with RTLD_GLOBAL
    Libdl.dlopen(joinpath(cvlibdir,i), Libdl.RTLD_GLOBAL)
end

# Now include C++ header files
addHeaderDir(joinpath(cvheaderdir,"opencv2"), kind = C_System )
addHeaderDir(joinpath(cvheaderdir,"opencv2/core"), kind = C_System )
cxxinclude(joinpath(cvheaderdir,"opencv2/opencv.hpp"))
    # => opencv.hpp calls all the main headers
    #include "opencv2/core.hpp"
    #include "opencv2/imgproc.hpp"
    #include "opencv2/photo.hpp"
    #include "opencv2/video.hpp"
    #include "opencv2/features2d.hpp"
    #include "opencv2/objdetect.hpp"
    #include "opencv2/calib3d.hpp"
    #include "opencv2/imgcodecs.hpp"
    #include "opencv2/videoio.hpp"
    #include "opencv2/highgui.hpp"
    #include "opencv2/ml.hpp"

cxxinclude(joinpath(cvheaderdir,"opencv2/core/opengl.hpp"))            # enable OpenGL
cxxinclude(joinpath(cvheaderdir,"opencv2/core/ocl.hpp"))               # enable OpenCL
cxxinclude(joinpath(cvheaderdir,"opencv2/video/background_segm.hpp"))  # enable bg/fg segmentation
cxxinclude(joinpath(cvheaderdir,"opencv2/video/tracking.hpp"))         # enable tracking
cxxinclude(joinpath(cvheaderdir,"opencv2/shape.hpp"))
cxxinclude(joinpath(cvheaderdir,"opencv2/stitching.hpp"))
cxxinclude(joinpath(cvheaderdir,"opencv2/superres.hpp"))
cxxinclude(joinpath(cvheaderdir,"opencv2/videostab.hpp"))

# TO DO: we need to include paths for opencv-contrib folders/subfolders inside opencv main directory
# Currently system-specific!
# try
#     cxxinclude("/Users/maximilianosuster/opencv/opencv_contrib-master/modules/bgsegm/include/opencv2/bgsegm.hpp")
# catch
#     warn("opencv_contrib module headers could not be found, set the paths manually in src/OpenCV.jl to use advanced functions")
# end

# Include C++ headers
cxx"""
 #include <iostream>
 #include <unistd.h>
 #include <cstdlib>
 #include <cstdio>
 #include <cstddef>
 #include <cstring>
 #include <cfloat>
 #include <vector>
 #include <ctime>
 #include <map>
 #include <utility>
 #include <exception>
"""

# Load Qt framework
# BUG: see deps/Qt_support.jl, so currently disabled
# include(joinpath(Pkg.dir("OpenCV"), "./deps/Qt_support.jl"))

# Load header constants and typedefs
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_hpp.jl"))

# Load OpenCV bindings
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_modules.jl"))

# Load custom utility functions
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_util.jl"))

# Load Videoplayer
include(joinpath(Pkg.dir("OpenCV"), "./src/Videoprocessor.jl"))

# Load support for Images
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_ImagesSupport.jl"))

# Load demos -- currently Cxx versions
function run_tests()
    include(joinpath(Pkg.dir("OpenCV"), "./test/cxx/demos.jl"))
end
