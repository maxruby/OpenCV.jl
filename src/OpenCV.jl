################################################################################################
# OpenCV
#
# Loading interface for calling OpenCV (C++) directly in Julia
#
# If OpenCV is already installed, or you wish to use different version of the libraries,
# change the library and header paths to your local directories, e.g.,
#      /usr/local/lib/
#      /usr/local/include/
#
# 2014, Maximiliano Suster
# See README.md for how to install and use
#################################################################################################

using Cxx

# Check Julia version before continuing (also checked later in Cxx)
(VERSION >= v"0.4-") ? nothing :
     throw(ErrorException("Julia $VERSION does not support C++ FFI"))


# Load C++ FFI interface, headers and shared libraries
include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_libs.jl"))


# Include directories and paths for headers and libs
@osx_only begin
    const cvlibdir = joinpath(Pkg.dir("OpenCV"), "./deps/usr/lib/")
    const cvheaderdir = joinpath(Pkg.dir("OpenCV"), "./deps/usr/include/")

    addHeaderDir(cvlibdir; kind = C_System)

    # load OpenCV shared libraries
    # IMPORTANT: if necessary, make sure to link symbols accross libraries with RTLD_GLOBAL
    for i in opencv_libraries
       dlopen_e(joinpath(cvlibdir,i), RTLD_GLOBAL)==C_NULL ? throw(ArgumentError("Skip loading $(i)")): nothing
    end

    #Now include header files
    addHeaderDir(joinpath(cvheaderdir,"opencv2"), kind = C_System)
    addHeaderDir(joinpath(cvheaderdir,"opencv2/core"), kind = C_System)
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

    cxxinclude(joinpath(cvheaderdir,"opencv2/shape.hpp"))
    cxxinclude(joinpath(cvheaderdir,"opencv2/stitching.hpp"))
    cxxinclude(joinpath(cvheaderdir,"opencv2/superres.hpp"))
    cxxinclude(joinpath(cvheaderdir,"opencv2/videostab.hpp"))

    #load OpenCV.jl interface
    include(joinpath(Pkg.dir("OpenCV"), "./src/OpenCV_hpp.jl"))

    #load demo function
    function run_tests()
        include(joinpath(Pkg.dir("OpenCV"), "./test/demos.jl"))
    end
end
