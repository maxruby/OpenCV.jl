# Loading config for OpenCV 3.2.0 shared libraries (.dylib, .so)
# Supports OSX (10.12.3) and Linux
# see README for details

cvlibdir = "/usr/local/lib/"
cvheaderdir = "/usr/local/include/"

version = "3.2.0"

libprefix = "libopencv_"

libNames = [
              "shape",
              "stitching",
              "objdetect",
              "superres",
              "videostab",
              "calib3d",
              "features2d",
              "highgui",
              "videoio",
              "imgcodecs",
              "video",
              "photo",
              "ml",
              "imgproc",
              #"flann",   // TODO: resolve typeid error due to rtti flag
              "viz"
            ]

function getFullLibNames()
    return map(x -> "$(libprefix)$(x).$(version)", libNames)
end
