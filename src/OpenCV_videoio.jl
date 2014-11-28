####################################################################################################
# videoio. Media I/O
####################################################################################################

# 1. Reading and Writing Video

# Create the VideoCapture structures
cxx""" cv::VideoCapture VideoCapture(){ cv::VideoCapture capture = cv::VideoCapture(); return(capture); }"""
videoCapture() = @cxx VideoCapture()

cxx""" cv::VideoCapture VideoCapture(const char *filename){ cv::VideoCapture capture = cv::VideoCapture(filename); return(capture); }"""
videoCapture(filename::String) = @cxx VideoCapture(pointer(filename))

cxx""" cv::VideoCapture VideoCapture(int device){ cv::VideoCapture capture = cv::VideoCapture(device); return(capture); }"""
videoCapture(device::Int) = @cxx VideoCapture(device)   # autodetect = 0

# Functions for opening capture and grabbing frames
openVideo(capture,filename::String) = @cxx capture->open(pointer(filename))
openVideo(capture, device::Int) = @cxx capture->open(device)
isOpened(capture) = @cxx capture->isOpened()

# Useful for multi-camera environments
grab(capture) = @cxx capture->grab()
# Decodes and returns the grabbed video frame
retrieve(capture, image, flag=0) = @cxx capture->retrieve(image, flag)
# automatically called by cv::VideoCapture->open
release(capture) = @cxx capture->release()  # automatically called

# Grab, decode and return the next video frame
cxx""" cv::VideoCapture&  videoread(cv::VideoCapture& capture){ cv::Mat frame; capture >> frame; return(capture); } """
videoRead(capture) = @cxx videoread(capture)
videoRead(capture, image) = @cxx capture->read(image)

# Return the specified VideoCapture property
getVideoId(capture, propId::Int) = @cxx capture->get(propId)
# CAP_PROP_POS_MSEC       Current position of the video file (msec or timestamp)
# CAP_PROP_POS_FRAMES     0-based index of the frame to be decoded/captured next
# CAP_PROP_POS_AVI_RATIO  Relative position of the video file: 0 - start of the film, 1 - end of the film
# CAP_PROP_FRAME_WIDTH    Width of the frames in the video stream
# CAP_PROP_FRAME_HEIGHT   Height of the frames in the video stream
# CAP_PROP_FPS            frame rate
# CAP_PROP_FOURCC         4-character code of codec
# CAP_PROP_FRAME_COUNT    Number of frames in the video file
# CAP_PROP_FORMAT         Format of the Mat objects returned by retrieve()
# CAP_PROP_MODE           Backend-specific value indicating the current capture mode
# CAP_PROP_BRIGHTNESS     Brightness of the image (only for cameras)
# CAP_PROP_CONTRAST       Contrast of the image (only for cameras)
# CAP_PROP_SATURATION     Saturation of the image (only for cameras)
# CAP_PROP_HUE            Hue of the image (only for cameras)
# CAP_PROP_GAIN           Gain of the image (only for cameras)
# CAP_PROP_EXPOSURE       Exposure (only for cameras)
# CAP_PROP_CONVERT_RGB    Boolean flags indicating whether images should be converted to RGB
# CAP_PROP_WHITE_BALANCE  Currently not supported
# CAP_PROP_RECTIFICATION  Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

# Sets a property in the VideoCapture
setVideoId(capture, propId::Int, value::Float64) = @cxx capture->set(propId, value)

# Create the VideoWriter structures
cxx""" cv::VideoWriter VideoWriter(){ cv::VideoWriter writer = cv::VideoWriter(); return(writer); }"""
videoWriter() = @cxx VideoWriter()

cxx""" cv::VideoWriter VideoWriter(const char *filename, int fourcc, double fps, cv::Size frameSize,
    bool isColor=true){ cv::VideoWriter writer(filename, fourcc, fps, frameSize, isColor); return(writer); }"""
videoWriter(filename::String, fourcc::Int, fps::Float64, frameSize, isColor=true) =
    @cxx VideoWriter(pointer(filename), fourcc, fps, frameSize, isColor)
# Parameters
# filename  – Name of the output video file.
# fourcc    – Fourcc codec, e.g., fourcc('M','J','P','G')
# fps       – Framerate
# frameSize – Size of the video frames
# isColor   –  only supported for Windows

openWriter(writer, filename::String, fourcc::Int, fps::Float64, frameSize, isColor=true) =
    @cxx writer->open(pointer(filename), fourcc, fps, frameSize, isColor)

isOpened(writer) = @cxx writer->isOpened()

# Write the next video frame
writeVideo(writer, image) = @cxx writer->write(image)

# Video write with operator >>
cxx""" cv::VideoWriter&  writeframe(cv::VideoWriter& videowriter){ cv::Mat frame; videowriter << frame; return(videowriter); } """
writeVideo(videowrite) = @cxx writeframe(videowrite)


fcc = CV_FOURCC_MPEG # const Array(Ptr{Uint8}, 4)
# CV_FOURCC_IYUV  #for yuv420p into an uncompressed AVI
# CV_FOURCC_DIV3  #for DivX MPEG-4 codec
# CV_FOURCC_MP42  #for MPEG-4 codec
# CV_FOURCC_DIVX  #for DivX codec
# CV_FOURCC_PIM1  #for MPEG-1 codec
# CV_FOURCC_I263  #for ITU H.263 codec
# CV_FOURCC_MPEG  #for MPEG-1 codec

# Get int for the FourCC codec code
cxx""" int fourcc(const char *cc1, const char *cc2, const char *cc3, const char *cc4) {
           int code = cv::VideoWriter::fourcc (*cc1, *cc2, *cc3, *cc4);
           return(code);
   }
"""

fourcc(fcc::Array) = @cxx fourcc(fcc[1],fcc[2], fcc[3], fcc[4])

