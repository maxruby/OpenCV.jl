################################################################################################
#
# demo 1: Create a blank color image and display it
#
################################################################################################

# blue color vector (BGR)
color = cvScalar(255, 0, 0)

# Create an 8-bit unsigned RGB image, rows(height) x columns(width)
img = Mat(300, 300, CV_8UC3)

# Display in window
imdisplay(img, "Blue image")
closeWindows(0,27,"")  # press ESC(27) to exit


#################################################################################################
#
# demo 2: Convert an image from .png to .jpg format (with compression)
#
#################################################################################################

inputfile = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
outfile = joinpath(homedir(), "lena_copy.jpeg")
img = imread(inputfile)

#  compression parameters
compression_params = stdvector(0,cint(0))
push_back(compression_params, cint(IMWRITE_JPEG_QUALITY))
push_back(compression_params, cint(98))
# jpeg-> png
# push_back(compression_params, cint(IMWRITE_PNG_COMPRESSION))
# push_back(compression_params, cint(9))

!(imwrite(outfile, img, compression_params)) ? throw(ArgumentError("Can not save to disk!")) : nothing
imdisplay(img, "Converted .jpeg image")
closeWindows(0,27,"")  # press ESC(27) to exit


#################################################################################################
#
# demo 3: Binary thresholding
#
#################################################################################################

filename = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
img = imread(filename)
dst = Mat(int(rows(img)), int(cols(img)), CV_8UC1)
cvtColor(img, dst, COLOR_BGR2GRAY)
gaussianBlur(dst, dst, cvSize(5,5))
thresh = Mat()
threshold(dst, thresh, 120, 255, THRESH_BINARY)  # thresh = 0, max = 255
 #THRESH_OTSU
 #THRESH_BINARY_INV
 #THRESH_TRUNC
 #THRESH_TOZERO
 #THRESH_TOZERO_INV
imdisplay(img, "Original", "ON")
imdisplay(dst, "Gaussian blur", "ON")
imdisplay(thresh, "Thresholded", "ON")
closeWindows(0,27,"")  # press ESC(27) to exit



################################################################################################
#
# demo 4: Live video capture (webcam)
#
################################################################################################

# see src/OpenCV_util.jl for details

videocam()    # press ESC to stop




################################################################################################
#
# demo 5: video player: set video properties
#
################################################################################################

fvideo = joinpath(Pkg.dir("OpenCV"), "./test/images/movie.avi")
vid = videoCapture(fvideo)
!isOpened(vid) ? throw(ArgumentError("Can not open video stream!")) : nothing

setVideoId(vid, CAP_PROP_POS_MSEC, 300.0)     #300 ms into the video
# setVideoId(vid, CAP_PROP_FRAME_WIDTH, 600.0)
# setVideoId(vid, CAP_PROP_FRAME_HEIGHT, 400.0)
# setVideoId(vid, CAP_PROP_FPS, 5.0)

fps = getVideoId(vid, CAP_PROP_FPS)
println("Frames per second: ", fps)

namedWindow("Video player")
frame = Mat()

# Loop until user presses ESC or frame is empty
while(true)
   if !(videoRead(vid, frame))
      throw(ArgumentError("Can not acquire video stream!"))
      break
   end

   imshow("Video player", frame)

   if (waitkey(30) == 27)
       destroyAllWindows()
       break
   end
end


######################################################################################
#
# demo 6: Interactive video player with threshold/brightness/contrast control
#
######################################################################################




