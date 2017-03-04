################################################################################################
#
# test 1: Create a blue color image and display it
#
################################################################################################

println("\nOpenCV.jl tests: press ESC to continue")

function test1()
    println("test 1: Create a blank color image and display it")
    # blue color vector (BGR)
    color = cvScalar(255, 0, 0)

    # Create an 8-bit unsigned RGB image, rows(height) x columns(width)
    img = Mat(300, 300, CV_8UC3, color)

    # Display in window
    imdisplay(img, "Blue image")
    closeWindows(0,27,"")  # press ESC(27) to exit
end

test1()

#################################################################################################
#
# test 2: Convert an image from .png to .jpg format (with compression)
#
#################################################################################################

function test2()
    println("test 2: Convert an image from .png to .jpg format (with compression)")
    inputfile = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
    outfile = joinpath(homedir(), "lena_copy.jpeg")
    img = imread(inputfile)

    #  compression parameters
    compression_params = stdvec(0,cint(0))
    stdpush!(compression_params, cint(IMWRITE_JPEG_QUALITY))
    stdpush!(compression_params, cint(98))
    # jpeg-> png
    # push_back(compression_params, cint(IMWRITE_PNG_COMPRESSION))
    # push_back(compression_params, cint(9))

    !(imwrite(outfile, img, compression_params)) ? throw(ArgumentError("Can not save to disk!")) : nothing
    imdisplay(img, "Converted .jpeg image")
    closeWindows(0,27,"")  # press ESC(27) to exit
end

test2()

#################################################################################################
#
# test 3: Binary thresholding
#
#################################################################################################

function test3()
    println("test 3: Binary thresholding")
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
    imdisplay(img, "Original")
    imdisplay(dst, "Gaussian blur")
    imdisplay(thresh, "Thresholded")
    closeWindows(0,27,"")  # press ESC(27) to exit
end

test3()

################################################################################################
#
# test 4: Live video capture (webcam)
#
################################################################################################

# see src/OpenCV_util.jl for details

function test4()
    println("test 4: Live video capture")
    videocam()    # press ESC to stop
end

test4()

################################################################################################
#
# test 5: Basic video display: set video properties
#
################################################################################################

function test5()
   println("test 5: Basic video display: set video properties")
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
end

test5()

################################################################################################
#
# test 6: Conversion of images from Images.jl to OpenCV Mat 
#
################################################################################################

function test6()
   println("test 6: Conversion of images from Images.jl to OpenCV Mat")
   filename = joinpath(Pkg.dir("OpenCV"), "./test/images/mandrill.jpg")
   image = imread(filename)
   dst = Mat(int(rows(image)), int(cols(image)), CV_8UC1)
   dst = convertToMat(image)
end

test6()