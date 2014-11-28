
# General utility functions
cint(x) = convert(Cint, x)
csize_t(x) = convert(Csize_t, x)


#-------------------------------------------------------------------------------------------------------------------#
# Image processing (imgproc)

# Support functions for image convolution
cxx"""
// Function getSum returns total sum of all the elements of given matrix.

float getKernelSum(cv::Mat kernel)
{
   float sum = 0;
   for(typeof(kernel.begin<float>()) it = kernel.begin<float>(); it!= kernel.end<float>() ; it++)
   {
      sum+=*it;
   }
   return sum;
}

// Normalize the mask (kernel)
cv::Mat normalizeKernel (cv::Mat kernel, double ksum) {
    kernel = kernel/ksum;
    return(kernel);
    }
"""

# kernel = ones(5,5,CV_32F)
getKernelSum(kernel) = @cxx getKernelSum(kernel)
# ksum = getKernelSum(kernel)
normalizeKernel(kernel, ksum) = @cxx normalizeKernel(kernel, ksum)

# TO_DO:  Similar utility functions are likely required for other functions in imgproc

#-------------------------------------------------------------------------------------------------------------------#
# GUI interface (highui)

# Display an image in a window and close upon key press (and delay)
# For concurrent multiple display, set multi = "ON"
function imdisplay(img, windowName::String, multi="OFF", flag=WINDOW_AUTOSIZE, delay=0, key=27)
    namedWindow(windowName, flag)
    imshow(windowName, img)
    (multi == "OFF" && waitkey(delay) == key) ? destroyWindow(windowName) : nothing
end

macro closeWindows(delay, key, windowName)
      (waitkey(delay) == key && windowName != "") ? destroyWindow(windowName) : destroyAllWindows()
end
# @closeWindows(0,27)

function im2tile(imgArray, windowName::String, flag=WINDOW_AUTOSIZE, delay=0, key=27)
    canvas = Mat()

    for i=1:length(imgArray)
        # check that images have same dims, format and channels
        if (i > 1)
            (cvtypeval(imgArray[i]) != cvtypeval(imgArray[i-1]) ||
             rows(imgArray[i]) != rows(imgArray[i-1]) ||
             cols(imgArray[i]) != cols(imgArray[i-1])) ?
          throw(ArgumentError("Images must have same dimensions and format")): nothing
        end
        push(canvas, imgArray[i])
    end

    imdisplay(canvas, windowName, flag, delay, key)
end


#-------------------------------------------------------------------------------------------------------------------#
# Video capture (video)

function videoCapture (device = CAP_ANY)
    cam = videoCapture(device)    # open Video device
    !isOpened(cam) ? throw(ArgumentError("Can not open camera!")) : nothing
    namedWindow("Welcome!")

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        imshow("Welcome!", converted)

        if (waitkey(30) == 27)
            destroyAllWindows()
            release(cam)
            break
       end
   end
end


function videoWrite (filename::String, fourcc::Int, fps::Float64, frameSize, isColor=true, device = CAP_ANY)
    cam = videoCapture(device)    # open Video device
    !(isOpened(cam)) ? throw(ArgumentError("Can not open camera!")) : nothing
    namedWindow("Welcome!")

    writer = videoWriter(filename, fourcc, fps, frameSize, isColor)
    openWriter(writer, filename, fourcc, fps, frameSize, isColor)
    !(isOpened(writer)) ? throw(ArgumentError("Can not open the video writer!")) : nothing

    # Set the frame size to the camera capture WIDTH and HEIGHT
    width = getVideoId(cam, CAP_PROP_FRAME_WIDTH)
    height = getVideoId(cam, CAP_PROP_FRAME_HEIGHT)

    # Initialize images
    frame = Mat(int(height), int(columns), CV_8UC3)

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        writeVideo(writer, frame)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        imshow("Welcome!", converted)

        if (waitkey(30) == 27)
            destroyAllWindows()
            release(cam)
            break
       end
   end
end




#-------------------------------------------------------------------------------------------------------------------#
# Interactive thresholding

cxx"""
class iThreshold {
   public:
     int threshold = 120;
     int max_val = 255;
     cv::Mat dst;
     cv::Mat gray;

     int getValue(void);
     void setValue(int thresh);
     void setTrackbar (const char* tn, const char* wn);
     void show (const char* wn, cv::Mat img, int flag = cv::THRESH_BINARY);
  };

  void iThreshold::setValue(int thresh) {
      threshold = thresh;
  }

  int iThreshold::getValue(void) {
      return(threshold);
  }

  void iThreshold::setTrackbar (const char* tn, const char* wn){
     std::string t = tn;
     std::string w = wn;
     cv::namedWindow(w, cv::WINDOW_AUTOSIZE);
     cv::createTrackbar(t, w, &threshold, max_val);

  }

  void iThreshold::show (const char* wn, cv::Mat img, int flag) {
     std::string w = wn;
     // Make grayscale
     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
     while(true) {
        // Threshold
        // cv::THRESH_BINARY
        // cv::THRESH_OTSU
        // cv::THRESH_BINARY_INV
        // cv::THRESH_TRUNC
        // cv::THRESH_TOZERO
        // cv::THRESH_TOZERO_INV
        cv::threshold(gray, dst, threshold, max_val, flag);
        // Show image
        cv::imshow(w, dst);
        if (cv::waitKey(30) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }
  }
"""

iThreshold() = @cxxnew iThreshold()
setValue(iThreshold, val::Int) = @cxx iThreshold->setValue(val)
getValue(iThreshold) = @cxx iThreshold->getValue()
setWindow(iThreshold, trackbarname::String, winname::String) =
     @cxx iThreshold->setTrackbar(pointer(trackbarname), pointer(winname))
showWindow(iThreshold, winname::String, img) =  @cxx iThreshold->show(pointer(winname), img)
