
# Julia: general utility functions
cint(x) = convert(Cint, x)
csize_t(x) = convert(Csize_t, x)

# Search directory
searchdir(path,key) = filter(x->contains(x,key), readdir(path))
swapext(f, new_ext) = "$(splitext(f)[1])$new_ext"

# C++ std::vector class thin-wrappings for Julia
cxx"""
template <typename T>
    std::vector<T> stdvector(int size, T x)
      {
        std::vector<T> cppvec(size, x);
        return (cppvec);
      }

template <typename T_>
    T_ at(std::vector<T_> cppvec, int index)
       {
           T_ value = cppvec[index];    // does not check for out of bounds (fast but dangerous)
           return(value);
       }

template <typename T_>
    T_ at_(std::vector<T_> cppvec, int index)
       {
          T_ value = cppvec.at(index); // checks for out of bounds (safe but slow)
          return(value);
       }

"""

stdvector(size, value) = @cxxnew stdvector(size, value)
stdassign(ccpvec, size, value) = @cxx ccpvec->assign(size,value)
stddata(cppvec) = @cxx ccpvec->data()     # Ptr to first elememt
stdempty(cppvec) = @cxx cppvec->empty()   # check if it is empty
stdcapacity(cppvec) = int(@cxx cppvec->capacity())
stdpush_back(cppvec, value) = @cxx cppvec->push_back(value)
stdpop_back(cppvec) = @cxx cppvec->pop_back()
stdsize(cppvec) = int(@cxx cppvec->size())
stdresize(cppvec, n::Int) = @cxx cppvec->resize(n)
stdshrink(cppvec) = @cxx cppvec->shrink_to_fit()
stdswap(cppvec1, cppvec2) = @cxx cppvec1->swap(cppvec2)

function at(cppvec, index::Int)
   (index < 0 || index > stdsize(cppvec)) ? throw(ArgumentError("index is out of bounds")) : nothing
   @cxx at(cppvec, index)
end
function at_(cppvec, index::Int)
  (index < 0 || index > stdsize(cppvec)) ? throw(ArgumentError("index is out of bounds")) : nothing
   @cxx at_(cppvec, index)   # out of bounds check
end
clear(cppvec) = @cxx cppvec->clear()


# C++ string handling
# julia string => std::string
# std::string => julia string

cxx"""
std::string stdstring(const char* word)
      {
        std::string stdword = word;
        return(stdword);
      }
"""

stdstring(word::String) = @cxx stdstring(pointer(word))
juliastring(std_string) = bytestring(@cxx std_string->c_str())
stdstrdata(std_string) = @cxx std_string->data()     # Ptr to first elememt
stdstrempty(std_string) = @cxx std_string->empty()
stdstrsize(std_string) = @cxx std_string->size()
stdstrlength(std_string) = @cxx std_string->length()
stdstrclear(std_string) = @cxx std_string->clear()

#-------------------------------------------------------------------------------------------------------------------#
# Image processing (imgproc)

# Support for image convolution
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
function imdisplay(img, windowName::String, flag=WINDOW_AUTOSIZE)
    namedWindow(windowName, flag)
    imshow(windowName, img)
end

function closeWindows(delay, key, windowName)
      (waitkey(delay) == key && windowName != "") ? destroyWindow(windowName) : destroyAllWindows()
end
# closeWindows(0,27,"")

function im2tile(imgArray, windowName::String, flag=WINDOW_AUTOSIZE)
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

    imdisplay(canvas, windowName, flag)
end


#-------------------------------------------------------------------------------------------------------------------#
# Video capture (video)

function videocam (device = CAP_ANY)
    cam = videoCapture(device)    # open Video device
    !isOpened(cam) ? throw(ArgumentError("Can not open camera!")) : nothing
    namedWindow("Video")
    frame = Mat()

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        imshow("Video", frame)

        if (waitkey(30) == 27)
            destroyAllWindows()
            break
       end
   end
  release(cam)
end

# Webstreaming
# src: full http link to the video source
# e.g., NASA TV: vid = "http://www.nasa.gov/multimedia/nasatv/NTV-Public-IPS.m3u8"
function webstream (src::String)
    cam = videoCapture(src)    # open Video device
    !isOpened(cam) ? throw(ArgumentError("Can not open stream!")) : nothing
    namedWindow("Web stream")
    frame = Mat()

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video stream!"))
            break
        end

        imshow("Web stream", frame)

        if (waitkey(30) == 27)
            destroyAllWindows()
            break
       end
   end
  release(cam)
end


function videoWrite (cam, filename::String, fps::Float64, nframes = 0, frameSize=cvSize(0,0), fourcc=-1, isColor=true)
    !(isOpened(cam)) ? throw(ArgumentError("Can is not opened!")) : nothing

    # Set the video capture frame size based on camera input WIDTH and HEIGHT
    width = getVideoId(cam, CAP_PROP_FRAME_WIDTH)
    height = getVideoId(cam, CAP_PROP_FRAME_HEIGHT)

    frame = Mat(int(height), int(width), CV_8UC3)

    # Set the frameSize of output video frames
    (frameSize.data[1] == 0) ? frameSize = cvSize(int(width), int(height)) :
         throw(ArgumentError("Output frame dimension is wrong"))

    # Initialize and open video writer
    writer = videoWriter(filename, fourcc, fps, frameSize, isColor)
    openWriter(writer, filename, fourcc, fps, frameSize, isColor)
    !(isOpened(writer)) ? throw(ArgumentError("Can not open the video writer!")) : nothing

    # create a window for display
    namedWindow("Welcome!")

    # Loop until user presses ESC or frame is empty
    count = 0
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        writeVideo(writer, frame)
        imshow("Welcome!", frame)

        (nframes > 0) ? count += 1 : nothing
        if ((waitkey(30) == 27) || (count == nframes))
            destroyAllWindows()
            break
       end
   end
   release(cam)
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
