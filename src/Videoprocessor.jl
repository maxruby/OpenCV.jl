######################################################################################
#
# Videoprocessor.jl
#
# OpenCV.jl videoplayer with trackbar control for interactive image processing
#
# Processing functions
# threshold
# brightness
# contrast
#
######################################################################################


const THRESHOLD = cint(0)
const BRIGHTNESS = cint(1)
const CONTRAST = cint(2)

cxx"""
class Videoprocessor
{
    public:
        int framesToProcess = -1; // infinity
        bool stop = false;

        class Threshold
        {
        public:
        int threshold = 120;
        int max_val = 255;
        int flag = cv::THRESH_BINARY;
              // THRESH_OTSU, THRESH_BINARY_INV,
              // THRESH_TRUNC, THRESH_TOZERO,
              // THRESH_TOZERO_INV
        std::string tn = "threshold";
        };

        class Brightness
        {
          public:
              int brightness = 50;
              int max_val = 100;
              std::string tn = "brightness";
        };

        class Contrast
        {
          public:
              int contrast = 50;
              int max_val = 100;
              std::string tn = "contrast";
        };

        // Instantiation of processing classses
        Threshold Thresh;
        Contrast Contr;
        Brightness Bright;

        // Videoplayer class methods
        void setDuration(int frames);
        void setParameters(int process, int param, int  max_, int flags);
        int getParameters(int process);
        void setTrackbar(int process, const std::string& w);
        cv::Mat Process(int process, cv::Mat src, cv::Mat dst);
        int Display(int process, const std::string& w, const char* filename, bool video, int device);
};


void Videoprocessor::setDuration(int frames)
{
     framesToProcess = frames;
}

void Videoprocessor::setParameters(int process, int  param, int  max_, int flags)
{
    switch(process)
    {
        case(0) :
            Thresh.threshold = param;
            Thresh.max_val = max_;
            Thresh.flag = flags;
            break;
        case(1) :
            Bright.brightness = param;
            Bright.max_val = max_;
            break;
        case(2)  :
            Contr.contrast = param;
            Contr.max_val = max_;
            break;
        default :
            std::cout <<"ERROR: wrong input" << std::endl;
            break;
    }
}

int Videoprocessor::getParameters(int process)
{
      switch(process)
      {
        case(0) :
            return(Thresh.threshold);
            break;
        case(1) :
            return(Bright.brightness);
            break;
        case(2) :
            return(Contr.contrast);
            break;
        default :
            std::cout <<"ERROR: wrong input" << std::endl;
            return(-1);
            break;
     }
}

void Videoprocessor::setTrackbar(int process, const std::string& w)
{
    //std::string w = wn;
    cv::namedWindow(w, cv::WINDOW_AUTOSIZE);

    switch(process)
    {
        case(0) :
            cv::createTrackbar(Thresh.tn, w, &Thresh.threshold, Thresh.max_val);
            break;
        case(1) :
            cv::createTrackbar(Bright.tn, w, &Bright.brightness, Bright.max_val);
            break;
        case(2) :
            cv::createTrackbar(Contr.tn, w, &Contr.contrast, Contr.max_val);
            break;
        default :
            std::cout <<"ERROR: wrong input" << std::endl;
            break;
     }
}

cv::Mat Videoprocessor::Process(int process, cv::Mat src, cv::Mat dst)
{

   switch(process)
   {
        case(0) :
            // Convert the image to Gray
            cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
            // Gaussian filtering
            cv::GaussianBlur(dst, dst, cv::Size(5,5), 0);
            cv::threshold(dst, dst, Thresh.threshold, Thresh.max_val, cv::THRESH_BINARY + cv::THRESH_OTSU);
            return(dst);
            break;
        case(1) :
            src.convertTo(dst, -1, 1, (Bright.brightness -50));
            return(dst);
            break;
        case(2) :
            src.convertTo(dst, -1, (Contr.contrast/50), 0);
            return(dst);
            break;
        default :
            std::cout <<"Unprocessed image!" << std::endl;
            return(dst);
            break;
     }
}

int Videoprocessor::Display(int process, const std::string& w, const char* filename = 0, bool video = true, int device = cv::CAP_ANY)
{
     cv::VideoCapture cam = cv::VideoCapture();
     cv::Mat src;

     // Open video file only if filename is provided
     if (video)
     {
         if (filename==NULL)
         {
              // open Video device
              cam.open(device);

              //Check that capture device is open
              if (!cam.isOpened())
              {
                  std::cout<< "Can not open camera!";
                  return(-1);  // error
              }
         }
         else
         {
              // open Video file
              cam.open(filename);
              //Check that video file is open
              if (!cam.isOpened())
              {
                  std::cout<< "Can not open video file!";
                  return(-2);  // error
              }
         }

         // Set the frame size to the camera capture WIDTH and HEIGHT
         double dWidth = cam.get(cv::CAP_PROP_FRAME_WIDTH);
         double dHeight = cam.get(cv::CAP_PROP_FRAME_HEIGHT);
         int Height = static_cast<int>(dHeight);
         int Width = static_cast<int>(dWidth);
         // Initialize the cv::Mat structures
         src = cv::Mat(Height, Width, CV_8UC3);
         cv::Mat dst = clone(src);
    }

    else   // Open single image
    {
         src = cv::imread(filename);
         if (src.empty())
         {
             std::cout<< "Cannot read the file!";
             return(-3);  // error
         }

         // Check that input image is RGB
         if (src.channels() != 3)
         {
            std::cout<< "Color image is required!";
            return(-4);  // error
         }
    }

    // Main processing loop

    // create a destination image same as src
    cv::Mat dst (src.rows,src.cols, src.type());
    int count = 0;  // keep a frame counter

    while(!stop)
    {
         count +=1;

         bool Success = cam.read(src);
         if (!Success)
         {
               break;
         }
         // select image processing operation
           dst = Videoprocessor::Process(0, src, dst);
        // dst = Videoprocessor::Process(1, src, dst);
        // dst = Videoprocessor::Process(2, src, dst);
         cv::imshow(w, dst);
         if (count == framesToProcess)
         {
             cv::destroyAllWindows();
             break;
         }

         if (cv::waitKey(0)==27)
         {
             std::cout << "Processed " << count << " frames." << std::endl;
             cv::destroyAllWindows();
             break;
         }
    }

    // reset stop
    stop = false;

    return (0);
}

"""

# How to use:

# Initialize the player
videoprocessor() = @cxxnew Videoprocessor()
# Set recording duration
setDuration(videoprocessor, frames::Int) = @cxx videoprocessor->setDuration(frames)
# Set the parameters for the processing operation(s)
setParameters(videoprocessor, proc::Int32, threshold::Int, max_val::Int, flag::Int) =
        @cxx videoprocessor->setParameters(proc, threshold, max_val, flag)
# Check parameters
getParameters(videoprocessor,proc::Int32) = @cxx videoprocessor->getParameters(THRESHOLD)
# Create a trackbar
setTrackbar(videoprocessor,proc::Int32, winname::String) = @cxx videoprocessor->setTrackbar(proc, pointer(winname))
# Diplay image/video and apply image processing
Display(videoprocessor,proc::Int32, winname::String) = @cxx videoprocessor->Display(proc, pointer(winname))

# other paramters to access
framesToProcess(videoprocessor) = @cxx videoprocessor->framesToProcess
stop(videoprocessor) = @cxx videoprocessor->stop

# julia> vid = videoprocessor()
# CppPtr{:Videoprocessor,()}(Ptr{Void} @0x00007ff81f91f7f0)
# julia> setDuration(vid, 10000)
# julia> setParameters(vid, THRESHOLD, 120, 255, THRESH_BINARY)
# julia> getParameters(vid, THRESHOLD)
# 120
# julia> setTrackbar(vid, THRESHOLD, "Demo")
# julia> Display(vid, THRESHOLD, "Demo")
# Processed 1 frames.
# Cleaned up camera.

# TO_DO:
# Adding a timestamp on the image
# cv::VideoCapture x( "path/to/video.avi" );
# int video_timestamp = x.get( CV_CAP_PROP_POS_MSEC );
# CvFont font;
# cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.3,0.3, 0, 1, CV_AA);
# cvPutText(img, "Hello!", cvPoint(50, 50), &font, CV_RGB(255,255,255));
