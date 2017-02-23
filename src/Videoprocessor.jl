######################################################################################
#
# Videoprocessor.jl
#
# OpenCV.jl videoplayer with trackbar control for interactive image processing
#
# Image processing functions
# BRIGHTNESS
# CONTRAST
# THRESHOLD
#
# Easy to add new filters/algorithms for live image processing
#
######################################################################################


# procesess
const BRIGHTNESS = cint(0)
const CONTRAST = cint(1)
const THRESHOLD = cint(2)
const GAUSSIAN = cint(3)

cxx"""
class Videoprocessor
{
    public:

        int framesToProcess = -1; // infinity
        bool stop = false;
        unsigned int delay = 0;  // microseconds

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
        int T = Thresh.threshold;
        Contrast Contr;
        int C = Contr.contrast;
        Brightness Bright;
        int B = Bright.brightness;

        // Videoplayer class methods
        void setDuration(int frames);
        void setDelay(int tsleep);
        void setParameters(int process, int param, int  max_, int flags);
        int getParameters(int process);
        void setTrackbar(int process, const std::string& w);
        cv::Mat ProcessMulti(int process, cv::Mat src, cv::Mat dst);
        cv::Mat ProcessSingle(int process, cv::Mat src, cv::Mat dst);
        int Display(std::vector<int> processes, const std::string& w, const std::string& filename, bool video, int device);
};

void Videoprocessor::setDuration(int frames)
{
     framesToProcess = frames;
}

void Videoprocessor::setDelay(int tsleep)
{
     delay = tsleep; // microseconds
}

void Videoprocessor::setParameters(int process, int  param, int  max_, int flags)
{

    switch(process)
    {

        case(0) :
            B = param;
            Bright.max_val = max_;
            break;
        case(1)  :
            C = param;
            Contr.max_val = max_;
            break;
        case(2) :
            T = param;
            Thresh.max_val = max_;
            Thresh.flag = flags;
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
            return(B);
            break;
        case(1) :
            return(C);
            break;
        case(2) :
            return(T);
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
            cv::createTrackbar(Bright.tn, w, &B, Bright.max_val);
            break;
        case(1) :
            cv::createTrackbar(Contr.tn, w, &C, Contr.max_val);
            break;
        case(2) :
            cv::createTrackbar(Thresh.tn, w, &T, Thresh.max_val);
            break;
        default :
            std::cout <<"ERROR: wrong input" << std::endl;
            break;
     }
}

cv::Mat Videoprocessor::ProcessMulti(int process, cv::Mat src, cv::Mat dst)
{

   switch(process)
   {
        case(0) :
            src.convertTo(dst, -1, 1, (B -50));
            return(dst);
            break;
        case(1) :
            src.convertTo(dst, -1, (C/50), 0);
            return(dst);
            break;
        case(2) :
            // Convert the image to Gray
            cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
            // Gaussian filtering
            cv::GaussianBlur(dst, dst, cv::Size(5,5), 0);
            cv::threshold(dst, dst, T, Thresh.max_val, Thresh.flag);
            return(dst);
            break;
        default :
            std::cout <<"Unprocessed image!" << std::endl;
            return(dst);
            break;
    }
}


cv::Mat Videoprocessor::ProcessSingle(int process, cv::Mat src, cv::Mat dst)
{
   switch(process)
   {
        case(0) :
            src.convertTo(dst, -1, 1, (B -50));
            return(dst);
            break;
        case(1) :
            src.convertTo(dst, -1, (C/50), 0);
            return(dst);
            break;
        case(2) :
            cv::threshold(src, dst, T, Thresh.max_val, Thresh.flag);
            return(dst);
            break;
        case(3) :
             // Convert the image to Gray
            cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
            // Gaussian filtering
            cv::GaussianBlur(dst, dst, cv::Size(5,5), 0);
            return(dst);
            break;
        default :
            std::cout <<"Unprocessed image!" << std::endl;
            return(dst);
            break;
   }
}

int Videoprocessor::Display(std::vector<int> processes, const std::string& w, const std::string& filename, bool video = true, int device = cv::CAP_ANY)
{

     cv::VideoCapture cam = cv::VideoCapture();
     cv::Mat src;

     // Open video file only if filename is provided
     if (video)
     {
         if (filename.size()==0)
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
              // open Video file;
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
    }

    else   // Open and display single image
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

    // Main processing

    // create a destination image same as src
    cv::Mat dst = cv::Mat(src.rows,src.cols, src.type());

    if (video) // video file/webstream/camera
    {
       int count = 0;  // keep a frame counter

       while(true)
       {
          count +=1;

          bool Success = cam.read(src);
          if (!Success)
          {
              break;
          }

          // select image processing operation
          for (std::size_t i = 0; i < processes.size(); i++)
          {
              src = Videoprocessor::ProcessMulti(processes[i], src, dst);
          }

          cv::imshow(w, src);

          if (count == framesToProcess)
          {
              cv::destroyAllWindows();
              break;
          }

	        // delay in microseconds
          if (delay > 0)
          { usleep(delay); }

          if (cv::waitKey(30)==27)
          {
              std::cout << "Processed " << count << " frames." << std::endl;
              cv::destroyAllWindows();
              break;
          }
       }
      cam.release();
    }

    else
    {
         cv::Mat gray = cv::Mat(src.rows,src.cols, CV_8UC1);
         cv::Mat thresh = gray.clone();


         bool converted = false;
         while(true)
         {
              // select image processing operation
              for (std::size_t i = 0; i < processes.size(); i++)
              {
                   if (processes[i] < 2)
                   {
                       dst  = Videoprocessor::ProcessSingle(processes[i], src, dst);
                   }
                   if (processes[i] ==2)
                   {
                        if (!converted)
                        {
                             gray = Videoprocessor::ProcessSingle(3, src, dst);  // make grayscale and apply Gaussian blur
                             converted = true;
                        }
                        thresh = Videoprocessor::ProcessSingle(processes[i], gray, thresh);
                   }
               }


               if (!converted)
               {
                    cv::imshow(w, dst);
               }
               else
               {
                    cv::imshow(w, thresh);
               }

               if (cv::waitKey(30) == 27)
               {
                  cv::destroyAllWindows();
                  break;
               }
         }
     }

    return (0);
}

std::vector<int> RunVideoprocessor(std::vector<int> processes, const std::string& winname, const std::string &filename,
         int frames = -1, int tsleep=0, int brightness = 30, int contrast = 30, int threshold = 120, int max_val = 255,
         int flag = cv::THRESH_BINARY, bool video = true, int device = cv::CAP_ANY)
{
   // Initialize the videoprocessor
   Videoprocessor videoprocessor;

   // Set how many frames to record, default = -1 (continuous loop);
   if (frames > 1)
   {
        videoprocessor.setDuration(frames);
   }

   // Set interframe delay in microseconds for video streams
   if (tsleep > 0)
   {
       videoprocessor.setDelay(tsleep);
   }

   // Set initial filter parameters (threshold, contrast, brightness)
   videoprocessor.setParameters(0, videoprocessor.B, videoprocessor.Bright.max_val, 0);
   videoprocessor.setParameters(1, videoprocessor.C, videoprocessor.Contr.max_val, 0);
   videoprocessor.setParameters(2, videoprocessor.T, videoprocessor.Thresh.max_val, flag);

   // Set trackbar (BRIGHTNESS, CONTRAST, THRESHOLD)
   for (std::size_t i = 0; i < processes.size(); i++)
   {
        videoprocessor.setTrackbar(processes[i], winname);
   }

   // Set display
   videoprocessor.Display(processes, winname, filename, video, device);

   // Get parameters upon termination
   std::vector<int> final_params;
   for (std::size_t i = 0; i < processes.size(); i++)
   {
        final_params.push_back(videoprocessor.getParameters(processes[i]));
   }
   return(final_params);
}
"""

# Instructions:

# 1. Create an std::vector with processing operations [BRIGHTNESS, CONTRAST, THRESHOLD]
# processes = stdvector(cint(0),cint(0))
# # Always add THRESHOLD last!
# stdpush_back(processes, BRIGHTNESS)
# stdpush_back(processes, CONTRAST)
# stdpush_back(processes, THRESHOLD)

# 2. Arguments:
# processes                     std::vector<int> (see above)
# winname                       window name (string)
# filename= Ptr{UInt8}[0][1]    file to open
# frames=-1                     number of frames to record
# delay=0                       delay in microseconds (uses system usleep), e.g., 1000 = 1s
# brightness=30                 initial brightness value
# contrast=30                   initial contrast value
# threshold=120                 initial threshold value
# max_val=255                   maximum value (paired with process, e.g., BRIGHTNESS)
# flag=THRESH_BINARY            flag for thresholding algorithm
# video=true                    if video source is used
# device=CAP_ANY                flag for video device

videoprocessor(processes, winname::String, filename="", frames=-1, delay=0, brightness=30,
    contrast=30, threshold=120, max_val=255, flag=THRESH_BINARY, video=true, device=CAP_ANY) =
    @cxx RunVideoprocessor(processes, pointer(winname), pointer(filename), frames, delay, brightness, contrast,
        threshold, max_val, flag, video, device)



