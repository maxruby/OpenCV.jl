################################################################################################
#
# demo 4: Live video capture (webcam)
#
################################################################################################

# Julia variables
device_index = CAP_ANY
winname = "Welcome"

# C++ OpenCV code
cxx"""
void videocapture(int device_index, const char *winname) {

   // convert const char* => string
   std::string wname = winname;

   // open Video device
   cv::VideoCapture capture(device_index);

   // check that device is opened
   if (!capture.isOpened())
    {
        std::cout<< "Can not open device!";
    }

 while(true) {
   // Create a Mat frame and show it in a window
    cv::Mat frame;
    cv::namedWindow(wname, 0);
    bool Success = capture.read(frame);

    if (!Success)
    {
        std::cout << "Failure to acquire stream!";
        break;
    }

    cv::imshow(wname, frame);

    if (cv::waitKey(30) == 27)
    {
        std::cout << "Press esc to end acquisition.";
        cv::destroyWindow(wname);
        break;
    }

  }
}
"""

@cxx videocapture(device_index, pointer(winname))

############################################################################################
