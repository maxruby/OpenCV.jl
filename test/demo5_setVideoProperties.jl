################################################################################################
#
# demo 5: set video properties
#
################################################################################################

# Julia variables
winname = "Welcome"
fname = joinpath(Pkg.dir("OpenCV"), "./test/images/movie.avi")

# C++ headers
cxx""" #include <iostream> """

# C++ OpenCV code
cxx"""
void setVideoProperties(const char *winname, const char *fname)
{
    // convert const char* => string
    std::string wname = winname;
    std::string fvideo = fname;

    // Open video file
    cv::VideoCapture capture(fvideo);

    if (!capture.isOpened())
    {
        std::cout << "Cannot open the video file!" << std::endl;
        exit(0);
    }

//    Properties to control:
//    int propID
//    cv::CAP_PROP_POS_MSEC - current position of the video in milliseconds
//    cv::CAP_PROP_POS_FRAMES - current position of the video in frames
//    cv::CAP_PROP_FRAME_WIDTH - width of the frame of the video stream
//    cv::CAP_PROP_FRAME_HEIGHT - height of the frame of the video stream
//    cv::CAP_PROP_FPS - frame rate (frames per second)
//    cv::CAP_PROP_FOURCC - four character code  of codec

    capture.set(cv::CAP_PROP_POS_MSEC, 300); //300 ms into the video
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 600);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 400);
    capture.set(cv::CAP_PROP_FPS, 5);

    // Get the fps
    double fps = capture.get(cv::CAP_PROP_FPS);

    std::cout << "Frame per second: "<< fps << std::endl;
    cv::namedWindow("Welcome", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;

    while(true)
    {
        bool bSuccess = capture.read(frame);
        if (!bSuccess)
        {
             std::cout << "Can not read the video frame!"<< std::endl;
             break;
        }

        cv::imshow("Welcome", frame);

        if (cv::waitKey(30)== 27)  // wait for esc key(27) press for (30) msec
        {
            std::cout << "Pressed esc to stop."<< std::endl;
            cv::destroyWindow("Welcome");
            break;
        }
    }
}
"""

@cxx setVideoProperties(pointer(winname), pointer(fname))
