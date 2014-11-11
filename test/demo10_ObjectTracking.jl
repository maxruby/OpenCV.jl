#################################################################################################
#
# demo 10:  Live object tracking based on color thresholding
#
# based on http://opencv-srf.blogspot.no/2010/09/object-detection-using-color-seperation.html
# We use moments to calculate position of a single red object with HSV values
#
#################################################################################################

device_index = CAP_ANY

cxx"""
#include <iostream>
#include <stdio.h>
"""

cxx"""
void ObjectTracking(int device_index)
{
    cv::VideoCapture capture(device_index);  // capture the video from webcam

    if (!capture.isOpened())   //  if no success, exit program
    {
        std::cout << "Cannot open the web cam";
        exit(0);
    }

    cv::namedWindow("Control",  cv::WINDOW_AUTOSIZE); //create a window called "Control"
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE); //create a window called "Original"
    cv::namedWindow("Thresholded Image", cv::WINDOW_AUTOSIZE); //create a window called "Original"

    int iLowH = 170;
    int iHighH = 179;

    int iLowS = 150;
    int iHighS = 255;

    int iLowV = 60;
    int iHighV = 255;

    //For HSV codes, go to http://www.rapidtables.com/web/color/color-picker.htm

    //Create trackbars in "Control" window
    cv::createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cv::createTrackbar("HighH", "Control", &iHighH, 179);

    cv::createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cv::createTrackbar("HighS", "Control", &iHighS, 255);

    cv::createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
    cv::createTrackbar("HighV", "Control", &iHighV, 255);

    int iLastX = -1;
    int iLastY = -1;

    //Capture a temporary image from the camera
    cv::Mat imgTmp;
    capture.read(imgTmp);

    //Create a black image with the size as the camera output
    cv::Mat imgLines = cv::Mat::zeros(imgTmp.size(), CV_8UC3);

    cv::Mat imgOriginal;

    while (true)
    {
        // read a new frame from video
        bool bSuccess = capture.read(imgOriginal);

        if (!bSuccess) //if not success, break loop
        {
            std::cout << "Cannot read a frame from video stream";
            break;
        }

        cv::Mat imgHSV;

        cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        cv::Mat imgThresholded;

        cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
        cv::dilate(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

        //Calculate the moments of the thresholded image
        cv::Moments oMoments = moments(imgThresholded);

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        // if the area <= 10000
        if (dArea > 10000)
        {
            //calculate the position of the object
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
            {
                //Draw a red line from the previous point to the current point
                cv::line(imgLines, cv::Point(posX, posY), cv::Point(iLastX, iLastY), cv::Scalar(0,0,255), 2);
            }

            iLastX = posX;
            iLastY = posY;
        }

        imgOriginal = imgOriginal + imgLines;
        cv::imshow("Original", imgOriginal);
        cv::imshow("Thresholded Image", imgThresholded); //show the thresholded image

        //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        if (cv::waitKey(30) == 27)
        {
            std::cout << "esc key is pressed by user";
            cv::destroyAllWindows();
            break;
        }
     }
}
"""

@cxx ObjectTracking(device_index)

