#################################################################################################
# Image Histogram with OpenCV.jl
#
# http://docs.opencv.org/trunk/modules/imgproc/doc/histograms.html?highlight=hist#threshhist
#################################################################################################

# Julia filename with full path
inputfile = "/Users/maximilianosuster/programming/ComputerVision/testimages/mandrill.jpg"
outfile = "/Users/maximilianosuster/programming/ComputerVision/testimages/imghistogram.jpg"

# header files
cxx"""
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
"""

#Main code
cxx"""
void OpenCV_ImgHist(const char *inputfile, const char *outfile) {

    const std::string ifname = inputfile;
    const std::string ofname = outfile;

    cv::Mat src, hsv;
    src = cv::imread(ifname);

    if(src.empty()) {
        std::cout << "Can not load image!" << std::endl;
        exit(0);
    }

    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    cv::MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    cv::calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scale = 20;
    cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*20, CV_8UC3);

    for( int h = 0; h < hbins; h++ )
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            cv::rectangle(histImg, cv::Point(h*scale, s*scale),
                        cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                        cv::Scalar::all(intensity),
                        cv::FILLED);
        }

    cv::namedWindow("Source", 1);
    cv::imshow("Source", src);

    cv::namedWindow("H-S Histogram", 1);
    cv::imshow("H-S Histogram",  histImg);

    cv::waitKey(0);
    cv::imwrite(ofname, histImg);
    cv::destroyAllWindows();
}
"""

jlOpenCV_ImgHist = @cxx OpenCV_ImgHist(pointer(inputfile), pointer(outfile))
