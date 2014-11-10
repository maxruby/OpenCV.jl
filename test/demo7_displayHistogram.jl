#################################################################################################
#
# demo 7:  Image Histogram in OpenCV.jl
#
# https://github.com/bsdnoobz/opencv-code/blob/master/display-histogram.cpp
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

cxx"""
void showHistogram(cv::Mat& img)
{
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels

	std::vector<cv::Mat> hist(nc);      // histogram arrays

	// Initalize histogram arrays
	for (int i = 0; i < hist.size(); i++)
		hist[i] = cv::Mat::zeros(1, bins, CV_32SC1);

	// Calculate the histogram of the image
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<cv::Vec3b>(i,j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}

	// For each histogram arrays, obtain the maximum (peak) value
	// Needed to normalize the display later
	int hmax[3] = {0,0,0};
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < bins-1; j++)
			hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = {"blue", "green", "red"};
	cv::Scalar colors[3] = {cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255)};

	std::vector<cv::Mat> canvas(nc);

	// Display each histogram in a canvas
	for (int i = 0; i < nc; i++)
	{
		canvas[i] = cv::Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
		{
			cv::line(
				canvas[i],
				cv::Point(j, rows),
				cv::Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
				nc == 1 ? cv::Scalar(200,200,200) : colors[i],
				1, 8, 0
			);
		}

		cv::imshow(nc == 1 ? "value" : wname[i], canvas[i]);
	}
}

// Test the `showHistogram` function above
void imHist(const char *inputfile, const char *outfile)
{
	cv::Mat src = cv::imread(inputfile);
	if (src.empty()) {
     std::cout << "Can not open the file!" << std::endl;
	   exit(0);
  }
	showHistogram(src);
	cv::imshow("src", src);
	cv::waitKey(0);
  cv::destroyAllWindows();
}
"""

jl_imHist = @cxx imHist(pointer(inputfile), pointer(outfile))
