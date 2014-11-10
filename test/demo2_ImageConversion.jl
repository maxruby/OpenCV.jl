#################################################################################################
#
# demo2: Convert a image from .png to .jpg format (with compression)
#
#################################################################################################

# Julia filename with full path
inputfile = "/Users/maximilianosuster/lena.png"
outfile = "/Users/maximilianosuster/lena.jpeg"

cxx"""
#include <iostream>
"""

cxx"""
void imageConversion(const char *inputfile, const char *outfile)
{
    const std::string ifname = inputfile;
    const std::string ofname = outfile;

    // Open video file
     cv::Mat img = cv::imread(ifname);

     if (img.empty())
     {
          std::cout<< "Cannot read the file!";
     }

     //cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

     cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
     cv::imshow("Original", img);  // jpeg, actually same image displayed in both windows,

     std::vector<int> compression_params; // vector that stores compression params
     compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
     compression_params.push_back(98); // specify % compression quality

     //compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
     //compression_params.push_back(9);

     // not working!
     bool Success = cv::imwrite(ofname, img, compression_params);
     //

     if(!Success)
     {
         std::cout<< "Failed to save the image!";
     }

     cv::namedWindow("Modified", cv::WINDOW_AUTOSIZE);
     cv::imshow("Modified", img); //  jpeg

     cv::waitKey(0);

     cv::destroyAllWindows();
}
"""

# Run the script
res = @cxx imageConversion(pointer(inputfile), pointer(outfile))
################################################################################################
