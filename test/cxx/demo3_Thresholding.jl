#################################################################################################
# demo 3: Thresholding with OpenCV.jl
#
# http://docs.opencv.org/trunk/modules/imgproc/doc/miscellaneous_transformations.html#threshold
# http://docs.opencv.org/trunk/doc/tutorials/imgproc/threshold/threshold.html#basic-threshold
#################################################################################################

# Julia filename with full path
inputfile = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.png")
outfile = joinpath(Pkg.dir("OpenCV"), "./test/images/lena.jpeg")

#Main code
cxx"""
// global variables
int threshold_value = 0;
int const max_value = 255;

void imThreshold(const char *inputfile, const char *outfile) {

  const std::string ifname = inputfile;
  const std::string ofname = outfile;

  cv::Mat src = cv::imread(ifname);
  cv::Mat img_gray, img_thresh;

  if (src.empty()) {
    std::cout << "Can not open the file!" << std::endl;
    exit(0);
  }

  // Create windows to display images
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Blur", cv::WINDOW_AUTOSIZE);

  // Convert the image to Gray
  cv::cvtColor(src, img_gray, cv::COLOR_RGB2GRAY);

  // Gaussian filtering
  cv::GaussianBlur(img_gray, img_gray, cv::Size(5,5), 0);

  // Display original and blurred images
  cv::imshow("Original", src);
  cv::imshow("Blur", img_gray);

       // cv::THRESH_BINARY
       // cv::THRESH_OTSU
       // cv::THRESH_BINARY_INV
       // cv::THRESH_TRUNC
       // cv::THRESH_TOZERO
       // cv::THRESH_TOZERO_INV

  cv::threshold(img_gray, img_thresh, threshold_value, max_value, cv::THRESH_BINARY + cv::THRESH_OTSU);

  imshow("Thresholded", img_thresh);

  cv::waitKey(0);
  cv::imwrite(ofname, img_thresh);
  cv::destroyAllWindows();
}
"""

@cxx imThreshold(pointer(inputfile), pointer(outfile))
