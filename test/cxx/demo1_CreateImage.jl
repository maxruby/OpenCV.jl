################################################################################################
#
# demo1: Create a blank color image and display it
#
################################################################################################

# Image properties
width = cint(300)
height = cint(300)

# Color
R=cint(255)
G=cint(0)
B=cint(0)

cxx"""
void create_image_and_display(int width, int height, int B, int G, int R)
   {
   // Create a new image width x height, 8-bit unsigned RGB (BGR in OpenCV)
   cv::Mat img(width, height, CV_8UC3, cv::Scalar(B,G,R));  // Blue, Green, Red (0:255)

   // Create a new window named "Welcome"
   cv::namedWindow("Welcome", cv::WINDOW_AUTOSIZE);

   // Show the image in window
   cv::imshow("Welcome", img);

   // Writing a message to the REPL from Julia
   std::cout << "\nTo end this test, press any key" << std::endl;

   // Wait for key press
   cv::waitKey(0);

   // Destroy the window
   cv::destroyWindow("Welcome");

 }
"""

@cxx create_image_and_display(width, height, B, G, R)
