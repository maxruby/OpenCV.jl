################################################################################################
#
# demo1: Create a blank image (specifying color channels) and display
#
################################################################################################

# Image properties
width = 300
height = 300

# Color
R=255
G=0
B=0

# Window name
fun = "New Image"

cxx"""
void create_image_and_display()
   {
   // Create a new image width x height, 8-bit unsigned RGB (BGR in OpenCV)
   cv::Mat img($width, $height, CV_8UC3, cv::Scalar($B,$G,$R));  // Blue, Green, Red (0:255)

   // Create a new window named "Welcome"
   cv::namedWindow("Welcome", cv::WINDOW_AUTOSIZE);

   // Show the image in window
   cv::imshow("Welcome", img);

   // Writing a message to the REPL from Julia
   $:(println("\nTo end this test, press any key")::Nothing);

   // Wait for key press
   cv::waitKey(0);

   // Destroy the window
   cv::destroyWindow("Welcome");

 }
"""

@cxx create_image_and_display()
