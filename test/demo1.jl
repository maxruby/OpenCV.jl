################################################################################################
#
# demo1: Create a blank image (specifying color channels) and display it
#
################################################################################################

# Image properties
width = 300
height = 300
color = cvScalar(255, 0, 0)  # BGR: blue image

# Window name
winname = "New Image"

# Create a new image width x height, 8-bit unsigned RGB
img = Mat(width, height, CV_8UC3)
imdisplay(img, winname, WINDOWS_AUTOSIZE, 0, 27)  # waitkey(0) = 27  (ESC)

