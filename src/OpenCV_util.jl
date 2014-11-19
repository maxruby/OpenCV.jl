
# Custom utility functions

# Make a Dict() for lookup of image formats (CV_Mat types)
CV_MAT_TYPE= Dict( 0 => "CV_8UC1",
                   8 => "CV_8UC2",
                  16 => "CV_8UC3",
                  24 => "CV_8UC4",
                   1 => "CV_8SC1",
                   9 => "CV_8SC2",
                  17 => "CV_8SC3",
                  25 => "CV_8SC4",
                   2 => "CV_16UC1",
                  10 => "CV_16UC2",
                  18 => "CV_16UC3",
                  26 => "CV_16UC4",
                   3 => "CV_16SC1",
                  11 => "CV_16SC2",
                  19 => "CV_16SC3",
                  27 => "CV_16SC4",
                   4 => "CV_32SC1",
                  12 => "CV_32SC2",
                  20 => "CV_32SC3",
                  28 => "CV_32SC4",
                   5 => "CV_32FC1",
                  13 => "CV_32FC2",
                  21 => "CV_32FC3",
                  29 => "CV_32FC4",
                   6 => "CV_64FC1",
                  14 => "CV_64FC2",
                  22 => "CV_64FC3",
                  30 => "CV_64FC4")

# Display an image in a window and close upon key press (and delay)
function imdisplay(img, windowName::String, flag::WindowProperty, delay, key)
    namedWindow(pointer(windowName), flag)
    imshow(pointer(windowName), img)    #img::CppValue{symbol("cv::Mat"),()}
    waitkey(delay) == key ? destroyWindow(pointer(windowName)) : nothing
end



