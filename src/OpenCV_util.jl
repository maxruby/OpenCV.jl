
# Custom utility functions

# Display an image in a window and close upon key press (and delay)
function imdisplay(img, windowName::String, flag::WindowProperty, delay, key)
    namedWindow(pointer(windowName), flag)
    imshow(pointer(windowName), img)    #img::CppValue{symbol("cv::Mat"),()}
    waitkey(delay) == key ? destroyWindow(pointer(windowName)) : nothing
end



