####################################################################################################
# highgui. High-level GUI and Media I/O
####################################################################################################

# User Interface

# createTrackbar: Creates a trackbar and attaches it to the specified window
# 1. create a TrackbarCallback function in C++ and wrap with @cxx macro
# 2. call createTrackbar() below
createTrackbar(trackbarname::Ptr{Uint8},winname::Ptr{Uint8}, value::Ptr{Int}, count::Int, onChange,
    userdata=Ptr{Void}[0]) = @cxx cv::createTrackbar(trackbarname,winname, value, count, onChange, userdata)
# trackbarname – Name of the created trackbar
# winname      – Name of the window that will be used as a parent of the created trackbar
# value        – Optional pointer to an integer variable whose value reflects the position of the slider
# count        – Maximal position of the slider. The minimal position is always 0.
# onChange     – Pointer to the function to be called every time the slider changes position.
#                 This function should be prototyped as void Foo(int,void*)
# userdata     – User data that is passed as is to the callback

# getTrackbarPos: Returns the trackbar position
getTrackbarPos(trackbarname::Ptr{Uint8},winname::Ptr{Uint8}) = @cxx cv::getTrackbarPos(trackbarname,winname)

# namedWindow(const String& winname, int flags=WINDOW_AUTOSIZE)
namedWindow(windowName::Ptr{Uint8}, flags=WINDOW_AUTOSIZE) = @cxx cv::namedWindow(windowName, flags)

# void imshow(const String& winname, InputArray mat)
imshow(windowName::Ptr{Uint8}, img) = @cxx cv::imshow(windowName, img)

# waitKey
waitkey(delay) = @cxx cv::waitKey(delay)

# destroy GUI window(s)
destroyWindow(windowName::Ptr{Uint8}) = @cxx cv::destroyWindow(windowName)
destroyAllWindows() = @cxx cv::destroyAllWindows()

# MoveWindow
moveWindow(winname::Ptr{Uint8}, x::Int, y::Int) = @cxx cv::moveWindow(winname, x, y)
# x – The new x-coordinate of the window
# y – The new y-coordinate of the window

# ResizeWindow
resizeWindow(winname::Ptr{Uint8}, width::Int, height::Int) = @cxx cv::resizeWindow(winname, width, height)

# updateWindow
updateWindow(winname::Ptr{Uint8}) = @cxx cv::updateWindow(winname)

# SetMouseCallback: Sets mouse handler for the specified window
# create MouseCallback function wrapped in @cxx
setMouseCallback(winname::Ptr{Uint8}, onMouse, userdata=Ptr{Void}[0]) =
    @cxx cv::setMouseCallback(winname, onMouse, userdata)

# setTrackbarPos (by value)
setTrackbarPos(trackbarname::Ptr{Uint8}, winname::Ptr{Uint8}, pos::Int) =
   @cxx cv::setTrackbarPos(trackbarname, winname, pos)

