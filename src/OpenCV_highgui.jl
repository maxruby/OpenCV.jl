####################################################################################################
# highgui. High-level GUI and Media I/O
####################################################################################################

# User Interface

# createTrackbar: Creates a trackbar and attaches it to the specified window
# 1. create a TrackbarCallback function in C++ and wrap with @cxx macro
# 2. call createTrackbar() below

createTrackbar(trackbarname::String, winname::String, value::Ptr{Int32}, count::Int) =
     @cxx cv::createTrackbar(pointer(trackbarname),pointer(winname), value, count)

# createTrackbar(trackbarname::String, winname::String, value::Int, count::Int, onChange=C_NULL,
#     userdata=Ptr{Void}[0][1]) = @cxx cv::createTrackbar(pointer(trackbarname),pointer(winname), pointer([cint(value)]), count, onChange, userdata)

# trackbarname – Name of the created trackbar
# winname      – Name of the window that will be used as a parent of the created trackbar
# value        – Optional pointer to an integer variable whose value reflects the position of the slider
# count        – Maximal position of the slider. The minimal position is always 0.
# onChange     – Pointer to the function (TrackbarCallback) to be called every time the slider changes position.
#                 This function should be prototyped as void Foo(int,void*)
# userdata     – User data (void*) that is passed as is to the callback

# getTrackbarPos: Returns the trackbar position
getTrackbarPos(trackbarname::String,winname::String) = @cxx cv::getTrackbarPos(pointer(trackbarname),pointer(winname))

# namedWindow(const String& winname, int flags=WINDOW_AUTOSIZE)
namedWindow(windowName::String, flags=WINDOW_AUTOSIZE) = @cxx cv::namedWindow(pointer(windowName), flags)

# void imshow(const String& winname, InputArray mat)
imshow(windowName::String, img) = @cxx cv::imshow(pointer(windowName), img)

# waitKey
waitkey(delay) = @cxx cv::waitKey(delay)

# destroy GUI window(s)
destroyWindow(windowName::String) = @cxx cv::destroyWindow(pointer(windowName))
destroyAllWindows() = @cxx cv::destroyAllWindows()

# MoveWindow
moveWindow(winname::String, x::Int, y::Int) = @cxx cv::moveWindow(pointer(winname), x, y)
# x – The new x-coordinate of the window
# y – The new y-coordinate of the window

# ResizeWindow
resizeWindow(winname::String, width::Int, height::Int) = @cxx cv::resizeWindow(pointer(winname), width, height)

# SetMouseCallback: Sets mouse handler for the specified window
# create MouseCallback function wrapped in @cxx
setMouseCallback(winname::String, onMouse, userdata=Ptr{Void}[0]) =
    @cxx cv::setMouseCallback(pointer(winname), onMouse, userdata)

# Gets the mouse-wheel motion delta (multiple of 120)
if is_windows()
    # EVENT_MOUSEWHEEL and EVENT_MOUSEHWHEEL
    getMouseWheelDelta(flags::Int) = @cxx getMouseWheelDelta(flags)
end

# setTrackbarPos (by value)
setTrackbarPos(trackbarname::String, winname::String, pos::Int) =
   @cxx cv::setTrackbarPos(pointer(trackbarname), pointer(winname), pos)
