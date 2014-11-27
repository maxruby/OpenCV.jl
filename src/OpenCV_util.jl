
# General utility functions
cint(x) = convert(Cint, x)
csize_t(x) = convert(Csize_t, x)


#-------------------------------------------------------------------------------------------------------------------#
# Image processing (imgproc)

# Support functions for image convolution
cxx"""
// Function getSum returns total sum of all the elements of given matrix.

float getKernelSum(cv::Mat kernel)
{
   float sum = 0;
   for(typeof(kernel.begin<float>()) it = kernel.begin<float>(); it!= kernel.end<float>() ; it++)
   {
      sum+=*it;
   }
   return sum;
}

// Normalize the mask (kernel)
cv::Mat normalizeKernel (cv::Mat kernel, double ksum) {
    kernel = kernel/ksum;
    return(kernel);
    }
"""

# kernel = ones(5,5,CV_32F)
getKernelSum(kernel) = @cxx getKernelSum(kernel)
# ksum = getKernelSum(kernel)
normalizeKernel(kernel, ksum) = @cxx normalizeKernel(kernel, ksum)



#-------------------------------------------------------------------------------------------------------------------#
# GUI interface (highui)

# Display an image in a window and close upon key press (and delay)
# For concurrent multiple display, set multi = "ON"
function imdisplay(img, windowName::String, multi="OFF", flag=WINDOW_AUTOSIZE, delay=0, key=27)
    namedWindow(windowName, flag)
    imshow(windowName, img)
    (multi == "OFF" && waitkey(delay) == key) ? destroyWindow(windowName) : nothing
end

macro closeWindows(delay, key, windowName)
      (waitkey(delay) == key && windowName != "") ? destroyWindow(windowName) : destroyAllWindows()
end
# @closeWindows(0,27)

function im2canvas(imgArray, windowName::String, flag=WINDOW_AUTOSIZE, delay=0, key=27)
    canvas = Mat()

    for i=1:length(imgArray)
        # check that images have same dims, format and channels
        if (i > 1)
            (cvtypeval(imgArray[i]) != cvtypeval(imgArray[i-1]) ||
             rows(imgArray[i]) != rows(imgArray[i-1]) ||
             cols(imgArray[i]) != cols(imgArray[i-1])) ?
          throw(ArgumentError("Images must have same dimensions and format")): nothing
        end
        push(canvas, imgArray[i])
    end

    imdisplay(canvas, windowName, flag, delay, key)
end
