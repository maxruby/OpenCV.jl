
# type conversion
cint(n) = convert(Cint, n)    # same as int32
csize_t(n) = convert(Csize_t, n)
cchar(n) = convert(Int8,n)
cuchar(n) = convert(Cuchar, n)
cshort(n) = convert(Cshort, n)
cushort(n) = convert(Cushort, n)

# Search directory
searchdir(path,key) = filter(x->contains(x,key), readdir(path))
swapext(f, new_ext) = "$(splitext(f)[1])$new_ext"


# C++ std::vector class thin-wrappings for Julia
cxx"""
    template <typename T_>
    std::vector<T_> stdvector(int size, T_ x)
    {
        std::vector<T_> cppvec(size, x);
        return (cppvec);
    }

    template <typename T_>
    std::vector<std::vector<T_>> stdvector2(int rows, std::vector<T_> colVec)
    {
        std::vector<std::vector<T_>> cppvec2(rows, colVec);
        return (cppvec2);
    }

    template <typename T_>
    std::vector<std::vector<std::vector<T_>>> stdvector3(int rows, int cols, std::vector<T_> colVec)
    {
        std::vector<std::vector<T_>> vec2 = stdvector2(cols, colVec);
        std::vector<std::vector<std::vector<T_>>> cppvec3(rows, vec2);
        return (cppvec3);
    }

    template <typename T_>
    std::vector<T_> stdvectorSzt(std::size_t size, T_ x)
    {
        std::vector<T_> cppvec(size, x);
        return (cppvec);
    }

    template <typename T_>
    T_ at(std::vector<T_> cppvec, int index)
    {
         T_ value = cppvec[index];    // does not check for out of bounds (fast but dangerous)
         return(value);
    }

    template <typename T_>
    T_ at_(std::vector<T_> cppvec, int index)
    {
        T_ value = cppvec.at(index); // checks for out of bounds (safe but slow)
        return(value);
    }

    template <typename T_>
    void stdset(std::vector<T_>& cppvec, int index, T_ value)
    {
        cppvec[index] = value;
    }

    template <typename T_>
    void stdset_(std::vector<T_>& cppvec, int index, T_ value)
    {
        cppvec.at(index) = value; // checks for out of bounds (safe but slow)
    }

    template <typename T_>
    void stdvec2set_(std::vector<std::vector<T_>>& cppvec2, int row, int col, T_ value)
    {
         cppvec2[row][col] = static_cast<T_>(value);
    }

    template <typename T_>
    T_ stdvec2get_(std::vector<std::vector<T_>>& cppvec2, int row, int col)
    {
         // debugging
         std::cout << cppvec2[row][col] << std::endl;
         return cppvec2[row][col];
    }

    template <typename T1, typename T2>
    void stdvec3set_(std::vector<std::vector<std::vector<T1>>>& cppvec3, int row, int col, std::vector<T2> value)
    {
         std::vector<T1> convertedValue(value.begin(), value.end());
         cppvec3[row][col] = convertedValue;
         // debugging
         std::cout << convertedValue[0] << ", " << convertedValue[1] << ", " << convertedValue[2] << std::endl;
    }

    template <typename T_>
    std::vector<T_> stdvec3get_(std::vector<std::vector<std::vector<T_>>>& cppvec3, int row, int col)
    {
         // debugging
         std::cout << cppvec3[row][col][0] << ", " << cppvec3[row][col][1] << ", " << cppvec3[row][col][2] << std::endl;
         return cppvec3[row][col];
    }
    // Crucial to this implementation
    // - cv::DataType<T>::type
    // - efficient access to vector using pointers
    // Reference for cv::DataType<T>::type:
    // http://stackoverflow.com/questions/14291165/template-initialization-of-opencv-mat-from-vector
    template <typename T>
    cv::Mat stdvector2Mat(std::vector<std::vector<T>> vec2)
    {
          cv::Mat img = cv::Mat::zeros(vec2.size(), vec2[0].size(), cv::DataType<T>::type);

          for(int row = 0; row<vec2.size(); ++row) {
              T* p = img.template ptr<T>(row);
              for(int col = 0; col<vec2[0].size(); ++col) {
                 //points to each pixel value
                  *p++ = vec2[row][col];
              }
          }

          return img;
     }

     // conversion to cv::Mat_<T>
     template <typename T>
     cv::Mat_<T> stdvector2Mat_(std::vector<std::vector<T>> vec2)
     {
           cv::Mat_<T> img = cv::Mat_<T>(vec2.size(), vec2[0].size(), cv::DataType<T>::type);

           for(int row = 0; row<vec2.size(); ++row) {
               T* p = img.template ptr<T>(row);
               for(int col = 0; col<vec2[0].size(); ++col) {
                  // points to each pixel value
                   *p++ = vec2[row][col];
               }
           }

           return img;
     }

     template <typename T>
     cv::Mat stdvector3Mat(std::vector<std::vector<std::vector<T>>> vec3)
     {
          cv::Mat img = cv::Mat::zeros(vec3.size(), vec3[0].size(), cv::DataType<T>::type);

          // copy data
          for (int row=0; row<vec3.size(); row++)
          {
              for (int col=0; col<vec3[0].size(); col++)
              {
                 cv::Point3_<T>* p = img.template ptr<cv::Point3_<T>>(row,col);
                 p->x = vec3[row][col][0];  //B
                 p->y = vec3[row][col][1];  //G
                 p->z = vec3[row][col][2];  //R
              }
          }

          return img;
    }

     template <typename T>
     cv::Mat_<T> stdvector3Mat_(std::vector<std::vector<std::vector<T>>> vec3)
     {
          cv::Mat_<T> img = cv::Mat_<T>(vec3.size(), vec3[0].size(), cv::DataType<T>::type);

          // copy data
          for (int row=0; row<vec3.size(); row++)
          {
              for (int col=0; col<vec3[0].size(); col++)
              {
                cv::Point3_<T> p(vec3[row][col][0], vec3[row][col][1], vec3[row][col][2]);
                img(row, col) = p;
              }
          }

          return img;
    }

"""

stdvec(size, value) = @cxxnew stdvector(size, value)
stdvec2(rows, colVec) = @cxxnew stdvector2(rows, colVec)
stdvec3(rows, cols, colVec2) = @cxxnew stdvector3(rows, cols, colVec2)
stdvec2Mat_(vec2) = @cxx stdvector2Mat_(vec2)
stdvec3Mat_(vec3) = @cxx stdvector3Mat_(vec3)
stdvec2Mat(vec2) = @cxx stdvector2Mat(vec2)
stdvec3Mat(vec3) = @cxx stdvector3Mat(vec3)
stdvec2get(vec2, row, col) = @cxxnew stdvec2get_(vec2, row, col)
stdvec3get(vec3, row, col) = @cxxnew stdvec3get_(vec3, row, col)
stdvec2set(vec2, row, col, val) = @cxxnew stdvec2set_(vec2, row, col, val)
stdvec3set(vec2, row, col, val) = @cxxnew stdvec3set_(vec2, row, col, val)
stdvecSzt(size, value) = @cxxnew stdvectorSzt(csize_t(size), value)
stdassign(ccpvec, size, value) = @cxx ccpvec->assign(size,value)
stddata(cppvec) = @cxx cppvec->data()     # Ptr to first elememt
stdempty!(cppvec) = @cxx cppvec->empty()   # check if it is empty
stdcapacity(cppvec) = Int(@cxx cppvec->capacity())
stdpush!(cppvec, value) = @cxx cppvec->push_back(value)
stdpop!(cppvec) = @cxx cppvec->pop_back()
stdsize(cppvec) = Int(@cxx cppvec->size())
stdresize!(cppvec, n::Int) = @cxx cppvec->resize(n)
stdshrink!(cppvec) = @cxx cppvec->shrink_to_fit()
stdswap!(cppvec1, cppvec2) = @cxx cppvec1->swap(cppvec2)

# make sure index is stdsize -1 (C++ indexing starts at 0)
function at(cppvec, index::Int)
   (index < 0 || index > (stdsize(cppvec)-1)) ? throw(ArgumentError("index is out of bounds")) : nothing
   @cxx at(cppvec, index)
end
function at_(cppvec, index::Int)
  # make sure index is stdsize -1 (C++ indexing starts at 0)
  (index < 0 || index > (stdsize(cppvec)-1)) ? throw(ArgumentError("index is out of bounds")) : nothing
   @cxx at_(cppvec, index)   # out of bounds check  (will lead to crash if out of bounds)
end

function set!(cppvec, index, value)
   (index < 0 || index > (stdsize(cppvec)-1)) ? throw(ArgumentError("index is out of bounds")) : nothing
   @cxx stdset(cppvec, index, value)
end

clear(cppvec) = @cxx cppvec->clear()

# Converting julia Array{T,N} to std::vector<T> or std::vector<std::vector<T>>
# accepted input array types
# ndims = 1
#  [1,2,3,4]

# ndims = 2
#  [1,2,3,4,
#   5,6,7,8 ]

# ndims = 3
#  [ [200, 155, 200], [200, 155, 200], [200, 155, 200],
#    [200, 155, 200], [200, 155, 200], [200, 155, 200] ]


function tostdvec{T, N}(jl_vector::Array{T,N})

    if (ndims(jl_vector) === 1)
        # C++ compiler must deduce type from template functions
        vec = stdvec(0,jl_vector[1])

        for i=1:length(jl_vector)
           stdpush!(vec, jl_vector[i])  # index -1 (C++ has 0-indexing)
        end
        return(vec)

    elseif (ndims(jl_vector) === 2)
        rows = size(jl_vector, 1)
        cols = size(jl_vector, 2)

        # C++ compiler must deduce type from template functions
        colVec = stdvec(cols, jl_vector[1, 1])
        vec2 = stdvec2(rows, colVec)

        for row = 1: rows
            for col = 1: cols
                stdvec2set(vec2, row-1, col-1, jl_vector[row, col])
            end
        end
        return(vec2)

    elseif (ndims(jl_vector) === 3)
        rows = size(jl_vector, 2)
        cols = size(jl_vector, 3)

        # C++ compiler must deduce type from template functions
        vec = tostdvec(jlimg_[:, 1, 1])
        colVec3 = stdvec2(cols, vec)
        vec3 = stdvec3(rows, colVec3)

        for row = 1: rows
            for col = 1: cols
                stdvec3set(vec3, row-1, col-1, jl_vector[:, row, col])
            end
        end
        return(vec3)

    else
        throw(ArgumentError("Incompatible array dimensions"))
    end

end

# C++ string handling
# julia string => std::string
# std::string => julia string

cxx"""
std::string stdstring(const char* word)
      {
        std::string stdword = word;
        return(stdword);
      }
"""

stdstring(word::String) = @cxx stdstring(pointer(word))
juliastring(std_string) = bytestring(@cxx std_string->c_str())
stdstrdata(std_string) = @cxx std_string->data()     # Ptr to first elememt
stdstrempty(std_string) = @cxx std_string->empty()
stdstrsize(std_string) = @cxx std_string->size()
stdstrlength(std_string) = @cxx std_string->length()
stdstrclear(std_string) = @cxx std_string->clear()


#-------------------------------------------------------------------------------------------------------------------#
# Basic core utility functions

# Print Mat uchar array (very rough)
cxx"""
void printMat(const cv::Mat& mat)
{
    for(int i = 0; i < mat.rows; i++)
    {
        for(int j=0; j < mat.cols; j++)
        {
            std::cout << (int) mat.at<uchar>(i,j) << " "; break;
        }
    }
}
"""

printMat(img) = @cxx printMat(img)

#-------------------------------------------------------------------------------------------------------------------#
# Image processing (imgproc)

# imreplace - uses a rectangular ROI to copy a region from one image to another
cxx"""
cv::Mat imreplace(cv::Mat& src,cv::Mat& dst, cv::Rect roi)
{
    //initialize the ROI (location to copy)
    cv::Mat dstROI(dst(roi));
    //perform the copy
    src.copyTo(dstROI);
    return(dst);
}
"""
imreplace(src, dst, roi) = @cxx imreplace(src, dst, roi)


# Support for image convolution
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

# TO_DO:  Similar utility functions are likely required for other functions in imgproc


#-------------------------------------------------------------------------------------------------------------------#
# GUI interface (highui)

# Display an image in a window and close upon key press (and delay)
# For concurrent multiple display, set multi = "ON"
function imdisplay(img, windowName::String, flag=WINDOW_AUTOSIZE)
    namedWindow(windowName, flag)
    imshow(windowName, img)
end

function closeWindows(delay, key, windowName)
      (waitkey(delay) == key && windowName != "") ? destroyWindow(windowName) : destroyAllWindows()
end
# closeWindows(0,27,"")

function im2tile(imgArray, windowName::String, flag=WINDOW_AUTOSIZE)
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

    imdisplay(canvas, windowName, flag)
end


#-------------------------------------------------------------------------------------------------------------------#
# Video capture (video)

function videocam(device = CAP_ANY)
    cam = videoCapture(device)    # open Video device
    !isOpened(cam) ? throw(ArgumentError("Can not open camera!")) : nothing
    namedWindow("Video")
    frame = Mat()

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        imshow("Video", frame)

        if (waitkey(30) == 27)
            destroyAllWindows()
            break
       end
   end
  release(cam)
end

# Webstreaming
# src: full http link to the video source
# e.g., NASA TV: vid = "http://www.nasa.gov/multimedia/nasatv/NTV-Public-IPS.m3u8"
function webstream(src::String)
    cam = videoCapture(src)    # open Video device
    !isOpened(cam) ? throw(ArgumentError("Can not open stream!")) : nothing
    namedWindow("Web stream")
    frame = Mat()

    # Loop until user presses ESC or frame is empty
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video stream!"))
            break
        end

        imshow("Web stream", frame)

        if (waitkey(30) == 27)
            destroyAllWindows()
            break
       end
   end
  release(cam)
end


function videoWrite(cam, filename::String, fps::Float64, nframes = 0, frameSize=cvSize(0,0), fourcc=-1, isColor=true)
    !(isOpened(cam)) ? throw(ArgumentError("Video source is not available!")) : nothing

    # Set the video capture frame size based on camera input WIDTH and HEIGHT
    width = getVideoId(cam, CAP_PROP_FRAME_WIDTH)
    height = getVideoId(cam, CAP_PROP_FRAME_HEIGHT)

    frame = Mat(Int(height), Int(width), CV_8UC3)

    # Set the frameSize of output video frames
    (frameSize.data[1] == 0) ? frameSize = cvSize(Int(width), Int(height)) :
         throw(ArgumentError("Output frame dimension is wrong"))

    # Initialize and open video writer
    writer = videoWriter(filename, fourcc, fps, frameSize, isColor)
    openWriter(writer, filename, fourcc, fps, frameSize, isColor)
    !(isOpened(writer)) ? throw(ArgumentError("Can not open the video writer!")) : nothing

    # create a window for display
    namedWindow("Welcome!")

    # Loop until user presses ESC or frame is empty
    count = 0
    while(true)
        if !(videoRead(cam, frame))
            throw(ArgumentError("Can not acquire video!"))
            break
        end

        writeVideo(writer, frame)
        imshow("Welcome!", frame)

        (nframes > 0) ? count += 1 : nothing
        if ((waitkey(30) == 27) || (count == nframes))
            destroyAllWindows()
            break
       end
   end
   release(cam)
end
