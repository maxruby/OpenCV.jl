################################################################################################
#
# OpenCV_cxx.jl
#
# Wrapper for OpenCV structures and functions in Julia with Cxx.jl
#
# Mainly returns a pointer to OpenCV C++ objects which Julia can handle
################################################################################################

# Basic Structures
# Mat::Mat
# modules/core/include/opencv2/core/mat.hpp

# class CV_EXPORTS Mat
# {
# public:
#     // ... a lot of methods ...
#     ...

#     /*! includes several bit-fields:
#          - the magic signature
#          - continuity flag
#          - depth
#          - number of channels
#      */
#     int flags;
#     //! the array dimensionality, >= 2
#     int dims;
#     //! the number of rows and columns or (-1, -1) when the array has more than 2 dimensions
#     int rows, cols;
#     //! pointer to the data
#     uchar* data;

#     //! pointer to the reference counter;
#     // when array points to user-allocated data, the pointer is NULL
#     int* refcount;

#     // other members
#     ...
# };

# (rows = pixels, columns = pixels, format, Scalar(B = 0:255,G = 0:255,R = 0:255) e.g., CV_8UC3
# formats:
# CV_8U       8-bit unsigned (0:255)
# CV_8S       8-bit signed (-128:127)
# CV_16U     16-bit unsigned
# CV_16S     16-bit signed
# CV_32S     32-bit signed
# CV_32F     32-bit float
# CV_64F     64-bit float

# Parameters:
# ndims –  dimension
# rows –   Number of rows in a 2D array
# cols –   Number of columns in a 2D array

# roi –    Region of interest
# size –   2D array size: Size(cols, rows)
# sizes –  Integer array
# type –   Array type CV_8UC(n), ..., CV_64FC(n)

# s  –     l value to initialize each matrix element with.
#          To set all the matrix elements to the particular value after the construction,
#          use the assignment operator Mat::operator=(const Scalar& value).

# data –   Pointer to the user data.
# step –   Number of bytes each matrix row occupies, if missing set to AUTO_STEP.
#          cols*elemSize(). See Mat::elemSize().
#          The value should include the padding bytes at the end of each row.
# steps –  Array of ndims-1 steps in case of a multi-dimensional array
#          (the last step is always set to the element size).
#          If not specified, the matrix is assumed to be continuous.

# m –      Array that (as a whole or partly) is assigned to the constructed matrix.
#          No data is copied by these constructors.  Instead, the header pointing to m data or its sub-array is constructed and associated with it. The reference counter, if any, is incremented. So, when you modify the matrix formed using such a constructor, you also modify the corresponding elements of m . If you want to have an independent copy of the sub-array, use Mat::clone() .

# img –    Pointer to the old-style IplImage image structure.

# vec –    STL vector whose elements form the matrix.
#          1-column matrix =>  n rows = n vector elem
#          matrix Type => matches type of vector elem
#          arbitrary types OK, as long as declared DataType.
#          elems => primitive numbers or uni-type numerical tuples of numbers.
#          ***Mixed-type structures are not supported.***
#          "Mat(vec)" declared explicitly.
#          if copyData=true (see next), no new elements will be added to the vector

# copyData – Flag for STL vector, true = copied, false =  shared the newly constructed matrix.
#            (old-style CvMat or IplImage)
# rowRange – Range of the m rows to take.    Range::all()=> take all the rows.
# colRange – Range of the m columns to take. Range::all()=> take all the columns.
# ranges –   Array of selected ranges of m along each dimensionality.
################################################################################################

# go through the documentation here
# http://docs.opencv.org/trunk/index.html

#DataTypes

# size
cvSize(width::Int, height::Int) = @cxx cv::Size(width,heigth)

# Scalar
cvScalar(blue::Int, green::Int, red::Int) = @cxx cv::Scalar(blue,green,red)

# Mat constructors

# Mat::Mat()
cvMat() = @cxx Mat()

# Mat::Mat(int rows, int cols, int type)
cvMat(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat(rows, cols, matType)

# Mat::Mat(Size size, int type)
cvMat(cvSize, matType::CV_MatType) = @cxx cv::Mat(cvSize, matType)

# Mat::Mat(int rows, int cols, int type, const Scalar& s)
cvMat(rows::Int, cols::Int, matType::CV_MatType, cvScalar) = @cxx cv::Mat(rows, cols, matType, cvScalar)

# Mat::Mat(Size size, int type, const Scalar& s)
cvMat(cvSize, matType::CV_MatType, cvScalar) = @cxx cv::Mat(cvSize, matType, cvScalar)

# Mat::Mat(const Mat& m)
# cvMat() = @cxx Mat(const Mat& m)

# Mat::Mat(int ndims, const int* sizes, int type)
const psizes = pointer([sizes::Int])
cvMat(ndims::Int, psizes, matType::CV_MatType) = @cxx cv::Mat(ndims, psizes, matType)

# Mat::Mat(int ndims, const int* sizes, int type, const Scalar& s)
const psizes = pointer([sizes::Int])
cvMat(ndims::Int, psizes, matType::CV_MatType, cvScalar) = @cxx cv::Mat(ndims, psizes, matType, cvScalar)

# Mat::Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0)

# Mat::Mat(const Mat& m, const Rect& roi)

# Mat::zeros()
cvMatzeros(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::zeros(rows, cols, matType)
# Mat::ones()
cvMatones(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::ones(rows, cols, matType)
# Mat::eye()
cvMateye(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::eye(rows, cols, matType)

# C++: Mat::Mat(const Mat& m)
# C++: Mat::Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP)
# C++: Mat::Mat(Size size, int type, void* data, size_t step=AUTO_STEP)
# C++: Mat::Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all() )
# C++: template<typename T, int n> explicit Mat::Mat(const Vec<T, n>& vec, bool copyData=true)
# C++: template<typename T, int m, int n> explicit Mat::Mat(const Matx<T, m, n>& vec, bool copyData=true)
# C++: template<typename T> explicit Mat::Mat(const vector<T>& vec, bool copyData=false)
# C++: Mat::Mat(const Mat& m, const Range* ranges)


# Mat::convertTo
img = @cxx Mat()
@cxx img->convertTo(cv::OutputArray m, int rtype, double alpha=1, double beta=0)

# GUI functions
# void imshow(const String& winname, InputArray mat)
imshow(windowName::Ptr{Uint8}, img) = @cxx cv::imshow(windowName, img)




imshow(img, windowName::Ptr{Uint8}, set::WindowProperty, key::Int, delay::Int) = @cxx cv::imshow(img, windowName, set, key, delay)


cxx"""
   void imshow(cv::Mat *img, const char *winname, int WindowProperty, int key, int delay) {

       // Create a new window named
       const std::string winName = winname;

       //WINDOW_NORMAL => user can resize the window (no constraint)
       //WINDOW_AUTOSIZE => the window size is automatically adjusted
       //WINDOW_OPENGL => OpenGL support

       cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);

       // Show the image in window
       cv::imshow(winName, *img);

       //  Wait for command to close window
       if (cv::waitKey(delay) == key) {
           cv::destroyWindow(winName);
       }
    }
"""


cvSize(width::Int, height::Int)
cvScalar(blue::Int, green::Int, red::Int)


# Create a gray image 600 x 600
img = cvMat(600, 600, CV_8UC1)

# Show the image
imshow(img, pointer("Welcome!"), WINDOW_AUTOSIZE, 27, 30)
