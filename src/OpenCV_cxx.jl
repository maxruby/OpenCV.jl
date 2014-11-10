################################################################################################
#
# OpenCV_cxx.jl
# Wrapping OpenCV structures and functions in Julia using Cxx.jl
#
# Mainly returns a pointer to a C++ object which Julia can handle
################################################################################################

# Basic Structures
# Mat::Mat
# modules/core/include/opencv2/core/mat.hpp

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


# 1. Mat::Mat()
cxx"""
    cv::Mat *emptyMat() {
        cv::Mat *Mat = new cv::Mat;
        return (Mat);
    }
"""

# 2. Mat::Mat(int rows, int cols, int type)
cxx"""
    cv::Mat *basicMat(int rows, int cols, int type) {
        cv::Mat *Mat = new cv::Mat(rows, cols, type);
        return (Mat);
    }
"""

# 3. Mat::Mat(int rows, int cols, int type, const Scalar& s)
cxx"""
    cv::Mat *colorMat(int rows, int cols, int type, int Red, int Green, int Blue) {
        cv::Mat *Mat = new cv::Mat(rows, cols, type, cv::Scalar(Blue, Green, Red));
        return (Mat);
    }
"""

# 4. cv::imshow()
cxx"""
   void imshow(cv::Mat *img, const char *winname, uint32 flags, int key, int delay) {

       // Create a new window named
       const std::string winName = winname;

       //WINDOW_NORMAL => user can resize the window (no constraint)
       //WINDOW_AUTOSIZE => the window size is automatically adjusted
       //WINDOW_OPENGL => OpenGL support

       cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

       // Show the image in window
       cv::imshow(winName, *img);

       //  Wait for command to close window
       if (cv::waitKey(delay) == key) {
          cv::destroyWindow()
       }
   }
"""


# 4.




# Julia declarations
# See http://julia.readthedocs.org/en/latest/manual/functions/

jl_Scalar = Dict{Symbol,Int}(:B => 0, :G => 0, :R => 0)
# # pcpp"cv::Scalar(255,0,0)"
# rcpp"cv::Scalar(255,0,0)"

function Mat()
    img = @cxx Mat()
end

function Mat(width::Int, height::Int, MatType::CV_MatType, jl_Scalar::Dict{Symbol,Int})
    R = jl_Scalar[:R]
    B = jl_Scalar[:B]
    G = jl_Scalar[:G]
    img = @cxx Mat($width, $height, CV_MatType, rcpp"cv::Scalar($B,$G,$R)");
end

