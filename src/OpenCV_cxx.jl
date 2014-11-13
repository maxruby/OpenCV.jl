################################################################################################
#
# OpenCV_cxx.jl
# Julia wrapper of OpenCV structures and functions
#
#
################################################################################################

# Mat: The core image array structure in OpenCV

# Parameters:
# ndims –  dimension
# rows –   Number of rows in a 2D array
# cols –   Number of columns in a 2D array

# roi –    Region of interest
# size –   2D array size: Size(cols, rows)
# sizes –  Integer array
# type –   # select using CV_MatType in OpenCV_hpp.jl

           # Array type CV_8UC(n), ..., CV_64FC(n)
           # CV_8U       8-bit unsigned (0:255)
           # CV_8S       8-bit signed (-128:127)
           # CV_16U     16-bit unsigned
           # CV_16S     16-bit signed
           # CV_32S     32-bit signed
           # CV_32F     32-bit float
           # CV_64F     64-bit float
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
# DataTypes
################################################################################################

# 1) Point
cvPoint(x, y) = @cxx cv::Point(x,y)
cvPoint2f(x, y) = @cxx cv::Point2f(float32(x),float32(y))
cvPoint2d(x::Float64, y::Float64) = @cxx cv::Point2d(x,y)
# cvPoint3(x, y, z) = @cxx cv::Point3(x,y,z)

# 2) Size
cvSize(width, height) = @cxx cv::Size(width,height)
cvSize2f(width, height) = @cxx cv::Size(float32(width),float32(height))

# 3) Scalar
cvScalar(blue::Int, green::Int, red::Int) = @cxx cv::Scalar(blue,green,red)

# 4) Range
cvRange(start::Int, tend::Int) = @cxx cv::Range(start, tend)
cvRange_all(range) = @cxx range->all()

# 5) Rectangle
cvRect(x::Int, y::Int, width::Int, height::Int) = @cxx cv::Rect(x, y, width, height)

# 6) Rotated Rectangle
cvRotatedRect() =  @cxx cv::RotatedRect()
# const center = cvPoint2f(x,y)
# const size = cv::Size2f(width,height)
cvRotatedRect(center, size, angle::Float32) = @cxx cv::RotatedRect(center, size, angle)
# const point1 = cvPoint2f(x,y)
# const point2 = cvPoint2f(x,y)
# const point3 = cvPoint2f(x,y)
cvRotatedRect(point1, point2, point3) = @cxx cv::RotatedRect(point1, point2, point3)
# arrpts(x,y) = [cvPoint2f(x,y)]
cvRotatedRectPoints(arrpts) = @cxx cvRotatedRect.points(arrpts)
# rRect = cvRotatedRect(pts, size, angle)
cvRotatedRectBoundingRect(rRect) = @cxx rRect.boundingRect()

################################################################################################
# Mat constructors
################################################################################################

# 1) Mat::Mat()
cvMat() = @cxx cv::Mat()

# 2) Mat::Mat(int rows, int cols, int type)
cvMat(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat(rows, cols, matType)

# 3) Mat::Mat(Size size, int type)
cvMat(size, matType::CV_MatType) = @cxx cv::Mat(size, matType)

# 4) Mat::Mat(int rows, int cols, int type, const Scalar& s)
cvMat(rows::Int, cols::Int, matType::CV_MatType, scalar) = @cxx cv::Mat(rows, cols, matType, scalar)

# 5) Mat::Mat(Size size, int type, const Scalar& s)
cvMat(size, matType::CV_MatType, scalar) = @cxx cv::Mat(size, matType, scalar)

# 6) Mat::Mat(const Mat& m)
cvMat(img) = @cxx cv::Mat(img)

# 7) Mat::Mat(int ndims, const int* sizes, int type)
# const psizes(sizes) = pointer([sizes::Int])
cvMat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat(ndims, psizes, matType)

# 8) Mat::Mat(int ndims, const int* sizes, int type, const Scalar& s)
cvMat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType, cvScalar) = @cxx cv::Mat(ndims, psizes, matType, cvScalar)

# 9) Mat::Mat(const Mat& m, const Rect& roi)
cvMat(img, roi) = @cxx cv::Mat(img, roi)

# 10) Mat::Mat(const Mat& m, const Range* ranges)
#     const ranges = pointer(range)
cvMat(img, ranges) = @cxx cv::Mat(img, ranges)


# Mat operators
################################################################################################
# add
cxx"""
    cv::Mat add(cv::Mat img1, cv::Mat img2) {
       cv::Mat result = img1 + img2;
       return(result);
    }
"""
add(img1, img2) = @cxx add(img1, img2)

# substract
cxx"""
    cv::Mat substract(cv::Mat img1, cv::Mat img2) {
       cv::Mat result = img1 - img2;
       return(result);
    }
"""
substract(img1, img2) = @cxx substract(img1, img2)

# multiply
cxx"""
    cv::Mat multiply(cv::Mat img1, cv::Mat img2) {
       cv::Mat result = img1 * img2;
       return(result);
    }
"""
multiply(img1, img2) = @cxx multiply(img1, img2)

# #scale
# cxx"""
#     cv::Mat scale(cv::Mat img1, cv::Scalar alpha) {
#        cv::Mat result = img1 * alpha;
#        return(result);
#     }
# """
# scale(img, alpha) = @cxx scale(img, alpha)

# row
cxx"""
    cv::Mat row(cv::Mat img, int y) {
       cv::Mat mrow = img.row(y);
       return(mrow);
    }
"""

# col
cxx"""
    cv::Mat col(cv::Mat img, int x) {
       cv::Mat mcol = img.col(x);
       return(mcol);
    }
"""
row(img, x::Int) = @cxx row(img, x)
col(img, y::Int) = @cxx col(img, y)

# rowRange
cxx"""
   cv::Mat rowRange(cv::Mat img, int startrow, int endrow) {
       cv::Mat result = img.rowRange(startrow, endrow);
       return(result);
   }
"""
cxx"""
   cv::Mat rowRange(cv::Mat img, const cv::Range& r) {
       cv::Mat result = img.rowRange(r);
       return(result);
   }
"""
# colRange
cxx"""
    cv::Mat colRange(cv::Mat img, int startcol, int endcol) {
       cv::Mat result = img.colRange(startcol, endcol);
       return(result);
    }
"""
cxx"""
    cv::Mat colRange(cv::Mat img, const cv::Range& r) {
       cv::Mat result = img.colRange(r);
       return(result);
    }
"""
# const range = cvRange(start::Int, tend::Int)
imrow(img, startrow::Int, endrow::Int) = @cxx rowRange(img, startrow, endrow)
imrow(img, range) = @cxx rowRange(img, range)
imcol(img, startcol::Int, endcol::Int) = @cxx colRange(img, startcol, endcol)
imcol(img, range) = @cxx colRange(img, range)

# Mat Mat::diag(int d=0)
cxx"""
    cv::Mat diag(cv::Mat img, int d=0) {
       cv::Mat result = img.diag(d);
       return(result);
    }
"""
diag(img, d::Int) = @cxx diag(img, d)

# static Mat Mat::diag(const Mat& d)
# cxx"""
#     cv::Mat diag(cv::Mat img) {
#        cv::Mat result = img.diag(d);
#        return(result);
#     }
# """
# diag(img, d) = @cxx diag(img, d)

# Mat Mat::clone()
cxx"""
    cv::Mat clone(cv::Mat img) {
        cv::Mat copy = img.clone();
        return(copy);
    }
"""
clone(img) = @cxx clone(img)

# void Mat::copyTo(OutputArray m) const
# void Mat::copyTo(OutputArray m, InputArray mask) const
cxx"""
    void copyTo(cv::Mat img, cv::Mat copy) {
        img.copyTo(copy);
    }
"""
copy(img, copy) = @cxx copy(img, copy)
#copy(mask, copy) = @cxx copy(mask, copy)

# Mat::convertTo
# cxx"""
#     void convertTo(cv::Mat img, cv::Mat m, int rtype, double alpha=1, double beta=0) {
#         // m – output matrix
#         // rtype - depth, rtype < 0 output = input type
#         // alpha – optional scale factor.
#         // beta – optional delta added to the scaled values.
#         img.covertTo(m, rtype, alpha, beta);
#     }
# """
#convertTo(img, m, rtype, double, beta) = @cxx convertTo(img, m, rtype, double, beta)
# Mat::convertTo
convert(img1, img2, rtype::Int, alpha=1, beta=0) = @cxx img1->convertTo(img2, rtype, alpha, beta)

# Mat::assignTo
cxx"""
    void  assignTo(cv::Mat img, cv::Mat& m, int type=-1) {
        // m – Destination array.
        // type – Desired destination array depth
        img.assignTo(m, type);
   }
"""
assignTo(img, m, rtype) = @cxx convertTo(img, m, rtype)

# Mat::setTo
# value(cvScalar), mask (same size as img)
set(img, value) = @cxx img->setTo(value)

# Mat::reshape
cxx"""
    // set new N channels N rows. 0 = no change
    cv::Mat reshape(cv::Mat img, int cn, int rows=0) {
        cv::Mat result = img.reshape(cn, rows);
        return(result);
    }
"""
reshape(img, ch::Int, rows=0) = @cxx reshape(img, ch, rows)

# MatExpr Mat::t()
cxx"""
    cv::Mat transpose(cv::Mat img, double lambda) {
        cv::Mat result = img.t()*lambda;
        return(result);
    }
"""
transpose(img, lambda) = @cxx transpose(img, lambda)

# Mat::inv
cxx"""
   cv::Mat inv(cv::Mat img, int method= cv::DECOMP_LU) {
       cv::Mat result = img.inv(method);
       return(result);
   }
"""
inv(img, method=DECOMP_LU) = @cxx inv(img, method)


# Mat::mul
cxx"""
   cv::Mat mul(cv::Mat img, double scale=1) {
       cv::Mat result = img.mul(scale);
       return(result);
   }
"""
mul(img, scale=1) = @cxx inv(img, scale)

# Mat::cross
cxx"""
   cv::Mat cross(cv::Mat img, cv::Mat m) {
      // computes a cross-product of two 3-element vectors
      // vectors must be 3-element floating-point vectors of the same shape and size
      // output: 3-element vector of the same shape and type as operands
      cv::Mat result = img.cross(m);
      return(result);
   }
"""
cross(img, m) = @cxx cross(img, m)

# Mat::dot
cxx"""
    double dot(cv::Mat img, cv::Mat m) {
       // Computes a dot-product of two matrices
       // vectors must have the same size and type
       return(img.dot(m));
   }
"""
dot(img, m) = @cxx dot(img, m)

# Mat::zeros(), Mat::ones(), Mat::eye(), Mat::create()

# ndims – Array dimensionality.
# rows – Number of rows.
# cols – Number of columns.
# size – Alternative to the matrix size specification Size(cols, rows) .
# sz – Array of integers specifying the array shape.
# type – Created matrix type.

zeros(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::zeros(rows, cols, matType)
zeros(size, matType::CV_MatType) = @cxx cv::Mat::zeros(size, matType)
zeros(ndims::Int, sz::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat::zeros(ndims, sz, matType)

ones(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::ones(rows, cols, matType)
ones(size, matType::CV_MatType) = @cxx cv::Mat::ones(size, matType)
ones(ndims::Int, sz::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat::ones(ndims, sz, matType)

eye(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::eye(rows, cols, matType)
eye(size, matType::CV_MatType) = @cxx cv::Mat::eye(size, matType)
eye(ndims::Int, sz::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat::eye(ndims, sz, matType)

# Mat::addref
addref(img) = @cxx img->addref()

# Mat::release
destroy(img) = @cxx img->release()

# Mat::resize
# sz – new n rows,  s – value added
# const s = cvScalar()
# void Mat::resize(size_t sz)
# void Mat::resize(size_t sz, const Scalar& s)
resize(img, sz::Csize_t) = @cxx img->resize(sz)
resize(img, s, sz::Csize_t) = @cxx img->resize(sz, s)

# Mat::reserve
# void Mat::reserve(size_t sz)
reserve(img,sz::Csize_t) = @cxx img->reserve(sz)

# Mat::push_back
# const Mat& m
# const T& elem
push(img, m) = @cxx img->push_back(m)
push(img, elem) = @cxx img->push_back(elem)

# Mat::pop_back
pop(img, nelems::Csize_t) = @cxx img->pop_back(nelems)

# Mat::locateROI
# void Mat::locateROI(Size& wholeSize, Point& ofs)
locateROI(img, wholeSize, ofs) = @cxx img->locateROI(wholeSize, ofs)

# Mat::adjustROI
# Mat& Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
# dtop – Shift of the top submatrix boundary upwards.
# dbottom – Shift of the bottom submatrix boundary downwards.
# dleft – Shift of the left submatrix boundary to the left.
# dright – Shift of the right submatrix boundary to the right.
adjustROI(img, dtop::Int, dbottom::Int, dleft::Int, dright::Int) = @cxx img->adjustROI(dtop, dbottom, dleft, dright)


# Accessing Mat parameters
################################################################################################
total(img) = @cxx img->total()                               # returns size_t: number of array elements
dims(img) = @cxx img->dims                                   # ndims
cxx""" cv::Size size(cv::Mat img) {  return(img.size()); } """
size(img) = @cxx size(img)                                   # returns Size(cols, rows), if matrix > 2d, size = (-1,-1)
rows(img) = @cxx img->rows                                   # rows
cols(img) = @cxx img->cols                                   # cols
isContinuous(img) = @cxx img->isContinuous()                 # bool
elemSize(img) = @cxx img->elemSize()                         # step (size_t), e.g., CV_16SC3 = 6 (3*sizeof(short))
elemSize1(img) = @cxx img->elemSize1()                       # size of element e.g., CV_16SC3 = 2 (sizeof(short))

function findlabel(value::Int)
     return CV_MAT_TYPE[value]                               # e.g., CV_8UC1 type CV_MatType in OpenCV_hpp.jl
end

function cvtype(img)
    cxx""" int cvtype(cv::Mat img) { return(img.type()); } """
    val = @cxx cvtype(img)
    findlabel(int(val))
end

depth(img) = @cxx img->depth()
cxx""" int channels(cv::Mat img) { return(img.channels()); } """
channels(img) = @cxx channels(img)                           # number of matrix channels
step(img) = elemSize(img)*cols(img)                          # assumed no padding
step1(img, i) = @cxx img->step1(i)                           # Returns a normalized step, default i = 1
flags(img) = @cxx img->flags                                 # array dimensionality, >= 2
data(img) = @cxx img->data                                   # data – Pointer to the user data
refcount(img) = @cxx img->refcount                           # if user-allocated data, pointer is NULL
empty(img) = @cxx img->empty()                               # bool
ptr(img, row) = @cxx img->ptr(row)                           # return uchar* or typed pointer for matrix row

# Mat::at
# Parameters:
# i – Index along the dimension 0
# j – Index along the dimension 1
# k – Index along the dimension 2
# pt – Element position specified as Point(j,i) .
# idx – Array of Mat::dims indices.

cxx"""
     uchar at(cv::Mat img, int i, int j) {
         return(img.at<uchar>(i,j));
     }
"""

at(img, i::Int) = @cxx at(img, i)
at(img, i::Int, j::Int) = @cxx at(img,i,j)
at(img, i::Int, j::Int, k::Int) = @cxx at(img,i,j,k)
at(img, point) = @cxx at(img,point)
at(img, point::Ptr{Int}) = @cxx at(img,idx)    # const int* idx

# Mat::begin
cxx"""
      template<typename T>cv::MatIterator_<T> mbegin(cv::Mat img) {
         out = img.begin<T>());
      }
"""

mbegin(img) = @cxx mbegin(img)


# Mat::end
# Mat::forEach

# Graphical user interface (highgui)
################################################################################################

# Mat imread(const String& filename, int flags=IMREAD_COLOR)
imread(filename::Ptr{Uint8}, flags=IMREAD_COLOR) = @cxx cv::imread(filename, flags)

# namedWindow(const String& winname, int flags=WINDOW_AUTOSIZE)
namedWindow(windowName::Ptr{Uint8}, flags=WINDOW_AUTOSIZE) = @cxx cv::namedWindow(windowName, flags)

# void imshow(const String& winname, InputArray mat)
imshow(windowName::Ptr{Uint8}, img) = @cxx cv::imshow(windowName, img)

# waitKey
waitkey(delay) = @cxx cv::waitKey(delay)

# destroy GUI window(s)
destroyWindow(windowName::Ptr{Uint8}) = @cxx cv::destroyWindow(windowName)
destroyAllWindows() = @cxx cv::destroyAllWindows()

