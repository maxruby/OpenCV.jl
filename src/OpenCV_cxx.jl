################################################################################################
#
# OpenCV_cxx.jl
# Julia wrapper of OpenCV structures and functions
#
#
################################################################################################

# Mat: The core image array structure in OpenCV

# Parameters:

# ndims    –>   dimension
# rows     –>   Number of rows in a 2D array
# cols     –>   Number of columns in a 2D array
# roi      –>   Region of interest
# size     –>   2D array size: Size(cols, rows)
# sizes    –>   Integer array
# type     –>   # select using CV_MatType in OpenCV_hpp.jl
                # Array type CV_8UC(n), ..., CV_64FC(n)
                # CV_8U       8-bit unsigned (0:255)
                # CV_8S       8-bit signed (-128:127)
                # CV_16U     16-bit unsigned
                # CV_16S     16-bit signed
                # CV_32S     32-bit signed
                # CV_32F     32-bit float
                # CV_64F     64-bit float
# s        –>   l value to initialize each matrix element with.
#              To set all the matrix elements to the particular value after the construction,
#              use the assignment operator Mat::operator=(const Scalar& value).
# data     –>   Pointer to the user data.

# step     –>   Number of bytes each matrix row occupies, if missing set to AUTO_STEP.
#              cols*elemSize(). See Mat::elemSize().
#              The value should include the padding bytes at the end of each row.
# steps    –>   Array of ndims-1 steps in case of a multi-dimensional array
#              (the last step is always set to the element size).
#              If not specified, the matrix is assumed to be continuous
# m        –>  Array that (as a whole or partly) is assigned to the constructed matrix.
#              No data is copied by these constructors.
#              Instead, the header pointing to m data or its sub-array is constructed and associated with it.
#              The reference counter, if any, is incremented.
# img      –>  Pointer to the old-style IplImage image structure.
# vec      –>  STL vector whose elements form the matrix.
#              1-column matrix =>  n rows = n vector elem
#              matrix Type => matches type of vector elem
#              arbitrary types OK, as long as declared DataType.
#              elems => primitive numbers or uni-type numerical tuples of numbers.
#              ***Mixed-type structures are not supported.***
#              "Mat(vec)" declared explicitly.
#              if copyData=true (see next), no new elements will be added to the vector
# copyData  –> Flag for STL vector, true = copied, false =  shared the newly constructed matrix.
#              (old-style CvMat or IplImage)
# rowRange  –> Range of the m rows to take.    Range::all()=> take all the rows.
# colRange  –> Range of the m columns to take. Range::all()=> take all the columns.
# ranges    –> Array of selected ranges of m along each dimensionality.


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
# const center = cvPoint2f(x,y), const size = cv::Size2f(width,height)
cvRotatedRect(center, size, angle::Float32) = @cxx cv::RotatedRect(center, size, angle)
# const point1 = cvPoint2f(x,y), const point2 = cvPoint2f(x,y), const point3 = cvPoint2f(x,y)
cvRotatedRect(point1, point2, point3) = @cxx cv::RotatedRect(point1, point2, point3)
# arrpts(x,y) = [cvPoint2f(x,y)]
cvRotatedRectPoints(arrpts) = @cxx cvRotatedRect.points(arrpts)
# rRect = cvRotatedRect(pts, size, angle)
cvRotatedRectBoundingRect(rRect) = @cxx rRect.boundingRect()

# 7) TermCriteria
TermCriteria() = @cxx cv::TermCriteria::TermCriteria()
TermCriteria(rtype::Int, maxCount::Int, epsilon::Float64) =
     @cxx cv::TermCriteria::TermCriteria(rtype, maxCount, epsilon)
# type     – TermCriteria::COUNT, TermCriteria::EPS or TermCriteria::COUNT + TermCriteria::EPS
# maxCount – The maximum number of iterations or elements to compute
# epsilon  – The desired accuracy or change in parameters


################################################################################################
# Mat constructors
################################################################################################

# 1) Mat::Mat()
Mat() = @cxx cv::Mat()

# 2) Mat::Mat(int rows, int cols, int type)
Mat(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat(rows, cols, matType)

# 3) Mat::Mat(Size size, int type)
Mat(size, matType::CV_MatType) = @cxx cv::Mat(size, matType)

# 4) Mat::Mat(int rows, int cols, int type, const Scalar& s)
Mat(rows::Int, cols::Int, matType::CV_MatType, s) = @cxx cv::Mat(rows, cols, matType, s)

# 5) Mat::Mat(Size size, int type, const Scalar& s)
Mat(size, matType::CV_MatType, s) = @cxx cv::Mat(size, matType, s)

# 6) Mat::Mat(const Mat& m)
Mat(m) = @cxx cv::Mat(m)

# 7) Mat::Mat(int ndims, const int* sizes, int type)
# const psizes(sizes) = pointer([sizes::Int])
Mat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat(ndims, psizes, matType)

# 8) Mat::Mat(int ndims, const int* sizes, int type, const Scalar& s)
Mat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType, s) = @cxx cv::Mat(ndims, psizes, matType, s)

# 9) Mat::Mat(const Mat& m, const Rect& roi)
Mat(img, roi) = @cxx cv::Mat(img, roi)

# 10) Mat::Mat(const Mat& m, const Range* ranges)
# const ranges = pointer(range)
Mat(img, ranges) = @cxx cv::Mat(img, ranges)


################################################################################################
# Mat operators
################################################################################################

# addition
cxx""" cv::Mat add(cv::Mat img1, cv::Mat img2) { return(img1 + img2); } """
imadd(img1, img2) = @cxx add(img1, img2)

# substract
cxx""" cv::Mat substract(cv::Mat img1, cv::Mat img2) { return(img1 - img2); } """
imsubstract(img1, img2) = @cxx substract(img1, img2)

# multiply
cxx""" cv::Mat multiply(cv::Mat img1, cv::Mat img2) { return(img1 * img2); } """
immultiply(img1, img2) = @cxx multiply(img1, img2)

# Mat::scale
# cxx""" cv::Mat scale(cv::Mat img, cv::Scalar alpha) { return(img * alpha); } """
# scale(img, alpha) = @cxx scale(img, alpha)

# Mat::row
cxx""" cv::Mat row(cv::Mat img, int y) { return(img.row(y)); } """
row(img, x::Int) = @cxx row(img, x)

# Mat::col
cxx""" cv::Mat col(cv::Mat img, int x) { return(img.col(x)); } """
col(img, y::Int) = @cxx col(img, y)

# Mat::rowRange
cxx""" cv::Mat rowRange(cv::Mat img, int startrow, int endrow) { return(img.rowRange(startrow, endrow)); } """
imrow(img, startrow::Int, endrow::Int) = @cxx rowRange(img, startrow, endrow)

cxx""" cv::Mat rowRange(cv::Mat img, const cv::Range& r) { return(img.rowRange(r)); } """
imrow(img, range) = @cxx rowRange(img, range)

# Mat::colRange
# const range = cvRange(start::Int, tend::Int)
cxx""" cv::Mat colRange(cv::Mat img, int startcol, int endcol) { return(img.colRange(startcol, endcol)); } """
imcol(img, startcol::Int, endcol::Int) = @cxx colRange(img, startcol, endcol)

cxx""" cv::Mat colRange(cv::Mat img, const cv::Range& r) { return(img.colRange(r)); } """
imcol(img, range) = @cxx colRange(img, range)

# Mat::diag
# d=0 is the main diagonal
# d>0 is a diagonal from the lower half
# d=1 below the main one
# d<0 is a diagonal from the upper half

cxx""" cv::Mat diag(cv::Mat img, int d=0) { return(img.diag(d)); } """
diag(img, d::Int) = @cxx diag(img, d)

cxx""" cv::Mat diag(cv::Mat img, const cv::Mat& m) { return(img.diag(m)); } """
diag(img, m) = @cxx diag(img, m)

# Mat::clone()
cxx""" cv::Mat clone(cv::Mat img) { return(img.clone()); } """
clone(img) = @cxx clone(img)

# Mat::copyTo
cxx""" void copyTo(cv::Mat img, cv::Mat copy) { img.copyTo(copy); } """
copy(img, copy) = @cxx copy(img, copy)
copy(mask, copy) = @cxx copy(mask, copy)

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
cxx""" void  assignTo(cv::Mat img, cv::Mat& m, int type=-1) { img.assignTo(m, type); } """
assignTo(img, m, rtype) = @cxx assignTo(img, m, rtype)

# Mat::setTo
# value(cvScalar), mask (same size as img)
set(img, value) = @cxx img->setTo(value)

# Mat::reshape
# rows = 0 (no change)
cxx""" cv::Mat reshape(cv::Mat img, int cn, int rows=0) { return(img.reshape(cn, rows)); } """
reshape(img, ch::Int, rows=0) = @cxx reshape(img, ch, rows)

# Mat::t() (transpose)
cxx""" cv::Mat transpose(cv::Mat img, double lambda) { return(img.t()*lambda); } """
transpose(img, lambda) = @cxx transpose(img, lambda)

# Mat::inv (invert)
cxx""" cv::Mat inv(cv::Mat img, int method= cv::DECOMP_LU) { return(img.inv(method)); } """
inv(img, method=DECOMP_LU) = @cxx inv(img, method)

# Mat::mul (mutiply)
cxx""" cv::Mat mul(cv::Mat img, double scale=1) { return(img.mul(scale)); } """
mul(img, scale=1) = @cxx inv(img, scale)

# Mat::cross (cross-product of 2 Vec<float,3>)
cxx""" cv::Mat cross(cv::Mat img, cv::Mat m) { return(img.cross(m)); } """
cross(img, m) = @cxx cross(img, m)

# Mat::dot (Computes a dot-product of two equally sized matrices)
cxx"""  double dot(cv::Mat img, cv::Mat m) { return(img.dot(m)); } """
dot(img, m) = @cxx dot(img, m)

# Mat::zeros()
# ndims –> Array dimensionality.
# rows  –> Number of rows.
# cols  –> Number of columns.
# size  –> Alternative to the matrix size specification Size(cols, rows) .
# sz    –> Array of integers specifying the array shape.
# type  –> Created matrix type.

zeros(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::zeros(rows, cols, matType)
zeros(size, matType::CV_MatType) = @cxx cv::Mat::zeros(size, matType)
zeros(ndims::Int, sz::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat::zeros(ndims, sz, matType)

# Mat::ones()
ones(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::ones(rows, cols, matType)
ones(size, matType::CV_MatType) = @cxx cv::Mat::ones(size, matType)
ones(ndims::Int, sz::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat::ones(ndims, sz, matType)

# Mat::eye()
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
resize(img, sz::Csize_t) = @cxx img->resize(sz)         # resize by a factor (Uint64)
resize(img, sz::Csize_t, s) = @cxx img->resize(sz, s)   # resize by vector s

# Mat::reserve
# void Mat::reserve(size_t sz)
reserve(img,sz::Csize_t) = @cxx img->reserve(sz)

# Mat::push_back
# const Mat& m, const T& elem
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


#########################################################################################################################
# Accessing Mat parameters
#########################################################################################################################

total(img) = @cxx img->total()                               # returns size_t: number of array elements
dims(img) = @cxx img->dims                                   # ndims
rows(img) = @cxx img->rows                                   # rows
cols(img) = @cxx img->cols                                   # cols
isContinuous(img) = @cxx img->isContinuous()                 # bool
elemSize(img) = @cxx img->elemSize()                         # step (size_t), e.g., CV_16SC3 = 6 (3*sizeof(short))
elemSize1(img) = @cxx img->elemSize1()                       # size of element e.g., CV_16SC3 = 2 (sizeof(short))
depth(img) = @cxx img->depth()
step(img) = elemSize(img)*cols(img)                          # assumes no padding
step1(img, i) = @cxx img->step1(i)                           # normalized step, default i = 1
flags(img) = @cxx img->flags                                 # array dimensionality >= 2
data(img) = @cxx img->data                                   # pointer to the user data
refcount(img) = @cxx img->refcount                           # reference count, pointer is NULL (if user-allocated data)
empty(img) = @cxx img->empty()                               # bool
ptr(img, row) = @cxx img->ptr(row)                           # return uchar* or typed pointer for matrix row

cxx""" cv::Size size(cv::Mat img) {  return(img.size()); } """
size(img) = @cxx size(img)                                   # returns Size(cols, rows), if matrix > 2d, size = (-1,-1)

cxx""" int channels(cv::Mat img) { return(img.channels()); } """
channels(img) = @cxx channels(img)                           # number of matrix channels

findlabel(value::Int) = CV_MAT_TYPE[value]                   # e.g., CV_8UC1 type CV_MatType in OpenCV_hpp.jl
cxx""" int cvtype(cv::Mat img) { return(img.type()); } """
cvtypeval(img) = @cxx cvtype(img)
cvtypelabel(img) = findlabel(int(cvtypeval(img)))

# Mat::at  => get and set functions
# Parameters:
# i – Index along the dimension 0
# j – Index along the dimension 1
# k – Index along the dimension 2
# pt – Element position specified as Point(j,i)
# idx – Array of Mat::dims indices

cxx"""
template <typename _Tp>

_Tp get(const cv::Mat_<_Tp>& mat, int row, int col) {
    typedef typename cv::DataType<_Tp>::work_type _wTp;
    if (mat.dims==1) {
        return((_wTp) mat.template at<_Tp>(row,col));
    }
    elseif(mat.dims==3) {row
        cv:Vec<_Tp,3> cvec;
        cvec[0] = mat.template at<cv::Vec<_Tp,3>>(row,col)[0];
        cvec[1] = mat.template at<cv::Vec<_Tp,3>>(row,col)[1];
        cvec[2] = mat.template at<cv::Vec<_Tp,3>>(row,col)[2];
        return((_wTp) cvec);
    }
}

template <typename _Rs>
_Rs imget(cv::Mat img, int row, int col) {
 typedef typename cv::DataType<_Rs>::work_type _wRs;

    if (img.dims!=1 | img.dims!=3) {
       std::cout << "Argument error: only images with 1 and 3 dimensions are supported." << std::endl;
       return(-1);
    }

    int cvtype = img.type()
    switch(cvtype) {
    case CV_8SC1:  cv::Mat_<char>dst(img.rows, img.cols,img.type()); break;
    case CV_8UC1:  cv::Mat_<uchar>dst(img.rows, img.cols,img.type()); break;
    case CV_16SC1: cv::Mat_<short>dst(img.rows, img.cols,img.type()); break;
    case CV_16UC1: cv::Mat_<unsigned short>dst(img.rows, img.cols,img.type()); break;
    case CV_32SC1: cv::Mat_<int>dst(img.rows, img.cols,img.type()); break;
    case CV_32FC1: cv::Mat_<float>dst(img.rows, img.cols,img.type()); break;
    case CV_64FC1: cv::Mat_<double>dst(img.rows, img.cols,img.type()); break;
    case CV_8SC3:  cv::Mat_<cv::Vec<char,3>>dst(img.rows, img.cols,img.type()); break;
    case CV_8UC3:  cv::Mat_<cv::Vec3b>dst(img.rows, img.cols,img.type()); break;
    case CV_16SC3: cv::Mat_<cv::Vec3s>dst(img.rows, img.cols,img.type()); break;
    case CV_16UC3: cv::Mat_<cv::Vec<unsigned short,3>>dst(img.rows, img.cols,img.type()); break;
    case CV_32SC3: cv::Mat_<cv::Vec3i>dst(img.rows, img.cols,img.type()); break;
    case CV_32FC3: cv::Mat_<cv::Vec3f>dst(img.rows, img.cols,img.type()); break;
    case CV_64FC3: cv::Mat_<cv::Vec3d>dst(img.rows, img.cols,img.type()); break;
    }

    img.assignTo(dst);
    return((_wRs) get(dst, row, col));
}
"""

cxx"""
template <typename _Tp>

void set(const cv::Mat_<_Tp>& mat, int row, int col, cv::Vec<_Tp,3> cvec) {
    if (mat.dims==1) {
         mat.template at<_Tp>(row,col) = cvec[0];
    }
    elseif(mat.dims==3) {
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[0];
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[1];
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[2];
    }
}

template <typename _Rs>
void imset(cv::Mat img, int row, int col, cv::Vec<_Rs,3> cvec)
    if (img.dims!=1 | img.dims!=3) {
       std::cout << "Argument error: only images with 1 and 3 dimensions are supported." << std::endl;
       return(-1);
    }

    int cvtype = img.type()
    switch(cvtype) {
    case CV_8SC1:  cv::Mat_<char>dst(img.rows, img.cols,img.type()); break;
    case CV_8UC1:  cv::Mat_<uchar>dst(img.rows, img.cols,img.type()); break;
    case CV_16SC1: cv::Mat_<short>dst(img.rows, img.cols,img.type()); break;
    case CV_16UC1: cv::Mat_<unsigned short>dst(img.rows, img.cols,img.type()); break;
    case CV_32SC1: cv::Mat_<int>dst(img.rows, img.cols,img.type()); break;
    case CV_32FC1: cv::Mat_<float>dst(img.rows, img.cols,img.type()); break;
    case CV_64FC1: cv::Mat_<double>dst(img.rows, img.cols,img.type()); break;
    case CV_8SC3:  cv::Mat_<cv::Vec<char,3>>dst(img.rows, img.cols,img.type()); break;
    case CV_8UC3:  cv::Mat_<cv::Vec3b>dst(img.rows, img.cols,img.type()); break;
    case CV_16SC3: cv::Mat_<cv::Vec3s>dst(img.rows, img.cols,img.type()); break;
    case CV_16UC3: cv::Mat_<cv::Vec<unsigned short,3>>dst(img.rows, img.cols,img.type()); break;
    case CV_32SC3: cv::Mat_<cv::Vec3i>dst(img.rows, img.cols,img.type()); break;
    case CV_32FC3: cv::Mat_<cv::Vec3f>dst(img.rows, img.cols,img.type()); break;
    case CV_64FC3: cv::Mat_<cv::Vec3d>dst(img.rows, img.cols,img.type()); break;
    }

    img.assignTo(dst);
    set(dst, row, col, cvec);
}
"""

####################################################################################################
# Operations on Arrays
####################################################################################################

abs(img) = @cxx cv::abs(img)
absdiff(src1, src2, dst) = @cxx absdiff(src1, src2, dst)

add(src1, src2, dst, int dtype=-1) = @cxx cv::add(src1, src2, dst, dtype)
# mask – optional operation mask - 8-bit single channel array, default mask = noArray()
# dtype – optional depth of the output array

addWeighted(src1, alpha, src2, beta::Float64, gamma::Float64, dst, int dtype=-1) =
    @cxx  cv::addWeighted(src1, alpha, src2, beta::Float64, gamma::Float64, dst, int dtype=-1)
# beta – weight of the second array elements
# gamma – scalar added to each sum

bitwise_and(src1, src2, dst) = @cxx cv::bitwise_and(src1, src2, dst)
bitwise_not(src, dst) = @cxx cv::bitwise_not(src, dst)
bitwise_or(src1, src2, dst) = @cxx cv::bitwise_or(src1, src2, dst)
bitwise_xor(src1, src2, dst) = @cxx cv::bitwise_xor(src1, src2, dst)
# mask – optional operation mask - 8-bit single channel array, default mask = noArray()

calcCovarMatrix(pointer(samples), nsamples::Int, covar, mean, flags::Int, ctype=CV_64F) =
    @cxx cv::calcCovarMatrix(pointer(samples), nsamples, covar, mean, flags, ctype)
# samples – samples stored either as separate matrices or as rows/columns of a single matrix
# covar   – output covariance matrix of the type ctype and square size
# mean    – input or output (depending on the flags) array
# vects   – a set of vectors
# flags:
#    CV_COVAR_SCRAMBLED
#    CV_COVAR_USE_AVG
#    CV_COVAR_SCALE
#    CV_COVAR_ROWS
#    CV_COVAR_COLS

cartToPolar(x, y, magnitude, angle, angleInDegrees=false) =
    @cxx cv::cartToPolar(x, y, magnitude, angle, angleInDegrees=false)
polarToCart(magnitude, angle, x, y, angleInDegrees=false) =
    @cxx cv::polarToCart(magnitude, angle, x, y, angleInDegrees=false)
# x              – array of x-coordinates; single-precision or double-precision floating-point array
# y              – array of y-coordinates, same size and same type as x
# magnitude      – output array of magnitudes of the same size and type as x
# angle          – output array of angles that has the same size and type as x;
#                   in radians (from 0 to 2*Pi) or in degrees (0 to 360 degrees)
# angleInDegrees – bool flag = false (radians), flag = true (degrees)

checkRange(a, quiet=true, point, double minVal=-DBL_MAX, double maxVal=DBL_MAX) =
    @cxx cv::checkRange(a, quiet, point, minVal, maxVal)
# a       – input array
# quiet   – bool flag to throw error if array elements are out of range (true/false)
# pos     – optional output parameter, pos of first outlier is stored
#           when not NULL,it must be a pointer to array of src.dims elements
# minVal  – inclusive lower boundary of valid values range
# maxVal  – exclusive upper boundary of valid values range

compare(src1, src2, dst, cmpop::Int) = @cxx cv::compare(rc2, dst, cmpop)
# cmpop     –  flag, that specifies correspondence between the arrays:
# CMP_EQ if src1 = src2
# CMP_GT if src1 > src2
# CMP_GE src1 >= src2
# CMP_LT src1 < src2
# CMP_LE src1 =< src2
# CMP_NE src1 != src2
# e.g., Mat dst2 = src1 < 8;

completeSymm(mtx, lowerToUpper=false) = @cxx cv::completeSymm(mtx, lowerToUpper)
# mtx          – input-output floating-point square matrix
# lowerToUpper – operation flag; if true, the lower half is copied to the upper half
#                Otherwise, the upper half is copied to the lower half

convertScaleAbs(src, dst, alpha=1.0, beta=float(0)) = @cxx cv::convertScaleAbs(src, dst, alpha, beta)
# src   – input array
# dst   – output array
# alpha – optional scale factor
# beta  – optional delta added to the scaled values

countNonZero(src) = @cxx cv::countNonZero(src)

dct(src, dst, flags=0) = @cxx cv::dct(src, dst, flags)
# forward or inverse discrete Cosine transform of 1D or 2D array
# flags  - transformation flags
# DCT_INVERSE
# DCT_ROWS

dft(src, dst, flags=0, nonzeroRows=0) = @cxx cv::dft(src, dst, flags, nonzeroRows)
# forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array

divide(src1, src2, dst, scale=1.0, dtype=-1) = @cxx cv::divide(src1, src2, dst, scale, dtype)
# per-element division of two arrays or a scalar by an array

determinant(mtx) = @cxx cv::determinant(mtx)
# determinant of a square floating-point matrix

eigen(src, eigenvalues) = @cxx cv::eigen(src, eigenvalues)
# src           – input matrix that must have CV_32FC1 or CV_64FC1
# eigenvectors  – default =noArray()
# eigenvalues   – output vector [eigenvalues, type = src], stored in descending order
# lowindex      – optional index of largest eigenvalue/-vector to calculate
# highindex     – optional index of smallest eigenvalue/-vector to calculate

exp(src, dst) = @cxx cv::exp(src, dst)

flip(src, dst, flipCode) = @cxx cv::flip(src, dst, flipCode)
# flipCode – a flag to specify how to flip the array
#   0 -> x-axis
#  >0 -> y-axis
#  <0 -> x,y axes

gemm(src1, src2, alpha, src3, beta, dst, flags=0) = @cxx cv::gemm(src1, src2, alpha, src3, beta, dst, flags)
# src1      – first multiplied input matrix that could be real(CV_32FC1, CV_64FC1) or complex(CV_32FC2, CV_64FC2)
# src2      – second multiplied input matrix of the same type as src1
# alpha     – weight of the matrix product
# src3      – third optional delta matrix added to the matrix product;
# beta      – weight of src3
# dst       – output matrix
# flags     –
# GEMM_1_T transposes src1
# GEMM_2_T transposes src2
# GEMM_3_T transposes src3

getOptimalDFTSize(vecsize::Int) = @cxx cv::getOptimalDFTSize(vecsize)
# optimal DFT size for a given vector size

idct(src, dst, flags=0) = @cxx cv::idct(src, dst, flags)
# inverse Discrete Cosine Transform of a 1D or 2D array
# flags – operation flags

idft(src, dst, flags=0, nonzeroRows=0) = @cxx cv::idft(src, dst, flags, nonzeroRows)
# flags – operation flags
# nonzeroRows – number of dst rows to process

inRange(src, lowerb, upperb, dst) = @cxx cv::inRange(src, lowerb, upperb, dst)
# src    – first input array
# lowerb – inclusive lower boundary array or a scalar
# upperb – inclusive upper boundary array or a scalar
# dst    – output array of the same size as src and CV_8U type

# same as inv for cv::Mat
invert(src, dst, flags=DECOMP_LU) = @cxx cv::invert(src, dst, flags)

# natural logarithm of every array element
log(src, dst) = @cxx cv::log(src, dst)

# look-up table transform of an array
LUT(src, lut, dst) = @cxx cv::LUT(src, lut, dst)
# src – input array of 8-bit elements
# dst – output array of the same size
# lut – look-up table of 256 elements
# multi-channel input array, single channel (same lut table for all channel)
# or the same number of channels as in the input array.

# magnitude of 2D vectors
magnitude(x, y, magnitude) = @cxx cv::magnitude(x, y, magnitude)
# x         – floating-point array of x-coordinates of the vectors
# y         – floating-point array of y-coordinates of the vectors; it must have the same size as x
# magnitude – output array of the same size and type as x

# Mahalanobis distance between two vectors
Mahalanobis(v1, v2, icovar) = @cxx cv::Mahalanobis(v1, v2, icovar)
# vec1   – first 1D input vector
# vec2   – second 1D input vector
# icovar – inverse covariance matrix

max(a, b) = @cxx cv::max(a,b)
max(a, scalar) = @cxx cv::max(a,scalar)

mean(src) = @cxx cv::mean(src)
# mask  – optional operation mask, default mask=noArray()

meanStdDev(src, mean, stddev) = @cxx cv::meanStdDev(src, mean, stddev)
# mask  – optional operation mask, default mask=noArray()

merge(mv, count::Uint64, dst) = @cxx cv::merge(mv, count, dst)
# mv  => const pointer(cv::Mat) or const Mat* mv
# dst – output array of the same size

min(a, b) = @cxx cv::min(a,b)
min(a, scalar) = @cxx cv::min(a,scalar)

# global minimum and maximum in an array
minMaxIdx(src, minVal::Ptr{Float64}, maxIdx=pointer([0]), iminIdx=pointer([0]), maxIdx=pointer([0])) =
    @cxx cv::minMaxIdx(src, minVal::Ptr{Float64}, maxIdx, iminIdx, maxIdx)
# maxVal::Ptr{Float64}
# minIdx::Ptr{Int}
# maxIdx::Ptr{Int}
# src – input single-channel array.
# minVal – pointer to the returned minimum value; NULL is used if not required.
# maxVal – pointer to the returned maximum value; NULL is used if not required.
# minIdx – minIdx is not NULL, it must have at least 2 elements (as well as maxIdx),

# global minimum and maximum in an array
minMaxLoc(src, minVal::Ptr{Float64}, maxVal=pointer([0]), minLoc, maxLoc) =
    @cxx cv::minMaxLoc(src, minVal, maxVal, minLoc, maxLoc)
# minLoc = convert(C_NULL, pointer(cvPoint))
# maxLoc = convert(C_NULL, pointer(cvPoint))
# mask  – optional operation mask, default mask=noArray()
# minVal – pointer to the returned minimum value; NULL is used if not required.
# maxVal – pointer to the returned maximum value; NULL is used if not required.
# minLoc – pointer to the returned minimum location (in 2D case); NULL is used if not required.
# maxLoc – pointer to the returned maximum location (in 2D case); NULL is used if not required.
# mask – optional mask used to select a sub-array.

mixChannels(src, dst, fromTo::Ptr{Int}, npairs::Uint64) = @cxx cv::mixChannels(src, dst, fromTo, npairs)
# const fromTo::Ptr{Int}
# fromTo = const std::vector<int>&

# per-element multiplication of two Fourier spectrums
mulSpectrums(a, b, c, flags=DFT_ROWS, conjB=false) = @cxx cv::mulSpectrums(a, b, c, flags, conjB)
# flags – currently, the only supported flag is DFT_ROWS

# per-element scaled product of two arrays
multiply(src1, src2, dst, scale=1.0, dtype=-1) = @cxx cv::multiply(src1, src2, dst, scale, dtype)

# product of a matrix and its transposition
mulTransposed(src, dst, aTa::bool, scale=1.0, dtype=-1) = @cxx cv::mulTransposed(src, dst, aTa, scale, dtype)
# InputArray delta=noArray()

# absolute array norm, an absolute difference norm, or a relative difference norm
norm(src1, normType=NORM_L2) = @cxx cv::norm(src1, normType=NORM_L2)
norm(src1, src2, normType=NORM_L2) = @cxx cv::norm(src1, src2, normType=NORM_L2)
# InputArray mask=noArray()

# norm or value range of an array
normalize(src, dst, alpha=1.0, beta=0, norm_type=NORM_L2, dtype=-1) = @cxx cv::normalize(src, dst, alpha, beta, norm_type, dtype)
# InputArray mask=noArray()

# PCA: Principal Component Analysis class
# Constructors
pca() = cv::PCA::PCA()
pca(data, mean, flags::Int, maxComponents=0) =  @cxx cv::PCA::PCA(data, mean, flags, maxComponents)
pca(data, mean, flags::Int, retainedVariance::Float64) = @cxx cv::PCA::PCA(data, mean, flags, retainedVariance)
# Run PCA
pca_operator() = @cxx cv::PCA::PCA->operator()
# data             – input samples stored as matrix rows or matrix columns.
# mean             – optional mean value; if the matrix is empty (noArray()), the mean is computed from the data.
# flags            – operation flags; currently the parameter is only used to specify the data layout:
#                     CV_PCA_DATA_AS_ROW stored as matrix rows
#                     CV_PCA_DATA_AS_COL stored as matrix columns
# maxComponents    – maximum number of components that PCA should retain; by default, all the components are retained.
# retainedVariance – Percentage of variance that PCA should retain.
#                    Using this parameter will let the PCA decided how many components to retain
#                    but it will always keep at least 2.


# Projects vector(s) to the principal component subspace
pca_project(vec) =  @cxx cv::PCA::project(vec)                  # returns cv::Mat
pca_project(vec, result) =  @cxx cv::PCA::project(vec, result)  # void
# vec    – input vector(s), if CV_PCA_DATA_AS_ROW then vec.cols==data.cols
# result – output vectors, if CV_PCA_DATA_AS_COL then result.cols==vec.cols

# Reconstructs vectors from their PC projections
pca_backProject(vec) = @cxx cv::PCA::backProject(vec)
pca_backProject(vec, result) = @cxx cv::PCA::backProject(vec, result)

# Performs the perspective matrix transformation of vectors
perspectiveTransform(src, dst, m) = @cxx cv::perspectiveTransform(src, dst, m)
# src   – input two-channel or three-channel floating-point array
# dst   – output array of the same size and type as src
# m     – 3x3 or 4x4 floating-point transformation matrix

# rotation angle of 2D vectors
phase(x, y, angle, angleInDegrees=false) = @cxx cv::phase(x, y, angle, angleInDegrees)

# Raises every array element to a power
pow(src, power::Float64, dst) = @cxx cv::pow(src, power, dst)

# Random number generator
rng() = @cxx cv::RNG::RNG()
rng(state::Uint64) = @cxx cv::RNG::RNG(state)
rng_next() = @cxx cv::RNG::RNG->next()
rng_uniform(a::Int, b::Int) = @cxx cv::RNG::RNG->uniform(a, b)
rng_uniform(a::Float32, b::Float32) = @cxx cv::RNG::RNG->uniform(a, b)
rng_uniform(a::Float64, b::Float64) = @cxx cv::RNG::RNG->uniform(a, b)
rng_gaussian(sigma::Float64) = @cxx cv::RNG::RNG->gaussian(sigma)
rng_fill(mat, distType, a, b, saturateRange=false) = @cxx cv::RNG::RNG->fill(mat, distType, a, b, saturateRange)
# distType = RNG::UNIFORM or RNG::NORMAL

# array with normally distributed random numbers
randn(dst, mean, stddev) = @cxx cv::randn(dst, mean, stddev)

# Shuffles the array elements randomly
theRNG() = @cxx cv::theRNG()
randShuffle(dst, iterFactor=1.0, rng = theRNG()) = cv::randShuffle(dst, iterFactor, rng)
# RNG* rng=0

# Reduces a matrix to a vector
reduce(src, dst, dim::Int, rtype::Int, dtype=-1) = @cxx cv::reduce(src, dst, dim, rtype, dtype=-1)

# output array with repeated copies of the input array
repeat(src, ny::Int, nx::Int, dst) = @cxx cv::repeat(src, ny, nx, dst)
repeat(src, ny::Int, nx::Int) = @cxx cv::repeat(src, ny, nx)

# sum of a scaled array and another array
scaleAdd(src1, alpha::Float64, src2, dst) = @cxx cv::scaleAdd(src1, alpha, src2, dst)

# Initializes a scaled identity matrix
setIdentity(mtx, s=cvScalar(1, 0, 0)) =  @cxx cv::setIdentity(mtx, s)

# Solves one or more linear systems or least-squares problems
solve(src1, src2, dst, flags=DECOMP_LU) = @cxx cv::solve(src1, src2, dst, flags)
# DECOMP_LU
# DECOMP_CHOLESKY
# DECOMP_EIG
# DECOMP_SVD
# DECOMP_QR
# DECOMP_NORMAL

# real roots of a cubic equation
solveCubic(coeffs, roots) = @cxx cv::solveCubic(coeffs, roots)
#coeffs – equation coefficients, an array of 3 or 4 elements
#roots – output array of real roots that has 1 or 3 elements

# real or complex roots of a polynomial equation
solvePoly(coeffs, roots, maxIters=300) = @cxx cv::solvePoly(coeffs, roots, maxIters)

# Sorts each row or each column of a matrix
# CV_SORT_EVERY_ROW
# CV_SORT_EVERY_COLUMN
# CV_SORT_ASCENDING
# CV_SORT_DESCENDING
sort(src, dst, flags::Int) = @cxx cv::sort(src, dst, flags)
sortIdx(src, dst, flags::Int) = @cxx cv::sortIdx(src, dst, flags)

# Divides a multi-channel array into several single-channel arrays
split(src, mvbegin) = @cxx cv::split(src, mvbegin)    #mvbegin = Mat*
split(m, mv) = @cxx cv::split(src, mvbegin)

# square root of array elements
sqrt(src, dst) = @cxx cv::sqrt(src, dst)
# src - input floating-point array
# dst – output array

subtract(src1, src2, dst, dtype=-1) = @cxx cv::subtract(src1, src2, dst, dtype)
# InputArray mask=noArray()

# Singular Value Decomposition of a floating-point matrix
svd() = cv::SVD::SVD()
svd(src, flags=0) = @cxx cv::SVD::SVD(src, flags)
# src – decomposed matrix
# flags -
# SVD::MODIFY_A  # modify the decomposed matrix
# SVD::NO_UV     # use only singular values
# SVD::FULL_UV   # u and vt are full-size square orthogonal matrices

# Perform SVD of a matrix
svd_compute(src, w, u, vt, flags=0) = @cxx cv::SVD::compute(src, w, u, vt, flags)
svd_compute(src, w, flags=0) = @cxx cv::SVD::compute(src, w, flags)
# src – decomposed matrix
# w   – calculated singular values
# u   – calculated left singular vectors
# V   – calculated right singular vectors
# vt  – transposed matrix of right singular values

# Solves an under-determined singular linear system
svd_solveZ(src, dst) = @cxx cv::SVD::solveZ(src, dst)

# sum of array elements
sum(src) = @cxx cv::sum(src)

# trace of a matrix
trace(mtx) = @cxx cv::trace(mtx)

# matrix transformation of every array element
transform(src, dst, m) = @cxx cv::transform(src, dst, m)

# Transpose a matrix
transposeMat(src, dst) = @cxx cv::transpose(src, dst)

# Computes the source location of an extrapolated pixel
borderInterpolate(p::Int, len::Int, borderType::Int) = @cxx cv::borderInterpolate(p, len, borderType)
# Forms a border around an image
copyMakeBorder(src, dst, top::Int, bottom::Int, left::Int, right::Int, borderType::Int, value=cvScalar()) =
    @cxx cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value)
# borderType ->
# BORDER_CONSTANT
# BORDER_TRANSPARENT
# BORDER_ISOLATED

# Clustering
# Finds centers of clusters and groups input samples around the clusters

kmeans(data, K::Int, bestLabels, criteria, attempts::Int, flags::Int) =
    @cxx cv::kmeans(data, K, bestLabels, criteria, attempts, flags)
# OutputArray centers=noArray()
# TermCriteria
# partition


# Utility and System Functions and Macros

# angle of a 2D vector in degrees
fastAtan2(y::Float64, x::Float64) = @cxx cv::fastAtan2(y, x)

# cube root of an argument
cubeRoot(val::Float64) =  @cxx cv::cubeRoot(val)

# heckHardwareSupport(int feature)
checkHardwareSupport(feature::Int) = @cxx cv::checkHardwareSupport(feature)
# CV_CPU_MMX
# CV_CPU_SSE
# CV_CPU_SSE2
# CV_CPU_SSE3
# CV_CPU_SSSE3
# CV_CPU_SSE4_1
# CV_CPU_SSE4_2
# CV_CPU_POPCNT
# CV_CPU_AVX

getNumberOfCPUs() = @cxx cv::getNumberOfCPUs()
getNumThreads() = @cxx cv::getNumThreads()
setNumThreads(nthreads::Int) = @cxx cv::setNumThreads(nthreads)

# Returns the number of ticks after a certain event
getTickCount() = @cxx cv::getTickCount()
getTickFrequency() = @cxx cv::getTickFrequency()
getCPUTickCount() = @cxx cv::getCPUTickCount()

# Enables or disables the optimized code
cvUseOptimized(on_off::Int) = @cxx cv::cvUseOptimized(on_off)
useOptimized() = @cxx cv::useOptimized()  #true/false status


#####################################################################################################
# Image processing (imgproc)
#####################################################################################################

# bilateralFilter
# void bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace,
# int borderType=BORDER_DEFAULT )
# Parameters:
# src        – Source 8-bit or floating-point, 1-channel or 3-channel image.
# dst        – Destination image of the same size and type as src .
# d          – Diameter of each pixel neighborhood that is used during filtering. If it is
#              non-positive, it is computed from sigmaSpace.
# sigmaColor – Filter sigma in the color space. A larger value of the parameter means that
#              farther colors within the pixel neighborhood
#              (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
# sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means
#              that farther pixels will influence
#              each other as long as their colors are close enough (see sigmaColor ). When d>0 ,
#              it specifies the neighborhood size
#              regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace
# Setting parameters
# Sigma values: if sigmaColor and sigmaSpace < 10, not much effect
#               if sigmaColor and sigmaSpace > 150,the image look will look “cartoonish”
# Filter size:   d > 5 is slow
#                d=5 for real-time applications
#                d=9 for offline applications that need heavy noise filtering

bilateralFilter(src, dst, d::Int, sigmaColor::Float64, sigmaSpace::Float64, borderType=BORDER_DEFAULT) =
     @cxx cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)

# Blur an image using the normalized box filter
# void blur(InputArray src, OutputArray dst, Size ksize, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT )
# Parameters:
# src – input image; it can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
# dst – output image of the same size and type as src.
# ksize – blurring kernel size.
# anchor – anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
# borderType – border mode used to extrapolate pixels outside of the image

blur(src, dst, ksize, anchor=cvPoint(-1,-1), borderType=BORDER_DEFAULT) = @cxx cv::blur(src, dst, ksize, anchor, borderType)

# Blur an image using the box filter
# void boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), bool normalize=true,
# int borderType=BORDER_DEFAULT )
# Parameters:
# src – input image.
# dst – output image of the same size and type as src.
# ddepth – the output image depth (-1 to use src.depth()).
# ksize – blurring kernel size.
# anchor – anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
# normalize – flag, specifying whether the kernel is normalized by its area or not.
# borderType – border mode used to extrapolate pixels outside of the image.

boxFilter(src, dst, ddepth::Int, ksize, anchor=cvPoint(-1,-1), normalize=true, borderType=BORDER_DEFAULT) =
    @cxx cv::boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType)

# Construct the Gaussian pyramid for an image
# void buildPyramid(InputArray src, OutputArrayOfArrays dst, int maxlevel, int borderType=BORDER_DEFAULT )
# Parameters:
# src        – Source image. Check pyrDown() for the list of supported types.
# dst        – Destination vector of maxlevel+1 images of the same type as src . dst[0] will be the same as src .
#               dst[1] is the next pyramid layer, a smoothed and down-sized src , and so on.
# maxlevel   – 0-based index of the last (the smallest) pyramid layer. It must be non-negative.
# borderType – Pixel extrapolation method (BORDER_CONSTANT don’t supported). See borderInterpolate for details.

buildPyramid(src, dst, maxlevel::Int, borderType=BORDER_DEFAULT) = @cxx cv::buildPyramid(src, dst, maxlevel, borderType)

# Dilate an image by using a specific structuring element
# void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT,
#      const Scalar& borderValue=morphologyDefaultBorderValue() )
# Parameters:
# src         – input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F` or ``CV_64F.
# dst         – output image of the same size and type as src.
# kernel      – structuring element used for dilation; if elemenat=Mat() ,
#               a 3 x 3 rectangular structuring element is used. Kernel can be created using getStructuringElement()
# anchor      – position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
# iterations  – number of times dilation is applied.
# borderType  – pixel extrapolation method (see borderInterpolate for details)
# borderValue – border value in case of a constant border  => cv::morphologyDefaultBorderValue()

dilate(src, dst, kernel, anchor=cvPoint(-1,-1), iterations=1, borderType=BORDER_DEFAULT) = @cxx cv::dilate(src, dst, kernel, anchor, iterations, borderType)

# Erode an image by using a specific structuring element
# void erode(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT,
#    const Scalar& borderValue=morphologyDefaultBorderValue() )
# Parameters:
# src         – input image; the number of channels can be arbitrary, but the depth should be one of CV_8U, CV_16U, CV_16S, CV_32F` or ``CV_64F.
# dst         – output image of the same size and type as src.
# kernel      – structuring element used for erosion; if element=Mat()
#               a 3 x 3 rectangular structuring element is used.
#               Kernel can be created using getStructuringElement().
# anchor      – position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.
# iterations  – number of times erosion is applied.
# borderType  – pixel extrapolation method (see borderInterpolate for details)
# borderValue – border value in case of a constant border

erode(src, dst, kernel, anchor=cvPoint(-1,-1), iterations=1, borderType=BORDER_DEFAULT) = @cxx cv::erode(src, dst, kernel, anchor, iterations, borderType)

# Convolve an image with the kernel
# void filter2D(InputArray src, OutputArray dst, int ddepth, InputArray kernel,
#    Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )
# Parameters:
# src         – input image.
# dst         – output image of the same size and the same number of channels as src.
# ddepth      – desired depth of the destination image
#               if < 0, it will be the same as src.depth() => supported
#               src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
#               src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
#               src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
#               src.depth() = CV_64F, ddepth = -1/CV_64F
#               if ddepth=-1, output image depth = source depth
# kernel      – convolution kernel (or rather a correlation kernel), a single-channel floating point matrix;
#               if you want to apply different kernels to different channels, split the image into separate color planes
#               using split() and process them individually.
# anchor      – anchor of the kernel that indicates the relative position of a filtered point within the kernel;
#               the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center.
# delta       – optional value added to the filtered pixels before storing them in dst.
# borderType  – pixel extrapolation method (see borderInterpolate for details)

filter2D(src, dst, ddepth, kernel, anchor=cvPoint(-1,-1), delta=float(0), borderType=BORDER_DEFAULT) = @cxx cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType)

# Blur an image using a Gaussian filter
# void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT)
# Parameters:
# src         – input image; the image can have any number of channels, which are processed independently,
#                but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
# dst         – output image of the same size and type as src.
# ksize       – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd.
#               Or, they can be zero’s and then they are computed from sigma* .
# sigmaX      – Gaussian kernel standard deviation in X direction.
# sigmaY      – Gaussian kernel standard deviation in Y direction;
#              if sigmaY==zero, sigma = sigmaX
#              if sigmaY==zero && sigmaX==zero both sigmas
#              computed from ksize.width and ksize.height
# borderType – pixel extrapolation method (see borderInterpolate for details

gaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType=BORDER_DEFAULT) = @cxx cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType)

# Return filter coefficients for computing spatial image derivatives
# void getDerivKernels(OutputArray kx, OutputArray ky, int dx, int dy, int ksize, bool normalize=false, int ktype=CV_32F )
# Parameters:
# kx        – Output matrix of row filter coefficients. It has the type ktype .
# ky        – Output matrix of column filter coefficients. It has the type ktype .
# dx        – Derivative order in respect of x.
# dy        – Derivative order in respect of y.
# ksize     – Aperture size. It can be CV_SCHARR , 1, 3, 5, or 7.
# normalize – Flag indicating whether to normalize (scale down) the filter coefficients or not.
#             Theoretically, the coefficients should have the denominator  =2^{ksize*2-dx-dy-2} .
#             If you are going to filter floating-point images, you are likely to use the normalized kernels.
#             But if you compute derivatives of an 8-bit image, store the results in a 16-bit image,
#             and wish to preserve all the fractional bits, you may want to set normalize=false .
# ktype     – Type of filter coefficients. It can be CV_32f or CV_64F .

getDerivKernels(kx, ky, dx, dy, ksize, normalize=false, ktype=CV_32F) = @cxx cv::getDerivKernels(kx, ky, dx, dy, ksize, normalize, ktype)

# Return Gaussian filter coefficients
# Mat getGaussianKernel(int ksize, double sigma, int ktype=CV_64F)
# Parameters:
# ksize – Aperture size. It should be odd ( \texttt{ksize} \mod 2 = 1 ) and positive.
# sigma – Gaussian standard deviation. If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
# ktype – Type of filter coefficients. It can be CV_32F or CV_64F .

getGaussianKernel(ksize, sigma::Float64, ktype=CV_64F) = @cxx cv::getDerivKernels(kx, ky, dx, dy, ksize, normalize, ktype)

# Return Gabor filter coefficients
# Mat getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi=CV_PI*0.5, int ktype=CV_64F )
# Parameters:
# ksize – Size of the filter returned.
# sigma – Standard deviation of the gaussian envelope.
# theta – Orientation of the normal to the parallel stripes of a Gabor function.
# lambd – Wavelength of the sinusoidal factor.
# gamma – Spatial aspect ratio.
# psi   – Phase offset.
# ktype – Type of filter coefficients. It can be CV_32F or CV_64F .

getGaborKernel(ksize, sigma::Float64, theta::Float64, lambd::Float64, gamma::Float64, psi=CV_PI*0.5, ktype=CV_64F) =
  cv::getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)

# Get structuring element of the specified size and shape for morphological operations
getStructuringElement(shape::Int, ksize, anchor=cvPoint(-1,-1)) = @cxx cv::getStructuringElement(shape, ksize, anchor)
# shape –
       # MORPH_RECT
       # MORPH_ELLIPSE
       # MORPH_CROSS
# ksize  – Size
# cols   – Width
# rows   – Height
# anchor – Anchor position, anchor_x, anchor_y

# Blur an image using the median filter
medianBlur(src, dst, ksize::Int) = @cxx cv::medianBlur(src, dst, ksize)
# src   – input 1-, 3-, or 4-channel image; when ksize is 3 or 5
# dst   – destination array
# ksize – aperture linear size

# Advanced morphological transformations
morphologyEx(src, dst, op::Int, kernel, anchor=cvPoint(-1,-1), iterations=1, borderType=BORDER_CONSTANT) =
    @cxx cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType)
# optional: const Scalar& borderValue=morphologyDefaultBorderValue()
# src         – Source image: CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
# dst         – Destination image
# kernel      – Structuring element obtained with getStructuringElement()
# anchor      – Anchor position with the kernel
# op –
#   MORPH_OPEN
#   MORPH_CLOSE
#   MORPH_GRADIENT
#   MORPH_TOPHAT
#   MORPH_BLACKHAT
# iterations   – Number of times erosion and dilation are applied
# borderType   – Pixel extrapolation method
# borderValue  – Border value in case of a constant border

# Laplacian of an image
laplacian(src, dst, ddepth::Int, ksize=1, scale=1.0, delta=float(0), borderType=BORDER_DEFAULT) =
    @cxx cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType)

# Blur image and then downsample
pyrDown(src, dst, dstsize, borderType=BORDER_DEFAULT) = @cxx cv::pyrDown(src, dst, dstsize, borderType)
# dstsize = cvSize()

# Upsample image and then blur it
pyrUp(src, dst, dstsize, borderType=BORDER_DEFAULT) = @cxx cv::pyrUp(src, dst, dstsize, borderType)
# dstsize = cvSize()

# meanshift segmentation of an image
# pyrMeanShiftFiltering(src, dst, sp::Float64, sr::Float64, maxLevel=1,
#      termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) )

# Separable linear filter to an image
sepFilter2D(src, dst, ddepth::Int, kernelX, kernelY, anchor=cvPoint(-1,-1), delta=float(0), borderType=BORDER_DEFAULT) =
    @cxx cv::sepFilter2D(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType)
# supported ddepth
# src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
# src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_64F, ddepth = -1/CV_64F
# kernelX – Coefficients for filtering each row
# kernelY – Coefficients for filtering each column
# anchor  – Anchor position within the kernel
# delta   – Value added to the filtered results

# Edge detection with Sobel and Scharr operators
# Calculate the first, second, third, or mixed image derivatives using an extended Sobel operator
sobel(src, dst, ddepth::Int, dx::Int, dy::Int, ksize=3, scale=1.0, delta=float(0), borderType=BORDER_DEFAULT) =
     @cxx cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType)

# Calculate the first x- or y- image derivative using Scharr operator (edge detection)
scharr(src, dst, ddepth::Int, dx::Int, dy::Int, scale=1.0, delta=float(0), borderType=BORDER_DEFAULT) =
   @cxx cv::Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)
# xorder – order of the derivative x
# yorder – order of the derivative y
# ksize  – size of the extended Sobel kernel; it must be 1, 3, 5, or 7
# scale  – optional scale factor for the computed derivative values



# Geometric transformations

# Convert image transformation maps from one representation to another
convertMaps(map1, map2, dstmap1, dstmap2, dstmap1type::Int, nninterpolation=false) =
     @cxx cv::convertMaps(map1, map2, dstmap1, dstmap2, dstmap1type, nninterpolation)
# map1            – The first input map of type CV_16SC2, CV_32FC1, or CV_32FC2
# map2            – The second input map of type CV_16UC1 , CV_32FC1
# dstmap1         – The first output map
# dstmap2         – The second output map
# dstmap1type     – CV_16SC2 , CV_32FC1 , or CV_32FC2
# nninterpolation – Flag indicating whether the fixed-point maps are used

# Calculate an affine transform from three pairs of the corresponding points
getAffineTransform(src, dst) = @cxx cv::getAffineTransform(src, dst)

# Calculate a perspective transform from four pairs of the corresponding points
getPerspectiveTransform(src, dst) = @cxx cv::getPerspectiveTransform(src, dst)
# src: InputArray or const Point2f []
# dst: InputArray or const Point2f []

# Retrieve a pixel rectangle from an image with sub-pixel accuracy
getRectSubPix(image, patchSize, center, patch, patchType=-1) =  @cxx cv::getRectSubPix(image, patchSize, center, patch, patchType)
# src       – Source image
# patchSize – Size of the extracted patch
# center    – Floating point coordinates of the center of the extracted rectangle within the source image.
# dst       – Extracted patch that has the size patchSize and the same number of channels as src
# patchType – Depth of the extracted pixels

# Calculate an affine matrix of 2D rotation
getRotationMatrix2D(center, angle::Float64, scale::Float64) = @cxx cv::getRotationMatrix2D(center, angle, scale)
# center     – Center of the rotation in the source image
# angle      – Rotation angle in degrees
# scale      – Isotropic scale factor
# map_matrix – The output affine transformation, 2x3 floating-point matrix

# Invert an affine transformation.
invertAffineTransform(M, iM) = @cxx cv::invertAffineTransform(M, iM)

# Apply a generic geometrical transformation to an image
remap(src, dst, map1, map2, interpolation::Int, borderMode=BORDER_CONSTANT) =
    @cxx cv::remap(src, dst, map1, map2, interpolation, borderMode)
# map1          – The first map of either (x,y)
# map2          – The second map of y values having the type CV_16UC1 , CV_32FC1
# interpolation – Interpolation method
        # INTER_NEAREST - a nearest-neighbor interpolation
        # INTER_LINEAR - a bilinear interpolation (used by default)
        # INTER_AREA - resampling using pixel area relation
        # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        # NTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood
# borderMode    – Pixel extrapolation method

# Resize an image
resize(src, dst, dsize, fx=float(0), fy=float(0), interpolation=INTER_LINEAR) =
    @cxx cv::resize(src, dst, dsize, fx, fy, interpolation)

# Apply an affine transformation to an image
warpAffine(src, dst, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT) =
    @cxx cv::warpAffine(src, dst, M, dsize, flags, borderMode)
# optional: const Scalar& borderValue=Scalar()
# M – 2 x 3 transformation matrix

# Apply a perspective transformation to an image
warpPerspective(src, dst, M, dsize, flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT) =
    @cxx cv::warpPerspective(src, dst, M, dsize, flag, int borderMode)
# optional: const Scalar& borderValue=Scalar()

# Compute the undistortion and rectification transformation map
initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type::Int, map1, map2) =
    @cxx cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type, map1, map2)
# cameraMatrix     – Input camera matrix
# distCoeffs       – Input vector of distortion coefficients, 4, 5, or 8 elements
# R                – Optional rectification transformation in the object space (3x3 matrix).
# newCameraMatrix  – New camera matrix
# size             – Undistorted image size
# m1type           – Type of the first output map that can be CV_32FC1 or CV_16SC2
# map1             – The first output map
# map2             – The second output map

# Returns the default new camera matrix
getDefaultNewCameraMatrix(cameraMatrix, imgsize, centerPrincipalPoint=false) =
    @cxx cv::getDefaultNewCameraMatrix(cameraMatrix, imgsize, centerPrincipalPoint)
# imgsize              – Camera view image size in pixels
# centerPrincipalPoint – Location of the principal point in the new camera matrix


# Transforms an image to compensate for lens distortion
undistort(src, dst, cameraMatrix, distCoeffs) = @cxx cv::undistort(src, dst, cameraMatrix, distCoeffs)
# optional: newCameraMatrix=noArray()

# Compute the ideal point coordinates from the observed point coordinates
undistortPoints(src, dst, cameraMatrix, distCoeffs) = @cxx cv::undistortPoints(src, dst, cameraMatrix, distCoeffs)
# optional: InputArray R=noArray(), InputArray P=noArray()
# R – Rectification transformation in the object space (3x3 matrix)
# P – New camera matrix (3x3) or new projection matrix (3x4)

# Apply an adaptive threshold to an array
adaptiveThreshold(src, dst, maxValue::Float64, adaptiveMethod::Int, thresholdType::Int, blockSize::Int, C::Float64) =
    @cxx cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# adaptiveMethod:  ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
# thresholdType:  THRESH_BINARY  or THRESH_BINARY_INV

# Convert an image from one color space to another
cvtColor(src, dst, code::Int, dstCn=0) = @cxx cv::cvtColor(src, dst, code, dstCn)
# code – color space conversion code, e.g., COLOR_BGR2GRAY

# Calculate the distance to the closest zero pixel for each pixel of the source
distanceTransform(src, dst, distanceType::Int, maskSize::Int, dstType=CV_32F) =
     @cxx cv::distanceTransform(src, dst, distanceType, maskSize, dstType=CV_32F)
# src          – 8-bit, single-channel (binary) source image
# dst          – Output image with calculated distances. 8-bit or 32-bit floating-point.
# distanceType – Type of distance.
#     DIST_L1
#     DIST_L2
#     DIST_C
# maskSize     – Size of the distance transform mask. It can be 3, 5, or DIST_MASK_PRECISE
# labels       – Optional output 2D array of labels (the discrete Voronoi diagram) - CV_32SC1
# labelType    – Type of the label array to build

# Fill a connected component with the given color
floodFill(image, seedPoint, newVal, rect, loDiff, SupDiff, flags=4) =
     @cxx cv::floodFill(image, seedPoint, newVal, rect, loDiff, SupDiff, flags)
# Rect* rect=0
# image        – Input/output 1- or 3-channel, 8-bit, or floating-point image
# mask         – Operation mask that should be a single-channel 8-bit image,
#                2 pixels wider and 2 pixels taller than image
# seedPoint    – Starting point
# newVal       – New value of the repainted domain pixels
# loDiff       – Maximal lower brightness/color difference
# upDiff       – Maximal upper brightness/color difference
# rect         – Optional output parameter set by the function to the minimum bounding rectangle
# flags        –
#       FLOODFILL_FIXED_RANGE
#       FLOODFILL_MASK_ONLY


# Calculate the integral of an image
integral(src, sum, sdepth=-1) = @cxx cv::integral(src, sum, sdepth)
integral(src, sum, sqsum, sdepth=-1, sqdepth=-1) = @cxx cv::integral(src, sum, sqsum, sdepth, sqdepth)
# image   – input image as W x H, 8-bit or floating-point (32f or 64f).
# sum     – integral image as (W+1) x (H+1), 32-bit integer or floating-point (32f or 64f).
# sqsum   – integral image for squared pixel values
# tilted  – integral for the image rotated by 45 degrees
# sdepth  – depth of the integral and the tilted integral images, CV_32S, CV_32F, or CV_64F
# sqdepth – depth of the integral image of squared pixel values, CV_32F or CV_64F

# Apply a fixed-level threshold to each array element
threshold(src, dst, thresh::Float64, maxval::Float64, ttype::Int) =
    @cxx cv::threshold(src, dst, thresh, maxval, ttype)
# src     – input array (single-channel, 8-bit or 32-bit floating point)
# dst     – output array
# thresh  – threshold value
# maxval  – maximum value with THRESH_BINARY and THRESH_BINARY_INV
# ttype   – thresholding type
#      THRESH_BINARY
#      THRESH_BINARY_INV
#      THRESH_TRUNC
#      THRESH_TOZERO
#      THRESH_TOZERO_INV

# Marker-based image segmentation using the watershed algorithm
watershed(image, markers) = @cxx cv::watershed(image, markers)
# image   – Input 8-bit 3-channel image
# markers – Input/output 32-bit single-channel image (map) of markers

# Run the GrabCut algorithm
grabCut(img, mask, rect, bgdModel, fgdModel, iterCount::Int, mode=GC_EVAL) =
     @cxx cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)
# img       – Input 8-bit 3-channel image
# mask –
      # GC_BGD = obvious background pixels
      # GC_FGD = obvious foreground (object) pixel
      # GC_PR_BGD = possible background pixel
      # GC_PR_FGD = possible foreground pixel
# rect      – ROI containing a segmented object
# bgdModel  – Temporary array for the background model
# fgdModel  – Temporary arrays for the foreground model
# iterCount – Number of iterations the algorithm should make before returning the result
# mode –
     # GC_INIT_WITH_RECT
     # GC_INIT_WITH_MASK
     # GC_EVAL



# Drawing functions

# Draw a circle
circle(img, center, radius::Int, color, thickness=1, lineType=LINE_8, shift=0) =
    @cxx cv::circle(img, center, radius, color, thickness, lineType, shift)
# Clip the line against the image rectangle
clipLine(imgSize, pt1, pt2) = @cxx cv::clipLine(imgSize, pt1, pt2)  #cvSize
clipLine(imgRect, pt1, pt2) = @cxx cv::clipLine(imgRect, pt1, pt2)  #cvRect
# Draw a simple or thick elliptic arc or fills an ellipse sector
ellipse(img, center, axes, angle::Float64, startAngle::Float64, endAngle::Float64, color, thickness=1, lineType=LINE_8, shift=0) =
    @cxx cv::ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift)
# img         – Image
# center      – Center of the ellipse
# axes        – Half of the size of the ellipse main axes
# angle       – Ellipse rotation angle in degrees
# startAngle  – Starting angle of the elliptic arc in degrees
# endAngle    – Ending angle of the elliptic arc in degrees
# box         – Alternative ellipse representation
# color       – Ellipse color
# thickness   – Thickness of the ellipse arc outline
# lineType    – Type of the ellipse boundary
# shift       – Number of fractional bits in the coordinates of the center and values of axes

# Approximate an elliptic arc with a polyline
ellipse2Poly(center, axes, angle::Int, arcStart::Int, arcEnd::Int, delta::Int, pts) =
     @cxx cv::ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta, pts)
# pts = vector<Point>&
# center   – Center of the arc.
# axes     – Half of the size of the ellipse main axes
# angle    – Rotation angle of the ellipse in degrees
# arcStart – Starting angle of the elliptic arc in degrees
# arcEnd   – Ending angle of the elliptic arc in degrees
# delta    – Angle between the subsequent polyline vertices
# pts      – Output vector of polyline vertices

# Fills a convex polygon
fillConvexPoly(img, pts, npts::Int, color, lineType=LINE_8, shift=0) = @cxx cv::fillConvexPoly(img, pts, npts, color, lineType, shift)
# pts = const Point*
fillConvexPoly(img, points, color, lineType=LINE_8, shift=0) = @cxx cv::fillConvexPoly(img, points, color, lineType, shif)

# Fill the area bounded by one or more polygons
fillPoly(img, pts, npts::Ptr{Int}, ncontours::Int, color, lineType=LINE_8, shift=0) =
    @cxx cv::fillPoly(img, pts, npts, ncontours, color, lineType, shift)
# pts = const Point**
# npts = const int*
# optional: Point offset=Point()

fillPoly(img, pts, color, lineType=LINE_8, shift=0) = @cxx cv::fillPoly(img, pts, color, lineType, shift)
# pts = InputArrayOfArrays

# Calculate the width and height of a text string
getTextSize(text::Ptr{Uint8}, fontFace::Int, fontScal::Float64, thickness::Int, baseLine::Ptr{Int}) =
    @cxx cv::getTextSize(text, fontFace, fontScal, thickness, baseLine)
# text       – Input text string
# fontFace   – Font to use
# fontScale  – Font scale
# thickness  – Thickness of lines
# baseLine   – Output parameter - y-coordinate of the baseline
# text_size  – Output parameter - The size of a box that contains the specified text


# Draw a line segment connecting two points
line(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0) =
     @cxx cv::line(img, pt1, pt2, color, thickness, lineType, shift)
# img       – Image
# pt1       – First point
# pt1       – First point
# color     – Line color
# thickness – Line thickness
# lineType  –
#        LINE_8
#        LINE_4
#        LINE_AA
# shift     – Number of fractional bits in the point coordinates

# Draw a arrow segment pointing from the first point to the second one
arrowedLine(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, tipLength=0.1) =
    @cxx cv::arrowedLine(img, pt1, pt2, color, thickness, lineType, shift, tipLength)
# tipLength – The length of the arrow tip in relation to the arrow length

# Class for iterating pixels on a raster line
# cv::LineIterator
# see http://docs.opencv.org/trunk/modules/imgproc/doc/drawing_functions.html

# Draw a simple, thick, or filled up-right rectangle
rectangle(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0) =
     @cxx cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift)

# Draws several polygonal curves
polylines(img, pts, npts::Ptr{Int}, ncontours::Int, isClosed, color, thickness=1, lineType=LINE_8, shift=0) =
     @cxx cv::polylines(img, pts, npts, ncontours, isClosed, color, thickness, lineType, shift)
# pts = const Point* const*

polylines(img, pts, isClosed,color, thickness=1, lineType=LINE_8, shift=0) =
     @cxx cv::polylines(img, pts, isClosed,color, thickness, lineType, shift)
# pts – Array of polygonal curves

# Draw contours outlines or filled contours
drawContours(image, contours, contourIdx::Int, color, thickness=1, lineType=LINE_8, maxLevel=INT_MAX)
# optional:  hierarchy=noArray()
# optional:  Point offset=Point()
# INT_MAX = 0, draw only the specified contour is drawn
# INT_MAX = 1, draws the contour(s) and all the nested contours
# INT_MAX = 2, draws the contours, all the nested contours,and so on...
# offset   - Optional contour shift parameter (x,y)

# Draws a text string
putText(img, text::Ptr{Uint8}, org, fontFace::Int, fontScale::Float64, color, thickness=1, lineType=LINE_8,
  bottomLeftOrigin=false) =  @cxx cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
# fontFace – Font type
# FONT_HERSHEY_SIMPLEX
# FONT_HERSHEY_PLAIN
# FONT_HERSHEY_DUPLEX
# FONT_HERSHEY_COMPLEX
# FONT_HERSHEY_TRIPLEX
# FONT_HERSHEY_COMPLEX_SMALL
# FONT_HERSHEY_SCRIPT_SIMPLEX
# FONT_HERSHEY_SCRIPT_COMPLEX
# FONT_ITALIC

# ColorMaps in OpenCV
# Appl a GNU Octave/MATLAB equivalent colormap on a given image
applyColorMap(src, dst, colormap::Int) = @cxx cv::applyColorMap(src, dst, colormap)
# colormap =
#     COLORMAP_AUTUMN
#     COLORMAP_BONE
#     COLORMAP_JET
#     COLORMAP_WINTER
#     COLORMAP_RAINBOW
#     COLORMAP_OCEAN
#     COLORMAP_SUMMER
#     COLORMAP_SPRING
#     COLORMAP_COOL
#     COLORMAP_HSV
#     COLORMAP_PINK
#     COLORMAP_HOT

# Histograms
calcHist(images, nimages::Int, channels::Ptr{Int}, mask, hist, int dims, histSize::Ptr{Int}, ranges::Ptr{Ptr{Float64}}, uniform=true,
    accumulate=false) = @cxx cv::calcHist(images, nimages, channels, mask, hist, int dims, histSize, ranges, uniform, accumulate)
# images = const Mat*
# Version1    -> hist = Mat&
# Version2    -> hist = SparseMat&
# nimages     – Number of source images
# channels    – The list of channels used to compute the back projection.
# hist        – Input histogram that can be dense or sparse
# backProject – Destination back projection array
# ranges      – Array of arrays of the histogram bin boundaries in each dimension
# scale       – Optional scale factor for the output back projection
# uniform     – Flag indicating whether the histogram is uniform or not
# accumulate  – Accumulation flag: compute a single histogram from several sets of arrays, or to update the histogram in time

# Calculate the back projection of a histogram
calcBackProject(images, nimages::Int, channels::Ptr{Int}, hist, backProject, ranges::Ptr{Ptr{Float64}}, scale=1.0, uniform=true) =
    @cxx cv::calcBackProject(images, nimages, channels, hist, backProject, ranges, scale, uniform)
# parameters same as above

# Compare two histograms
compareHist(H1, H2, method::Int) = @cxx cv::compareHist(H1, H2, method::Int)
# Version1 -> H1, H2 = Mat&
# Version2 -> H1, H2 = SparseMat&
# method –
# HISTCMP_CORREL        Correlation
# HISTCMP_CHISQR        Chi-Square
# HISTCMP_CHISQR_ALT    Alternative Chi-Square
# HISTCMP_INTERSECT     Intersection
# HISTCMP_BHATTACHARYYA Bhattacharyya distance
# HISTCMP_HELLINGER     Synonym for CV_COMP_BHATTACHARYYA
# HISTCMP_KL_DIV        Kullback-Leibler divergence

# EMD: Computes the “minimal work” distance between two weighted point configurations
emd(signature1, signature2, distType::Int, lowerBound=[pointer(float(0))]) =
    @cxx cv::emd(signature1, signature2, distType::Int, lowerBound=[pointer(float(0))])
# optional: InputArray cost=noArray()
# optional: OutputArray flow=noArray()

# equalizeHist: Equalizes the histogram of a grayscale image
equalizeHist(src, dst) = @cxx cv::equalizeHist(src, dst)



# Structural Analysis and Shape Descriptors

# moments
moments(array, binaryImage=false) = @cxx cv::moments(array, binaryImage)
# array       – Raster image (single-channel, 8-bit or floating-point 2D array)
# binaryImage – If it is true, all non-zero image pixels are treated as 1’s
# moments     – Output moments

# Calculates seven Hu invariant
huMoments(m, hu) = @cxx cv::HuMoments(m, hu)

cxx"""
void humoments(const cv::Moments& moments) {
    std::vector<double>hu(7);
    cv::HuMoments(moments, hu)}
"""
huMoments(moments) = @cxx humoments(moments)

# connectedComponents
connectedComponents(image, labels, connectivity=8, ltype=CV_32S) = @cxx cv::connectedComponents(image, labels, connectivity, ltype)
connectedComponentsWithStats(image, labels, stats, centroids, connectivity=8, ltype=CV_32S) =
    @cxx cv::connectedComponentsWithStats(image, labels, stats, centroids, connectivity, ltype)
# image        – the image to be labeled
# labels       – destination labeled image
# connectivity – 8 or 4 for 8-way or 4-way connectivity respectively
# ltype        – output image label type. Currently CV_32S and CV_16U are supported.
# statsv –
#     CC_STAT_LEFT    leftmost (x) of bounding box in the horizontal direction
#     CC_STAT_TOP     topmost (y)  of the bounding box in the vertical direction
#     CC_STAT_WIDTH   horizontal size of the bounding box
#     CC_STAT_HEIGHT  vertical size of the bounding box
#     CC_STAT_AREA    The total area (in pixels) of the connected component
# centroids     – floating point centroid (x,y) output for each label, including the background label

# findContours: Finds contours in a binary image
findContours(image, contours, hierarchy, mode::Int, method::Int) = @cxx cv::findContours(image, contours, hierarchy, mode, method)
findContours(image, contours, mode::Int, method::Int) = @cxx cv::findContours(image, contours, mode, method)
# optional: Point offset=Point()
# Parameters
# image     – Source, an 8-bit single-channel image
# contours  – Detected contours. Each contour is stored as a vector of points
# hierarchy – Optional output vector, containing information about the image topology
# mode –
# RETR_EXTERNAL        extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours
# RETR_LIST            all contours without establishing any hierarchical relationships
# RETR_CCOMP           all contours and organizes them into a two-level hierarchy
# RETR_TREE            all contours and reconstructs a full hierarchy of nested contours
# method –
# CHAIN_APPROX_NONE       all the contour points.
# CHAIN_APPROX_SIMPLE     compresses horizontal, vertical, and diagonal segments and leaves only their end points
# CHAIN_APPROX_TC89_L1
# CHAIN_APPROX_TC89_KCOS  Teh-Chin chain approximation algorithm
# offset      –           Optional offset by which every contour point is shifted
#                         useful if the contours are extracted from the image ROI


# approxPolyDP: Approximate a polygonal curve(s) with the specified precision
approxPolyDP(curve, approxCurve, epsilon::Float64, closed=false) = @cxx cv::approxPolyDP(curve, approxCurve, epsilon, closed)
# curve         –    Input vector of a 2D point stored in: std::vector or Mat
# approxCurve   –    Result of the approximation. The type should match the type of the input curve
# epsilon       –    Parameter specifying the approximation accuracy
# closed        –    bool = true, then approximated curve is closed. Otherwise, not closed
# header_size   –    Header size of the approximated curve. Normally, sizeof(CvContour)
# storage       –    Memory storage where the approximated curve is stored
# method        –    Contour approximation algorithm. Only CV_POLY_APPROX_DP is supported.
# recursive     –    Recursion flag.

# arcLength: Calculate a contour perimeter or a curve length
arcLength(curve, closed) = @cxx cv::arcLength(curve, closed)
# curve  – Input vector of 2D points, stored in std::vector or Mat
# closed – Flag indicating whether the curve is closed or not

# boundingRect: Calculate the up-right bounding rectangle of a point set
boundingRect(points) = @cxx cv::boundingRect(points)

# contourArea: Calculates a contour area
contourArea(contour, oriented=false) = @cxx cv::contourArea(contour, oriented)
# contour      – Input vector of 2D points (contour vertices), stored in std::vector or Mat
# oriented     – Oriented area flag, default = false
#                if true, signed area value depending on clockwise or counter-clockwise
#                used to determine orientation of a contour by taking the sign of an area

# convexHull: Finds the convex hull of a point set
convexHull(points, hull, clockwise=false, returnPoints=true) = @cxx cv::convexHull(points, hull, clockwise, returnPoints)
# points       – Input 2D point set, stored in std::vector or Mat
# hull         – Output convex hull: integer vector of indices or vector of points
# clockwise    – Orientation flag. If true, the output convex hull is oriented clockwise. Otherwise, counter-clockwise.
# returnPoints – Operation flag. In case of a matrix, when the flag is true, the function returns convex hull points.

# convexityDefects: Finds the convexity defects of a contour
convexityDefects(contour, convexhull, convexityDefects) = @cxx cv::convexityDefects(contour, convexhull, convexityDefects)
# contour          – Input contour
# convexhull       – Convex hull obtained using convexHull()
# convexityDefects – The output vector of convexity defects: 4-element integer vector (a.k.a. cv::Vec4i)

# fitEllipse: Fits an ellipse around a set of 2D points
fitEllipse(points) = @cxx cv::fitEllipse(points)
# points – Input 2D point set, stored in std::vector<> or Mat
# returns: cvRotatedRect

# fitLine: Fits a line to a 2D or 3D point set
fitLine(points, line, distType::Int, param::Float64, reps::Float64, aeps::Float64) =
    @cxx cv::fitLine(points, line, distType, param, reps, aeps)
# points   – Input vector of 2D or 3D points, stored in std::vector<> or Mat.
# line     – Output line parameters.
#             2D: 4 elements (like Vec4f) normalized vector colinear, (x0, y0) is on the line
#             3D: 6 elements (like Vec6f) normalized vector colinear, (x0, y0, z0) is on the line
# distType – Distance used by the M-estimator
#     DIST_L2
#     DIST_L1
#     DIST_FAIR
#     DIST_WELSCH
#     DIST_HUBER
# param    – Numerical parameter (C) for some types of distances. If 0, optimal is chosen.
# reps     – radius accuracy, set default = 0.01 (distance between the coordinate origin and the line)
# aeps     – angle accuracy, set default = 0.01

# isContourConvex: Tests a contour convexity
isContourConvex(contour) = @cxx cv::isContourConvex(contour)
# contour – 2D std::vector<> or Mat

# minAreaRect: Find a rotated rectangle of the minimum area enclosing the input 2D point set
minAreaRect(points) = @cxx cv::minAreaRect(points)
# contour – 2D std::vector<> or Mat
# returns RotatedRect

# boxPoints: Find the four vertices of a rotated rect. Useful to draw the rotated rectangle
boxPoints(box, points) = @cxx cv::boxPoints(box, points)
# box      – The input rotated rectangle
# points   – The output array of four vertices of rectangles.

# minEnclosingTriangle: Finds a triangle of minimum area enclosing a 2D point set and returns its area
minEnclosingTriangle(points, triangle) = @cxx cv::minEnclosingTriangle(points, triangle)
# points   – Input vector of 2D points with depth CV_32S or CV_32F in std::vector<> or Mat
# triangle – Output vector of three 2D points defining the vertices of the triangle, must be CV_32F

# minEnclosingCircle: Finds a circle of the minimum area enclosing a 2D point set
minEnclosingCircle(points, center, radius) = @cxx cv::minEnclosingCircle(points, center, radius)
# points     – Input vector of 2D points in std::vector<> or Mat
# center     – Output center of the circle, cvPoint2f&
# radius     – Output radius of the circle, double&

# matchShapes: Compares two shapes
matchShapes(contour1, contour2,method::Int, parameter::Float64) =
     @cxx cv::matchShapes(contour1, contour2,method::Int, parameter::Float64)
# object1 – First contour or grayscale image
# object2 – Second contour or grayscale image
# method –  all 3 use the Hu invariants
#    CV_CONTOURS_MATCH_I1
#    CV_CONTOURS_MATCH_I2
#    CV_CONTOURS_MATCH_I3

# pointPolygonTest: Performs a point-in-contour test
pointPolygonTest(contour, pt, measureDist) = @cxx cv::pointPolygonTest(contour, pt, measureDist)
# contour     – Input contour
# pt          – Point tested against the contour
# measureDist – If true, estimates signed distance from the point to the nearest contour edge

# rotatedRectangleIntersection: Find out if there is any intersection between two rotated rectangles
rotatedRectangleIntersection(rect1, rect2, intersectingRegion) =
    @cxx cv::rotatedRectangleIntersection(rect1, rect2, intersectingRegion)
# rect1                – First rectangle  (cvRotatedRect)
# rect2                – Second rectangle (cvRotatedRect)
# intersectingRegion   – The output array of the verticies of the intersecting region
#                        max 8 vertices => std::vector<cv::Point2f> or cv::Mat as Mx1 of type CV_32FC2
# pointCount           – The number of vertices
# Output
# INTERSECT_NONE=0     – No intersection
# INTERSECT_PARTIAL=1  – There is a partial intersection
# INTERSECT_FULL=2     – One of the rectangle is fully enclosed in the other

# Motion Analysis and Object

# accumulate: Adds an image to the accumulator
accumulate(src, dst) = @cxx cv::accumulate(src, dst)
# accumulateSquare: Adds the square of a source image to the accumulator
accumulateSquare(src, dst) = @cxx cv::accumulateSquare(src, dst)
# optional: InputArray mask=noArray()
# src         – Input image as 1- or 3-channel, 8-bit or 32-bit floating point
# dst         – Accumulator image with 32-bit or 64-bit floating-point (same channels as src)

# accumulateProduct: Adds the per-element product of two input images to the accumulator
accumulateProduct(src1, src2, dst) = @cxx cv::accumulateProduct(src1, src2, dst)
# optional: InputArray mask=noArray()
# src1, src2  – Input images as 1- or 3-channel, 8-bit or 32-bit floating point
# dst         – Accumulator image with 32-bit or 64-bit floating-point (same channels as src)

# accumulateWeighted: Updates a running average
accumulateWeighted(src, dst, alpha::Float64) = @cxx cv::accumulateWeighted(src, dst, alpha)
# optional: InputArray mask=noArray()
# src1, src2  – Input images as 1- or 3-channel, 8-bit or 32-bit floating point
# dst         – Accumulator image with 32-bit or 64-bit floating-point (same channels as src
# alpha       – Weight of the input image

# phaseCorrelate: detect translational shifts that occur between two images
phaseCorrelate(src1, src2, response=pointer(float(0))) = @cxx cv::phaseCorrelate(src1, src2, response)
# optional: InputArray mask=noArray()
# src1        – Source floating point array (CV_32FC1 or CV_64FC1)
# src2        – Source floating point array (CV_32FC1 or CV_64FC1)
# window      – Floating point array with windowing coefficients to reduce edge effects (optional)
# response    – Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional)

# createHanningWindow: computes a Hanning window coefficients in two dimensions
createHanningWindow(dst, winSize, rtype::Int) = @cxx cv::createHanningWindow(dst, winSize, rtype)
# dst         – Destination array to place Hann coefficients in
# winSize     – The window size specifications
# winSize     – The window size specifications



# Feature Detection

# Canny: Finds edges in an image using the [Canny86] algorithm
Canny(image, edges, threshold1::Float64, threshold2::Float64, apertureSize=3, L2gradient=false) =
    @cxx cv::Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)
# image         – 8-bit input image
# edges         – output edge map; single channels 8-bit image, which has the same size as image
# threshold1    – first threshold for the hysteresis procedure
# threshold2    – second threshold for the hysteresis procedure
# apertureSize  – aperture size for the Sobel() operator
# L2gradient    – a flag, indicating whether a more accurate

# cornerEigenValsAndVecs
cornerEigenValsAndVecs(src, dst, blockSize::Int, ksize::Int, borderType=BORDER_DEFAULT) =
    @cxx cv::cornerEigenValsAndVecs(src, dst, blockSize, ksize, borderType)
# src          – Input single-channel 8-bit or floating-point image
# dst          – Image to store the results. It has the same size as src and the type CV_32FC(6)
# blockSize    – Neighborhood size
# ksize        – Aperture parameter for the Sobel() operator
# borderType   – Pixel extrapolation method. See borderInterpolate()

# cornerHarris: Harris corner detector
cornerHarris(src, dst, blockSize::Int, ksize::Int, k::Float64, borderType=BORDER_DEFAULT) =
    @cxx cv::cornerHarris(src, dst, blockSize, ksize, k, borderType)
# src          – Input single-channel 8-bit or floating-point image
# dst          – Image to store the Harris detector responses. It has the type CV_32FC1
# blockSize    – Neighborhood size
# ksize        – Aperture parameter for the Sobel() operator
# k            – Harris detector free parameter
# borderType   – Pixel extrapolation method

# cornerMinEigenVal: Calculates the minimal eigenvalue of gradient matrices for corner detection
cornerMinEigenVal(src, dst, blockSize::Int,ksize=3, borderType=BORDER_DEFAULT) =
     @cxx cv::cornerMinEigenVal(src, dst, blockSize,ksize, borderType)
# src          – Input single-channel 8-bit or floating-point image
# dst          – Image to store the Harris detector responses. It has the type CV_32FC1
# blockSize    – Neighborhood size
# ksize        – Aperture parameter for the Sobel() operator
# borderType   – Pixel extrapolation method

# cornerSubPix: Refines the corner locations
cornerSubPix(image, corners, winSize, zeroZone, criteria) =
     @cxx cv::cornerSubPix(image, corners, winSize, zeroZone, criteria)
# image         – Input image
# corners       – Initial coordinates of the input corners and refined coordinates provided for output
# winSize       – Half of the side length of the search window e.g., winSize=Size(5,5)
# zeroZone      – Half of the size of the dead region in the middle of the search zone
# criteria      – Criteria for termination of the iterative process of corner refinement  (TermCriteria)

# goodFeaturesToTrack:
goodFeaturesToTrack(image, corners, maxCorners::Int, qualityLevel::Float64, minDistance::Float64,
     blockSize=3, useHarrisDetector=false, k=0.04) = @cxx cv::goodFeaturesToTrack(image, corners,
         maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k)
# image             – Input 8-bit or floating-point 32-bit, single-channel image.
# corners           – Output vector of detected corners
# maxCorners        – Maximum number of corners to return
# qualityLevel      – Parameter characterizing the minimal accepted quality of image corners
# minDistance       – Minimum possible Euclidean distance between the returned corners
# blockSize         – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
# useHarrisDetector – Parameter indicating whether to use a Harris detector (see cornerHarris()) or cornerMinEigenVal().
# optional:           Region of interest - InputArray mask=noArray()
# k                 – Free parameter of the Harris detector


# HoughCircles: Finds circles in a grayscale image using the Hough transform
houghCircles(image, circles, method=HOUGH_GRADIENT, dp::Float64, minDist::Float64, param1=100.0, param2=100.0,
      minRadius=0, maxRadius=0) = @cxx cv::HoughCircles(image, circles, method, dp, minDist, param1,
            param2, minRadius, maxRadius)
# image      – 8-bit, single-channel, grayscale input image.
# circles    – Output vector of found circles. 3-element floating-point vector (x, y, radius)
# method     – Detection method = HOUGH_GRADIENT, described in [Yuen90].
# dp         – Inverse ratio of the accumulator resolution to the image resolution
#             dp=1 : resolution unchanged
#             dp=2 : 1/2 resolution of original
# minDist    – Minimum distance between the centers of the detected circles
# param1     – First method-specific parameter (Canny() edge detector threshold)
# param2     – Second method-specific parameter (accumulator threshold for circle centers)
# minRadius  – Minimum circle radius
# maxRadius  – Maximum circle radius

# HoughLines: Finds lines in a binary image using the standard Hough transform
houghLines(image, lines, rho::Float64, theta::Float64, threshold::Int, srn=float(0),
           stn=float(0), min_theta =float(0), max_theta=CV_PI)
# image       – 8-bit, single-channel binary source image
# lines       – Output vector of lines
# rho         – Distance resolution of the accumulator in pixels
# theta       – Angle resolution of the accumulator in radians
# threshold   – Accumulator threshold parameter
# srn         – For the multi-scale Hough transform, it is a divisor for the distance resolution rho
# stn         – For the multi-scale Hough transform, it is a divisor for the distance resolution theta
# min_theta   – minimum angle to check for lines - range {0, max_theta}
# min_theta   – maximum angle to check for lines - range {min_theta, CV_PI}
# method –
#         HOUGH_STANDARD
#         HOUGH_PROBABILISTIC
#         HOUGH_MULTI_SCALE

# HoughLinesP: Finds line segments in a binary image using the probabilistic Hough transform
houghLinesP(image, lines, rho::Float64, theta::Float64, threshold::Int, minLineLength=float(0), maxLineGap=float(0)) =
     @cxx cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap)
# same as above, but with the following added
# minLineLength   – Minimum line length. Line segments shorter than that are rejected
# maxLineGap      – Maximum allowed gap between points on the same line to link them

# createLineSegmentDetector
createLineSegmentDetector(int _refine=LSD_REFINE_STD, _scale=0.8, _sigma_scale=0.6, _quant=2.0, _ang_th=22.5,
    _log_eps=float(0), _density_th=0.7, _n_bins=1024) = @cxx cv::createLineSegmentDetector(int _refine,
          _scale, _sigma_scale, _quant, _ang_th, _log_eps, _density_th, _n_bins)
# _refine  -
# LSD_REFINE_NONE
# LSD_REFINE_STD
# LSD_REFINE_ADV
# scale       – The scale of the image that will be used to find the lines. Range (0..1]
# sigma_scale – Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
# quant       – Bound to the quantization error on the gradient norm.
# ang_th      – Gradient angle tolerance in degrees
# log_eps     – Detection threshold: -log10(NFA) > log_eps. (ONLY REFINE_ADV)
# density_th  – Minimal density of aligned region points in the enclosing rectangle
# n_bins      – Number of bins in pseudo-ordering of gradient modulus

# LineSegmentDetector::detect
detectLines(linesegmentDetector, _image, _lines) = @cxx linesegmentDetector->detect(_image, _lines)
# _lines        – A vector of Vec4i elements specifying the beginning and ending point of a line
# nfa           – Vector containing number of false alarms
#                 -1 =>  10 mean false alarms
#                  0 =>   1 mean false alarm
#                  1 => 0.1 mean false alarms
# Default:
# OutputArray width=noArray()
# OutputArray prec=noArray()
# OutputArray nfa=noArray()

# LineSegmentDetector::drawSegments
drawSegments(linesegmentDetector, _image, lines)  = @cxx linesegmentDetector->drawSegments(_image, lines)
# image – The image, where the liens will be drawn
# lines – A vector of the lines that needed to be drawn

# LineSegmentDetector::compareSegments
compareSegments(linesegmentDetector, size, lines1, lines2, image)  = @cxx linesegmentDetector->compareSegments(size, lines1, lines2, image)
# size     – The size of the image, where lines1 and lines2 were found.
# lines1   – The first group of lines that needs to be drawn. It is visualized in blue color.
# lines2   – The second group of lines. They visualized in red color.
# image    – Optional image, where the lines will be drawn.

# preCornerDetect: Calculates a feature map for corner detection
preCornerDetect(src, dst, ksize::Int, borderType=BORDER_DEFAULT) = @cxx cv::preCornerDetect(src, dst, ksize, borderType)
# src – Source single-channel 8-bit of floating-point image
# dst – Output image that has the type CV_32F and the same size as src
# ksize – Aperture size of the Sobel()
# borderType – Pixel extrapolation method



# Object Detection

# matchTemplate: Compares a template against overlapped image regions
matchTemplate(image, templ, result, method::Int) = @cxx cv: matchTemplate(image, templ, result, method)
# image    – Image where the search is running. It must be 8-bit or 32-bit floating-point
# templ    – Searched template. It must be not greater than the source image and have the same data type.
# result   – Map of comparison results. It must be single-channel 32-bit floating-point
# method   – Parameter specifying the comparison method
#    TM_SQDIFF
#    TM_SQDIFF_NORMED
#    TM_CCORR
#    TM_CCORR_NORMED
#    TM_CCOEFF
#    TM_CCOEFF_NORMED





####################################################################################################
# Reading and Writing Video (videoio.hpp)
####################################################################################################

# Create the VideoCapture structures
cxx""" cv::VideoCapture VideoCapture(){ cv::VideoCapture() capture; return(capture); }"""
videoCapture() = @cxx VideoCapture()

cxx""" cv::VideoCapture VideoCapture(const char *filename){ cv::VideoCapture(filename) capture; return(capture); }"""
videoCapture(filename::Ptr{Uint8}) = @cxx VideoCapture(filename)

cxx""" cv::VideoCapture VideoCapture(int device){ cv::VideoCapture(device) capture; return(capture); }"""
videoCapture(device::Int) = @cxx VideoCapture(device)   # autodetect = 0

# Functions for opening capture and grabbing frames
openVideo(capture,filename::Ptr{Uint8}) = @cxx capture->open(filename)
openVideo(capture, device::Int) = @cxx capture->open(device)
isOpened(capture) = @cxx capture->isOpened()

# Useful for multi-camera environments
grab(capture) = @cxx capture->grab()
# Decodes and returns the grabbed video frame
retrieve(capture, image, flag=0) = @cxx capture->retrieve(image, flag)
# automatically called by cv::VideoCapture->open
release(capture) = @cxx capture->release()  # automatically called

# Grab, decode and return the next video frame
cxx""" cv::VideoCapture&  videoread(cv::VideoCapture& capture){ cv::Mat& image; capture >> image; return(capture); } """
videoRead(capture) = @cxx videoread(capture)
videoRead(capture, image) = @cxx capture->read(image)

# Return the specified VideoCapture property
getVideoId(capture, propId::Int) = @cxx capture->get(propId)
# CAP_PROP_POS_MSEC       Current position of the video file (msec or timestamp)
# CAP_PROP_POS_FRAMES     0-based index of the frame to be decoded/captured next
# CAP_PROP_POS_AVI_RATIO  Relative position of the video file: 0 - start of the film, 1 - end of the film
# CAP_PROP_FRAME_WIDTH    Width of the frames in the video stream
# CAP_PROP_FRAME_HEIGHT   Height of the frames in the video stream
# CAP_PROP_FPS            frame rate
# CAP_PROP_FOURCC         4-character code of codec
# CAP_PROP_FRAME_COUNT    Number of frames in the video file
# CAP_PROP_FORMAT         Format of the Mat objects returned by retrieve()
# CAP_PROP_MODE           Backend-specific value indicating the current capture mode
# CAP_PROP_BRIGHTNESS     Brightness of the image (only for cameras)
# CAP_PROP_CONTRAST       Contrast of the image (only for cameras)
# CAP_PROP_SATURATION     Saturation of the image (only for cameras)
# CAP_PROP_HUE            Hue of the image (only for cameras)
# CAP_PROP_GAIN           Gain of the image (only for cameras)
# CAP_PROP_EXPOSURE       Exposure (only for cameras)
# CAP_PROP_CONVERT_RGB    Boolean flags indicating whether images should be converted to RGB
# CAP_PROP_WHITE_BALANCE  Currently not supported
# CAP_PROP_RECTIFICATION  Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

# Sets a property in the VideoCapture
setVideoId(capture, propId::Int, value::Float64) = @cxx capture->set(propId, value)

# Create the VideoWriter structures
cxx""" cv::VideoWriter VideoWriter(){ cv::VideoWriter() writer; return(writer); }"""
videoWriter() = @cxx VideoWriter()

cxx""" cv::VideoWriter VideoWriter(const char *filename, int fourcc, double fps, Size frameSize,
    bool isColor=true){ cv::VideoWriter writer(filename, fourcc, fps, frameSize, isColor); return(capture); }"""
videoWriter(filename::Ptr{Uint8}, fourcc::Int, fps::Float64, frameSize, isColor=true) =
    @cxx VideoCapture(filename, fourcc, fps, frameSize, isColor)
# Parameters
# filename  – Name of the output video file.
# fourcc    – Fourcc codec, e.g., fourcc('M','J','P','G')
# fps       – Framerate
# frameSize – Size of the video frames
# isColor   –  only supported for Windows

openWriter(filename::Ptr{Uint8}, fourcc::Int, fps::Float64, frameSize, isColor=true) =
    @cxx writer->open(filename, fourcc, fps, frameSize, isColor)

isOpened(writer) = @cxx writer->isOpened()

# Write the next video frame
videoWrite(writer, image) = @cxx writer->

# Video write with operator >>
cxx""" cv::VideoCapture&  videoread(cv::VideoCapture& capture){ cv::Mat& image; capture >> image; return(capture); } """
videoRead(capture) = @cxx videoread(capture)

# Get int for the FourCC codec code
fourcc(writer, fcc::Array(Ptr{Uint8},4)) = @cxx writer->fourcc(fcc[1], fcc[2], fcc[3], fcc[4])
# CV_FOURCC_IYUV  #for yuv420p into an uncompressed AVI
# CV_FOURCC_DIV3  #for DivX MPEG-4 codec
# CV_FOURCC_MP42  #for MPEG-4 codec
# CV_FOURCC_DIVX  #for DivX codec
# CV_FOURCC_PIM1  #for MPEG-1 codec
# CV_FOURCC_I263  #for ITU H.263 codec
# CV_FOURCC_MPEG  #for MPEG-1 codec


####################################################################################################
# Graphical user interface (highgui)
####################################################################################################

# createTrackbar: Creates a trackbar and attaches it to the specified window
# create a TrackbarCallback function in C++ and wrap with @cxx macro
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

# MoveWindow
moveWindow(winname::Ptr{Uint8}, x:Int, y::Int) = @cxx cv::moveWindow(winname, x, y)
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

