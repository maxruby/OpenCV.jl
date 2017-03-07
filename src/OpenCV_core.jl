################################################################################################
# core. The Core Functionality
################################################################################################

# 1.  Basic Structures
#     Point
#     Size
#     Rect
#     RotatedRect
#     TermCriteria
#     TermCriteria::TermCriteria
#     Vec
#     Scalar
#     Range
#     Mat::Mat
#     Mat::~Mat
#     Mat::operator =
#     Mat::row
#     Mat::col
#     Mat::rowRange
#     Mat::colRange
#     Mat::diag
#     Mat::clone
#     Mat::copyTo
#     Mat::convertTo
#     Mat::assignTo
#     Mat::setTo
#     Mat::reshape
#     Mat::t
#     Mat::inv
#     Mat::mul
#     Mat::cross
#     Mat::dot
#     Mat::zeros
#     Mat::ones
#     Mat::eye
#     Mat::create
#     Mat::addref
#     Mat::release
#     Mat::resize
#     Mat::reserve
#     Mat::push_back
#     Mat::pop_back
#     Mat::locateROI
#     Mat::adjustROI
#     Mat::operator()
#     Mat::total
#     Mat::isContinuous
#     Mat::elemSize
#     Mat::elemSize1
#     Mat::type
#     Mat::depth
#     Mat::channels
#     Mat::step1
#     Mat::size
#     Mat::empty
#     Mat::ptr
#     Mat::at
#     Mat::begin
#     Mat::end
#     Mat::forEach
#     Mat_
#     InputArray
#     OutputArray
#     MatIterator

# 2. Operations on Arrays
#     abs
#     absdiff
#     add
#     addWeighted
#     bitwise_and
#     bitwise_not
#     bitwise_or
#     bitwise_xor
#     calcCovarMatrix
#     cartToPolar
#     checkRange
#     compare
#     completeSymm
#     convertScaleAbs
#     countNonZero
#     cvarrToMat
#     dct
#     dft
#     divide
#     determinant
#     eigen
#     exp
#     extractImageCOI
#     insertImageCOI
#     flip
#     gemm
#     getOptimalDFTSize
#     idct
#     idft
#     inRange
#     invert
#     log
#     LUT
#     magnitude
#     Mahalanobis
#     max
#     mean
#     meanStdDev
#     merge
#     min
#     minMaxIdx
#     minMaxLoc
#     mixChannels
#     mulSpectrums
#     multiply
#     mulTransposed
#     norm
#     normalize
#     PCA
#     PCA::PCA
#     PCA::operator ()
#     PCA::project
#     PCA::backProject
#     perspectiveTransform
#     phase
#     polarToCart
#     pow
#     RNG
#     RNG::RNG
#     RNG::next
#     RNG::operator T
#     RNG::operator ()
#     RNG::uniform
#     RNG::gaussian
#     RNG::fill
#     randu
#     randn
#     randShuffle
#     reduce
#     repeat
#     scaleAdd
#     setIdentity
#     solve
#     solveCubic
#     solvePoly
#     sort
#     sortIdx
#     split
#     sqrt
#     subtract
#     SVD
#     SVD::SVD
#     SVD::operator ()
#     SVD::compute
#     SVD::solveZ
#     SVD::backSubst
#     sum
#     theRNG
#     trace
#     transform
#     transpose
#     borderInterpolate
#     copyMakeBorder

# 3.  Clustering
#     kmeans
#     partition

# 4.  Utility and System Functions and Macros
#     alignPtr
#     alignSize
#     allocate
#     deallocate
#     fastAtan2
#     cubeRoot
#     Ceil
#     Floor
#     Round
#     IsInf
#     IsNaN
#     CV_Assert
#     error
#     Exception
#     fastMalloc
#     fastFree
#     format
#     getBuildInformation
#     checkHardwareSupport
#     getNumberOfCPUs
#     getNumThreads
#     getThreadNum
#     getTickCount
#     getTickFrequency
#     getCPUTickCount
#     saturate_cast
#     setNumThreads
#     setUseOptimized
#     useOptimized

# import base functions to explicitly extend
import Base.copy,
       Base.convert,
       Base.split

#-------------------------------------------------------------------------------------------------------------------#
# 1. Basic structures

cxx"""
// UInt8 => Cuchar
cv::Vec2b vec2b(uchar a, uchar b) { return cv::Vec2b(a,b); }
cv::Vec3b vec3b(uchar a, uchar b, uchar c) { return cv::Vec3b(a,b,c); }
cv::Vec4b vec4b(uchar a, uchar b, uchar c, uchar d) { return cv::Vec4b(a,b,c,d); }

// Int32 => Cint
cv::Vec2i vec2i(int a, int b) { return cv::Vec2i(a,b); }
cv::Vec3i vec3i(int a, int b, int c) { return cv::Vec3i(a,b,c); }
cv::Vec4i vec4i(int a, int b, int c, int d) { return cv::Vec4i(a,b,c,d); }

// Float32
cv::Vec2f vec2f(float a, float b) { return cv::Vec2f(a,b); }
cv::Vec3f vec3f(float a, float b, float c) { return cv::Vec3f(a,b,c); }
cv::Vec4f vec4f(float a, float b, float c, float d) { return cv::Vec4f(a,b,c,d); }
cv::Vec6f vec6f(float a, float b, float c, float d, float e, float f) { return cv::Vec6f(a,b,c,d,e,f); }

// Float64 => Cdouble
cv::Vec2d vec2d(double a, double b) { return cv::Vec2d(a,b); }
cv::Vec3d vec3d(double a, double b, double c) { return cv::Vec3d(a,b,c); }
cv::Vec4d vec4d(double a, double b, double c, double d) { return cv::Vec4d(a,b,c,d); }
cv::Vec6d vec6d(double a, double b, double c, double d, double e, double f) { return cv::Vec6d(a,b,c,d,e,f); }
"""

vec2b(a,b) = @cxx vec2b(a, b)
vec3b(a,b,c) = @cxx vec3b(a, b, c)
vec4b(a,b,c,d) = @cxx vec4b(a, b, c, d)

vec2i(a,b) = @cxx vec2i(cint(a), cint(b))
vec3i(a,b,c) = @cxx vec3i(cint(a), cint(b), cint(c))
vec4i(a,b,c,d) = @cxx vec4i(cint(a), cint(b), cint(c), cint(d))

vec2f(a,b) = @cxx vec2f(float32(a), float32(b))
vec3f(a,b,c) = @cxx vec3f(float32(a), float32(b), float32(c))
vec4f(a,b,c,d) = @cxx vec4f(float32(a), float32(b), float32(c), float32(d))
vec6f(a,b,c,d,e,f) = @cxx vec6f(float32(a),float32(b), float32(c), float32(d), float32(e), float32(f))

vec2d(a,b) = @cxx vec2d(float(a), float(b))
vec3d(a,b,c) = @cxx vec3d(float(a), float(b), float(c))
vec4d(a,b,c,d) = @cxx vec4d(float(a), float(b), float(c), float(d))
vec6d(a,b,c,d,e,f) = @cxx vec6d(float(a), float(b), float(c), float(d), float(e), float(f))

# Point
cvPoint(x, y) = @cxx cv::Point(x,y)
cvPoint2f(x, y) = @cxx cv::Point2f(float32(x),float32(y))
cvPoint2d(x, y) = @cxx cv::Point2d(float(x),float(y))
cvPoint3i(x, y, z) = @cxx cv::Point3f(cint(x),cint(y),cint(z))
cvPoint3f(x, y, z) = @cxx cv::Point3f(float32(x),float32(y),float32(z))
cvPoint3d(x, y, z) = @cxx cv::Point3d(float64(x),float64(y),float64(z))

# Size
cvSize(width, height) = @cxx cv::Size(width,height)
cvSize2f(width, height) = @cxx cv::Size2f(width,height)

# Scalar
cvScalar(blue::Int, green::Int, red::Int) = @cxx cv::Scalar(blue,green,red)

# Range
cvRange(start::Int, tend::Int) = @cxx cv::Range(start, tend)
cvRange_all(range) = @cxx range->all()

# Rectangle
cvRect(x::Int, y::Int, width::Int, height::Int) = @cxx cv::Rect(x, y, width, height)

# Rotated Rectangle
cvRotatedRect() =  @cxx cv::RotatedRect()
# const center = cvPoint2f(x,y), const size = cv::Size2f(width,height)
cvRotatedRect(center, size, angle::Float32) = @cxx cv::RotatedRect(center, size, angle)
# const point1 = cvPoint2f(x,y), const point2 = cvPoint2f(x,y), const point3 = cvPoint2f(x,y)
cvRotatedRect(point1, point2, point3) = @cxx cv::RotatedRect(point1, point2, point3)
# arrpts(x,y) = [cvPoint2f(x,y)]
cvRotatedRectPoints(arrpts) = @cxx rRect->points(arrpts)
# rRect = cvRotatedRect(pts, size, angle)
cvRotatedRectBoundingRect(rRect) = @cxx rRect->boundingRect()

#cv::String class
cvString() = @cxx cv::String()
cvString(str) = @cxx cv::String(str)
cvString(str, pos, len) = @cxx cv::String(str, csize_t(pos), csize_t(len))
cvString(s::String) = @cxx cv::String(pointer(s))
cvString(s::String, n) = @cxx cv::String(s::String, csize_t(n))
cvString(first::String, last::String) = @cxx cv::String(pointer(first), pointer(last))

#TermCriteria
# type     – TermCriteria::COUNT, TermCriteria::EPS or TermCriteria::COUNT + TermCriteria::EPS
# maxCount – The maximum number of iterations or elements to compute
# epsilon  – The desired accuracy or change in parameters

TermCriteria()      = @cxx cv::TermCriteria::TermCriteria()
TermCriteria(rtype::Int, maxCount::Int, epsilon::Float64) =
     @cxx cv::TermCriteria::TermCriteria(rtype, maxCount, epsilon)
TermCriteriaCount() = @cxx cv::TermCriteria::COUNT  # ::CppEnum  (not int)!
TermCriteriaEPS()   = @cxx cv::TermCriteria::EPS      # ::CppEnum  (not int)!

# to get the value of TermCriteriaCount and TermCriteriaEPS, e.g.,
# TermCriteriaCount().val or TermCriteriaEPS().val
# e.g., criteria=TermCriteria(int(TermCriteriaCount().val+TermCriteriaEPS().val), 30, 0.01)

###################################################
# Mat: The core image array structure in OpenCV
###################################################
# Mat array constructors
# Parameters:
# ndims    –>   dimension
# rws      –>   Number of rows in a 2D array
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


# Mat::Mat()
Mat() = @cxx cv::Mat()

# Mat::Mat(int rows, int cols, int type)
Mat(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat(rows, cols, matType)

# Mat::Mat(Size size, int type)
#cxx""" cv::Mat Mat(int rows, int cols, int matType){ cv::Mat img(cv::Size(rows,cols), matType); return(img); }"""
Mat(size, matType::CV_MatType) = @cxx cv::Mat(size, matType)

# Mat::Mat(int rows, int cols, int type, const Scalar& s)
#cxx""" cv::Mat Mat(int rows, int cols, int matType){ cv::Mat img(cv::Size(rows,cols), matType); return(img); }"""
Mat(rows::Int, cols::Int, matType::CV_MatType, s) = @cxx cv::Mat(rows, cols, matType, s)

# Mat::Mat(Size size, int type, const Scalar& s)
Mat(size, matType::CV_MatType, s) = @cxx cv::Mat(size, matType, s)

# Mat::Mat(const Mat& m)
Mat(m) = @cxx cv::Mat(m)

# Mat::Mat(int ndims, const int* sizes, int type)
# const psizes(sizes) = pointer([sizes::Int])
Mat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType) = @cxx cv::Mat(ndims, psizes, matType)

# Mat::Mat(int ndims, const int* sizes, int type, const Scalar& s)
Mat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType, s) = @cxx cv::Mat(ndims, psizes, matType, s)

# Mat::Mat(const Mat& m, const Rect& roi)
Mat(img, roi) = @cxx cv::Mat(img, roi)

# Mat::Mat(const Mat& m, const Range* ranges)
# const ranges = pointer(range)
Mat(img, ranges) = @cxx cv::Mat(img, ranges)

# Mat class methods
# addition
cxx""" cv::Mat add(cv::Mat img1, cv::Mat img2) { return(img1 + img2); } """
imadd(img1, img2) = @cxx add(img1, img2)

# substract
cxx""" cv::Mat substract(cv::Mat img1, cv::Mat img2) { return(img1 - img2); } """
imsubstract(img1, img2) = @cxx substract(img1, img2)

# Mat::row
cxx""" cv::Mat row(cv::Mat img, int y) { return(img.row(y)); } """
row(img, x::Int) = @cxx row(img, x)

# Mat::col
cxx""" cv::Mat col(cv::Mat img, int x) { return(img.col(x)); } """
col(img, y::Int) = @cxx col(img, y)

# Mat::rowRange
cxx""" cv::Mat rowRange(cv::Mat img, int startrow, int endrow) { return(img.rowRange(startrow, endrow)); } """
rowRange(img, startrow::Int, endrow::Int) = @cxx rowRange(img, startrow, endrow)

cxx""" cv::Mat rowRange(cv::Mat img, const cv::Range& r) { return(img.rowRange(r)); } """
rowRange(img, range) = @cxx rowRange(img, range)

# Mat::colRange
# const range = cvRange(start::Int, tend::Int)
cxx""" cv::Mat colRange(cv::Mat img, int startcol, int endcol) { return(img.colRange(startcol, endcol)); } """
colRange(img, startcol::Int, endcol::Int) = @cxx colRange(img, startcol, endcol)

cxx""" cv::Mat colRange(cv::Mat img, const cv::Range& r) { return(img.colRange(r)); } """
colRange(img, range) = @cxx colRange(img, range)

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
cxx""" void copy(cv::Mat out, cv::Mat img) { img.copyTo(out); } """
cxx""" void copyTomask(cv::Mat img, cv::Mat mask, cv::Mat out) { img.copyTo(out, mask); } """
copy(out, img) = @cxx copy(out, img)
copyTomask(img, mask, out) = @cxx copyTomask(img, mask, out)

cxx""" cv::Mat imageROI(cv::Mat img, cv::Rect roi) { return img(roi); } """
ROImage(img, ROI) = @cxx imageROI(img, ROI)

# Mat::convertTo
# rtype - depth, rtype < 0 output = input type
# alpha – optional scale factor.
# beta – optional delta added to the scaled values.
convert(img1, img2, rtype::Int, alpha=1, beta=0) = @cxx img1->convertTo(img2, rtype, alpha, beta)

# Mat::assignTo
cxx""" void  assignTo(cv::Mat img, cv::Mat& m, int type) { img.assignTo(m, type); } """
assignTo(img, m, rtype=-1) = @cxx assignTo(img, m, rtype)

# Mat::setTo
# value(cvScalar), mask (same size as img)
set(img, value) = @cxx img->setTo(value, mask)

# Mat::reshape
# rows = 0 (no change)
cxx""" cv::Mat reshape(cv::Mat img, int cn, int rows) { return(img.reshape(cn, rows)); } """
reshape(img, ch::Int, rows=0) = @cxx reshape(img, ch, rows)

# Mat::t() (transpose)
cxx""" cv::Mat transpose(cv::Mat img, double lambda) { return(img.t()*lambda); } """
transpose(img, lambda) = @cxx transpose(img, lambda)

# Mat::inv (invert)
cxx""" cv::Mat inv(cv::Mat img, int method) { return(img.inv(method)); } """
inv(img, method=DECOMP_LU) = @cxx inv(img, method)

# Mat::mul (mutiply)   # element-wise only
cxx""" cv::Mat mul(cv::Mat img, double scale) { return(img.mul(scale)); } """
mul(img, scale=1) = @cxx mul(img, scale)

# Mat::cross (cross-product of 2 Vec<float,3>)
cxx""" cv::Mat cross(cv::Mat img, cv::Mat m) { return(img.cross(m)); } """
cross(img, m) = @cxx cross(img, m)

# Mat::dot (Computes a dot-product of two equally sized matrices)
cxx"""  double dot(cv::Mat img, cv::Mat m) { return(img.dot(m)); } """
dot(img, m) = @cxx dot(img, m)

# Empty array (NULL pointer)
noArray() = @cxx cv::noArray()

# Mat::zeros()
# ndims –> Array dimensionality.
# rows  –> Number of rows.
# cols  –> Number of columns.
# size  –> Alternative to the matrix size specification Size(cols, rows) .
# sz    –> Array of integers specifying the array shape.
# type  –> Created matrix type.

cxx""" cv::Mat zerosM(int rows, int cols, int matType) {cv::Mat A = cv::Mat::zeros(rows, cols, matType); return (A); }"""
cxx""" cv::Mat zerosM(cv::Size size, int matType) {cv::Mat A = cv::Mat::zeros(size, matType); return (A); }"""
cxx""" cv::Mat zerosM(int ndims, int* sz, int matType) {cv::Mat A = cv::Mat::zeros(ndims, sz, matType); return (A); }"""
zerosM(rows::Int, cols::Int, matType::CV_MatType) = @cxx zerosM(rows, cols, matType)
zerosM(size, matType::CV_MatType) = @cxx zerosM(size, matType)
zerosM(ndims::Int, sz::Ptr{Int32}, matType::CV_MatType) = @cxx zerosM(ndims, sz, matType)

# Mat::ones()
ones(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::ones(rows, cols, matType)
ones(size, matType::CV_MatType) = @cxx cv::Mat::ones(size, matType)
ones(ndims::Int, sz::Ptr{Int32}, matType::CV_MatType) = @cxx cv::Mat::ones(ndims, sz, matType)

# Mat::eye()
eye(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::Mat::eye(rows, cols, matType)
eye(size, matType::CV_MatType) = @cxx cv::Mat::eye(size, matType)

# Mat::addref
addref(img) = @cxx img->addref()

# Mat::release
destroy(img) = @cxx img->release()

# Mat::resize
resizeMat(img, sz) = @cxx img->resize(csize_t(sz))         # sz – new n rows (UInt64), s = cvScalar()
resizeMat(img, sz, s) = @cxx img->resize(csize_t(sz), s)

# Mat::reserve
# Reserves space for the certain number of rows
reserve(img,sz) = @cxx img->reserve(csize_t(sz))

# Mat::push_back
# Adds elements to the bottom of the matrix:
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

cxx""" cv::Size sizeofimage(cv::Mat img) {  return(img.size()); } """
sizeofimage(img) = @cxx sizeofimage(img)                                   # returns Size(cols, rows), if matrix > 2d, size = (-1,-1)

cxx""" int channels(cv::Mat img) { return(img.channels()); } """
channels(img) = @cxx channels(img)                           # number of matrix channels

findlabel(value::Int) = CV_MAT_TYPE[value]                   # e.g., CV_8UC1 type CV_MatType in OpenCV_hpp.jl
cxx""" int cvtype(cv::Mat img) { return(img.type()); } """
cvtypeval(img) = @cxx cvtype(img)
cvtypelabel(img) = findlabel(int(cvtypeval(img)))


# SparseMat: multi-dimensional sparse numerical arrays
# Parameters:
# m – Source matrix for copy constructor
#    If m is dense matrix (ocv:class:Mat) -> sparse representation
# dims – Array dimensionality
# _sizes – Sparce matrix size on all dementions, i.e.,  const int*
# _type – Sparse matrix data type, i.e., const SparseMat&

SparseMat() = @cxx cv::SparseMat()
SparseMat(dims::Int, _sizes::Ptr{Int32}, _type::Int) = @cxx cv::SparseMat::SparseMat(dims, _sizes, _type)
SparseMat(sparseM) = @cxx cv::SparseMat::SparseMat(sparseM)
SparseMat(M) = @cxx cv::SparseMat::SparseMat(M)

# clone
cxx""" cv::SparseMat cloneSparse(cv::SparseMat img) { return(img.clone()); } """
cloneSparse(SparseM) = @cxx cloneSparse(SparseM)

# SparseMat::copyTo
cxx""" void copyToSparse(cv::SparseMat img, cv::SparseMat copy) { img.copyTo(copy); } """
cxx""" void copyToSparse(cv::SparseMat img, cv::Mat copy) { img.copyTo(copy); } """
copySparse(SparseM, copy) = @cxx copySparse(SparseM, copy)
copySparse(M, copy) = @cxx copySparse(M, copy)

# SparseMat::convertTo
convertSparse(sparseM1, sparseM2, rtype::Int, alpha=1, beta=0) = @cxx img1->convertToSparse(sparseM1, sparseM2, rtype, alpha, beta)
SparseToMat(sparseM, M, rtype::Int, alpha=1, beta=0) = @cxx img1->SparseToMat(sparseM, M, rtype, alpha, beta)

# SparseMat::create
createSparse(sparseM) = @cxx sparseM->create(dims, _sizes, _type)

# SparseMat::clear
clearSparse(sparseM) = @cxx sparseM->clear()
addrefSparse(sparseM) = @cxx sparseM->addref()
destroySparse(sparseM) = @cxx sparseM->release()              # ~SparseMat()
elemSizeSparse(sparseM) = @cxx sparseM->elemSize()
elemSize1Sparse(sparseM) = @cxx sparseM->elemSize1()

cxx""" int typeSparse(cv::SparseMat img) { return(img.type()); } """
typeSparse(sparseM) = @cxx typeSparse(sparseM)
typeSparselabel(sparseM) = findlabel(int(typeSparse(sparseM)))
depthSparse(sparseM) = @cxx sparseM->depth()

cxx""" int channelsSparse(cv::SparseMat img) { return(img.channels()); } """
channelsSparse(sparseM) = @cxx channelsSparse(sparseM)         # number of matrix channels


#-------------------------------------------------------------------------------------------------------------------#
# Command Line Parser (incomplete)
# The CommandLineParser class is designed for command line arguments parsing

# CommandLineParser::CommandLineParser(int argc, const char* const argv[], const String& keys)
# Parameters:
# argc –
# argv –
# keys –
CommandLineParser(argc::Int, argv::Ptr{Ptr{UInt8}}, keys::String) =
       @cxxnew cv::CommandLineParser(argc, argv, pointer(keys))

# bool CommandLineParser::has(const String& name)
has(parser, name::String) = @cxx parser->has(stdstring(name))


#-------------------------------------------------------------------------------------------------------------------#
# 2. Operations on Arrays

abs(img) = @cxx cv::abs(img)
absdiff(src1, src2, dst) = @cxx absdiff(src1, src2, dst)

add(src1, src2, dst, mask = noArray(), dtype=-1) = @cxx cv::add(src1, src2, dst, mask, dtype)
# mask – optional operation mask - 8-bit single channel array, default mask = noArray()
# dtype – optional depth of the output array

addWeighted(src1, alpha, src2, beta::Float64, gamma::Float64, dst, dtype=-1) =
    @cxx cv::addWeighted(src1, alpha, src2, beta, gamma, dst, dtype)
# beta – weight of the second array elements
# gamma – scalar added to each sum

bitwise_and(src1, src2, dst,  mask = noArray()) = @cxx cv::bitwise_and(src1, src2, dst, mask)
bitwise_not(src, dst, mask = noArray()) = @cxx cv::bitwise_not(src, dst, mask)
bitwise_or(src1, src2, dst, mask = noArray()) = @cxx cv::bitwise_or(src1, src2, dst,mask)
bitwise_xor(src1, src2, dst, mask = noArray()) = @cxx cv::bitwise_xor(src1, src2, dst,mask)
# mask – optional operation mask - 8-bit single channel array, default mask = noArray()

calcCovarMatrix(sample, nsamples::Int, covar, mean, flags::Int, ctype=CV_64F) =
    @cxx cv::calcCovarMatrix(samples, nsamples, covar, mean, flags, ctype)
# samples – samples stored either as separate matrices or as rows/columns of a single matrix - pointer(mat)
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

checkRange(a, point, quiet=true, minVal=-DBL_MAX, maxVal=DBL_MAX) =
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

eigen(src, eigenvalues, eigenvectors=noArray()) = @cxx cv::eigen(src, eigenvalues,eigenvectors)
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

mean(src,mask=noArray()) = @cxx cv::mean(src, mask)
# mask  – optional operation mask, default mask=noArray()

meanStdDev(src, mean, stddev, mask=noArray()) = @cxx cv::meanStdDev(src, mean, stddev, mask)
# mask  – optional operation mask, default mask=noArray()

merge(mv, count::UInt64, dst) = @cxx cv::merge(mv, count, dst)
# mv  => const pointer(cv::Mat) or const Mat* mv
# dst – output array of the same size

min(a, b) = @cxx cv::min(a,b)
min(a, scalar) = @cxx cv::min(a,scalar)

# global minimum and maximum in an array
minMaxIdx(src, minVal::Ptr{Float64}, maxVal=pointer([float(0)]), iminIdx=pointer([0]), maxIdx=pointer([0]), mask=noArray()) =
    @cxx cv::minMaxIdx(src, minVal, maxVal, iminIdx, maxIdx, mask)
# maxVal::Ptr{Float64}
# minIdx::Ptr{Int}
# maxIdx::Ptr{Int}
# src – input single-channel array.
# minVal – pointer to the returned minimum value; NULL is used if not required.
# maxVal – pointer to the returned maximum value; NULL is used if not required.
# minIdx – minIdx is not NULL, it must have at least 2 elements (as well as maxIdx),

# global minimum and maximum in an array
minMaxLoc(src, minVal::Ptr{Float64}, maxVal=pointer([float(0)]), minLoc=pointer([0]), maxLoc=pointer([0]), mask=noArray()) =
    @cxx cv::minMaxLoc(src, minVal, maxVal, minLoc, maxLoc, mask)
# minLoc = convert(C_NULL, pointer(cvPoint))
# maxLoc = convert(C_NULL, pointer(cvPoint))
# mask  – optional operation mask, default mask=noArray()
# minVal – pointer to the returned minimum value; NULL is used if not required.
# maxVal – pointer to the returned maximum value; NULL is used if not required.
# minLoc – pointer to the returned minimum location (in 2D case); NULL is used if not required.
# maxLoc – pointer to the returned maximum location (in 2D case); NULL is used if not required.
# mask – optional mask used to select a sub-array.

mixChannels(src, dst, fromTo::Ptr{Int}, npairs::UInt64) = @cxx cv::mixChannels(src, dst, fromTo, npairs)
# const fromTo::Ptr{Int}
# fromTo = const std::vector<int>&

# per-element multiplication of two Fourier spectrums
mulSpectrums(a, b, c, flags=DFT_ROWS, conjB=false) = @cxx cv::mulSpectrums(a, b, c, flags, conjB)
# flags – currently, the only supported flag is DFT_ROWS

# per-element scaled product of two arrays
multiply(src1, src2, dst, scale=1.0, dtype=-1) = @cxx cv::multiply(src1, src2, dst, scale, dtype)

# product of a matrix and its transposition
mulTransposed(src, dst, aTa::Bool, delta=noArray(), scale=1.0, dtype=-1) = @cxx cv::mulTransposed(src, dst, aTa,delta, scale, dtype)
# InputArray delta=noArray()

# absolute array norm, an absolute difference norm, or a relative difference norm
norm(src1, normType=NORM_L2,mask=noArray()) = @cxx cv::norm(src1, normType=NORM_L2, mask)
norm(src1, src2, normType=NORM_L2,mask=noArray()) = @cxx cv::norm(src1, src2, normType=NORM_L2, mask)
# InputArray mask=noArray()

# norm or value range of an array
normalize(src, dst, alpha=1.0, beta=0, norm_type=NORM_L2, dtype=-1, mask=noArray()) = @cxx cv::normalize(src, dst, alpha, beta, norm_type, dtype, mask)
# InputArray mask=noArray()

# PCA: Principal Component Analysis class
# Constructors
pca() = cv::PCA::PCA()
pca(data, mean=noArray(), flags=CV_PCA_DATA_AS_ROW, maxComponents=0) =  @cxx cv::PCA::PCA(data, mean, flags, maxComponents)
pca(data, mean=noArray(), flags=CV_PCA_DATA_AS_ROW, retainedVariance=2.0) = @cxx cv::PCA::PCA(data, mean, flags, retainedVariance)
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
rng(state::UInt64) = @cxx cv::RNG::RNG(state)
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

subtract(src1, src2, dst, mask=noArray(), dtype=-1) = @cxx cv::subtract(src1, src2, dst, mask, dtype)
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

#-------------------------------------------------------------------------------------------------------------------#
# 3. Clustering

# Finds centers of clusters and groups input samples around the clusters
kmeans(data, K::Int, bestLabels, criteria, attempts::Int, flags::Int,centers=noArray()) =
    @cxx cv::kmeans(data, K, bestLabels, criteria, attempts, flags, centers)
# OutputArray centers=noArray()
# TermCriteria
# partition

#-------------------------------------------------------------------------------------------------------------------#
# 4. Utility and System Functions and Macros

# angle of a 2D vector in degrees
fastAtan2(y::Float64, x::Float64) = @cxx cv::fastAtan2(y, x)

# cube root of an argument
cubeRoot(val::Float64) =  @cxx cv::cubeRoot(val)

# fastMalloc
fastMalloc(bufSize::UInt64) = @cxx cv::fastMalloc(bufSize)  # returns void*

# fastFree
fastFree(ptr::Ptr{Void}) = @cxx cv::fastFree(ptr)
# ptr – Pointer to the allocated buffer

# format strings
cvString(text::String) = @cxx cv::format(pointer(text)) # returns cv::String

# convert cv::String to Julia String
julString(cvstr) = bytestring(@cxx cvstr->c_str())

# getBuildInformation
getBuildInformation() = @cxx cv::getBuildInformation()

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
