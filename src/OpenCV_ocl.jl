################################################################################################
# OpenCL support for GPU-accelerated OpenCV
################################################################################################

# Set up OpenCL
setUseOpenCL(select=true) = @cxx cv::ocl::setUseOpenCL(select)

# ocl::getOpenCLPlatforms
# Returns the list of OpenCL platforms
# int ocl::getOpenCLPlatforms(PlatformsInfo& platforms)
# cxx"""
# int getOCLPlatforms(void)
# {
#     cv::ocl::PlatformInfo& platforms;
#     return(getOpenCLPlatforms(platforms));
# }
# """
getOpenCLPlatforms() = @cxx getOpenCLPlatforms()

# ocl::getOpenCLDevices
# Returns the list of devices
# int ocl::getOpenCLDevices(DevicesInfo& devices, int deviceType=CVCL_DEVICE_TYPE_GPU,
#    const PlatformInfo* platform=NULL )
# Parameters:
# devices – Output variable
# deviceType – Bitmask of TYPE_GPU, TYPE_CPU or TYPE_DEFAULT
# deviceType
# TYPE_DEFAULT
# TYPE_CPU
# TYPE_GPU
# TYPE_ACCELERATOR
# TYPE_DGPU
# TYPE_IGPU
# platform – Specifies preferrable platform
# cxx"""
# int getOCLDevices(int deviceType)
# {
#     cv::cuda::DeviceInfo& devices;
#     const cv::ocl::PlatformInfo* platform=NULL;

#     return(getOpenCLDevices(devices, deviceType, platform));
# }
# """
getOpenCLDevices(C_NULL, deviceType=TYPE_GPU) =  @cxx getOpenCLDevices(C_NULL, deviceType)

# ocl::setDevice
# Initialize OpenCL computation context
# void ocl::setDevice(const DeviceInfo* info)
# cxx"""
# void setDevice(void)
# {
#     const cv::cuda::DeviceInfo* info;
#     cv::ocl::setDevice(info);
# }
# """
setDevice() = @cxx setDevice()

# ocl::initializeContext
# Alternative way to initialize OpenCL computation context
# This function can be used for context initialization with D3D/OpenGL interoperability.
# void ocl::initializeContext(void* pClPlatform, void* pClContext, void* pClDevice)
# Parameters:
# pClPlatform – selected platform_id (via pointer, parameter type is cl_platform_id*)
# pClContext – selected cl_context (via pointer, parameter type is cl_context*)
# pClDevice – selected cl_device_id (via pointer, parameter type is cl_device_id*)

initializeContext() = @cxx initializeContext()

# ocl::setBinaryPath
# void ocl::setBinaryPath(const char* path)
# Parameters:
# path – the path of OpenCL kernel binaries
setBinaryPath(path::String) = @cxx cv::ocl::setBinaryPath(pointer(path))


################################################################################################
# Data structures Universal Mat => "UMat"

# UMat::UMat()
UMat() = @cxx cv::UMat()

# UMat::UMat(int rows, int cols, int type)
UMat(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::UMat(rows, cols, matType)

# UMat::UMat(Size size, int type)
#cxx""" cv::UMat UMat(int rows, int cols, int matType){ cv::UMat img(cv::Size(rows,cols), matType); return(img); }"""
UMat(size, matType::CV_MatType) = @cxx cv::UMat(size, matType)

# UMat::UMat(int rows, int cols, int type, const Scalar& s)
#cxx""" cv::UMat UMat(int rows, int cols, int matType){ cv::UMat img(cv::Size(rows,cols), matType); return(img); }"""
UMat(rows::Int, cols::Int, matType::CV_MatType, s) = @cxx cv::UMat(rows, cols, matType, s)

# UMat::UMat(Size size, int type, const Scalar& s)
UMat(size, matType::CV_MatType, s) = @cxx cv::UMat(size, matType, s)

# UMat::UMat(const UMat& m)
UMat(m) = @cxx cv::UMat(m)

# UMat::UMat(int ndims, const int* sizes, int type)
# const psizes(sizes) = pointer([sizes::Int])
UMat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType) = @cxx cv::UMat(ndims, psizes, matType)

# UMat::UMat(int ndims, const int* sizes, int type, const Scalar& s)
UMat(ndims::Int, psizes::Ptr{Int}, matType::CV_MatType, s) = @cxx cv::UMat(ndims, psizes, matType, s)

# UMat::UMat(const UMat& m, const Rect& roi)
UMat(img, roi) = @cxx cv::UMat(img, roi)

# UMat::UMat(const UMat& m, const Range* ranges)
# const ranges = pointer(range)
UMat(img, ranges) = @cxx cv::UMat(img, ranges)

# UMat class methods
# addition
# cxx""" cv::UMat add(cv::UMat img1, cv::UMat img2) { return(img1 + img2); } """
# imadd(img1, img2) = @cxx add(img1, img2)

# substract
# cxx""" cv::UMat substract(cv::UMat img1, cv::UMat img2) { return(img1 - img2); } """
# imsubstract(img1, img2) = @cxx substract(img1, img2)

# UMat::row
cxx""" cv::UMat row(cv::UMat img, int y) { return(img.row(y)); } """
row(img, x::Int) = @cxx row(img, x)

# UMat::col
cxx""" cv::UMat col(cv::UMat img, int x) { return(img.col(x)); } """
col(img, y::Int) = @cxx col(img, y)

# UMat::rowRange
cxx""" cv::UMat rowRange(cv::UMat img, int startrow, int endrow) { return(img.rowRange(startrow, endrow)); } """
rowRange(img, startrow::Int, endrow::Int) = @cxx rowRange(img, startrow, endrow)

cxx""" cv::UMat rowRange(cv::UMat img, const cv::Range& r) { return(img.rowRange(r)); } """
rowRange(img, range) = @cxx rowRange(img, range)

# UMat::colRange
# const range = cvRange(start::Int, tend::Int)
cxx""" cv::UMat colRange(cv::UMat img, int startcol, int endcol) { return(img.colRange(startcol, endcol)); } """
colRange(img, startcol::Int, endcol::Int) = @cxx colRange(img, startcol, endcol)

cxx""" cv::UMat colRange(cv::UMat img, const cv::Range& r) { return(img.colRange(r)); } """
colRange(img, range) = @cxx colRange(img, range)

# UMat::diag
# d=0 is the main diagonal
# d>0 is a diagonal from the lower half
# d=1 below the main one
# d<0 is a diagonal from the upper half

cxx""" cv::UMat diag(cv::UMat img, int d=0) { return(img.diag(d)); } """
diag(img, d::Int) = @cxx diag(img, d)

cxx""" cv::UMat diag(cv::UMat img, const cv::UMat& m) { return(img.diag(m)); } """
diag(img, m) = @cxx diag(img, m)

# UMat::clone()
cxx""" cv::UMat clone(cv::UMat img) { return(img.clone()); } """
clone(img) = @cxx clone(img)

# UMat::copyTo
cxx""" void copy(cv::UMat out, cv::UMat img) { img.copyTo(out); } """
cxx""" void copyTomask(cv::UMat img, cv::UMat mask, cv::UMat out) { img.copyTo(out, mask); } """
copy(out, img) = @cxx copy(out, img)
copyTomask(img, mask, out) = @cxx copyTomask(img, mask, out)

cxx""" cv::UMat imageROI(cv::UMat img, cv::Rect roi) { return img(roi); } """
ROImage(img, ROI) = @cxx imageROI(img, ROI)

# UMat::convertTo

# UMat::assignTo
cxx""" void  assignTo(cv::UMat img, cv::UMat& m, int type) { img.assignTo(m, type); } """
assignTo(img, m, rtype=-1) = @cxx assignTo(img, m, rtype)

# UMat::reshape
# rows = 0 (no change)
cxx""" cv::UMat reshape(cv::UMat img, int cn, int rows) { return(img.reshape(cn, rows)); } """
reshape(img, ch::Int, rows=0) = @cxx reshape(img, ch, rows)

# UMat::t() (transpose)
# cxx""" cv::UMat transpose(cv::UMat img, double lambda) { return(img.t()*lambda); } """
# transpose(img, lambda) = @cxx transpose(img, lambda)

# UMat::inv (invert)
cxx""" cv::UMat inv(cv::UMat img, int method) { return(img.inv(method)); } """
inv(img, method=DECOMP_LU) = @cxx inv(img, method)

# UMat::mul (mutiply)   # element-wise only
cxx""" cv::UMat mul(cv::UMat img, double scale) { return(img.mul(scale)); } """
mul(img, scale=1) = @cxx mul(img, scale)

# UMat::cross (cross-product of 2 Vec<float,3>)
# cxx""" cv::UMat cross(cv::UMat img, cv::UMat m) { return(img.cross(m)); } """
# cross(img, m) = @cxx cross(img, m)

# UMat::dot (Computes a dot-product of two equally sized matrices)
cxx"""  double dot(cv::UMat img, cv::UMat m) { return(img.dot(m)); } """
dot(img, m) = @cxx dot(img, m)

# UMat::zeros()
# ndims –> Array dimensionality.
# rows  –> Number of rows.
# cols  –> Number of columns.
# size  –> Alternative to the matrix size specification Size(cols, rows) .
# sz    –> Array of integers specifying the array shape.
# type  –> Created matrix type.

cxx""" cv::UMat zerosUM(int rows, int cols, int matType) {cv::UMat A = cv::UMat::zeros(rows, cols, matType); return (A); }"""
cxx""" cv::UMat zerosUM(cv::Size size, int matType) {cv::UMat A = cv::UMat::zeros(size, matType); return (A); }"""
cxx""" cv::UMat zerosUM(int ndims, int* sz, int matType) {cv::UMat A = cv::UMat::zeros(ndims, sz, matType); return (A); }"""
zerosUM(rows::Int, cols::Int, matType::CV_MatType) = @cxx zerosUM(rows, cols, matType)
zerosUM(size, matType::CV_MatType) = @cxx zerosUM(size, matType)
zerosUM(ndims::Int, sz::Ptr{Int32}, matType::CV_MatType) = @cxx zerosUM(ndims, sz, matType)

# UMat::ones()
onesUM(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::UMat::ones(rows, cols, matType)
onesUM(size, matType::CV_MatType) = @cxx cv::UMat::ones(size, matType)
onesUM(ndims::Int, sz::Ptr{Int32}, matType::CV_MatType) = @cxx cv::UMat::ones(ndims, sz, matType)

# UMat::eye()
eyeUM(rows::Int, cols::Int, matType::CV_MatType) = @cxx cv::UMat::eye(rows, cols, matType)
eyeUM(size, matType::CV_MatType) = @cxx cv::UMat::eye(size, matType)

cxx""" cv::Size size(cv::UMat img) {  return(img.size()); } """
sizeU(img) = @cxx size(img)                                   # returns Size(cols, rows), if matrix > 2d, size = (-1,-1)

cxx""" int channels(cv::UMat img) { return(img.channels()); } """
channelsU(img) = @cxx channels(img)                           # number of matrix channels

cxx""" int cvtype(cv::UMat img) { return(img.type()); } """
cvtypevalU(img) = @cxx cvtype(img)

