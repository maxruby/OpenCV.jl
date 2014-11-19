#####################################################################################################
# imgproc. Image Processing
#####################################################################################################

# 1. Image Filtering
#     bilateralFilter
#     blur
#     boxFilter
#     buildPyramid
#     dilate
#     erode
#     filter2D
#     GaussianBlur
#     getDerivKernels
#     getGaussianKernel
#     getGaborKernel
#     getStructuringElement
#     medianBlur
#     morphologyEx
#     Laplacian
#     pyrDown
#     pyrUp
#     pyrMeanShiftFiltering
#     sepFilter2D
#     Smooth
#     Sobel
#     Scharr

# 2. Geometric Image Transformations
#     convertMaps
#     getAffineTransform
#     getPerspectiveTransform
#     getRectSubPix
#     getRotationMatrix2D
#     invertAffineTransform
#     LinearPolar
#     LogPolar
#     remap
#     resize
#     warpAffine
#     warpPerspective
#     initUndistortRectifyMap
#     getDefaultNewCameraMatrix
#     undistort
#     undistortPoints

# 3. Miscellaneous Image Transformations
#     adaptiveThreshold
#     cvtColor
#     distanceTransform
#     floodFill
#     integral
#     threshold
#     watershed
#     grabCut

# 4. Drawing Functions
#     circle
#     clipLine
#     ellipse
#     ellipse2Poly
#     fillConvexPoly
#     fillPoly
#     getTextSize
#     InitFont
#     line
#     arrowedLine
#     LineIterator
#     rectangle
#     polylines
#     drawContours
#     putText

# 5. ColorMaps in OpenCV
#     applyColorMap
#
# 6. Histograms
#     calcHist
#     calcBackProject
#     compareHist
#     EMD
#     equalizeHist
#     CalcBackProjectPatch
#     CalcProbDensity
#     ClearHist
#     CopyHist
#     CreateHist
#     GetMinMaxHistValue
#     MakeHistHeaderForArray
#     NormalizeHist
#     ReleaseHist
#     SetHistBinRanges
#     ThreshHist

# 7. Structural Analysis and Shape Descriptors
#     moments
#     HuMoments
#     connectedComponents
#     findContours
#     approxPolyDP
#     ApproxChains
#     arcLength
#     boundingRect
#     contourArea
#     convexHull
#     convexityDefects
#     fitEllipse
#     fitLine
#     isContourConvex
#     minAreaRect
#     boxPoints
#     minEnclosingTriangle
#     minEnclosingCircle
#     matchShapes
#     pointPolygonTest
#     rotatedRectangleIntersection

# 8. Motion Analysis and Object Tracking
#     accumulate
#     accumulateSquare
#     accumulateProduct
#     accumulateWeighted
#     phaseCorrelate
#     createHanningWindow

# 9. Feature Detection
#     Canny
#     cornerEigenValsAndVecs
#     cornerHarris
#     cornerMinEigenVal
#     cornerSubPix
#     goodFeaturesToTrack
#     HoughCircles
#     HoughLines
#     HoughLinesP
#     LineSegmentDetector
#     createLineSegmentDetector
#     LineSegmentDetector::detect
#     LineSegmentDetector::drawSegments
#     LineSegmentDetector::compareSegments
#     preCornerDetect

# 10. Object Detection
#     matchTemplate


#-------------------------------------------------------------------------------------------------------------------#
# 1. Image Filtering

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


#-------------------------------------------------------------------------------------------------------------------#
# 2. Geometric transformations

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
warpPerspective(src, dst, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT) =
    @cxx cv::warpPerspective(src, dst, M, dsize, flag, borderMode)
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


#-------------------------------------------------------------------------------------------------------------------#
# 3. Miscellaneous Image Transformations

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


#-------------------------------------------------------------------------------------------------------------------#
# 4. Drawing functions

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
# draw contours
INT_MAX = 0 # draw only the specified contour is drawn
# INT_MAX = 1 # draws the contour(s) and all the nested contours
# INT_MAX = 2 # draws the contours, all the nested contours,and so on...

drawContours(image, contours, contourIdx::Int, color, thickness=1, lineType=LINE_8, maxLevel=INT_MAX) =
    @cxx drawContours(image, contours, contourIdx, color, thickness, lineType, maxLevel)
# optional:  hierarchy=noArray()
# optional:  Point offset=Point()
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

#-------------------------------------------------------------------------------------------------------------------#
# 5. ColorMaps in OpenCV

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

#-------------------------------------------------------------------------------------------------------------------#
# 6. Histograms

calcHist(images, nimages::Int, channels::Ptr{Int}, mask, hist, dims::Int, histSize::Ptr{Int}, ranges::Ptr{Ptr{Float64}}, uniform=true,
    accumulate=false) = @cxx cv::calcHist(images, nimages, channels, mask, hist, dims, histSize, ranges, uniform, accumulate)
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
compareHist(H1, H2, method::Int) = @cxx cv::compareHist(H1, H2, method)
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
    @cxx cv::emd(signature1, signature2, distType, lowerBound)
# optional: InputArray cost=noArray()
# optional: OutputArray flow=noArray()

# equalizeHist: Equalizes the histogram of a grayscale image
equalizeHist(src, dst) = @cxx cv::equalizeHist(src, dst)


#-------------------------------------------------------------------------------------------------------------------#
# 7. Structural Analysis and Shape Descriptors

# moments
moments(array, binaryImage=false) = @cxx cv::moments(array, binaryImage)
# array       – Raster image (single-channel, 8-bit or floating-point 2D array)
# binaryImage – If it is true, all non-zero image pixels are treated as 1’s
# moments     – Output moments

# Calculates seven Hu invariant
huMoments(m, hu) = @cxx cv::HuMoments(m, hu)

cxx"""
void humoments(const cv::Moments& moments)
  {
    std::vector<double>hu(7);
    cv::HuMoments(moments, hu);
  }
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

#-------------------------------------------------------------------------------------------------------------------#
# 8. Motion Analysis and Object Tracking

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


#-------------------------------------------------------------------------------------------------------------------#
# 9. Feature Detection

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


method=HOUGH_GRADIENT  # only option now
# HoughCircles: Finds circles in a grayscale image using the Hough transform
houghCircles(image, circles, method, dp::Float64, minDist::Float64, param1=100.0, param2=100.0,
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
     stn=float(0), min_theta =float(0), max_theta=CV_PI) = @cxx cv::HoughLines(image, lines,
          rho, theta, threshold, srn, stn, min_theta, max_theta)
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
createLineSegmentDetector(_refine=LSD_REFINE_STD, _scale=0.8, _sigma_scale=0.6, _quant=2.0, _ang_th=22.5,
    _log_eps=float(0), _density_th=0.7, _n_bins=1024) = @cxx cv::createLineSegmentDetector(_refine,
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
detectLines(linesegmentDetector, _image, _lines) = @cxx cv::linesegmentDetector->detect(_image, _lines)
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
drawSegments(linesegmentDetector, _image, lines)  = @cxx cv::linesegmentDetector->drawSegments(_image, lines)
# image – The image, where the liens will be drawn
# lines – A vector of the lines that needed to be drawn

# LineSegmentDetector::compareSegments
compareSegments(linesegmentDetector, size, lines1, lines2, image)  = @cxx cv::linesegmentDetector->compareSegments(size, lines1, lines2, image)
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


#-------------------------------------------------------------------------------------------------------------------#
# 10. Object Detection

# matchTemplate: Compares a template against overlapped image regions
matchTemplate(image, templ, result, method::Int) = @cxx cv::matchTemplate(image, templ, result, method)
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

