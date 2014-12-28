################################################################################################################################
# video. Video Analysis
################################################################################################################################

# Motion Analysis and Object Tracking

#------------------------------------------------------------------------------------------------------------------------------#
# calcOpticalFlowPyrLK
# Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
# Lucas-Kanade optical flow in pyramids (Bouguet 2000)

# void calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts,
# OutputArray status, OutputArray err, Size winSize=Size(21,21), int maxLevel=3,
# TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4 )

# parameters:
# prevImg   – first 8-bit input image (InputArray) or pyramid constructed by buildOpticalFlowPyramid()
# nextImg   – second input image (InputArray) or pyramid of the same size and the same type as prevImg
# prevPts   – vector of 2D points (InputArray single-precision fp X,Y) to find flow
# nextPts   – output vector of 2D points (InputOutputArray single-precision fp X,Y) with new positions of input features in the second image
#             if OPTFLOW_USE_INITIAL_FLOW => Set size of vector = size of input vector
# status    – output status vector (uchar)
#             status = 1 if flow for the corresponding features has been found, else status = 0
# err       – output vector of errors; each element of the vector is set to an error for the corresponding feature
#             type of the error measure can be set in flags parameter
#             if no flow, error is not defined (status = 0)
# winSize   – cv::Size (size) of the search window at each pyramid level, default = cv::Size(21,21)
# maxLevel  – int 0-based maximal pyramid level number, default = 3
#             if maxLevel = 0, single level, i.e., pyramids are not used
#             if maxLevel = 1, two levels; etc
#             pyramids < maxLevel
# criteria  – TermCriteria parameter of the iterative search algorithm
#             stop when max iterations criteria.maxCount || search window moves < criteria.epsilon
#             default = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01)

# flags, int default =0
# OPTFLOW_USE_INITIAL_FLOW       =>  uses initial estimations, stored in nextPts
#                                    default: prevPts => nextPts
# OPTFLOW_LK_GET_MIN_EIGENVALS   =>  use minimum eigen values as an error measure (minEigThreshold);
#                                    default: L1 distance between patches around the original and a moved point,
#                                    divided by number of pixels in a window, is used as a error measure
# minEigThreshold (double)       =>  min eigen value of a 2x2 normal matrix of optical flow equations (spatial gradient matrix in [Bouguet00]/window pixels)
#                                    if eigen value < minEigThreshold, feature is filtered out and flow ignored
#                                    default = 1e-4
#-------------------------------------------------------------------------------------------------------------------#

# e.g., parameters
# prevImg::Mat(rows, columns, CV_8UC3)  || buildOpticalFlowPyramid()
# nextImg::Mat(rows, columns, CV_8UC3)
# prevPts = tostdvec([pts1::cvPoint2f, pts2::cvPoint2f, pts3::cvPoint2f, pts4::cvPoint2f, pts5::cvPoint2f . . .])
# nextPts = tostdvec([pts1::cvPoint2f, pts2::cvPoint2f, pts3::cvPoint2f, pts4::cvPoint2f, pts5::cvPoint2f . . .])

calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize=Size(21,21), maxLevel=3,
    criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), flags=0, minEigThreshold=1e-4) =
       @cxx cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold)

# buildOpticalFlowPyramid
# Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK().
# int buildOpticalFlowPyramid(InputArray img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives=true,
# int pyrBorder=BORDER_REFLECT_101, int derivBorder=BORDER_CONSTANT, bool tryReuseInputImage=true)
# Parameters:
# img                – 8-bit input image
# pyramid            – output pyramid
# winSize            – window size of optical flow algorithm, >= winSize in calcOpticalFlowPyrLK()
# maxLevel           – 0-based maximal pyramid level number
# withDerivatives    – set to precompute gradients for the every pyramid level
# pyrBorder          – the border mode for pyramid layers
# derivBorder        – the border mode for gradients
# tryReuseInputImage – put ROI of input image into the pyramid if possible
#                      if tryReuseInputImage = false, then forces data copying
# Output:
# number of levels in constructed pyramid, maybe < maxLevel

buildOpticalFlowPyramid(img, pyramid, winSize, maxLevel, withDerivatives=true, pyrBorder=BORDER_REFLECT_101,
    derivBorder=BORDER_CONSTANT, tryReuseInputImage=true) = @cxx cv::buildOpticalFlowPyramid(img, pyramid, winSize,
        maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage)

# calcOpticalFlowFarneback
# Computes a dense optical flow using the Gunnar Farneback’s algorithm
# void calcOpticalFlowFarneback(InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale,
# int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)

# Parameters:
# prev       – first 8-bit single-channel input image
# next       – second input image: size and type = prev
# flow       – CV_32FC2 computed flow image, size = prev
# pyr_scale  – image scale (<1) to build pyramids for each image
#              pyr_scale=0.5 => classical pyramid, next layer is 2X smaller than the previous one
# levels     – number of pyramid layers including the initial image
#              levels=1 => only the original images are used
# winsize    – averaging window size
#              larger => increase robustness, less noise and better fast motion detection), but yield more blurred motion field
# iterations – iterations/pyramid level
# poly_n     – typically poly_n =5 or 7
#              size of the pixel neighborhood used to find polynomial expansion in each pixel
#              larger => smoother surfaces, robust but more blurred motion field,
# poly_sigma – standard deviation of the Gaussian to smooth derivatives for the polynomial expansion
#              poly_n=5  => poly_sigma=1.1
#              poly_n=7  => poly_sigma=1.5
# flags      – same as calcOpticalFlowPyrLK

calcOpticalFlowFarneback(prev, next, flow, pyr_scale=0.5, levels=1, winsize=10, iterations=5, poly_n=5,
    poly_sigma=1.1, flags=0) = @cxx cv::calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize,
      iterations, poly_n, poly_sigma, flags)

# estimateRigidTransform
# Computes an optimal affine transformation between two 2D point sets
# Mat estimateRigidTransform(InputArray src, InputArray dst, bool fullAffine)

# Parameters:
# src – 2D point set stored in std::vector or Mat, or an image stored in Mat
# dst – 2D point set of the same size and the same type as A, or another image
# fullAffine =true  => affine transformation with no additional restrictions (6 dof) - degrees of freedom
# fullAffine =false => translation, rotation, and uniform scaling (5 dof)

estimateRigidTransform(src, dst, fullAffine=false) = @cxx cv::estimateRigidTransform(src, dst, fullAffine)

# findTransformECC
# Finds the geometric transform (warp) between two images in terms of the ECC criterion [EP08]
# double findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix,
# int motionType=MOTION_AFFINE, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001))

# Parameters:
# templateImage – single-channel template image, CV_8U or CV_32F array
# inputImage    – single-channel input image which should be warped with the final warpMatrix, type = temlateImage
# warpMatrix    – floating-point 2x3 or 3x3 mapping matrix (warp)
# motionType    – type of motion parameter
# MOTION_TRANSLATION  => translational motion model: warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated
# MOTION_EUCLIDEAN    => Euclidean (rigid) transformation as motion model: 2 parameters are estimated, warpMatrix is 2x3
# MOTION_AFFINE       => affine motion model (DEFAULT):                    6 parameters are estimated, warpMatrix is 2x3
# MOTION_HOMOGRAPHY   => homography as a motion model:                     8 parameters are estimated, warpMatrix is 3x3
# criteria      – termination criteria of the ECC algorithm
#                 criteria.epsilon = threshold of the increment in the correlation coefficient between two iterations
#                 criteria.epsilon < 0 => criteria.maxcount is termination criterion

findTransformECC(templateImage, inputImage, warpMatrix, motionType=MOTION_AFFINE, criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001)) =
      @cxx cv::findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria)

#-------------------------------------------------------------------------------------------------------------------#
# CamShift
# Finds an object center, size, and orientation
# CAMSHIFT object tracking algorithm [Bradski98]
# RotatedRect CamShift(InputArray probImage, Rect& window, TermCriteria criteria)

# Parameters:
# probImage – Back projection of the object histogram. See calcBackProject() .
# window    – Initial search window.
# criteria  – Stop criteria for the underlying meanShift() .
# Returns:
# RotatedRect

# e.g., input
# probImage = calcBackProject(...) see OpenCV_imgproc.jl
# window = cvRect
# criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01)
CamShift(probImage, window, criteria) = @cxx cv::CamShift(probImage, window, criteria)

# meanShift
# Finds an object on a back projection image
# iterative object search algorithm
# int meanShift(InputArray probImage, Rect& window, TermCriteria criteria)

# Parameters:
# probImage  – Back projection of the object histogram. See calcBackProject().
# window     – Initial search window.
# criteria   – Stop criteria for the iterative search algorithm
# Returns:
# Number of iterations CAMSHIFT took to converge

meanShift(probImage, window, criteria) = @cxx cv::meanShift(probImage, window, criteria)

#-------------------------------------------------------------------------------------------------------------------#
# KalmanFilter
# class
# implements a standard Kalman filter http://en.wikipedia.org/wiki/Kalman_filter, [Welch95]
# however, can be extended by modifying transitionMatrix, controlMatrix, and measurementMatrix

# Parameters:
# dynamParams   – Dimensionality of the state
# measureParams – Dimensionality of the measurement
# controlParams – Dimensionality of the control vector
# type          – Type of the created matrices that should be CV_32F or CV_64F

# constructors, return KF
KalmanFilter() = @cxxnew cv::KalmanFilter::KalmanFilter()
KalmanFilter(dynamParams::Int,measureParams::Int, controlParams=0, cvtype=CV_32F) =
     @cxxnew cv::KalmanFilter::KalmanFilter(dynamParams,measureParams, controlParams, cvtype)

# Re-initializes Kalman filter
KalmanInit(KF, dynamParams::Int,measureParams::Int, controlParams=0, cvtype=CV_32F) =
      @cxx KF->init(dynamParams,measureParams, controlParams, cvtype)

# Compute a predicted state (optional= cv::Mat)
KalmanPredict(KF, control=Mat()) = @cxx KF->predict(control)

# Update the predicted state from the measurement (measurement = cv::Mat)
KalmanCorrect(KF,measurement) = @cxx KF->correct(measurement)
#-------------------------------------------------------------------------------------------------------------------#

# BackgroundSubtractor class
# BackgroundSubtractor::apply
# Computes a foreground mask
# void BackgroundSubtractor::apply(InputArray image, OutputArray fgmask, double learningRate=-1)
# Parameters:
# image        – Next video frame
# fgmask       – The output foreground mask as an 8-bit binary image
# learningRate – 0-1 that indicates how fast the background model is learnt
#                  < 0 automatically chosen learning rate
#                    0 not updated at all
#                    1 reinitialized from the last frame

# Create a pointer to the abstract class BackgroundSubtractorMOG
cxx"""
   cv::bgsegm::BackgroundSubtractorMOG* BackgroundSubtractorMOG()
   {
       cv::bgsegm::BackgroundSubtractorMOG* bkgsub;
       return bkgsub;
   }
"""

# Constructor
bkgsubMOG() = @cxx BackgroundSubtractorMOG()

# Apply
bkgsubApply(bkgsubMOG, Iimage, fgmask, learningRate=-1.0) = @cxx bkgsubMOG->apply(image,fgmask, learningRate)

# Computes a background image
# input: backgroundImage = Mat(r,c,cvtype)
getBackgroundImage(bkgsubMOG, backgroundImage) = @cxx bkgsubMOG->getBackgroundImage(backgroundImage)

# BackgroundSubtractorMOG class: algorithm described in [KB2001]
# Create a mixture-of-gaussian background subtractor
# Ptr<BackgroundSubtractorMOG> createBackgroundSubtractorMOG(int history=200, int nmixtures=5, double backgroundRatio=0.7, double noiseSigma=0)
# Parameters:
# history           – length of the history
# nmixtures         – Number of Gaussian mixtures
# backgroundRatio   – Background ratio
# noiseSigma        – Noise strength (SD for brightness, each color channel). 0 = automatic value

createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0) =
     @cxx cv::bgsegm::createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma)

# BackgroundSubtractorMOG2
# Creates MOG2 Background Subtractor
# Ptr<BackgroundSubtractorMOG2> createBackgroundSubtractorMOG2(int history=500, double varThreshold=16, bool detectShadows=true )
# Parameters:
# history       – length of the history
# varThreshold  – Threshold on the squared Mahalanobis distance between the pixel
# detectShadows – if =true, detect shadows and mark them. slows down!, so set to false.

# http://www.developark.com/5098_19238765/

# Create a pointer to the abstract class BackgroundSubtractorMOG2
# cxx"""
# cv::BackgroundSubtractorMOG2* BackgroundSubtractorMOG2()
# {
#      cv::BackgroundSubtractorMOG2* bkgsub;
#      return bkgsub;
# }
# """

createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=true) =
    @cxx cv::createBackgroundSubtractorMOG(history, varThreshold, detectShadows)

# Returns the number of last frames that affect the background model
getHistory(bkgsubMOG2) = @cxx bkgsubMOG2->getHistory()

# Sets the number of last frames that affect the background model
setHistory(bkgsubMOG2, history::Int) = @cxx bkgsubMOG2->setHistory(history)

# Returns number of gaussian components in the background model
getNMixtures(bkgsubMOG2) = @cxx bkgsubMOG2->getNMixtures()

# Sets number of gaussian components in the background model, needs to be reinitialized
setNMixtures(bkgsubMOG2, nmixtures::Int) = @cxx bkgsubMOG2->setNMixtures(nmixtures)

# Returns the “background ratio” parameter of the algorithm
getBackgroundRatio(bkgsubMOG2) = @cxx bkgsubMOG2->getBackgroundRatio()

# Sets the “background ratio” parameter of the algorithm
setBackgroundRatio(bkgsubMOG2, ratio::Float64) = @cxx bkgsubMOG2->setBackgroundRatio(ratio)

# Returns the variance threshold for the pixel-model match
getVarThreshold(bkgsubMOG2) = @cxx bkgsubMOG2->getVarThreshold()

# Sets the variance threshold for the pixel-model match
setVarThreshold(bkgsubMOG2, varThreshold::Float64) = @cxx bkgsubMOG2-> setVarThreshold(varThreshold)

# Returns the variance threshold for the pixel-model match used for new mixture component generation
getVarThresholdGen(bkgsubMOG2) = @cxx bkgsubMOG2-> getVarThresholdGen()

# Set the variance threshold for the pixel-model match used for new mixture component generation
setVarThresholdGen(bkgsubMOG2, VarThreshold::Float64) = @cxx bkgsubMOG2-> setVarThresholdGen(VarThreshold)

# Returns the initial variance of each gaussian component
getVarInit(bkgsubMOG2) = @cxx bkgsubMOG2->getVarInit()

# Sets the initial variance of each gaussian component
setVarInit(bkgsubMOG2, varInit::Float64) = @cxx bkgsubMOG2->setVarInit(varInit)

# Returns the complexity reduction threshold
getComplexityReductionThreshold(bkgsubMOG2) = @cxx bkgsubMOG2->getComplexityReductionThreshold()

# Sets the complexity reduction threshold
setComplexityReductionThreshold(bkgsubMOG2, ct::Float64) = @cxx bkgsubMOG2->setComplexityReductionThreshold(ct)

# Returns the shadow detection flag
getDetectShadows(bkgsubMOG2) = @cxx bkgsubMOG2->getDetectShadows()

# Enables or disables shadow detection
setDetectShadows(bkgsubMOG2, detectShadows::Bool) = @cxx bkgsubMOG2->setDetectShadows(detectShadows)

# Returns the shadow value, Default value is 127
getShadowValue(bkgsubMOG2) = @cxx bkgsubMOG2->getShadowValue()

# Sets the shadow value
setShadowValue(bkgsubMOG2, value::Int) = @cxx bkgsubMOG2->setShadowValue(bkgsubMOG2, value)

# Returns the shadow threshold
getShadowThreshold(bkgsubMOG2) = @cxx bkgsubMOG2->getShadowThreshold()

# Sets the shadow threshold
setShadowThreshold(bkgsubMOG2, threshold::Float64) = @cxx bkgsubMOG2->setShadowThreshold(threshold)

# BackgroundSubtractorKNN
# K-nearest neigbours - based Background/Foreground Segmentation Algorithm
# The class implements the K-nearest neigbours background subtraction described in [Zivkovic2006]
# Very efficient if number of foreground pixels is low.

# C++: Ptr<BackgroundSubtractorKNN> createBackgroundSubtractorKNN(int history=500,
#     double dist2Threshold=400.0, bool detectShadows=true )

# Parameters:
# history        – Length of the history
# dist2Threshold – Threshold on the squared distance between the pixel
#                  This parameter does not affect the background update
# detectShadows  – If true, the algorithm will detect shadows and mark them
#                  set to false to speed up

createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0,detectShadows=true) =
    @cxx cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)

# Returns the number of last frames that affect the background model
getHistory() = @cxx bkgsubKNN->getHistory()

# Sets the number of last frames that affect the background model
setHistory(history::Int) = @cxx bkgsubKNN->setHistory(history)

# Returns the number of data samples in the background model
getNSamples() = @cxx bkgsubKNN->getNSamples()

# Sets the number of data samples in the background model. The model needs to be reinitalized to reserve memory
setNSamples(_nN::Int) = @cxx bkgsubKNN->setNSamples(_nN)

# Returns the threshold on the squared distance between the pixel and the sample
getDist2Threshold() = @cxx bkgsubKNN->getDist2Threshold()

# Sets the threshold on the squared distance
setDist2Threshold(_dist2Threshold::Float64) = @cxx bkgsubKNN->setDist2Threshold(_dist2Threshold)

# Returns the number of neighbours, the k in the kNN. K is the number of samples that need to be within
# dist2Threshold in order to decide that that pixel is matching the kNN background mode
getkNNSamples() = @cxx bkgsubKNN->getkNNSamples()

# Sets the k in the kNN. How many nearest neigbours need to match
setkNNSamples(_nkNN::Int) = @cxx bkgsubKNN->setkNNSamples(_nkNN)

# Returns the shadow detection flag
getDetectShadows() = @cxx bkgsubKNN->getDetectShadows()

# Enables or disables shadow detection
setDetectShadows(detectShadows::Bool) = @cxx bkgsubKNN->setDetectShadows(detectShadows)

# Returns the shadow value
getShadowValue() = @cxx bkgsubKNN->getShadowValue()

# Sets the shadow value
setShadowValue(value::Int) = @cxx bkgsubKNN->setShadowValue(value)

# Returns the shadow threshold
getShadowThreshold() = @cxx bkgsubKNN->getShadowThreshold()

# Sets the shadow threshold
setShadowThreshold(threshold::Float64) = @cxx bkgsubKNN->setShadowThreshold(threshold)


# BackgroundSubtractorGMG
# Background Subtractor module based on the algorithm given in [Gold2012]
# Creates a GMG Background Subtractor
# Ptr<BackgroundSubtractorGMG> createBackgroundSubtractorGMG(int initializationFrames=120,
# double decisionThreshold=0.8)

# Parameters:
# initializationFrames – number of frames used to initialize the background models
# decisionThreshold    – Threshold value, above which it is marked foreground, else background


createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8) =
   @cxx cv::bgsegm::createBackgroundSubtractorGMG(initializationFrames, decisionThreshold)

# Returns the number of frames used to initialize background model
getNumFrames(bkgsubGMG) = @cxx bkgsubGMG->getNumFrames()

# Sets the number of frames used to initialize background model
setNumFrames(nframes::Int) = @cxx bkgsubGMG->setNumFrames(nframes)

# Returns the learning rate of the algorithm
# It lies between 0.0 and 1.0. It determines how quickly features are “forgotten” from histograms
getDefaultLearningRate() = @cxx bkgsubGMG->getDefaultLearningRate()

# Sets the learning rate of the algorithm
setDefaultLearningRate(lr::Float64) = @cxx bkgsubGMG->setDefaultLearningRate(lr)

# Returns the value of decision threshold
# Decision value is the value above which pixel is determined to be FG
getDecisionThreshold() = @cxx bkgsubGMG->getDecisionThreshold()

# Sets the value of decision threshold
setDecisionThreshold(thresh::Float64) = @cxx bkgsubGMG->setDecisionThreshold(thresh)

# Returns total number of distinct colors to maintain in histogram
getMaxFeatures() = @cxx bkgsubGMG->getMaxFeatures()

# Sets total number of distinct colors to maintain in histogram
setMaxFeatures(maxFeatures::Int) = @cxx bkgsubGMG->setMaxFeatures(maxFeatures)

# Returns the parameter used for quantization of color-space.
# It is the number of discrete levels in each channel to be used in histograms.
getQuantizationLevels() = @cxx bkgsubGMG->getQuantizationLevels()

# Sets the parameter used for quantization of color-space
setQuantizationLevels(nlevels::Int) = @cxx bkgsubGMG->setQuantizationLevels(nlevels)

# Returns the kernel radius used for morphological operations
getSmoothingRadius() = @cxx bkgsubGMG->getSmoothingRadius()

# Sets the kernel radius used for morphological operations
setSmoothingRadius(radius::Int) = @cxx bkgsubGMG->setSmoothingRadius(radius)

# Returns the status of background model update
getUpdateBackgroundModel() = @cxx bkgsubGMG->getUpdateBackgroundModel()

# Sets the status of background model update
setUpdateBackgroundModel(update::Bool) = @cxx bkgsubGMG->setUpdateBackgroundModel(update)

# Returns the minimum value taken on by pixels in image sequence. Usually 0
getMinVal() = @cxx bkgsubGMG->getMinVal()

# Sets the minimum value taken on by pixels in image sequence
setMinVal(val::Float64) = @cxx bkgsubGMG->setMinVal(val)

# Returns the maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
getMaxVal() = @cxx bkgsubGMG->getMaxVal()

# Sets the maximum value taken on by pixels in image sequence
setMaxVal(val::Float64) = @cxx bkgsubGMG->setMaxVal(val)

# Returns the prior probability that each individual pixel is a background pixel
getBackgroundPrior() = @cxx bkgsubGMG->getBackgroundPrior()

# Sets the prior probability that each individual pixel is a background pixel
setBackgroundPrior(bgprior::Float64) = @cxx bkgsubGMG->setBackgroundPrior(bgprior)


#-------------------------------------------------------------------------------------------------------------------#
# createOptFlow_DualTVL1
# “Dual TV L1” optical flow algorithm described in [Zach2007] and [Javier2012]
#  See http://stackoverflow.com/questions/19309567/speeding-up-optical-flow-createoptflow-dualtvl1
# Ptr<DenseOpticalFlow> createOptFlow_DualTVL1()
createOptFlow_DualTVL1() = @cxxnew cv::createOptFlow_DualTVL1()

# DenseOpticalFlow::calc
# void DenseOpticalFlow::calc(InputArray I0, InputArray I1, InputOutputArray flow)
# Parameters:
# I0     – first 8-bit single-channel input image
# I1     – second input image of the same size and the same type as I0
# flow   – computed flow image that has the same size as I0 and type CV_32FC2

calcDenseOpticalFLow(tvl1, I0, I1, flow) = @cxx tvl1->calc(I0, I1, flow)

# DenseOpticalFlow::collectGarbage
# void DenseOpticalFlow::collectGarbage()
collectGarbage(tvl1) = tvl1->collectGarbage()

# how to use:
# tvl1 = createOptFlow_DualTVL1()
# tvl1->calc(Previous_Gray_Frame, Current_Gray_Frame, Optical_Flow)

# createOptFlow_DualTVL1 parameters (class members)
# double tau
    # Time step of the numerical scheme
    # tvl1->getDouble("tau");
    # tvl1->set("tau",0.125);

# double lambda
    # Weight parameter for the data term, attachment parameter.
    # This is the most relevant parameter, which determines the smoothness of the output.
    # The smaller this parameter is, the smoother the solutions we obtain.
    # It depends on the range of motions of the images, so its value should be adapted to each image sequence.
    # tvl1->getDouble("lambda");
    # tvl1->set("lambda",0.125);

# double theta
    # Weight parameter for (u - v)^2, tightness parameter.
    # It serves as a link between the attachment and the regularization terms.
    # In theory, it should have a small value in order to maintain both parts in correspondence.
    # The method is stable for a large range of values of this parameter.
    # tvl1->getDouble("theta");
    # tvl1->set("theta",0.125);

# int nscales
    # Number of scales used to create the pyramid of images.
    # tvl1->getInt("nscales");
    # tvl1->set("nscales",2);

# int warps
    # Number of warpings per scale. Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
    # This is a parameter that assures the stability of the method.
    # It also affects the running time, so it is a compromise between speed and accuracy.
    # tvl1->getInt("warps");
    # tvl1->set("warps",2);

# double epsilon
    # Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
    # A small value will yield more accurate solutions at the expense of a slower convergence.
    # tvl1->getDouble("epsilon");
    # tvl1->set("epsilon",0.5);

# int iterations
    # Stopping criterion iterations number used in the numerical scheme.
    # tvl1->getInt("iterations");
    # tvl1->set("iterations",20);
