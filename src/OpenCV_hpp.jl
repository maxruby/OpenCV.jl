################################################################################################
#
# OpenCV_hpp.jl
#
# Julia constants from OpenCV headers:
#
# cvdef.h
# core.h
# core.hpp
# types_c.h
# imgproc.hpp
# imgcodec.hpp
# videoio.hpp
# highgui.hpp
# features2d.hpp
# objectdetect.hpp
# calib3d.hpp (not in use)
#
################################################################################################

# C++ constant from <cfloat>
const DBL_MAX = 1.79769e+308
# constant declarations (not all)

# core.hpp
# Operations on arrays
const SORT_EVERY_ROW    = 0
const SORT_EVERY_COLUMN = 1
const SORT_ASCENDING    = 0
const SORT_DESCENDING   = 16

const COVAR_SCRAMBLED = 0
const COVAR_NORMAL    = 1
const COVAR_USE_AVG   = 2
const COVAR_SCALE     = 4
const COVAR_ROWS      = 8
const COVAR_COLS      = 16

# matrix decomposition types
const DECOMP_LU       = 0
const DECOMP_SVD      = 1
const DECOMP_EIG      = 2
const DECOMP_CHOLESKY = 3
const DECOMP_QR       = 4
const DECOMP_NORMAL   = 16

# norm types
const NORM_INF       = 1
const NORM_L1        = 2
const NORM_L2        = 4
const NORM_L2SQR     = 5
const NORM_HAMMING   = 6
const NORM_HAMMING2  = 7
const NORM_TYPE_MASK = 7
const NORM_RELATIVE  = 8
const NORM_MINMAX    = 32

# comparison types
const CMP_EQ = 0
const CMP_GT = 1
const CMP_GE = 2
const CMP_LT = 3
const CMP_LE = 4
const CMP_NE = 5

const GEMM_1_T = 1
const GEMM_2_T = 2
const GEMM_3_T = 4

const DFT_INVERSE        = 1
const DFT_SCALE          = 2
const DFT_ROWS           = 4
const DFT_COMPLEX_OUTPUT = 16
const DFT_REAL_OUTPUT    = 32
const DCT_INVERSE        = DFT_INVERSE
const DCT_ROWS           = DFT_ROWS

# Various border types, image boundaries are denoted with '|'
const BORDER_CONSTANT    = 0 # iiiiii|abcdefgh|iiiiiii  with some specified 'i'
const BORDER_REPLICATE   = 1 # aaaaaa|abcdefgh|hhhhhhh
const BORDER_REFLECT     = 2 # fedcba|abcdefgh|hgfedcb
const BORDER_WRAP        = 3 # cdefgh|abcdefgh|abcdefg
const BORDER_REFLECT_101 = 4 # gfedcb|abcdefgh|gfedcba
const BORDER_TRANSPARENT = 5 # uvwxyz|absdefgh|ijklmno
const BORDER_REFLECT101  = BORDER_REFLECT_101
const BORDER_DEFAULT     = BORDER_REFLECT_101
const BORDER_ISOLATED    = 16 # do not look outside of ROI

# Mat rtypes (core.hpp)
const CV_8U  = 0       # unsigned int8    (0:255)
const CV_8S  = 1       # signed int8      (-128:127)
const CV_16U = 2       # 16bit unsigned
const CV_16S = 3       # 16bit signed
const CV_32S = 4       # 32bit signed int
const CV_32F = 5       # 32bit float
const CV_64F = 6       # 64bit double

typealias CV_MatType Int
const CV_8UC1 = 0
const CV_8UC2 = 8
const CV_8UC3 = 16
const CV_8UC4 = 24
const CV_8SC1 = 1
const CV_8SC2 = 9
const CV_8SC3 = 17
const CV_8SC4 = 25
const CV_16UC1 = 2
const CV_16UC2 = 10
const CV_16UC3 = 18
const CV_16UC4 = 26
const CV_16SC1 = 3
const CV_16SC2 = 11
const CV_16SC3 = 19
const CV_16SC4 = 27
const CV_32SC1 = 4
const CV_32SC2 = 12
const CV_32SC3 = 20
const CV_32SC4 = 28
const CV_32FC1 = 5
const CV_32FC2 = 13
const CV_32FC3 = 21
const CV_32FC4 = 29
const CV_64FC1 = 6
const CV_64FC2 = 14
const CV_64FC3 = 22
const CV_64FC4 = 30
#end of CV_MatType

# Make a Dict() for lookup of image formats (CV_Mat types)
CV_MAT_TYPE= Dict( 0 => "CV_8UC1",
                   8 => "CV_8UC2",
                  16 => "CV_8UC3",
                  24 => "CV_8UC4",
                   1 => "CV_8SC1",
                   9 => "CV_8SC2",
                  17 => "CV_8SC3",
                  25 => "CV_8SC4",
                   2 => "CV_16UC1",
                  10 => "CV_16UC2",
                  18 => "CV_16UC3",
                  26 => "CV_16UC4",
                   3 => "CV_16SC1",
                  11 => "CV_16SC2",
                  19 => "CV_16SC3",
                  27 => "CV_16SC4",
                   4 => "CV_32SC1",
                  12 => "CV_32SC2",
                  20 => "CV_32SC3",
                  28 => "CV_32SC4",
                   5 => "CV_32FC1",
                  13 => "CV_32FC2",
                  21 => "CV_32FC3",
                  29 => "CV_32FC4",
                   6 => "CV_64FC1",
                  14 => "CV_64FC2",
                  22 => "CV_64FC3",
                  30 => "CV_64FC4")

# K-means
const KMEANS_RANDOM_CENTERS     = 0
const KMEANS_PP_CENTERS         = 2
const KMEANS_USE_INITIAL_LABELS = 1

# Line thickness
const FILLED  = -1
const LINE_4  = 4
const LINE_8  = 8
const LINE_AA = 16

# Font type and style
const FONT_HERSHEY_SIMPLEX        = 0
const FONT_HERSHEY_PLAIN          = 1
const FONT_HERSHEY_DUPLEX         = 2
const FONT_HERSHEY_COMPLEX        = 3
const FONT_HERSHEY_TRIPLEX        = 4
const FONT_HERSHEY_COMPLEX_SMALL  = 5
const FONT_HERSHEY_SCRIPT_SIMPLEX = 6
const FONT_HERSHEY_SCRIPT_COMPLEX = 7
const FONT_ITALIC                 = 16


# cvdef.h
const CV_CPU_NONE   = 0
const CV_CPU_MMX    = 1
const CV_CPU_SSE    = 2
const CV_CPU_SSE2   = 3
const CV_CPU_SSE3   = 4
const CV_CPU_SSSE3  = 5
const CV_CPU_SSE4_1 = 6
const CV_CPU_SSE4_2 = 7
const CV_CPU_POPCNT = 8
const CV_CPU_AVX    = 10
const CV_CPU_NEON   = 11

# core.h
const CV_REDUCE_SUM = 0
const CV_REDUCE_AVG = 1
const CV_REDUCE_MAX = 2
const CV_REDUCE_MIN = 3

const CV_SORT_EVERY_ROW = 0
const CV_SORT_EVERY_COLUMN = 1
const CV_SORT_ASCENDING = 0
const CV_SORT_DESCENDING = 16

# types_c.h
const CV_RNG_COEFF = 4164903690       #U = unsigned
const CV_PI = 3.1415926535897932384626433832795
const CV_LOG2 = 0.69314718055994530941723212145818
const CV_AUTO_STEP = 0x7fffffff
const CV_CN_MAX = 512
const CV_CN_SHIFT = 3
const CV_HIST_ARRAY = 0
const CV_HIST_SPARSE =  1
const CV_HIST_TREE  = CV_HIST_SPARSE
const CV_HIST_UNIFORM =  1
const CV_MAGIC_MASK =  0xFFFF0000
const CV_MAT_MAGIC_VAL =  0x42420000
const CV_MATND_MAGIC_VAL = 0x42430000
const CV_MAX_DIM = 32
const CV_NODE_EMPTY = 32
const CV_NODE_FLOW = 8
const CV_NODE_INT = 1
const CV_NODE_INTEGER = CV_NODE_INT
const CV_NODE_MAP =  6
const CV_NODE_NAMED = 64
const CV_NODE_NONE =  0
const CV_NODE_REAL =  2
const CV_NODE_REF =  4
const CV_NODE_SEQ = 5
const CV_NODE_SEQ_SIMPLE =   256
const CV_NODE_STR =   3
const CV_NODE_TYPE_MASK  =  7
const CV_NODE_USER =   16
const CV_SEQ_ELTYPE_BITS =  12
const CV_SEQ_ELTYPE_CODE = CV_8UC1
const CV_SEQ_ELTYPE_CONNECTED_COMP = 0
const CV_SEQ_ELTYPE_GENERIC =  0
const CV_SEQ_ELTYPE_GRAPH_EDGE = 0
const CV_SEQ_ELTYPE_GRAPH_VERTEX = 0
const CV_SEQ_ELTYPE_INDEX = CV_32SC1
const CV_SEQ_ELTYPE_POINT = CV_32SC2
const CV_SEQ_ELTYPE_POINT3D = CV_32FC3
const CV_SEQ_ELTYPE_TRIAN_ATR = 0
const CV_SEQ_KIND_BITS = 2
const CV_SEQ_MAGIC_VAL =  0x42990000
const CV_SET_MAGIC_VAL = 0x42980000
const CV_SPARSE_MAT_MAGIC_VAL  = 0x42440000
const CV_STORAGE_APPEND =  2
const CV_STORAGE_MAGIC_VAL  = 0x42890000
const CV_STORAGE_READ =  0
const CV_STORAGE_WRITE  = 1
const CV_STORAGE_WRITE_BINARY =  CV_STORAGE_WRITE
const CV_STORAGE_WRITE_TEXT  = CV_STORAGE_WRITE
const CV_SUBMAT_FLAG_SHIFT  = 15
const CV_TERMCRIT_EPS  = 2
const CV_TERMCRIT_ITER  = 1
const CV_USRTYPE1 =  7
const CV_WHOLE_SEQ_END_INDEX =  0x3fffffff

# Errors
const CV_StsOk = 0
const CV_StsBackTrace = -1
const CV_StsError = -2
const CV_StsInternal = -3
const CV_StsNoMem = -4
const CV_StsBadArg = -5
const CV_StsBadFunc = -6
const CV_StsNoConv = -7
const CV_StsAutoTrace = -8
const CV_HeaderIsNull = -9
const CV_BadImageSize = -10
const CV_BadOffset = -11
const CV_BadDataPtr = -12
const CV_BadStep = -13
const CV_BadModelOrChSeq = -14
const CV_BadNumChannels = -15
const CV_BadNumChannel1U = -16
const CV_BadDepth = -17
const CV_BadAlphaChannel = -18
const CV_BadOrder = -19
const CV_BadOrigin = -20
const CV_BadAlign = -21
const CV_BadCallBack = -22
const CV_BadTileSize = -23
const CV_BadCOI = -24
const CV_BadROISize = -25
const CV_MaskIsTiled = -26
const CV_StsNullPtr = -27,
const CV_StsVecLengthErr = -28
const CV_StsFilterStructContentErr = -29
const CV_StsKernelStructContentErr = -30
const CV_StsFilterOffsetErr = -31
const CV_StsBadSize = -201
const CV_StsDivByZero = -202
const CV_StsInplaceNotSupported = -203
const CV_StsObjectNotFound = -204
const CV_StsUnmatchedFormats = -205
const CV_StsBadFlag = -206
const CV_StsBadPoint = -207
const CV_StsBadMask = -208
const CV_StsUnmatchedSizes = -209
const CV_StsUnsupportedFormat = -210
const CV_StsOutOfRange = -211
const CV_StsParseError = -212
const CV_StsNotImplemented = -213
const CV_StsBadMemBlock = -214
const CV_StsAssert = -215
const CV_GpuNotSupported = -216
const CV_GpuApiCallError = -217
const CV_GpuNppCallError = -218
const CV_GpuCufftCallError = -219


# imgproc.hpp

# morphological operations
const MORPH_ERODE    = 0
const MORPH_DILATE   = 1
const MORPH_OPEN     = 2
const MORPH_CLOSE    = 3
const MORPH_GRADIENT = 4
const MORPH_TOPHAT   = 5
const MORPH_BLACKHAT = 6

# shape of the structuring element
const MORPH_RECT    = 0
const MORPH_CROSS   = 1
const MORPH_ELLIPSE = 2

# interpolation algorithm
const INTER_NEAREST        = 0 # nearest neighbor interpolation
const INTER_LINEAR         = 1 # bilinear interpolation
const INTER_CUBIC          = 2 # bicubic interpolation
const INTER_AREA           = 3 # area-based (or super) interpolation
const INTER_LANCZOS4       = 4 #  Lanczos interpolation over 8x8 neighborhood

const INTER_MAX            = 7 # mask for interpolation codes
const WARP_FILL_OUTLIERS   = 8
const WARP_INVERSE_MAP     = 16

const INTER_BITS      = 5
const INTER_BITS2     = INTER_BITS * 2
const INTER_TAB_SIZE  = 1 << INTER_BITS
const INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE

# Distance types for Distance Transform and M-estimators
const DIST_USER    = -1  # User defined distance
const DIST_L1      = 1   # distance = |x1-x2| + |y1-y2|
const DIST_L2      = 2   # the simple euclidean distance
const DIST_C       = 3   # distance = max(|x1-x2|,|y1-y2|)
const DIST_L12     = 4   # L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
const DIST_FAIR    = 5   # distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
const DIST_WELSCH  = 6   # distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
const DIST_HUBER   = 7   # distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345

# Mask size for distance transform
const DIST_MASK_3       = 3
const DIST_MASK_5       = 5
const DIST_MASK_PRECISE = 0

# type of the threshold operation
const THRESH_BINARY     = 0 # value = value > threshold ? max_value : 0
const THRESH_BINARY_INV = 1 # value = value > threshold ? 0 : max_value
const THRESH_TRUNC      = 2 # value = value > threshold ? threshold : value
const THRESH_TOZERO     = 3 #  value = value > threshold ? value : 0
const THRESH_TOZERO_INV = 4 #  value = value > threshold ? 0 : value
const THRESH_MASK       = 7
const THRESH_OTSU       = 8  #  use Otsu algorithm to choose the optimal threshold value

# adaptive threshold algorithm
const ADAPTIVE_THRESH_MEAN_C     = 0
const ADAPTIVE_THRESH_GAUSSIAN_C = 1

const PROJ_SPHERICAL_ORTHO  = 0
const PROJ_SPHERICAL_EQRECT = 1

# class of the pixel in GrabCut algorithm
const GC_BGD    = 0 #  background
const GC_FGD    = 1 # foreground
const GC_PR_BGD = 2 # most probably background
const GC_PR_FGD = 3  # most probably foreground

# GrabCut algorithm flags
const GC_INIT_WITH_RECT  = 0
const GC_INIT_WITH_MASK  = 1
const GC_EVAL            = 2

# distanceTransform algorithm flags
const DIST_LABEL_CCOMP = 0
const DIST_LABEL_PIXEL = 1

# floodfill algorithm flags
const FLOODFILL_FIXED_RANGE = 1 << 16
const FLOODFILL_MASK_ONLY   = 1 << 17

# type of the template matching operation
const TM_SQDIFF        = 0
const TM_SQDIFF_NORMED = 1
const TM_CCORR         = 2
const TM_CCORR_NORMED  = 3
const TM_CCOEFF        = 4
const TM_CCOEFF_NORMED = 5

# connected components algorithm output formats
const CC_STAT_LEFT   = 0
const CC_STAT_TOP    = 1
const CC_STAT_WIDTH  = 2
const CC_STAT_HEIGHT = 3
const CC_STAT_AREA   = 4
const CC_STAT_MAX    = 5

# mode of the contour retrieval algorithm
const RETR_EXTERNAL  = 0 # retrieve only the most external (top-level) contours
const RETR_LIST      = 1 # retrieve all the contours without any hierarchical information
const RETR_CCOMP     = 2 #  retrieve the connected components (that can possibly be nested)
const RETR_TREE      = 3 #  retrieve all the contours and the whole hierarchy
const RETR_FLOODFILL = 4

# the contour approximation algorithm
const CHAIN_APPROX_NONE      = 1
const CHAIN_APPROX_SIMPLE    = 2
const CHAIN_APPROX_TC89_L1   = 3
const CHAIN_APPROX_TC89_KCOS = 4

# Variants of a Hough transform
const HOUGH_STANDARD      = 0
const HOUGH_PROBABILISTIC = 1
const HOUGH_MULTI_SCALE   = 2
const HOUGH_GRADIENT      = 3

# Variants of Line Segment Detector
const LSD_REFINE_NONE = 0
const LSD_REFINE_STD  = 1
const LSD_REFINE_ADV  = 2

# Histogram comparison methods
const HISTCMP_CORREL        = 0
const HISTCMP_CHISQR        = 1
const HISTCMP_INTERSECT     = 2
const HISTCMP_BHATTACHARYYA = 3
const HISTCMP_HELLINGER     = HISTCMP_BHATTACHARYYA
const HISTCMP_CHISQR_ALT    = 4
const HISTCMP_KL_DIV        = 5

# the color conversion code
const COLOR_BGR2BGRA     = 0
const COLOR_RGB2RGBA     = COLOR_BGR2BGRA

const COLOR_BGRA2BGR     = 1
const COLOR_RGBA2RGB     = COLOR_BGRA2BGR

const COLOR_BGR2RGBA     = 2
const COLOR_RGB2BGRA     = COLOR_BGR2RGBA

const COLOR_RGBA2BGR     = 3
const COLOR_BGRA2RGB     = COLOR_RGBA2BGR

const COLOR_BGR2RGB      = 4
const COLOR_RGB2BGR      = COLOR_BGR2RGB

const COLOR_BGRA2RGBA    = 5
const COLOR_RGBA2BGRA    = COLOR_BGRA2RGBA

const COLOR_BGR2GRAY     = 6
const COLOR_RGB2GRAY     = 7
const COLOR_GRAY2BGR     = 8
const COLOR_GRAY2RGB     = COLOR_GRAY2BGR
const COLOR_GRAY2BGRA    = 9
const COLOR_GRAY2RGBA    = COLOR_GRAY2BGRA
const COLOR_BGRA2GRAY    = 10
const COLOR_RGBA2GRAY    = 11
const COLOR_BGR2BGR565   = 12
const COLOR_RGB2BGR565   = 13
const COLOR_BGR5652BGR   = 14
const COLOR_BGR5652RGB   = 15
const COLOR_BGRA2BGR565  = 16
const COLOR_RGBA2BGR565  = 17
const COLOR_BGR5652BGRA  = 18
const COLOR_BGR5652RGBA  = 19

const COLOR_GRAY2BGR565  = 20
const COLOR_BGR5652GRAY  = 21

const COLOR_BGR2BGR555   = 22
const COLOR_RGB2BGR555   = 23
const COLOR_BGR5552BGR   = 24
const COLOR_BGR5552RGB   = 25
const COLOR_BGRA2BGR555  = 26
const COLOR_RGBA2BGR555  = 27
const COLOR_BGR5552BGRA  = 28
const COLOR_BGR5552RGBA  = 29

const COLOR_GRAY2BGR555  = 30
const COLOR_BGR5552GRAY  = 31

const COLOR_BGR2XYZ      = 32
const COLOR_RGB2XYZ      = 33
const COLOR_XYZ2BGR      = 34
const COLOR_XYZ2RGB      = 35

const COLOR_BGR2YCrCb    = 36
const COLOR_RGB2YCrCb    = 37
const COLOR_YCrCb2BGR    = 38
const COLOR_YCrCb2RGB    = 39

const COLOR_BGR2HSV      = 40
const COLOR_RGB2HSV      = 41

const COLOR_BGR2Lab      = 44
const COLOR_RGB2Lab      = 45

const COLOR_BGR2Luv      = 50
const COLOR_RGB2Luv      = 51
const COLOR_BGR2HLS      = 52
const COLOR_RGB2HLS      = 53

const COLOR_HSV2BGR      = 54,
const COLOR_HSV2RGB      = 55,

const COLOR_Lab2BGR      = 56
const COLOR_Lab2RGB      = 57
const COLOR_Luv2BGR      = 58
const COLOR_Luv2RGB      = 59
const COLOR_HLS2BGR      = 60
const COLOR_HLS2RGB      = 61

const COLOR_BGR2HSV_FULL = 66
const COLOR_RGB2HSV_FULL = 67
const COLOR_BGR2HLS_FULL = 68
const COLOR_RGB2HLS_FULL = 69

const COLOR_HSV2BGR_FULL = 70
const COLOR_HSV2RGB_FULL = 71
const COLOR_HLS2BGR_FULL = 72
const COLOR_HLS2RGB_FULL = 73

const COLOR_LBGR2Lab     = 74
const COLOR_LRGB2Lab     = 75
const COLOR_LBGR2Luv     = 76
const COLOR_LRGB2Luv     = 77

const COLOR_Lab2LBGR     = 78
const COLOR_Lab2LRGB     = 79
const COLOR_Luv2LBGR     = 80
const COLOR_Luv2LRGB     = 81

const COLOR_BGR2YUV      = 82
const COLOR_RGB2YUV      = 83
const COLOR_YUV2BGR      = 84
const COLOR_YUV2RGB      = 85

# YUV 4:2:0 family to RGB
const COLOR_YUV2RGB_NV12  = 90
const COLOR_YUV2BGR_NV12  = 91
const COLOR_YUV2RGB_NV21  = 92
const COLOR_YUV2BGR_NV21  = 93
const COLOR_YUV420sp2RGB  = COLOR_YUV2RGB_NV21
const COLOR_YUV420sp2BGR  = COLOR_YUV2BGR_NV21

const COLOR_YUV2RGBA_NV12 = 94
const COLOR_YUV2BGRA_NV12 = 95
const COLOR_YUV2RGBA_NV21 = 96
const COLOR_YUV2BGRA_NV21 = 97
const COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21
const COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21

const COLOR_YUV2RGB_YV12  = 98
const COLOR_YUV2BGR_YV12  = 99
const COLOR_YUV2RGB_IYUV  = 100
const COLOR_YUV2BGR_IYUV  = 101
const COLOR_YUV2RGB_I420  = COLOR_YUV2RGB_IYUV
const COLOR_YUV2BGR_I420  = COLOR_YUV2BGR_IYUV
const COLOR_YUV420p2RGB   = COLOR_YUV2RGB_YV12
const COLOR_YUV420p2BGR   = COLOR_YUV2BGR_YV12

const COLOR_YUV2RGBA_YV12 = 102
const COLOR_YUV2BGRA_YV12 = 103
const COLOR_YUV2RGBA_IYUV = 104
const COLOR_YUV2BGRA_IYUV = 105
const COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV
const COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV
const COLOR_YUV420p2RGBA  = COLOR_YUV2RGBA_YV12
const COLOR_YUV420p2BGRA  = COLOR_YUV2BGRA_YV12

const COLOR_YUV2GRAY_420  = 106
const COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420
const COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420
const COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420
const COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420
const COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420
const COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420
const COLOR_YUV420p2GRAY  = COLOR_YUV2GRAY_420

# YUV 4:2:2 family to RGB
const COLOR_YUV2RGB_UYVY = 107
const COLOR_YUV2BGR_UYVY = 108
#COLOR_YUV2RGB_VYUY = 109
#COLOR_YUV2BGR_VYUY = 110
const COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY
const COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY
const COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY
const COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY

const COLOR_YUV2RGBA_UYVY = 111
const COLOR_YUV2BGRA_UYVY = 112
#COLOR_YUV2BGRA_VYUY = 114
const COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY
const COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY
const COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY
const COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY

const COLOR_YUV2RGB_YUY2 = 115
const COLOR_YUV2BGR_YUY2 = 116
const COLOR_YUV2RGB_YVYU = 117
const COLOR_YUV2BGR_YVYU = 118
const COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2
const COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2
const COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2
const COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2

const COLOR_YUV2RGBA_YUY2 = 119
const COLOR_YUV2BGRA_YUY2 = 120
const COLOR_YUV2RGBA_YVYU = 121
const COLOR_YUV2BGRA_YVYU = 122
const COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2
const COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2
const COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2
const COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2

const COLOR_YUV2GRAY_UYVY = 123
const COLOR_YUV2GRAY_YUY2 = 124
#CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY
const COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY
const COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY
const COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2
const COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2
const COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2

# alpha premultiplication
const COLOR_RGBA2mRGBA    = 125
const COLOR_mRGBA2RGBA    = 126

# RGB to YUV 4:2:0 family
const COLOR_RGB2YUV_I420  = 127
const COLOR_BGR2YUV_I420  = 128
const COLOR_RGB2YUV_IYUV  = COLOR_RGB2YUV_I420
const COLOR_BGR2YUV_IYUV  = COLOR_BGR2YUV_I420

const COLOR_RGBA2YUV_I420 = 129
const COLOR_BGRA2YUV_I420 = 130
const COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420
const COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420
const COLOR_RGB2YUV_YV12  = 131
const COLOR_BGR2YUV_YV12  = 132
const COLOR_RGBA2YUV_YV12 = 133
const COLOR_BGRA2YUV_YV12 = 134

#Demosaicing
const COLOR_BayerBG2BGR = 46
const COLOR_BayerGB2BGR = 47
const COLOR_BayerRG2BGR = 48
const COLOR_BayerGR2BGR = 49

const COLOR_BayerBG2RGB = COLOR_BayerRG2BGR
const COLOR_BayerGB2RGB = COLOR_BayerGR2BGR
const COLOR_BayerRG2RGB = COLOR_BayerBG2BGR
const COLOR_BayerGR2RGB = COLOR_BayerGB2BGR

const COLOR_BayerBG2GRAY = 86
const COLOR_BayerGB2GRAY = 87
const COLOR_BayerRG2GRAY = 88
const COLOR_BayerGR2GRAY = 89

# Demosaicing using Variable Number of Gradients
const COLOR_BayerBG2BGR_VNG = 62
const COLOR_BayerGB2BGR_VNG = 63
const COLOR_BayerRG2BGR_VNG = 64
const COLOR_BayerGR2BGR_VNG = 65

const COLOR_BayerBG2RGB_VNG = COLOR_BayerRG2BGR_VNG
const COLOR_BayerGB2RGB_VNG = COLOR_BayerGR2BGR_VNG
const COLOR_BayerRG2RGB_VNG = COLOR_BayerBG2BGR_VNG
const COLOR_BayerGR2RGB_VNG = COLOR_BayerGB2BGR_VNG

# Edge-Aware Demosaicing
const COLOR_BayerBG2BGR_EA  = 135
const COLOR_BayerGB2BGR_EA  = 136
const COLOR_BayerRG2BGR_EA  = 137
const COLOR_BayerGR2BGR_EA  = 138

const COLOR_BayerBG2RGB_EA  = COLOR_BayerRG2BGR_EA
const COLOR_BayerGB2RGB_EA  = COLOR_BayerGR2BGR_EA
const COLOR_BayerRG2RGB_EA  = COLOR_BayerBG2BGR_EA
const COLOR_BayerGR2RGB_EA  = COLOR_BayerGB2BGR_EA

const COLOR_COLORCVT_MAX  = 139

# COLORMAP
const COLORMAP_AUTUMN  = 0
const COLORMAP_BONE    = 1
const COLORMAP_JET     = 2
const COLORMAP_WINTER  = 3
const COLORMAP_RAINBOW = 4
const COLORMAP_OCEAN   = 5
const COLORMAP_SUMMER  = 6
const COLORMAP_SPRING  = 7
const COLORMAP_COOL    = 8
const COLORMAP_HSV     = 9
const COLORMAP_PINK    = 10
const COLORMAP_HOT     = 11

# types of intersection between rectangles
const INTERSECT_NONE = 0
const INTERSECT_PARTIAL  = 1
const INTERSECT_FULL  = 2

# highgui.hpp
typealias WindowProperty Uint32
const WINDOW_NORMAL     = 0x00000000 # the user can resize the window (no constraint)
const WINDOW_AUTOSIZE   = 0x00000001 # the user cannot resize the window, the size is constrainted by the image displayed
const WINDOW_OPENGL     = 0x00001000 # window with opengl support
const WINDOW_FULLSCREEN = 1          # change the window to fullscreen
const WINDOW_FREERATIO  = 0x00000100 # the image expends as much as it can (no ratio constraint)
const WINDOW_KEEPRATIO  = 0x00000000 # the ratio of the image is respected

typealias getWindowPropertyFlag Int64
const WND_PROP_FULLSCREEN   = 0  # fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN)
const WND_PROP_AUTOSIZE     = 1  # autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE)
const WND_PROP_ASPECT_RATIO = 2  # window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO);
const WND_PROP_OPENGL       = 3  # opengl support

typealias mouseEventFlag Int64
const EVENT_MOUSEMOVE      = 0
const EVENT_LBUTTONDOWN    = 1
const EVENT_RBUTTONDOWN    = 2
const EVENT_MBUTTONDOWN    = 3
const EVENT_LBUTTONUP      = 4
const EVENT_RBUTTONUP      = 5
const EVENT_MBUTTONUP      = 6
const EVENT_LBUTTONDBLCLK  = 7
const EVENT_RBUTTONDBLCLK  = 8
const EVENT_MBUTTONDBLCLK  = 9
const EVENT_MOUSEHWHEEL    = 11
const EVENT_FLAG_SHIFTKEY  = 16
const EVENT_FLAG_ALTKEY    = 32

# videoio.hpp
typealias captureFlag Int64
const CAP_ANY          = 0     # autodetect
const CAP_VFW          = 200   # platform native
const CAP_V4L          = 200
const CAP_V4L2         = CAP_V4L
const CAP_FIREWARE     = 300   # IEEE 1394 drivers
const CAP_FIREWIRE     = CAP_FIREWARE
const CAP_IEEE1394     = CAP_FIREWARE
const CAP_DC1394       = CAP_FIREWARE
const CAP_CMU1394      = CAP_FIREWARE
const CAP_QT           = 500   # QuickTime
const CAP_UNICAP       = 600   # Unicap drivers
const CAP_DSHOW        = 700   # DirectShow (via videoInput)
const CAP_PVAPI        = 800   # PvAPI, Prosilica GigE SDK
const CAP_OPENNI       = 900   # OpenNI (for Kinect)
const CAP_OPENNI_ASUS  = 910   # OpenNI (for Asus Xtion)
const CAP_ANDROID      = 1000  # Android
const CAP_XIAPI        = 1100  # XIMEA Camera API
const CAP_AVFOUNDATION = 1200  # AVFoundation framework for iOS (OS X Lion will have the same API)
const CAP_GIGANETIX    = 1300  # Smartek Giganetix GigEVisionSDK
const CAP_MSMF         = 1400  # Microsoft Media Foundation (via videoInput)
const CAP_INTELPERC    = 1500  # Intel Perceptual Computing SDK
const CAP_OPENNI2      = 1600  # OpenNI2 (for Kinect)

# generic properties (based on DC1394 properties)
typealias capturePROPflag Int64
const CAP_PROP_POS_MSEC       = 0
const CAP_PROP_POS_FRAMES     = 1
const CAP_PROP_POS_AVI_RATIO  = 2
const CAP_PROP_FRAME_WIDTH    = 3
const CAP_PROP_FRAME_HEIGHT   = 4
const CAP_PROP_FPS            = 5
const CAP_PROP_FOURCC         = 6
const CAP_PROP_FRAME_COUNT    = 7
const CAP_PROP_FORMAT         = 8
const CAP_PROP_MODE           = 9
const CAP_PROP_BRIGHTNESS    = 10
const CAP_PROP_CONTRAST      = 11
const CAP_PROP_SATURATION    = 12
const CAP_PROP_HUE           = 13
const CAP_PROP_GAIN          = 14
const CAP_PROP_EXPOSURE      = 15
const CAP_PROP_CONVERT_RGB   = 16
const CAP_PROP_WHITE_BALANCE_BLUE_U = 17
const CAP_PROP_RECTIFICATION = 18
const CAP_PROP_MONOCROME     = 19
const CAP_PROP_SHARPNESS     = 20
const CAP_PROP_AUTO_EXPOSURE = 21        # DC1394: exposure control
const CAP_PROP_GAMMA         = 22
const CAP_PROP_TEMPERATURE   = 23
const CAP_PROP_TRIGGER       = 24
const CAP_PROP_TRIGGER_DELAY = 25
const CAP_PROP_WHITE_BALANCE_RED_V = 26
const CAP_PROP_ZOOM          = 27
const CAP_PROP_FOCUS         = 28
const CAP_PROP_GUID          = 29
const CAP_PROP_ISO_SPEED     = 30
const CAP_PROP_BACKLIGHT     = 32
const CAP_PROP_PAN           = 33
const CAP_PROP_TILT          = 34
const CAP_PROP_ROLL          = 35
const CAP_PROP_IRIS          = 36
const CAP_PROP_SETTINGS      = 37

# FourCC codecs for VideoWriter
const CV_FOURCC_IYUV = [pointer("I"),pointer("Y"),pointer("U"),pointer("V")]   #for yuv420p into an uncompressed AVI
const CV_FOURCC_DIV3 = [pointer("D"),pointer("I"),pointer("V"),pointer("3")]   #for DivX MPEG-4 codec
const CV_FOURCC_MP42 = [pointer("M"),pointer("P"),pointer("4"),pointer("2")]   #for MPEG-4 codec
const CV_FOURCC_DIVX = [pointer("D"),pointer("I"),pointer("V"),pointer("X")]   #for DivX codec
const CV_FOURCC_PIM1 = [pointer("P"),pointer("I"),pointer("M"),pointer("1")]   #for MPEG-1 codec
const CV_FOURCC_I263 = [pointer("I"),pointer("2"),pointer("6"),pointer("3")]   #for ITU H.263 codec
const CV_FOURCC_MPEG = [pointer("M"),pointer("P"),pointer("E"),pointer("G")]   #for MPEG-1 codec

# imgcodecs.hpp
typealias imreadFlag Int64
const IMREAD_UNCHANGED  = -1         # 8bit, color or not
const IMREAD_GRAYSCALE  = 0          # 8bit, gray
const IMREAD_COLOR      = 1          # ?, color
const IMREAD_ANYDEPTH   = 2          # any depth, ?
const IMREAD_ANYCOLOR   = 4          # ?, any color
const IMREAD_LOAD_GDAL  = 8          # Use gdal driver

typealias imwriteFlag Int64
const IMWRITE_JPEG_QUALITY        = 1
const IMWRITE_JPEG_PROGRESSIVE    = 2
const IMWRITE_JPEG_OPTIMIZE       = 3
const IMWRITE_JPEG_RST_INTERVAL   = 4
const IMWRITE_JPEG_LUMA_QUALITY   = 5
const IMWRITE_JPEG_CHROMA_QUALITY = 6
const IMWRITE_PNG_COMPRESSION     = 16
const IMWRITE_PNG_STRATEGY        = 17
const IMWRITE_PNG_BILEVEL         = 18
const IMWRITE_PXM_BINARY          = 32
const IMWRITE_WEBP_QUALITY        = 64

const IMWRITE_PNG_STRATEGY_DEFAULT      = 0
const IMWRITE_PNG_STRATEGY_FILTERED     = 1
const IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2
const IMWRITE_PNG_STRATEGY_RLE          = 3
const IMWRITE_PNG_STRATEGY_FIXED        = 4

# objectdetect.hpp
# CvHaarClassifierCascade
const CASCADE_DO_CANNY_PRUNING    = 1
const CASCADE_SCALE_IMAGE         = 2
const CASCADE_FIND_BIGGEST_OBJECT = 4
const CASCADE_DO_ROUGH_SEARCH     = 8

# HOGDescriptor
const L2Hys = 0
const DEFAULT_NLEVELS = 64

# features2d.hpp
const DEFAULT = 0
const DRAW_OVER_OUTIMG = 1
const NOT_DRAW_SINGLE_POINTS = 2
const DRAW_RICH_KEYPOINTS = 4

# calib3d.hpp
const LMEDS  = 4 #  least-median algorithm
const RANSAC = 8 #  RANSAC algorithm

const SOLVEPNP_ITERATIVE = 0
const SOLVEPNP_EPNP      = 1  #  F.Moreno-Noguer et al, "EPnP: Efficient Perspective-n-Point Camera Pose Estimation"
const SOLVEPNP_P3P       = 2  #  X.S. Gao et al, "Complete Solution Classification for the Perspective-Three-Point Problem"
const SOLVEPNP_DLS       = 3  #  Joel A. Hesch et al, "A Direct Least-Squares (DLS) Method for PnP"
const SOLVEPNP_UPNP      = 4  #  A.Penate-Sanchez et al. "Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation"

const CALIB_CB_ADAPTIVE_THRESH = 1
const CALIB_CB_NORMALIZE_IMAGE = 2
const CALIB_CB_FILTER_QUADS    = 4
const CALIB_CB_FAST_CHECK      = 8

const CALIB_CB_SYMMETRIC_GRID  = 1
const CALIB_CB_ASYMMETRIC_GRID = 2
const CALIB_CB_CLUSTERING      = 4

const CALIB_USE_INTRINSIC_GUESS = 0x00001
const CALIB_FIX_ASPECT_RATIO    = 0x00002
const CALIB_FIX_PRINCIPAL_POINT = 0x00004
const CALIB_ZERO_TANGENT_DIST   = 0x00008
const CALIB_FIX_FOCAL_LENGTH    = 0x00010
const CALIB_FIX_K1              = 0x00020
const CALIB_FIX_K2              = 0x00040
const CALIB_FIX_K3              = 0x00080
const CALIB_FIX_K4              = 0x00800
const CALIB_FIX_K5              = 0x01000
const CALIB_FIX_K6              = 0x02000
const CALIB_RATIONAL_MODEL      = 0x04000
const CALIB_THIN_PRISM_MODEL    = 0x08000
const CALIB_FIX_S1_S2_S3_S4     = 0x10000

# only for stereo
const CALIB_FIX_INTRINSIC       = 0x00100
const CALIB_SAME_FOCAL_LENGTH   = 0x00200

# for stereo rectification
const CALIB_ZERO_DISPARITY      = 0x00400

# the algorithm for finding fundamental matrix
const FM_7POINT = 1 #  7-point algorithm
const FM_8POINT = 2 #  8-point algorithm
const FM_LMEDS  = 4 # least-median algorithm
const FM_RANSAC = 8  # RANSAC algorithm
