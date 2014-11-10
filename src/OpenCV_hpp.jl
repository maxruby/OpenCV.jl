################################################################################################
#
# OpenCV_hpp.jl
# single header OpenCV (.hpp) wrapper for Julia
#
################################################################################################

# constant declarations

# Mat types (core.hpp)
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

# highgui.hpp constant declarations
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


