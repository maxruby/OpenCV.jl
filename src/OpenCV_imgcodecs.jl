####################################################################################################
# imgcodecs. Image file reading and writing
####################################################################################################

# 1. Reading and Writing Images

# imdecode: Reads an image from a buffer in memory
imdecode(buf, flags) = @cxx cv::imdecode(buf, flags)  # returs Mat
# buf – Input array or vector of bytes
# flags – The same flags as in imread()

# imencode: Encodes an image into a memory buffer
imencode(ext, img, buf) = @cxx cv::imencode(ext, img, buf)
# ext    – File extension that defines the output format
# img    – Image to be written
# buf    – Output buffer resized to fit the compressed image  => vector<uchar>&
# params – Format-specific parameters. See imwrite()   => const vector<int>&, default = vector<int>()

# imread: Read images from file
imread(filename::Ptr{Uint8}, flags=IMREAD_COLOR) = @cxx cv::imread(filename, flags)
# IMREAD_UNCHANGED  # 8bit, color or not
# IMREAD_GRAYSCALE  # 8bit, gray
# IMREAD_COLOR      # ?, color
# IMREAD_ANYDEPTH   # any depth, ?
# IMREAD_ANYCOLOR   # ?, any color
# IMREAD_LOAD_GDAL

# imwrite: Saves an image to a specified file
imwrite(filename::Ptr{Uint8}, img) = @cxx cv::imwrite(filename, img)
# optional: const vector<int>& params=vector<int>()
# filename – Name of the file
# image    – Image to be saved
# params   –
# IMWRITE_JPEG_QUALITY      Default value is 95 {0,100}
# IMWRITE_WEBP_QUALITY      Default value is 100 {1,100}
# IMWRITE_PNG_COMPRESSION   Default value is 3 {0,9}
# IMWRITE_PXM_BINARY        Default value is 1 {0,1}
