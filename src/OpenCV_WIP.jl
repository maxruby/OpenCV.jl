# OpenCV 3.0 documentation
# @ http://docs.opencv.org/trunk/index.html

# Mat methods

# modules/core/include/opencv2/core/mat.hpp

# class CV_EXPORTS Mat
# {
# public:
#     // ... a lot of methods ...
#     ...

#     /*! includes several bit-fields:
#          - the magic signature
#          - continuity flag
#          - depth
#          - number of channels
#      */
#     int flags;
#     //! the array dimensionality, >= 2
#     int dims;
#     //! the number of rows and columns or (-1, -1) when the array has more than 2 dimensions
#     int rows, cols;
#     //! pointer to the data
#     uchar* data;

#     //! pointer to the reference counter;
#     // when array points to user-allocated data, the pointer is NULL
#     int* refcount;

#     // other members
#     ...
# };

# C++ header files
# cxx"""
# #include <iostream>
# #include <stdlib.h>
# #include <stdio.h>
# """


# In OpenCV image data is stored in a ROW-major order

# uchar* p = img.ptr(row)
# uchar* p   =>  row 1  |
#                cols   1  2  3  4  5  6  7  8  9  10
# Accessing data:
  # get a pointer to the first row with img.ptr(row) => uchar* p
  # iterate through all the cols of each row with *p++

# get a pointer to the data at the start of each row with .ptr()
# grayscale image
# cxx"""
#     for(int row = 0; row < img.rows; ++row) {
#         uchar* p = img.ptr(row);
#         for(int col = 0; col < img.cols; ++col) {
#            *p++  //points to each pixel value in turn assuming a CV_8UC1 greyscale image
# }
# """

# color image
# cxx"""
#     for(int col = 0; col < img.cols*3; ++col) {
#           *p++    //points to each pixel B,G,R value in turn assuming a CV_8UC3 color image
#     }
# """


# cv::Mat img = cv::imread("lenna.png");
# //Before changing

# cv::imshow("Before",img);
# //change some pixel value
# for(int j=0;j<img.rows;j++)
# {
#   for (int i=0;i<img.cols;i++)
#   {
#     if( i== j)
#        img.at<uchar>(j,i) = 255; //white
#   }
# }
# //After changing
# cv::imshow("After",img);
