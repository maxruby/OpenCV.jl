# OpenCV 3.0 documentation
# @ http://docs.opencv.org/trunk/index.html
# http://physics.nyu.edu/grierlab/manuals/opencv/core_2include_2opencv2_2core_2types__c_8h.html#a9d2ee1a8334733dea7482a47a88e0f87

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

# cv::Vector definition (Vec aliases)

# typedef Vec<uchar, 2> Vec2b;
# typedef Vec<uchar, 3> Vec3b;
# typedef Vec<uchar, 4> Vec4b;

# typedef Vec<short, 2> Vec2s;
# typedef Vec<short, 3> Vec3s;
# typedef Vec<short, 4> Vec4s;

# typedef Vec<int, 2> Vec2i;
# typedef Vec<int, 3> Vec3i;
# typedef Vec<int, 4> Vec4i;

# typedef Vec<float, 2> Vec2f;
# typedef Vec<float, 3> Vec3f;
# typedef Vec<float, 4> Vec4f;
# typedef Vec<float, 6> Vec6f;

# typedef Vec<double, 2> Vec2d;
# typedef Vec<double, 3> Vec3d;
# typedef Vec<double, 4> Vec4d;
# typedef Vec<double, 6> Vec6d;

# C++ header files
# cxx"""
# #include <iostream>
# #include <stdlib.h>
# #include <stdio.h>
# """



# Pass opencv inputarray and use it as std::vector
# http://stackoverflow.com/questions/25750041/pass-opencv-inputarray-and-use-it-as-stdvector


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


# Templated functions
cxx"""
      #include<iostream>
      #include<vector>

      template<typename T>

      void PrintmyTemplate(const T var) {
           std::cout << "Templated function:\n" << std::endl;
           std::cout << var << std::endl;
      }
"""

# See:
# http://bytefish.de/blog/opencv/code_snippets/

# http://www.wisegai.com/2012/12/06/using-templated-functions-in-c-and-opencv/
# http://stackoverflow.com/questions/3786360/confusing-template-error
cxx"""
template <typename T>
void drawCircles(InputArray _image, InputArray _points, Scalar color)
{
    Mat images = _image.getMat(), points = _points.getMat();  // getMat() =>  old C Ipl function
    CV_Assert(points.channels() == 2);

    for (int i = 0; i < points.cols; i++) {
        Vec<T,2>& v = points.at<Vec<T,2>>(0,i);

        Point2i p;
        p.x = cvRound(v[0]);
        p.y = cvRound(v[1]);

        circle(image, p, 5, color, 2, 8);
    }
}
"""

# // Usage:
# drawCircles<float>(frame, Points, Scalar(255, 255, 255));

# http://bytefish.de/blog/opencv/code_snippets/
cxx"""
#include <iostream>

template <typename _Tp>

void printMat(const cv::Mat_<_Tp>& mat) {
    typedef typename cv::DataType<_Tp>::work_type _wTp;
    for(int i = 0; i < mat.rows; i++)
        for(int j=0; j < mat.cols; j++)
            std::cout << (_wTp) mat(i,j) << " " << std::endl;
}

void printMat(const cv::Mat& mat) {
    for(int i = 0; i < mat.rows; i++)
        for(int j=0; j < mat.cols; j++)
            std::cout << (int) mat.at<uchar>(i,j) << " " << std::endl;
}

void funtest() {
    uchar a[] = {0,1,2,3,4,5,6,7,8};
    cv::Mat_<uchar> d0 = cv::Mat_<uchar>(9,1,a).clone();
    cv::Mat d1 = cv::Mat(9,1,CV_8UC1,a).clone();
    std::cout << "d0: " << std::endl;
    printMat(d0);
    std::cout << "d1: " << std::endl;
    printMat(d1);
}
"""

cxx"""
template <typename _Tp>
void histogram(const cv::Mat& input, cv::Mat& hist, int N) {
  hist = cv::Mat::zeros(1, N, CV_32SC1);
  for(int i = 0; i < input.rows; i++) {
    for(int j = 0; j < input.cols; j++) {
      int bin = input.at<_Tp>(i,j);
      hist.at<int>(0,bin) += 1;
    }
  }
}

void histogram(const cv::Mat& input, cv::Mat& hist, int N) {
  if(input.type() != CV_8SC1 && input.type() && CV_8UC1 && input.type() != CV_16SC1
            && input.type() != CV_16UC1 && input.type() != CV_32SC1)
  {
    CV_Error(CV_StsUnsupportedFormat, "Only Integer data is supported.");
    lf::logging::error("wrong type for histogram.");
  }

  switch(input.type()) {
    case CV_8SC1: histogram<char>(input, hist, N); break;
    case CV_8UC1: histogram<unsigned char>(input, hist, N); break;
    case CV_16SC1: histogram<short>(input, hist, N); break;
    case CV_16UC1: histogram<unsigned short>(input, hist, N); break;
    case CV_32SC1: histogram<int>(input, hist, N); break;
  }
}
"""

# get and set pixel values from RGB image
# Approach 1

# Mat image;
# Point3_<uchar>* p = image.ptr<Point3_<uchar> >(y,x);
# p->x //B
# p->y //G
# p->z //R

# Approach 2
# cv::Mat image = ...do some stuff...;
# image.at<cv::Vec3b>(y,x); gives you the RGB (it might be ordered as BGR) vector of type cv::Vec3b
# image.at<cv::Vec3b>(y,x)[0] = newval[0];
# image.at<cv::Vec3b>(y,x)[1] = newval[1];
# image.at<cv::Vec3b>(y,x)[2] = newval[2];


cxx"""
    template <typename T>

    int at(cv::Mat img, int i) {
       cv::Mat_<T> dst(img.rows, img.cols,img.type());
       img.assignTo(dst);
       return(static_cast<int>(dst.template at<T>(i)));
     }
"""

cxx"""
    int at(cv::Mat img, int i, int j) {
       cv::Mat_<uchar> dst(img.rows, img.cols,img.type());
       img.assignTo(dst);
       return(static_cast<int>(dst.at<uchar>(i,j)));
     }
"""

cxx"""
    int at(cv::Mat img, int i, int j, int k) {
       cv::Mat_<uchar> dst(img.rows, img.cols,img.type());
       img.assignTo(dst);
       return(static_cast<int>(dst.at<uchar>(i,j,k)));
     }
"""

cxx"""
    int at(cv::Mat img, cv::Point point) {
       cv::Mat_<uchar> dst(img.rows, img.cols,img.type());
       img.assignTo(dst);
       return(static_cast<int>(dst.at<uchar>(point)));
     }
"""

cxx"""
    int at(cv::Mat img, int *idx) {
       cv::Mat_<uchar> dst(img.rows, img.cols,img.type());
       img.assignTo(dst);
       return(static_cast<int>(dst.at<uchar>(idx)));
     }
"""

at(img, i::Int) = @cxx at(img, i)
at(img, i::Int, j::Int) = @cxx at(img,i,j)
at(img, i::Int, j::Int, k::Int) = @cxx at(img,i,j,k)
at(img, point) = @cxx at(img,point)
at(img, point::Ptr{Int}) = @cxx at(img,idx)    # const int* idx

# Mat::begin
# Mat::end

# bidirectional STL iterators - only useful in C++!
#
cxx"""
      template<typename _T1>

      cv::MatIterator_<_T1> beginIter(cv::Mat<_T1> img) {
          typedef typename cv::DataType<_T1>::work_type _wTp;
          cv::MatIterator_<_T1> iter = img.template begin<_T1>();
          return((_wT) iter);
      }

      cv::MatConstIterator_<_T1> beginConstIter(cv::Mat<_T1> img) {
          typedef typename cv::DataType<_T1>::work_type _wTp;
          cv::MatConstIterator_<_T1> iter = img.template begin<_T1>();
          return((_wT) iter);
      }
"""

# # Mat::end
cxx"""
      template<typename _T1>

      cv::MatIterator_<_T1> endIter(cv::Mat<_T1> img) {
          typedef typename cv::DataType<_T1>::work_type _wTp;
          cv::MatIterator_<_T1> iter = img.template end<_T1>();
          return((_wT) iter);
      }

      cv::MatConstIterator_<_T1> endConstIter(cv::Mat<_T1> img) {
          typedef typename cv::DataType<_T1>::work_type _wTp;
          cv::MatConstIterator_<_T1> iter = img.template end<_T1>();
          return((_wT) iter);
      }
"""

# Mat::forEach
