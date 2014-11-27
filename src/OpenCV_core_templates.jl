
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

    CV_Assert(mat.dims>1);
    CV_Assert(mat.dims<4);
    CV_Assert(mat.dims!=2);

    if (mat.dims==1)
    {
        return((_wTp) mat.template at<_Tp>(row,col));
    }
    if(mat.dims==3)
    {
        cv::Vec<_Tp,3> cvec;
        cvec[0] = mat.template at<cv::Vec<_Tp,3>>(row,col)[0];
        cvec[1] = mat.template at<cv::Vec<_Tp,3>>(row,col)[1];
        cvec[2] = mat.template at<cv::Vec<_Tp,3>>(row,col)[2];
        return((_wTp) cvec);
    }
}

template <typename _Rs>
_Rs imget(cv::Mat img, int row, int col) {
 typedef typename cv::DataType<_Rs>::work_type _wRs;

    CV_Assert(img.dims>1);
    CV_Assert(img.dims<4);
    CV_Assert(img.dims!=2);

    //  if (!(img.dims==1) | !(img.dims==3)) {
    //    std::cout << "Argument error: only images with 1 and 3 dimensions are supported." << std::endl
    //  }

 //   switch(img.type()) {
       // if(img.type()==CV_8SC1) {
        // case CV_8SC1:
        cv::Mat_<char>dst(img.rows, img.cols,img.type()); // break;
 //     case CV_8UC1:  cv::Mat_<uchar>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16SC1: cv::Mat_<short>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16UC1: cv::Mat_<unsigned short>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32SC1: cv::Mat_<int>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32FC1: cv::Mat_<float>dst(img.rows, img.cols,img.type()); break;
 //     case CV_64FC1: cv::Mat_<double>dst(img.rows, img.cols,img.type()); break;
 //     case CV_8SC3:  cv::Mat_<cv::Vec<char,3>>dst(img.rows, img.cols,img.type()); break;
 //     case CV_8UC3:  cv::Mat_<cv::Vec3b>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16SC3: cv::Mat_<cv::Vec3s>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16UC3: cv::Mat_<cv::Vec<unsigned short,3>>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32SC3: cv::Mat_<cv::Vec3i>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32FC3: cv::Mat_<cv::Vec3f>dst(img.rows, img.cols,img.type()); break;
 //     case CV_64FC3: cv::Mat_<cv::Vec3d>dst(img.rows, img.cols,img.type()); break;
    // }

    img.assignTo(dst,img.type());
    return((_wRs) get(dst, row, col));
}
"""

cxx"""
template <typename _Tp>

void set(const cv::Mat_<_Tp>& mat, int row, int col, cv::Vec<_Tp,3> cvec) {

    CV_Assert(mat.dims>1);
    CV_Assert(mat.dims<4);
    CV_Assert(mat.dims!=2);

    if (mat.dims==1)
    {
         mat.template at<_Tp>(row,col) = cvec[0];
    }
    if(mat.dims==3)
    {
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[0];
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[1];
         mat.template at<cv::Vec<_Tp,3>>(row,col)[0] = cvec[2];
    }
}

template <typename _Rs>
void imset(cv::Mat img, int row, int col, cv::Vec<_Rs,3> cvec) {

    CV_Assert(img.dims>1);
    CV_Assert(img.dims<4);
    CV_Assert(img.dims!=2);

   // if (!(img.dims==1) | !(img.dims==3)) {
   //    std::cout << "Argument error: only images with 1 and 3 dimensions are supported." << std::endl;
   //
   //  }

 //   switch(img.type()) {
    //   if(img.type()==CV_8SC1) {
        // case CV_8SC1:
        cv::Mat_<char>dst(img.rows, img.cols,img.type()); // break;
 //     case CV_8UC1:  cv::Mat_<uchar>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16SC1: cv::Mat_<short>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16UC1: cv::Mat_<unsigned short>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32SC1: cv::Mat_<int>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32FC1: cv::Mat_<float>dst(img.rows, img.cols,img.type()); break;
 //     case CV_64FC1: cv::Mat_<double>dst(img.rows, img.cols,img.type()); break;
 //     case CV_8SC3:  cv::Mat_<cv::Vec<char,3>>dst(img.rows, img.cols,img.type()); break;
 //     case CV_8UC3:  cv::Mat_<cv::Vec3b>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16SC3: cv::Mat_<cv::Vec3s>dst(img.rows, img.cols,img.type()); break;
 //     case CV_16UC3: cv::Mat_<cv::Vec<unsigned short,3>>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32SC3: cv::Mat_<cv::Vec3i>dst(img.rows, img.cols,img.type()); break;
 //     case CV_32FC3: cv::Mat_<cv::Vec3f>dst(img.rows, img.cols,img.type()); break;
 //     case CV_64FC3: cv::Mat_<cv::Vec3d>dst(img.rows, img.cols,img.type()); break;
  //   }

    img.assignTo(dst,img.type());
    set(dst, row, col, cvec);
}
"""
