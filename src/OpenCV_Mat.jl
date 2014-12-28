
# Get and set functions for access of pixel values in Mat image arrays
cxx"""
int at_u(cv::Mat &img, int row, int col)
{
     return(static_cast<int>(img.at<uchar>(row,col)));
}

void at_u(cv::Mat &img, int row, int col, double val)
{
    uchar value = static_cast<uchar>(val);
    img.at<uchar>(row,col) = value;
}

int at_s(cv::Mat &img, int row, int col)
{
    return(static_cast<int>(img.at<short>(row,col)));
}

void at_s(cv::Mat &img, int row, int col, double val)
{
     short value = static_cast<short>(val);
     img.at<short>(row,col) = value;
}

int at_us(cv::Mat &img, int row, int col)
{
    return(static_cast<int>(img.at<unsigned short>(row,col)));
}

void at_us(cv::Mat &img, int row, int col, double val)
{
    unsigned short value = static_cast<unsigned short>(val);
    img.at<unsigned short>(row,col) = value;
}

float at_f(cv::Mat &img, int row, int col)
{
    return(static_cast<float>(img.at<float>(row,col)));
}

void at_f(cv::Mat &img, int row, int col, double val)
{
    float value = static_cast<float>(val);
    img.at<float>(row,col) = value;
}

double at_d(cv::Mat &img, int row, int col)
{
     return(static_cast<int>(img.at<double>(row,col)));
}

void at_d(cv::Mat &img, int row, int col, double val)
{
     img.at<double>(row,col) = val;
}

std::vector<char> at_vc(cv::Mat &img, int row, int col)
{
         std::vector<char>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec<char,3>>(row,col)[i]; }; return vec;
}

void at_vc(cv::Mat& img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec<char,3>>(row,col)[i] = static_cast<char>(vec[i]); };
}

std::vector<uchar> at_v3b(cv::Mat &img, int row, int col)
{
         std::vector<uchar>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec3b>(row,col)[i]; }; return vec;
}

void at_v3b(cv::Mat &img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) {img.at<cv::Vec3b>(row,col)[i] = static_cast<uchar>(vec[i]); };
}

std::vector<short> at_v3s(cv::Mat &img, int row, int col)
{
         std::vector<short>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec3s>(row,col)[i]; }; return vec;
}

void at_v3s(cv::Mat &img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec3s>(row,col)[i] = static_cast<short>(vec[i]); };
}

std::vector<unsigned short> at_v3us(cv::Mat &img, int row, int col)
{
         std::vector<unsigned short>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec<unsigned short,3>>(row,col)[i]; }; return vec;
}

void at_v3us(cv::Mat &img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec<unsigned short,3>>(row,col)[i] =  static_cast<unsigned short>(vec[i]); };
}

std::vector<int> at_v3i(cv::Mat &img, int row, int col)
{
         std::vector<int>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec<unsigned short,3>>(row,col)[i]; }; return vec;
}

void at_v3i(cv::Mat &img, int row, int col,std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec<unsigned short,3>>(row,col)[i] = static_cast<int>(vec[i]); };
}

std::vector<float> at_v3f(cv::Mat &img, int row, int col)
{
         std::vector<float>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec3f>(row,col)[i]; }; return vec;
}

void at_v3f(cv::Mat &img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec3f>(row,col)[i] = static_cast<float>(vec[i]); };
}

std::vector<double> at_v3d(cv::Mat &img, int row, int col)
{
         std::vector<double>vec(3);
         for (int i = 0; i < 3; i++) { vec[i]= img.at<cv::Vec3d>(row,col)[i]; }; return vec;
}

void at_v3d(cv::Mat &img, int row, int col, std::vector<double> vec)
{
         for (int i = 0; i < 3; i++) { img.at<cv::Vec3d>(row,col)[i] = vec[i]; };
}
"""

function pixget(img, row::Int, col::Int)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing

  # Grayscale and binary images (returns int)
  if (cvtypeval(img) == CV_8UC1)
    val = @cxx at_u(img, row, col);
    #println(int(val)); return (val)
  elseif (cvtypeval(img) == CV_16SC1)
    val = @cxx at_s(img, row, col);
    println(int(val)); return (val)
  elseif (cvtypeval(img) == CV_32SC1)
    val = @cxx at_us(img, row, col);
    println(int(val)); return (val)
  elseif (cvtypeval(img) == CV_32FC1)
    val = @cxx at_f(img, row, col);
    println(val); return (val)
  elseif (cvtypeval(img) == CV_64FC1)
    val = @cxx at_d(img, row, col);
    println(val); return (val)
  # RGB images (returns vec with original types)
  elseif (cvtypeval(img) == CV_8SC3)
    vec = @cxx at_vc(img, row, col);
    println ([int(at(vec, 0)), int(at(vec, 1)),int(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_8UC3)
    vec = @cxx at_v3b(img, row, col);
    println ([int(at(vec, 0)), int(at(vec, 1)),int(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_16SC3)
    vec = @cxx at_v3s(img, row, col);
    println ([int(at(vec, 0)), int(at(vec, 1)),int(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_16UC3)
    vec = @cxx at_v3us(img, row, col);
    println ([int(at(vec, 0)), int(at(vec, 1)),int(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_32SC3)
    vec = @cxx at_v3i(img, row, col);
    println ([int(at(vec, 0)), int(at(vec, 1)),int(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_32FC3)
    vec = @cxx at_v3f(img, row, col);
    println ([float(at(vec, 0)), float(at(vec, 1)),float(at(vec, 2))]); return (vec)
  elseif (cvtypeval(img) == CV_64FC3)
    vec = @cxx at_v3d(img, row, col);
    println ([at(vec, 0), at(vec, 1),at(vec, 2)]); return (vec)
  else throw(ArgumentError("Image format not recognized!"))
  end
end


function pixset(img, row::Int, col::Int, value)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing
  # Grayscale and binary images (value:: double)
  if (cvtypeval(img) == CV_8UC1)
    @cxx at_u(img, row, col, value);
  elseif (cvtypeval(img) == CV_16SC1)
    @cxx at_s(img, row, col, value);
  elseif (cvtypeval(img) == CV_32SC1)
    @cxx at_us(img, row, col, value);
  elseif (cvtypeval(img) == CV_32FC1)
    @cxx at_f(img, row, col, value);
  elseif (cvtypeval(img) == CV_64FC1)
    @cxx at_d(img, row, col, value);
  # RGB images (value::  std::vector<double>)
  elseif (cvtypeval(img) == CV_8SC3)
    @cxx at_vc(img, row, col, value);
  elseif (cvtypeval(img) == CV_8UC3)
    @cxx at_v3b(img, row, col, value);
  elseif (cvtypeval(img) == CV_16SC3)
    @cxx at_v3s(img, row, col, value);
  elseif (cvtypeval(img) == CV_16UC3)
    @cxx at_v3us(img, row, col, value);
  elseif (cvtypeval(img) == CV_32SC3)
    @cxx at_v3i(img, row, col, value);
  elseif (cvtypeval(img) == CV_32FC3)
    @cxx at_v3f(img, row, col, value);
  elseif (cvtypeval(img) == CV_64FC3)
    @cxx at_v3d(img, row, col, value);
  else throw(ArgumentError("Image format not recognized!"))
  end
end

# Using pointers to efficiently scan and manipulate whole Mat images: set functions
cxx"""
// grayscale/binary image
void setgray(cv::Mat img, int val)
{
    for(int row = 0; row < img.rows; ++row) {
        uchar* p = img.ptr<uchar>(row);
        for(int col = 0; col < img.cols; ++col) {
           //points to each pixel value in turn assuming a CV_8UC1 greyscale image
            *p++ = val;
        }
    }
}

// color image
void setcolor(cv::Mat img, std::vector<int> color)
{
      int rows= img.rows;
      int cols= img.cols;

      for (int row=0; row<rows; row++)
      {
          for (int col=0; col<cols; col++)
          {
             cv::Point3_<uchar>* p = img.ptr<cv::Point3_<uchar>>(row,col);
             p->x = color[0];  //B
             p->y = color[1];  //G
             p->z = color[2];  //R
          }
      }
}
"""

# for grayscale/binary
function setgray(img, val)
  @cxx setgray(img, val)
end

# color/RGB
function setcolor(img, color)
  @cxx setcolor(img, color)
end


