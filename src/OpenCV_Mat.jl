
# Get and set functions for access of pixel values in Mat image arrays

cxx"""
// Grayscale and binary images 
template <typename T>
inline T ptr_val(cv::Mat &img, int row, int col)
{
  return(static_cast<int>(img.at<T>(row, col)));
}

template <typename T>
inline void ptr_val(cv::Mat &img, int row, int col, double val)
{
  T value = static_cast<T>(val);
    img.at<T>(row, col) = value;
}

// RGB images
template <typename T>
inline std::vector<T> ptr_val3(cv::Mat &img, int row, int col)
{
    std::vector<T>vec(3);
    for(int i = 0; i < 3; ++i)
    {
        vec[i] = img.at<cv::Vec<T, 3>>(row, col)[i];
    }
    return vec;
}

template <typename T>
inline void ptr_val3(cv::Mat &img, int row, int col, std::vector<double> vec)
{
    for(int i = 0; i < 3; ++i)
    {
        img.at<cv::Vec<T, 3>>(row, col)[i] = static_cast<T>(vec[i]);
    }
}

"""

function pixget(img, row::Int, col::Int)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothingcpp_templates
  
  # Grayscale and binary images
  if (cvtypeval(img) == CV_8UC1)
    val = @cxx ptr_val<unsigned char>(img, row, col);
  elseif (cvtypeval(img) == CV_16SC1)
    val = @cxx ptr_val<short>(img, row, col);
  elseif (cvtypeval(img) == CV_32SC1)
    val = @cxx ptr_val<unsigned short>(img, row, col);
  elseif (cvtypeval(img) == CV_32FC1)
    val = @cxx ptr_val<float>(img, row, col);
  elseif (cvtypeval(img) == CV_64FC1)
    val = @cxx ptr_val<double>(img, row, col);
  # RGB images
  elseif (cvtypeval(img) == CV_8SC3)
    vec = @cxx ptr_val3<char>(img, row, col);
  elseif (cvtypeval(img) == CV_8UC3)
    vec = @cxx ptr_val3<unsigned char>(img, row, col);
  elseif (cvtypeval(img) == CV_16SC3)
    vec = @cxx ptr_val3<short>(img, row, col);
  elseif (cvtypeval(img) == CV_16UC3)
    vec = @cxx ptr_val3<unsigned short>(img, row, col);
  elseif (cvtypeval(img) == CV_32SC3)
    vec = @cxx ptr_val3<int>(img, row, col);
  elseif (cvtypeval(img) == CV_32FC3)
    vec = @cxx ptr_val3<float>(img, row, col);
  elseif (cvtypeval(img) == CV_64FC3)
    vec = @cxx ptr_val3<double>(img, row, col);
  else throw(ArgumentError("Image format not recognized!"))
  end
end


function pixset(img, row::Int, col::Int, value)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing
  
  # Grayscale and binary images
  if (cvtypeval(img) == CV_8UC1)
    @cxx ptr_val<unsigned char>(img, row, col, value);
  elseif (cvtypeval(img) == CV_16SC1)
    @cxx ptr_val<short>(img, row, col, value);
  elseif (cvtypeval(img) == CV_32SC1)
    @cxx ptr_val<unsigned short>(img, row, col, value);
  elseif (cvtypeval(img) == CV_32FC1)
    @cxx ptr_val<float>(img, row, col, value);
  elseif (cvtypeval(img) == CV_64FC1)
    @cxx ptr_val<double>(img, row, col, value);
  # RGB images
  elseif (cvtypeval(img) == CV_8SC3)
    @cxx ptr_val3<char>(img, row, col, value);
  elseif (cvtypeval(img) == CV_8UC3)
    @cxx ptr_val3<unsigned char>(img, row, col, value);
  elseif (cvtypeval(img) == CV_16SC3)
    @cxx ptr_val3<short>(img, row, col, value);
  elseif (cvtypeval(img) == CV_16UC3)
    @cxx ptr_val3<unsigned short>(img, row, col, value);
  elseif (cvtypeval(img) == CV_32SC3)
    @cxx ptr_val3<int>(img, row, col, value);
  elseif (cvtypeval(img) == CV_32FC3)
    @cxx ptr_val3<float>(img, row, col, value);
  elseif (cvtypeval(img) == CV_64FC3)
    @cxx ptr_val3<double>(img, row, col, value);
  else throw(ArgumentError("Image format not recognized!"))
  end
end

# Using pointers to efficiently scan and manipulate whole Mat images: set functions
cxx"""
// grayscale/binary image
void setgray(cv::Mat img, int val)
{
    for(int row=0; row < img.rows; ++row) {
        uchar* p = img.ptr<uchar>(row);
        for(int col=0; col < img.cols; ++col){
            *p++ = val;
        }
    }
}

// color image
void setcolor(cv::Mat img, std::vector<int> color)
{
    int rows = img.rows;
    int cols = img.cols;

    for(int row=0; row < rows; ++row){
        for(int col=0; col < cols; ++col){
            cv::Point3_<uchar>* p = img.ptr<cv::Point3_<uchar>>(row, col);
            p->x = color[0]; //B
            p->y = color[1]; //G
            p->z = color[2]; //R
        }
    }
}
"""

# for grayscale / binary
function setgray(img, val)
  @cxx setgray(img, val)
end

# color / RGB
function setcolor(img, color)
  @cxx setcolor(img,  color)
end
