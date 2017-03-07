
# Get and set functions for access of pixel values in Mat image arrays

cxx"""
// Grayscale and binary images 
template <typename T1>
inline T1 ptr_val(cv::Mat &img, int row, int col)
{
  return(static_cast<int>(img.at<T1>(row, col)));
}

template <typename T2>
inline void ptr_val(cv::Mat &img, int row, int col, double val)
{
  T2 value = static_cast<T2>(val);
    img.at<T2>(row, col) = value;
}

// RGB images
template <typename T1>
inline std::vector<T1> ptr_val3(cv::Mat &img, int row, int col)
{
    std::vector<T1>vec(3);
    for(int i = 0; i < 3; ++i)
    {
        vec[i] = img.at<cv::Vec<T1, 3>>(row, col)[i];
    }
    return vec;
}

template <typename T2>
inline void ptr_val3(cv::Mat &img, int row, int col, std::vector<double> vec)
{
    for(int i = 0; i < 3; ++i)
    {
        img.at<cv::Vec<T2, 3>>(row, col)[i] = static_cast<T2>(vec[i]);
    }
}

"""

function pixget(img, row::Int, col::Int)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing
  cd = channels(img)
  
  if cd < 3
    # Grayscale and binary images
    val = @cxx ptr_val(img, row, col);
    println(val)
    return (val)
  else
    # RGB images
    vec = @cxx ptr_val3(img, row, col);
    println([int(at(vec, 0)), int(at(vec, 1)), int(at(vec, 2))])
    return (vec)
  end
end


function pixset(img, row::Int, col::Int, value)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing
  cd = channels(img)

  if cd < 3
    # Grayscale and binary images
    @cxx ptr_val(img, row, col, value);
  else
    # RGB images
    @cxx ptr_val3(img, row, col, value);
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
