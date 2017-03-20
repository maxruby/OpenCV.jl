
# Get and set functions for access of pixel values in Mat image arrays
cxx"""

template <typename T>
T getGray(cv::Mat &img, int row, int col, T value)
{
     return(img.template at<T>(row,col));
}
        
template <typename T>
void setGray(cv::Mat &img, int row, int col, T val)
{
     img.template at<T>(row,col) = static_cast<T>(val);
}

template <typename T>
std::vector<T> getRGB(cv::Mat &img, int row, int col, T value)
{
     std::vector<T>vec(3);
     for (int i = 0; i < 3; i++) {
        vec[i]= img.template at<cv::Vec<T,3>>(row,col)[i];
     }

     return vec;
}

template <typename T>
void setRGB(cv::Mat& img, int row, int col, std::vector<T> vec)
{
     for (int i = 0; i < 3; i++) {
        img.template at<cv::Vec<T,3>>(row,col)[i] = static_cast<T>(vec[i]);
     }
}

"""

# Setters using pointers
cxx"""
// grayscale/binary image
template<T>
void setgray(cv::Mat img, T val)
{
    for(int row = 0; row < img.rows; ++row) {
        T* p = img.ptr<T>(row);
        for(int col = 0; col < img.cols; ++col) {
           //points to each pixel value
            *p++ = val;
        }
    }
}

// color image
template <typename T>
void setcolor(cv::Mat img, std::vector<T> color)
{
      int rows= img.rows;
      int cols= img.cols;

      for (int row=0; row<rows; row++)
      {
          for (int col=0; col<cols; col++)
          {
             cv::Point3_<T>* p = img.ptr<cv::Point3_<T>>(row,col);
             p->x = color[0];  //B
             p->y = color[1];  //G
             p->z = color[2];  //R
          }
      }
}
"""

# Julia wrappers
function pixget(img, row::Int, col::Int)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing

  if (channels(img) === 2)
     val = @cxx getGray(img, row, col)
     return val
  elseif (channels(img) === 3)
     val = @cxx getRGB(img, row, col)
     return val
  end

end

function pixset(img, row::Int, col::Int, value)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing

  if (channels(img) === 2)
     val = @cxx setGray(img, row, col, value)
     return val
  elseif (channels(img) === 3)
     if (ndims(value) !== 3) throw(ArgumentError("RGB image input required."))
     val = @cxx setRGB(img, row, col, tostdvec(value))
  end

end

# for grayscale/binary
function setgray(img, val)
  @cxx setgray(img, val)
end

# color/RGB
function setcolor(img, color)
  @cxx setcolor(img, color)
end
