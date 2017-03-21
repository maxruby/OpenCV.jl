
# Getters and setters to work with template OpenCV cv::Mat_<T> images
cxx"""

/* get pixel value from grayscale images */
template <typename T>
T getPixel(cv::Mat_<T> img, int row, int col) {
    return img(row,col);
}

/* set pixel value in grayscale images */
template <typename T1, typename T2>
void setPixel(cv::Mat_<T1> img, int row, int col, T2 val) {
    img(row,col) = static_cast<T1>(val);
}

/* get pixel value in RGB images */
template <typename T>
std::vector<T> getPixel3d(cv::Mat_<T> img, int row, int col)
{
     std::vector<T>vec(3);
     for (int i = 0; i < 3; i++) {
        vec[i]= img(row,col)[i];
     }

     return vec;
}

/* set pixel value in RGB images */
template <typename T1, typename T2>
void setPixel3d(cv::Mat_<T1> img, int row, int col, std::vector<T2> vec)
{
     for (int i = 0; i < 3; i++) {
        img(row,col)[i] = static_cast<T1>(vec[i]);
     }
}

"""

# Setters for all grayscale and color pixels in cv::Mat images using pointer arithmetic

cxx"""
/* grayscale/binary image */
template<typename T>
void setgray(cv::Mat img, T val)
{
    for(int row = 0; row < img.rows; ++row) {
        T* p = img.template ptr<T>(row);
        for(int col = 0; col < img.cols; ++col) {
           //points to each pixel value
            *p++ = val;
        }
    }
}

/* color image */
template <typename T>
void setcolor(cv::Mat img, std::vector<T> color)
{
    int rows= img.rows;
    int cols= img.cols;

    for (int row=0; row<rows; row++)
    {
        for (int col=0; col<cols; col++)
        {
           cv::Point3_<T>* p = img.template ptr<cv::Point3_<T>>(row,col);
           p->x = color[0];  //B
           p->y = color[1];  //G
           p->z = color[2];  //R
        }
    }
}
"""

# Julia wrapper for getters
function getPixel(img, row::Int, col::Int)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing

  if (channels(img) === 2)
     val = @cxx getPixel(img, row, col)
     return val
  elseif (channels(img) === 3)
     val = @cxx getPixel3d(img, row, col)
     return val
  end
end

# Julia wrapper for setters
function setPixel(img, row::Int, col::Int, value)
  (row < 0 || col < 0 || row > rows(img) || col > cols(img)) ? throw(BoundsError()) : nothing
  if (channels(img) === 2)
     @cxx setPixel(img, row, col, value)
  elseif (channels(img) === 3)
     if (ndims(value) !== 3) throw(ArgumentError("RGB image input required.")) end
     @cxx setPixel3d(img, row, col, tostdvec(value))
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
