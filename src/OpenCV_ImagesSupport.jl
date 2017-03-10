# Preliminary (and partial) support for conversion of images from Images.jl to OpenCV Mat
# Images in Images.jl are encoded as 2d arrays of RGB4{Ufixed8}, Float64, Ufixed8, or Ufixed12

# The following function converts both grayscale and color images to Mat from Image.jl arrays
#   TO_DO: colorspace is NOT correctly transformed
#   TO_DO:  improve algorithm(currently, takes 300 ms for a 512x512 image
#   Use * pointer indexing instead of the "at" method implemented in pixset

function convertToMat(image)
    img = permuteddimsview(channelview(image), (2,3,1))
    cd = Base.size(channelview(image))[1] > 3 ? 1 : 3
    _rows = Base.size(image, 1)
    _cols = Base.size(image, 2)

    if (typeof(img[1,1,1].i) == UInt8)
       if (cd < 3); mat = Mat(_rows, _cols, CV_8UC1); end
       if (cd == 3); mat = Mat(_rows, _cols, CV_8UC3); end
    elseif (typeof(img[1,1,1].i) == Float32)
       if (cd < 3); mat = Mat(_rows, _cols, CV_32FC1); end
       if (cd == 3); mat = Mat(_rows, _cols, CV_32FC3); end
    elseif (typeof(img[1,1,1].i) == Float64)
       if (cd < 3); mat = Mat(_rows, _cols, CV_64FC1); end
       if (cd == 3); mat = Mat(_rows, _cols, CV_64FC3); end
    else
       throw(ArgumentError("Pixel format not supported!"))
    end

    if (cd < 3)   # grayscale or binary
        for j = 1:_rows     # index row first (Mat is row-major order)
            for k =1:_cols  # index column second
                # slow algorithm  - will try to use pointer method (C++)!
                pixset(mat, k, j, float(img[k,j,1].i))
            end
        end
    end

   if (cd == 3)   # color (RGB) image
        for j = 1:_rows     # index row first (Mat is row-major order)
            for k =1:_cols  # index column second
                colorvec = tostdvec([float(img[k,j,1].i),float(img[k,j,2].i),float(img[k,j,3].i)])
                pixset(mat, k-1, j-1, colorvec)   # -1 to have 0-indexing per C++
            end
        end
    end

    return(mat)
end
