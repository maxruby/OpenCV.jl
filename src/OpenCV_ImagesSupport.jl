# Support for conversion of images from Images.jl to template OpenCV cv::Mat_<T>

using Images
using Colors

# convert julia vector to cv::Mat_<T> using template
function jltoMat{T, N}(jl_array::Array{T, N})

    jlImg = channelview(jl_array)
    dims = ndims(jlImg)
    vec = tostdvec(jl_array)

    if (dims === 2)
        mat2Img = stdvec2Mat_(vec)
        return mat2Img
    elseif (dims === 3)
        mat3Img = stdvec3Mat_(vec)
        return mat3Img
    else
        throw(ArgumentError("Array format not supported"))
    end

end
