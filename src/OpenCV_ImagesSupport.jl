# Support for conversion of images from Images.jl to OpenCV Mat
# Images in Images.jl are encoded as 2d arrays as follows

# How to use:

# Create a grayscale image
# julia> imgg = rand(Gray{Float32}, 100, 100)
# julia> typeof(imgg)
#   100×100 Array{Gray{Float32},2}
# julia> imggv = channelview(imgg)
# julia> typeof(imggv)
#   Array{Float32,2}

# Create a color image
# julia> imgc = rand(RGB{Float32}, 300, 300)
# julia> typeof(imgc)
#   Array{ColorTypes.RGB{Float32},2}
# Convert to standard Array{Float32,3} before conversion to Mat
# julia> jlimg = channelview(imgc)
# julia> typeof(jlimg)
#   Array{Float32,3}

# The following function converts both grayscale and color image arrays to Mat
using Images

# convert julia vector to Mat using template
function arrToMat{T, N}(jl_array::Array{T, N})

    jlImg = channelview(jl_array)
    dims = ndims(jlImg)
    vec = tostdvec(jl_array)

    if (dims === 2)
        mat2Img = stdvec2Mat(vec)
        return mat2Img
    elseif (dims === 3)
        mat3Img = stdvec3Mat(vec)
        return mat3Img
    else
        throw(ArgumentError("Array format not supported"))
    end

end
