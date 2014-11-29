using OpenCV
using Base.Test

function run_tests()
    include(joinpath(Pkg.dir("OpenCV"), "./test/cxx/demos.jl"))
end

# create an image
# webstream
