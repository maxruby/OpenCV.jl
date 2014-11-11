using OpenCV
using Base.Test

function run_tests()
    include(joinpath(Pkg.dir("OpenCV"), "./test/demos.jl"))
end
