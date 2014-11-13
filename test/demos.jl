# OpenCV.jl tests (demos)
#################################################################################################

export run_demo

demos = [ "CreateImage",
          "ImageConversion",
          "Thresholding",
          "LiveVideo",
          "setVideoProperties",
          "LiveVideoWithTrackbars",
          "displayHistogram",
          "Drawing",
          "im2dots",
          "ObjectTracking"]

println("\nSelect a demo from the following options: \n")

for i in enumerate(demos)
    println("\t", i[1], ") ", i[2])
end

println("\nSelect and run the demo at the julia prompt: \n","e.g., run_demo(N)\n")

function run_demo(i::Int)
    include(joinpath(Pkg.dir("OpenCV"), string("./test/demo",i,"_",demos[i],".jl")))
end
