#################################################################################################
#
# demo 9: im2dots with OpenCV.jl
# Photoshop processing effect - overlay of small circles over image
# https://opencv-code.com/tutorials/photo-to-colored-dot-patterns-with-opencv/#more-754
#################################################################################################

# Julia filename with full path
inputfile =  joinpath(Pkg.dir("OpenCV"), "./test/images/julia.png")
outfile = joinpath(Pkg.dir("OpenCV"), "./test/images/dotjulia.png")

cxx"""
void im2dots(const char *inputfile, const char *outfile)
{
    const std::string ifname = inputfile;
    const std::string ofname = outfile;

    cv::Mat src = cv::imread(ifname);
    if (!src.data) {
        exit(0);
    }

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::Mat cir = cv::Mat::zeros(src.size(), CV_8UC1);
    int bsize = 10;

    for (int i = 0; i < src.rows; i += bsize)
    {
        for (int j = 0; j < src.cols; j += bsize)
        {
            cv::Rect rect = cv::Rect(j, i, bsize, bsize) &
                            cv::Rect(0, 0, src.cols, src.rows);

            cv::Mat sub_dst(dst, rect);
            sub_dst.setTo(cv::mean(src(rect)));

            cv::circle(
                cir,
                cv::Point(j+bsize/2, i+bsize/2),
                bsize/2-1,
                cv::Scalar(200,200,200), -1, 2
            );
        }
    }

    cv::Mat cir_32f;
    cir.convertTo(cir_32f, CV_32F);
    cv::normalize(cir_32f, cir_32f, 0, 1, cv::NORM_MINMAX);

    cv::Mat dst_32f;
    dst.convertTo(dst_32f, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(dst_32f, channels);
    for (int i = 0; i < channels.size(); ++i)
        channels[i] = channels[i].mul(cir_32f);

    cv::merge(channels, dst_32f);
    dst_32f.convertTo(dst, CV_8U);

    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::imwrite(ofname, dst);
    cv::destroyAllWindows();
}
"""

@cxx im2dots(pointer(inputfile), pointer(outfile))
