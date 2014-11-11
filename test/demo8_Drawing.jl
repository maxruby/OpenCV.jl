#################################################################################################
#
# demo 8:  Basic drawing example (adapated from OpenCV samples)
#
#################################################################################################

cxx"""
   #include <iostream>
   #include <stdio.h>
"""

cxx"""
static void help()
{
    printf("\nThis program demonstrates OpenCV drawing and text output functions.\n"
    "Usage:\n"
    "   ./drawing\n");
}

static cv::Scalar randomColor(cv::RNG& rng)
{
    int icolor = (unsigned)rng;
    return cv::Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

void drawing_test()
{
    help();
    std::string wndname = "Drawing Demo";
    const int NUMBER = 100;
    const int DELAY = 5;
    int lineType = cv::LINE_AA; // change it to LINE_8 to see non-antialiased graphics
    int i, width = 1000, height = 700;
    int x1 = -width/2, x2 = width*3/2, y1 = -height/2, y2 = height*3/2;
    cv::RNG rng(0xFFFFFFFF);

    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::imshow(wndname, image);
    cv::waitKey(DELAY);

    for (i = 0; i < NUMBER; i++)
    {
        cv::Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);

        cv::line( image, pt1, pt2, randomColor(rng), rng.uniform(1,10), lineType );

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 0; i < NUMBER; i++)
    {
        cv::Point pt1, pt2;
        pt1.x = rng.uniform(x1, x2);
        pt1.y = rng.uniform(y1, y2);
        pt2.x = rng.uniform(x1, x2);
        pt2.y = rng.uniform(y1, y2);
        int thickness = rng.uniform(-3, 10);

        cv::rectangle( image, pt1, pt2, randomColor(rng), MAX(thickness, -1), lineType );

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 0; i < NUMBER; i++)
    {
        cv::Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);
        cv::Size axes;
        axes.width = rng.uniform(0, 200);
        axes.height = rng.uniform(0, 200);
        double angle = rng.uniform(0, 180);

        cv::ellipse( image, center, axes, angle, angle - 100, angle + 200,
                 randomColor(rng), rng.uniform(-1,9), lineType );

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 0; i< NUMBER; i++)
    {
        cv::Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2);
        pt[0][1].y = rng.uniform(y1, y2);
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2);
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2);
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2);
        pt[1][2].y = rng.uniform(y1, y2);
        const cv::Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};

        cv::polylines(image, ppt, npt, 2, true, randomColor(rng), rng.uniform(1,10), lineType);

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 0; i< NUMBER; i++)
    {
        cv::Point pt[2][3];
        pt[0][0].x = rng.uniform(x1, x2);
        pt[0][0].y = rng.uniform(y1, y2);
        pt[0][1].x = rng.uniform(x1, x2);
        pt[0][1].y = rng.uniform(y1, y2);
        pt[0][2].x = rng.uniform(x1, x2);
        pt[0][2].y = rng.uniform(y1, y2);
        pt[1][0].x = rng.uniform(x1, x2);
        pt[1][0].y = rng.uniform(y1, y2);
        pt[1][1].x = rng.uniform(x1, x2);
        pt[1][1].y = rng.uniform(y1, y2);
        pt[1][2].x = rng.uniform(x1, x2);
        pt[1][2].y = rng.uniform(y1, y2);
        const cv::Point* ppt[2] = {pt[0], pt[1]};
        int npt[] = {3, 3};

        cv::fillPoly(image, ppt, npt, 2, randomColor(rng), lineType);

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 0; i < NUMBER; i++)
    {
        cv::Point center;
        center.x = rng.uniform(x1, x2);
        center.y = rng.uniform(y1, y2);

        cv::circle(image, center, rng.uniform(0, 300), randomColor(rng),
               rng.uniform(-1, 9), lineType);

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    for (i = 1; i < NUMBER; i++)
    {
        cv::Point org;
        org.x = rng.uniform(x1, x2);
        org.y = rng.uniform(y1, y2);

        cv::putText(image, "Testing text rendering", org, rng.uniform(0,8),
                rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);

        cv::imshow(wndname, image);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    cv::Size textsize = getTextSize("OpenCV forever!", cv::FONT_HERSHEY_COMPLEX, 3, 5, 0);
    cv::Point org((width - textsize.width)/2, (height - textsize.height)/2);

    cv::Mat image2;
    for( i = 0; i < 255; i += 2 )
    {
        image2 = image - cv::Scalar::all(i);
        cv::putText(image2, "OpenCV forever!", org, cv::FONT_HERSHEY_COMPLEX, 3,
                cv::Scalar(i, i, 255), 5, lineType);

        cv::imshow(wndname, image2);
        if(cv::waitKey(DELAY) >= 0)
            exit(0);
    }

    cv::waitKey();
    exit(0);
}
"""

@cxx drawing_test()
