Load images sequentially from the path and show them [(link)] (http://answers.opencv.org/question/43284/get-image-on-same-dialog-box-in-opencv-30/)

```c++
// Loading images from path sequentially
for (std::size_t i = 0, i < imgsPaths.size(); i++) 
// if you have a vector of images' paths
{
  cv::Mat img = cv::imread(imgsPaths[i]);
  cv::imshow("same window", img);
  cv::waitKey(); 
  // for seeing the displayed image until key pressed
}
```
Show multiple images in a single window
```c++
cv::Mat img1 = cv::imread("../img1.png");
cv::Mat img2 = cv::imread("../img2.png");
cv::Mat img3;
img3.push_back(img1);
img3.push_back(img2);
cv::imshow("img3", img3);
cv::waitKey();
```

