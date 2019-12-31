#include "opencv2/opencv.hpp"
#include<math.h>
#include<iostream>
#include<string>

using namespace cv;

void fct_fftshift(cv::Mat& src)
{
    int cx = src.cols/2;
    int cy = src.rows/2;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));   
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));  
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));  
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy)); 

    cv::Mat tmp;                           
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2); 
}

//DFTtransform实现
void DFTtransform(cv::Mat& image, cv::Mat& ph, cv::Mat& mag)
{
    image.convertTo(image, CV_32F);
    std::vector<cv::Mat> channels;
    split(image, channels);  //分离RGB通道
    cv::Mat image_B = channels[0];
    //expand input image to optimal size
    int m1 = cv::getOptimalDFTSize(image_B.rows);  //选取最适合做fft的宽和高
    int n1 = cv::getOptimalDFTSize(image_B.cols);
    cv::Mat padded;
    //填充0
    cv::copyMakeBorder(image_B, padded, 0, m1 - image_B.rows, 0, n1 - image_B.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);  //planes[0], planes[1]是实部和虚部

    cv::dft(complexI, complexI, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    cv::split(complexI, planes);

    //定义幅度谱和相位谱
    //cv::Mat ph, mag, idft;
    
    cv::Mat idft;
    cv::phase(planes[0], planes[1], ph);
    cv::magnitude(planes[0], planes[1], mag);  //由实部planes[0]和虚部planes[1]得到幅度谱mag和相位谱ph

    /*
    如果需要对实部planes[0]和虚部planes[1]，或者幅度谱mag和相位谱ph进行操作，在这里进行更改
    */
    //mag = Mat::zeros(mag.rows, mag.cols, mag.type()) + 1;

    //极坐标-笛卡尔变换
    cv::polarToCart(mag, ph, planes[0], planes[1]);  //由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
    cv::merge(planes, 2, idft);
    cv::dft(idft, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    image_B = idft(cv::Rect(0, 0, image.cols & -2, image.rows & -2));
    image_B.copyTo(channels[0]);
    merge(channels, image);
    image.convertTo(image, CV_8U);
}

void DFTtwice(cv::Mat& image, cv::Mat& ph, cv::Mat& mag)
{
    image.convertTo(image, CV_32F);
    std::vector<cv::Mat> channels;
    split(image, channels);  //分离RGB通道
    cv::Mat image_B = channels[0];
    //expand input image to optimal size
    int m1 = cv::getOptimalDFTSize(image_B.rows);  //选取最适合做fft的宽和高
    int n1 = cv::getOptimalDFTSize(image_B.cols);
    cv::Mat padded;
    //填充0
    cv::copyMakeBorder(image_B, padded, 0, m1 - image_B.rows, 0, n1 - image_B.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);  //planes[0], planes[1]是实部和虚部

    cv::dft(complexI, complexI, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    cv::split(complexI, planes);

    //定义幅度谱和相位谱
    //cv::Mat ph, mag, idft;
    
    cv::Mat idft;
    cv::phase(planes[0], planes[1], ph);
    cv::magnitude(planes[0], planes[1], mag);  //由实部planes[0]和虚部planes[1]得到幅度谱mag和相位谱ph

    /*
    如果需要对实部planes[0]和虚部planes[1]，或者幅度谱mag和相位谱ph进行操作，在这里进行更改
    */
    ph = Mat::ones(ph.rows, ph.cols, ph.type()) * 255;

    //极坐标-笛卡尔变换
    cv::polarToCart(mag, ph, planes[0], planes[1]);  //由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
    cv::merge(planes, 2, idft);
    cv::dft(idft, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    image_B = idft(cv::Rect(0, 0, image.cols & -2, image.rows & -2));
    image_B.copyTo(channels[0]);
    merge(channels, image);
    image.convertTo(image, CV_8U);
}

int main()
{    
    //cv::Mat img = cv::imread("../eye3.png", 0);
    //cv::blur(img, img, cv::Size(10, 10));
    int i = 0;
    while(1)
    {
        int n = 112;
        cv::Mat img = Mat::zeros(n, n, CV_8U);
        //cv::rectangle(img, Point(50, 50), Point(70, 70), cv::Scalar(255, 255, 255), 3, 4, 0);
        cv::circle(img, Point(img.cols / 2, img.rows / 2), i, cv::Scalar(255, 255, 255), 1);
        //cv::resize(img, img, cv::Size(112, 112));
        //定义幅度谱mag和相位谱ph
        cv::Mat ph, mag;
        DFTtransform(img, ph, mag);
        cv::imshow("DFT img", img);
        //normalize(ph, ph, 0, 255, NORM_MINMAX);   //归一化 方便显示，和实际数据没有关系
        //normalize(mag, mag, 0, 255, NORM_MINMAX);   //归一化 方便显示，和实际数据没有关系
        //fct_fftshift(ph);
        //fct_fftshift(mag);
        
        //cv::resize(mag, mag, cv::Size(500, 500));
        cv::imshow("ph", ph);
        cv::imshow("mag", mag);
        cv::imwrite("../what?"+std::to_string(i)+".jpg", 255*mag);
        cv::imwrite("../what?"+std::to_string(i)+"_0.jpg", 255*ph);
        
        i++;
        if(i > n*sqrt(2)/2) i = 0;
        
        //cv::imwrite("../ph.jpg", ph);
        if(cv::waitKey(30) == 27)
            break;
    }
    
    /*cv::Mat ph_new, mag_new;
    DFTtwice(ph, ph_new, mag_new);
    fct_fftshift(ph_new);
    fct_fftshift(mag_new);
    cv::imshow("ph_new", ph_new);
    cv::imshow("mag_new", mag_new);
    cv::imshow("ph_changed", ph);
    while(cv::waitKey() != 27)
        continue;*/
    
    return 0;
}




