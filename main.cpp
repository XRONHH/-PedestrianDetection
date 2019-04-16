
#include "opencv2/core/core.hpp"

#include "opencv2/objdetect/objdetect.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include <stdio.h>

using namespace std;

using namespace cv;

string kid_cascade_name = "G://cascade.xml";

//该文件存在于OpenCV安装目录下的\sources\data\haarcascades内，需要将该xml文件复制到当前工程目录下
CascadeClassifier kid_cascade;

//设置为全局变量
std::vector<Rect> hog;
Mat kid_gray;
//Mat imageLog(kid_gray.size(), CV_32FC3);
Mat image;

void detectAndDisplay(Mat frame);
int download_avi(Mat frame);

int main(int argc, char** argv) {
	////cascater test for image

	//静态图片识别
	//Mat image;

	//image = imread("F://img//kids//kid2.jpg");  //当前工程的image目录下的mm.jpg文件，注意目录符号

	//if (!kid_cascade.load(kid_cascade_name)) {

	//	printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");

	//	return -1;

	//}

	//detectAndDisplay(image); //调用人脸检测函数

	//waitKey(0);

	//视频识别部分
		Mat frame;
		VideoCapture capture("G://MyVideo_3.avi");
		//VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
		//capture.open("G:/video.avi");
		if (!capture.isOpened()) {
			printf("could not load video data...\n");
			return -1;
		}
		if (!kid_cascade.load(kid_cascade_name)) {

			printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");

			return -1;

		}
		//设置视频编写器
		VideoWriter write;
		string outVideo = "Show_Example3.avi";
		int fps = 30;//帧率double fps=cap.get(CV_CAP_PROP_FPS)；但是我调用该函数返回的是0
		int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));//读取视频的高度
		int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));//读取视频的宽度
																	   //目前opencv对保存视频的支持格式不是很多，我尝试了很多种，后面有格式和编码方式的补充。我这里使用的是.avi格式，编码方式选择XVID
		write.open(outVideo, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height), true);

		while (capture.read(image)) {
			imshow("input video", image);	
		
			Mat frame;
		detectAndDisplay(image); //调用人脸检测函数
		write.write(image);

		char c = waitKey(10);
		if (c == 27) {
			break;
		}
		}
		waitKey(0);
		system("pause");
		return 0;
		//暂停显示一下。
	
}
//	cascater test for video
//}
//下载处理好的视频
int download_avi(Mat frame)
{
	int c = 0;
	VideoCapture cap("G://MyVideo_1.avi");//调用摄像头，如果有多个摄像头，可以调整c（0,1,2，...）来选择到你想要的摄像头，
	if (!cap.isOpened())
	{
		return -1;
	}
	bool stop = false;
	int a = 0;
	//设置视频编写器
	VideoWriter write;
	string outVideo = "Show_Example.avi";
	int fps = 30;//帧率double fps=cap.get(CV_CAP_PROP_FPS)
	int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));//读取视频的高度
	int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));//读取视频的宽度
														
	write.open(outVideo, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height), true);
	while (!stop)
	{
		Mat frame;
		cap >> frame;
		blur(frame, frame, Size(3, 3), Point(-1, -1));
		imshow("当前视频", frame);
		write.write(frame);
		if (waitKey(30) >= 0)
			stop = true;
	}
	return 0;
}

void detectAndDisplay(Mat face) {

	//std::vector<Rect> hog;
	//Mat kid_gray;

	//拉普拉斯算子增强
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(face, kid_gray, CV_8UC3, kernel);

	Mat imageLog(kid_gray.size(), CV_32FC3);
	//基于对数log图像增强操作
	for (int i = 0; i < kid_gray.rows; i++)
	{
		for (int j = 0; j < kid_gray.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = log(1 + kid_gray.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + kid_gray.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + kid_gray.at<Vec3b>(i, j)[2]);
		}
	}

	for (int i = 0; i < kid_gray.rows; i++)
	{
		for (int j = 0; j < kid_gray.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = (kid_gray.at<Vec3b>(i, j)[0])*(kid_gray.at<Vec3b>(i, j)[0])*(kid_gray.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = (kid_gray.at<Vec3b>(i, j)[1])*(kid_gray.at<Vec3b>(i, j)[1])*(kid_gray.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = (kid_gray.at<Vec3b>(i, j)[2])*(kid_gray.at<Vec3b>(i, j)[2])*(kid_gray.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255    
	normalize(imageLog, imageLog, 0, 255, CV_MINMAX);
	//转换成8bit图像显示    
	convertScaleAbs(imageLog, imageLog);

	//rgb类型转换为灰度类型
	cvtColor(imageLog, kid_gray, CV_BGR2GRAY);

	//直方图均衡化
	equalizeHist(kid_gray, kid_gray);

	////膨胀腐蚀
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	dilate(kid_gray, kid_gray, structureElement, Point(-1, -1), 1, 1);//膨胀操作
	erode(kid_gray, kid_gray, structureElement);//腐蚀操作

	kid_cascade.detectMultiScale(kid_gray, hog, 1.5, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(200, 200));

	for (int i = 0; i < hog.size(); i++) {

		Point center(hog[i].x + hog[i].width*0.5, hog[i].y + hog[i].height*0.5);

		ellipse(face, center, Size(hog[i].width*0.5, hog[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2,7, 0);
		
	}
	namedWindow("hog_dect", CV_WINDOW_AUTOSIZE);
	//namedWindow("hog_dect", CV_WINDOW_NORMAL);
	imshow("hog_dect", face);

}

