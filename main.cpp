
#include "opencv2/core/core.hpp"

#include "opencv2/objdetect/objdetect.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include <stdio.h>

using namespace std;

using namespace cv;

string kid_cascade_name = "G://cascade.xml";

//���ļ�������OpenCV��װĿ¼�µ�\sources\data\haarcascades�ڣ���Ҫ����xml�ļ����Ƶ���ǰ����Ŀ¼��
CascadeClassifier kid_cascade;

//����Ϊȫ�ֱ���
std::vector<Rect> hog;
Mat kid_gray;
//Mat imageLog(kid_gray.size(), CV_32FC3);
Mat image;

void detectAndDisplay(Mat frame);
int download_avi(Mat frame);

int main(int argc, char** argv) {
	////cascater test for image

	//��̬ͼƬʶ��
	//Mat image;

	//image = imread("F://img//kids//kid2.jpg");  //��ǰ���̵�imageĿ¼�µ�mm.jpg�ļ���ע��Ŀ¼����

	//if (!kid_cascade.load(kid_cascade_name)) {

	//	printf("�������������󣬿���δ�ҵ��ļ����������ļ�������Ŀ¼�£�\n");

	//	return -1;

	//}

	//detectAndDisplay(image); //����������⺯��

	//waitKey(0);

	//��Ƶʶ�𲿷�
		Mat frame;
		VideoCapture capture("G://MyVideo_3.avi");
		//VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
		//capture.open("G:/video.avi");
		if (!capture.isOpened()) {
			printf("could not load video data...\n");
			return -1;
		}
		if (!kid_cascade.load(kid_cascade_name)) {

			printf("�������������󣬿���δ�ҵ��ļ����������ļ�������Ŀ¼�£�\n");

			return -1;

		}
		//������Ƶ��д��
		VideoWriter write;
		string outVideo = "Show_Example3.avi";
		int fps = 30;//֡��double fps=cap.get(CV_CAP_PROP_FPS)�������ҵ��øú������ص���0
		int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));//��ȡ��Ƶ�ĸ߶�
		int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));//��ȡ��Ƶ�Ŀ��
																	   //Ŀǰopencv�Ա�����Ƶ��֧�ָ�ʽ���Ǻܶ࣬�ҳ����˺ܶ��֣������и�ʽ�ͱ��뷽ʽ�Ĳ��䡣������ʹ�õ���.avi��ʽ�����뷽ʽѡ��XVID
		write.open(outVideo, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height), true);

		while (capture.read(image)) {
			imshow("input video", image);	
		
			Mat frame;
		detectAndDisplay(image); //����������⺯��
		write.write(image);

		char c = waitKey(10);
		if (c == 27) {
			break;
		}
		}
		waitKey(0);
		system("pause");
		return 0;
		//��ͣ��ʾһ�¡�
	
}
//	cascater test for video
//}
//���ش���õ���Ƶ
int download_avi(Mat frame)
{
	int c = 0;
	VideoCapture cap("G://MyVideo_1.avi");//��������ͷ������ж������ͷ�����Ե���c��0,1,2��...����ѡ������Ҫ������ͷ��
	if (!cap.isOpened())
	{
		return -1;
	}
	bool stop = false;
	int a = 0;
	//������Ƶ��д��
	VideoWriter write;
	string outVideo = "Show_Example.avi";
	int fps = 30;//֡��double fps=cap.get(CV_CAP_PROP_FPS)
	int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));//��ȡ��Ƶ�ĸ߶�
	int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));//��ȡ��Ƶ�Ŀ��
														
	write.open(outVideo, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height), true);
	while (!stop)
	{
		Mat frame;
		cap >> frame;
		blur(frame, frame, Size(3, 3), Point(-1, -1));
		imshow("��ǰ��Ƶ", frame);
		write.write(frame);
		if (waitKey(30) >= 0)
			stop = true;
	}
	return 0;
}

void detectAndDisplay(Mat face) {

	//std::vector<Rect> hog;
	//Mat kid_gray;

	//������˹������ǿ
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(face, kid_gray, CV_8UC3, kernel);

	Mat imageLog(kid_gray.size(), CV_32FC3);
	//���ڶ���logͼ����ǿ����
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
	//��һ����0~255    
	normalize(imageLog, imageLog, 0, 255, CV_MINMAX);
	//ת����8bitͼ����ʾ    
	convertScaleAbs(imageLog, imageLog);

	//rgb����ת��Ϊ�Ҷ�����
	cvtColor(imageLog, kid_gray, CV_BGR2GRAY);

	//ֱ��ͼ���⻯
	equalizeHist(kid_gray, kid_gray);

	////���͸�ʴ
	Mat structureElement = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	dilate(kid_gray, kid_gray, structureElement, Point(-1, -1), 1, 1);//���Ͳ���
	erode(kid_gray, kid_gray, structureElement);//��ʴ����

	kid_cascade.detectMultiScale(kid_gray, hog, 1.5, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(200, 200));

	for (int i = 0; i < hog.size(); i++) {

		Point center(hog[i].x + hog[i].width*0.5, hog[i].y + hog[i].height*0.5);

		ellipse(face, center, Size(hog[i].width*0.5, hog[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2,7, 0);
		
	}
	namedWindow("hog_dect", CV_WINDOW_AUTOSIZE);
	//namedWindow("hog_dect", CV_WINDOW_NORMAL);
	imshow("hog_dect", face);

}

