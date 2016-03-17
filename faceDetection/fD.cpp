#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

using namespace cv;
using namespace std;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale);

String cascadeName = "E://OpenCV//sources//data//haarcascades//haarcascade_frontalface_alt2.xml"; //������ѵ������

int main(int argc, char** argv)
{
	Mat image;
	CascadeClassifier cascade, nestedCascade, noseCascade, mouthCascade; //������������������
	double scale = 1.3;

	//image = imread("lena.jpg");

	image = imread("Crowd.jpg");

	if (!image.data)
	{
		cout << "No data!" << endl;
		return -1;
	}

	if (!cascade.load(cascadeName))
	{

		cerr << "Error:Could not load classifier cascade" << endl;
		return -2;
	}

	if (!image.empty())
	{
		detectAndDraw(image, cascade, scale);
		waitKey(0);
	}

	return 0;
}


void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale)
{

	int i = 0; 
	double t = 0;//����ͳ�Ƽ����㷨ִ��ʱ��
	vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };//�ò�ͬ����ɫ��ʾ��ͬ������

	Mat gray;
	Mat smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//��ͼƬ��С���ӿ����ٶ�

	cvtColor(img, gray, CV_BGR2GRAY); //imshow("pic_Gray",gray);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//��С�ߴ���1/scale�������Բ�ֵ
	//imshow("smallImg", smallImg);
	equalizeHist(smallImg, smallImg);//ֱ��ͼ���Ȼ�
	//imshow("ֱ��ͼ���Ȼ�", smallImg);
	t = (double)cvGetTickCount();//�㷨ִ��ǰʱ���¼

	//�������
	//detectMultiScale������smallImg��ʾ����Ҫ��������ͼ��ΪsmallImg��faces��ʾ��⵽������Ŀ�����У�1.1��ʾ
	//ÿ��ͼ��ߴ��С�ı���Ϊ1.1��2��ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�
	//С�����Լ�⵽����),CV_HAAR_SCALE_IMAGE��ʾ�������ŷ���������⣬��������ͼ��Size(30, 30)ΪĿ�����С���ߴ�

	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	t = (double)cvGetTickCount() - t;//���Ϊ�㷨ִ��ʱ��

	printf("detection time = %g ms\n", t / (double)cvGetTickFrequency() * 1000);
	printf("%d faces\n", (int)faces.size());

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		//cv::imshow(resultName[i], img);
		Mat smallImgROI;

		Point center;
		
		Scalar color = colors[i % 8];
		int radius;

		center.x = cvRound((r->x + r->width * 0.5) * scale);//��ԭ��ԭ���Ĵ�С
		center.y = cvRound((r->y + r->height * 0.5) * scale);
		radius = cvRound((r->width + r->height) * 0.25 * scale);
		circle(img, center, radius, color, 3, 8, 0);//��Բ:����Ϊ���ص�ͼ��Բ�ģ��뾶����ɫ����ϸ�����Σ�Բ������Ͱ뾶ֵ�÷���λ

		
		smallImgROI = smallImg(*r);

		
	}
	namedWindow("result", 1);
	cv::imshow("result", img);



}