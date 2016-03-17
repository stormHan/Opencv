#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

using namespace cv;
using namespace std;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale);

String cascadeName = "E://OpenCV//sources//data//haarcascades//haarcascade_frontalface_alt2.xml"; //人脸的训练数据

int main(int argc, char** argv)
{
	Mat image;
	CascadeClassifier cascade, nestedCascade, noseCascade, mouthCascade; //创建级联分类器对象
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
	double t = 0;//用于统计计算算法执行时间
	vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };//用不同的颜色表示不同的人脸

	Mat gray;
	Mat smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);//将图片缩小，加快检测速度

	cvtColor(img, gray, CV_BGR2GRAY); //imshow("pic_Gray",gray);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//缩小尺寸至1/scale，用线性插值
	//imshow("smallImg", smallImg);
	equalizeHist(smallImg, smallImg);//直方图均匀化
	//imshow("直方图均匀化", smallImg);
	t = (double)cvGetTickCount();//算法执行前时间记录

	//检测人脸
	//detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
	//每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
	//小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的最小最大尺寸

	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	t = (double)cvGetTickCount() - t;//相减为算法执行时间

	printf("detection time = %g ms\n", t / (double)cvGetTickFrequency() * 1000);
	printf("%d faces\n", (int)faces.size());

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		//cv::imshow(resultName[i], img);
		Mat smallImgROI;

		Point center;
		
		Scalar color = colors[i % 8];
		int radius;

		center.x = cvRound((r->x + r->width * 0.5) * scale);//还原成原来的大小
		center.y = cvRound((r->y + r->height * 0.5) * scale);
		radius = cvRound((r->width + r->height) * 0.25 * scale);
		circle(img, center, radius, color, 3, 8, 0);//画圆:参数为承载的图像，圆心，半径，颜色，粗细，线形，圆心坐标和半径值得分数位

		
		smallImgROI = smallImg(*r);

		
	}
	namedWindow("result", 1);
	cv::imshow("result", img);



}