#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include<fstream>
#include<sstream>
#include<vector>
#include<iostream>
#include<iomanip>
using namespace std;
using namespace cv;

#define ROWS 2848
#define COLS 4272
#define PIXSIZE 0.00519663

struct ControlPoint
{
	int flag;
	double X;
	double Y;
	double Z;
};

struct imgPoint
{
	int flag;
	double x;
	double y;
};

struct PointPair
{
	int flag;
	double X;
	double Y;
	double Z;
	double x;
	double y;
};

//外方位元素结构
struct orienParam {
	//内方位元素
	double x0 = 0;
	double y0 = 0;
	double f = 0;
	//畸变系数
	double k1 = 0;
	double k2 = 0;
	double p1 = 0;
	double p2 = 0;
	//外方位元素
	double Phi = 0;
	double Omega = 0;
	double Kappa = 0;
	double Xs = 0;
	double Ys = 0;
	double Zs = 0;
	//比例尺不一系数
	double dp = 0;
	double ds = 0;

};

void readClptData(char* file, vector<ControlPoint>& ControlPoints)
{
	ifstream inFile(file);
	if (!inFile)
	{
		cout << "文件读取失败，请检查文件路径" << endl;
		return;
	}
	string line;
	string firstLine;
	getline(inFile, firstLine);
	while (getline(inFile, line)) {
		ControlPoint clpt;
		istringstream iss(line);
		iss >> clpt.flag;
		iss >> clpt.Z;
		iss >> clpt.X;
		iss >> clpt.Y;
		clpt.Z = -1.0 * clpt.Z;
		ControlPoints.push_back(clpt);
	}
	inFile.close();
}

void readImgData(char* file, vector<imgPoint>& imgPoints)
{
	ifstream inFile(file);
	if (!inFile)
	{
		cout << "文件读取失败，请检查文件路径" << endl;
		return;
	}
	string line;
	while (getline(inFile, line)) {
		imgPoint imgpt;
		istringstream iss(line);
		iss >> imgpt.flag;
		iss >> imgpt.x;
		iss >> imgpt.y;
		imgPoints.push_back(imgpt);
	}
	inFile.close();
}

void cvtPix2Img(vector<imgPoint>& imgPoints)
{
	for (int i = 0; i < imgPoints.size(); i++)
	{
		imgPoints[i].x = (imgPoints[i].x - COLS / 2) * PIXSIZE;
		imgPoints[i].y = -1.0 * (imgPoints[i].y - ROWS / 2) * PIXSIZE;
	}
}

void GeneratePairs(vector<imgPoint>& imgPoints, vector<ControlPoint>& ControlPoints, vector<PointPair>& PointPairs)
{
	PointPair pair;
	for (int i = 0; i < imgPoints.size(); i++)
	{
		for (int j = 0; j < ControlPoints.size(); j++)
		{
			if (imgPoints[i].flag == ControlPoints[j].flag)
			{
				pair.flag = imgPoints[i].flag;
				pair.X = ControlPoints[j].X;
				pair.Y = ControlPoints[j].Y;
				pair.Z = ControlPoints[j].Z;
				pair.x = imgPoints[i].x;
				pair.y = imgPoints[i].y;
				PointPairs.push_back(pair);
				break;
			}
		}
	}
}

Mat cal_L_RoughValue(vector<PointPair> selectedPairs)
{
	//列方程（舍弃一个方程）
	Mat A = Mat::zeros(2 * selectedPairs.size(), 11, CV_64FC1);
	Mat Li = Mat::zeros(11, 1, CV_64FC1);
	Mat C = Mat::zeros(2 * selectedPairs.size(), 1, CV_64FC1);
	//填充系数阵和常数阵
	for (int i = 0; i < selectedPairs.size(); i++)
	{
		A.at<double>(i * 2, 0) = selectedPairs[i].X;
		A.at<double>(i * 2, 1) = selectedPairs[i].Y;
		A.at<double>(i * 2, 2) = selectedPairs[i].Z;
		A.at<double>(i * 2, 3) = 1.0;
		A.at<double>(i * 2, 4) = 0.0;
		A.at<double>(i * 2, 5) = 0.0;
		A.at<double>(i * 2, 6) = 0.0;
		A.at<double>(i * 2, 7) = 0.0;
		A.at<double>(i * 2, 8) = selectedPairs[i].x * selectedPairs[i].X;
		A.at<double>(i * 2, 9) = selectedPairs[i].x * selectedPairs[i].Y;
		A.at<double>(i * 2, 10) = selectedPairs[i].x * selectedPairs[i].Z;
		A.at<double>(i * 2 + 1, 0) = 0.0;
		A.at<double>(i * 2 + 1, 1) = 0.0;
		A.at<double>(i * 2 + 1, 2) = 0.0;
		A.at<double>(i * 2 + 1, 3) = 0.0;
		A.at<double>(i * 2 + 1, 4) = selectedPairs[i].X;
		A.at<double>(i * 2 + 1, 5) = selectedPairs[i].Y;
		A.at<double>(i * 2 + 1, 6) = selectedPairs[i].Z;
		A.at<double>(i * 2 + 1, 7) = 1.0;
		A.at<double>(i * 2 + 1, 8) = selectedPairs[i].y * selectedPairs[i].X;
		A.at<double>(i * 2 + 1, 9) = selectedPairs[i].y * selectedPairs[i].Y;
		A.at<double>(i * 2 + 1, 10) = selectedPairs[i].y * selectedPairs[i].Z;
		
		C.at<double>(i * 2, 0) = selectedPairs[i].x;
		C.at<double>(i * 2 + 1, 0) = selectedPairs[i].y;
	}
	//cout << A << endl;
	//求解
	Li = (A.t() * A).inv() * A.t() * C;
	return Li;
}

void cal_InteriorParams(Mat Li, orienParam& orien)
{
	double l1 = Li.at<double>(0, 0);
	double l2 = Li.at<double>(1, 0);
	double l3 = Li.at<double>(2, 0);
	double l5 = Li.at<double>(4, 0);
	double l6 = Li.at<double>(5, 0);
	double l7 = Li.at<double>(6, 0);
	double l9 = Li.at<double>(8, 0);
	double l10 = Li.at<double>(9, 0);
	double l11 = Li.at<double>(10, 0);
	orien.x0 = -1.0 * (l1 * l9 + l2 * l10 + l3 * l11) / (l9 * l9 + l10 * l10 + l11 * l11);
	orien.y0 = -1.0 * (l5 * l9 + l6 * l10 + l7 * l11) / (l9 * l9 + l10 * l10 + l11 * l11);
	double A = (l1 * l1 + l2 * l2 + l3 * l3) / (l9 * l9 + l10 * l10 + l11 * l11) - orien.x0 * orien.x0;
	double B = (l5 * l5 + l6 * l6 + l7 * l7) / (l9 * l9 + l10 * l10 + l11 * l11) - orien.y0 * orien.y0;
	double C = (l1 * l5 + l2 * l6 + l3 * l7) / (l9 * l9 + l10 * l10 + l11 * l11) - orien.x0 * orien.y0;
	orien.f = sqrt((A * B - C * C) / B);
	double dp = asin(sqrt(C * C / (A * B)));
	orien.ds = sqrt(A / B) - 1;
	if (dp * C > 0)
	{
		orien.dp = -1.0 * dp;
		return;
	}
	orien.dp = dp;
}

Mat cal_unpt_RoughValue(imgPoint left_pair, imgPoint right_pair, Mat left_Li, Mat right_Li)
{
	Mat unpt_RoughValue = Mat::zeros(3, 1, CV_64FC1);
	Mat A = Mat::zeros(4, 3, CV_64FC1);
	Mat C = Mat::zeros(4, 1, CV_64FC1);
	//填充系数阵和常数阵
	A.at<double>(0, 0) = left_Li.at<double>(0, 0) + left_pair.x * left_Li.at<double>(8, 0);
	A.at<double>(0, 1) = left_Li.at<double>(1, 0) + left_pair.x * left_Li.at<double>(9, 0);
	A.at<double>(0, 2) = left_Li.at<double>(2, 0) + left_pair.x * left_Li.at<double>(10, 0);
	A.at<double>(1, 0) = left_Li.at<double>(4, 0) + left_pair.y * left_Li.at<double>(8, 0);
	A.at<double>(1, 1) = left_Li.at<double>(5, 0) + left_pair.y * left_Li.at<double>(9, 0);
	A.at<double>(1, 2) = left_Li.at<double>(6, 0) + left_pair.y * left_Li.at<double>(10, 0);
	A.at<double>(2, 0) = right_Li.at<double>(0, 0) + right_pair.x * right_Li.at<double>(8, 0);
	A.at<double>(2, 1) = right_Li.at<double>(1, 0) + right_pair.x * right_Li.at<double>(9, 0);
	A.at<double>(2, 2) = right_Li.at<double>(2, 0) + right_pair.x * right_Li.at<double>(10, 0);
	A.at<double>(3, 0) = right_Li.at<double>(4, 0) + right_pair.y * right_Li.at<double>(8, 0);
	A.at<double>(3, 1) = right_Li.at<double>(5, 0) + right_pair.y * right_Li.at<double>(9, 0);
	A.at<double>(3, 2) = right_Li.at<double>(6, 0) + right_pair.y * right_Li.at<double>(10, 0);

	C.at<double>(0, 0) = -1.0 * (left_Li.at<double>(3, 0) + left_pair.x);
	C.at<double>(1, 0) = -1.0 * (left_Li.at<double>(7, 0) + left_pair.y);
	C.at<double>(2, 0) = -1.0 * (right_Li.at<double>(3, 0) + right_pair.x);
	C.at<double>(3, 0) = -1.0 * (right_Li.at<double>(7, 0) + right_pair.y);

	unpt_RoughValue = (A.t() * A).inv() * A.t() * C;
	return unpt_RoughValue;
}

void cal_L_AccurateValue(Mat Li, Mat& A, Mat& l, orienParam& orien, vector<PointPair> pairs)
{
	double l1 = Li.at<double>(0, 0);
	double l2 = Li.at<double>(1, 0);
	double l3 = Li.at<double>(2, 0);
	double l5 = Li.at<double>(4, 0);
	double l6 = Li.at<double>(5, 0);
	double l7 = Li.at<double>(6, 0);
	double l9 = Li.at<double>(8, 0);
	double l10 = Li.at<double>(9, 0);
	double l11 = Li.at<double>(10, 0);
	double f = orien.f;
	double x0 = orien.x0;
	double y0 = orien.y0;
	for (int i = 0; i < pairs.size(); i++)
	{
		double X = pairs[i].X;
		double Y = pairs[i].Y;
		double Z = pairs[i].Z;
		double x = pairs[i].x;
		double y = pairs[i].y;
		double a = l9 * pairs[i].X + l10 * pairs[i].Y + l11 * pairs[i].Z + 1;
		double r_2 = pow(x - x0, 2) + pow(y - y0, 2);
		A.at<double>(2 * i, 0) = -1.0 * X / a;
		A.at<double>(2 * i, 1) = -1.0 * Y / a;
		A.at<double>(2 * i, 2) = -1.0 * Z / a;
		A.at<double>(2 * i, 3) = -1.0 * 1 / a;
		A.at<double>(2 * i, 4) = 0;
		A.at<double>(2 * i, 5) = 0;
		A.at<double>(2 * i, 6) = 0;
		A.at<double>(2 * i, 7) = 0;
		A.at<double>(2 * i, 8) = -1.0 * x * X / a;
		A.at<double>(2 * i, 9) = -1.0 * x * Y / a;
		A.at<double>(2 * i, 10) = -1.0 * x * Z / a;
		A.at<double>(2 * i, 11) = -1.0 * (x - x0) * r_2;
		A.at<double>(2 * i, 12) = -1.0 * (x - x0) * r_2 * r_2;
		A.at<double>(2 * i, 13) = -1.0 * (r_2 + 2 * (x - x0) * (x - x0));
		A.at<double>(2 * i, 14) = -2.0 * (x - x0) * (y - y0);

		A.at<double>(2 * i + 1, 0) = 0.0;
		A.at<double>(2 * i + 1, 1) = 0.0;
		A.at<double>(2 * i + 1, 2) = 0.0;
		A.at<double>(2 * i + 1, 3) = 0.0;
		A.at<double>(2 * i + 1, 4) = -1.0 * X / a;
		A.at<double>(2 * i + 1, 5) = -1.0 * Y / a;
		A.at<double>(2 * i + 1, 6) = -1.0 * Z / a;
		A.at<double>(2 * i + 1, 7) = -1.0 * 1 / a;
		A.at<double>(2 * i + 1, 8) = -1.0 * y * X / a;
		A.at<double>(2 * i + 1, 9) = -1.0 * y * Y / a;
		A.at<double>(2 * i + 1, 10) = -1.0 * y * Z / a;
		A.at<double>(2 * i + 1, 11) = -1.0 * (y - y0) * r_2;
		A.at<double>(2 * i + 1, 12) = -1.0 * (y - y0) * r_2 * r_2;
		A.at<double>(2 * i + 1, 13) = -2.0 * (y - y0) * (x - x0);
		A.at<double>(2 * i + 1, 14) = -1.0 * (r_2 + 2 * (y - y0) * (y - y0));

		l.at<double>(2 * i, 0) = x / a;
		l.at<double>(2 * i + 1, 0) = y / a;
	}
}

void modifyImgPoints(vector<imgPoint>& PointPairs, orienParam orien)
{
	for (int i = 0; i < PointPairs.size(); i++)
	{
		double x = PointPairs[i].x;
		double y = PointPairs[i].y;
		double x0 = orien.x0;
		double y0 = orien.y0;
		double k1 = orien.k1;
		double k2 = orien.k2;
		double p1 = orien.p1;
		double p2 = orien.p2;
		double r_2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
		PointPairs[i].x = x + (x - x0) * (k1 * r_2 + k2 * r_2 * r_2) + p1 * (r_2 + 2 * (x - x0) * (x - x0)) + 2 * p2 * (x - x0) * (y - y0);
		PointPairs[i].y = y + (y - y0) * (k1 * r_2 + k2 * r_2 * r_2) + p2 * (r_2 + 2 * (y - y0) * (y - y0)) + 2 * p1 * (x - x0) * (y - y0);
	
	}
}

void cal_ExteriorParams(Mat AcLi, orienParam& orien, vector<PointPair> pairs) 
{
	double l1 = AcLi.at<double>(0, 0);
	double l2 = AcLi.at<double>(1, 0);
	double l3 = AcLi.at<double>(2, 0);
	double l4 = AcLi.at<double>(3, 0);
	double l5 = AcLi.at<double>(4, 0);
	double l6 = AcLi.at<double>(5, 0);
	double l7 = AcLi.at<double>(6, 0);
	double l8 = AcLi.at<double>(7, 0);
	double l9 = AcLi.at<double>(8, 0);
	double l10 = AcLi.at<double>(9, 0);
	double l11 = AcLi.at<double>(10, 0);
	Mat A = Mat::zeros(3, 3, CV_64FC1);
	Mat C = Mat::zeros(3, 1, CV_64FC1);
	A.at<double>(0, 0) = l1;
	A.at<double>(0, 1) = l2;
	A.at<double>(0, 2) = l3;
	A.at<double>(1, 0) = l5;
	A.at<double>(1, 1) = l6;
	A.at<double>(1, 2) = l7;
	A.at<double>(2, 0) = l9;
	A.at<double>(2, 1) = l10;
	A.at<double>(2, 2) = l11;

	C.at<double>(0, 0) = -1.0 * l4;
	C.at<double>(1, 0) = -1.0 * l8;
	C.at<double>(2, 0) = -1.0;

	Mat X = A.inv() * C;
	orien.Xs = X.at<double>(0, 0);
	orien.Ys = X.at<double>(1, 0);
	orien.Zs = X.at<double>(2, 0);

	double factor = sqrt(l9 * l9 + l10 * l10 + l11 * l11);
	double a3 = l9 / factor;
	double b3 = l10 / factor;
	double c3 = l11 / factor;

	double x0 = orien.x0;
	double y0 = orien.y0;
	double f = orien.f;
	double ds = orien.ds;
	double dp = orien.dp;
	double b1 = (l2 / factor + b3 * x0 + (y0 * b3 + l6) * (1 + ds) * sin(dp)) / f;
	double b2 = (l6 / factor + b3 * y0) * (1 + ds) * cos(dp) / f;

	orien.Phi = atan(-1.0 * a3 / c3);
	orien.Omega = asin(-1.0 * b3);
	orien.Kappa = atan(b1 / b2);

}


ControlPoint cal_unpt_AccurateValue(Mat left_Li,Mat right_Li ,ControlPoint pointInOS ,imgPoint left_Point, imgPoint right_Point, orienParam left_orien, orienParam right_orien)
{
	Mat A = Mat::zeros(4, 3, CV_64FC1);
	Mat l = Mat::zeros(4, 1, CV_64FC1);
	double l_l1 = left_Li.at<double>(0, 0);	double r_l1 = right_Li.at<double>(0, 0);
	double l_l2 = left_Li.at<double>(1, 0);	double r_l2 = right_Li.at<double>(1, 0);
	double l_l3 = left_Li.at<double>(2, 0);	double r_l3 = right_Li.at<double>(2, 0);
	double l_l4 = left_Li.at<double>(3, 0);	double r_l4 = right_Li.at<double>(3, 0);
 	double l_l5 = left_Li.at<double>(4, 0);	double r_l5 = right_Li.at<double>(4, 0);
	double l_l6 = left_Li.at<double>(5, 0);	double r_l6 = right_Li.at<double>(5, 0);
	double l_l7 = left_Li.at<double>(6, 0);	double r_l7 = right_Li.at<double>(6, 0);
	double l_l8 = left_Li.at<double>(7, 0);	double r_l8 = right_Li.at<double>(7, 0);
	double l_l9 = left_Li.at<double>(8, 0); double r_l9 = right_Li.at<double>(8, 0);
	double l_l10 = left_Li.at<double>(9, 0); double r_l10 = right_Li.at<double>(9, 0);
	double l_l11 = left_Li.at<double>(10, 0); double r_l11 = right_Li.at<double>(10, 0);
	double former_X = 0.0; double former_Y = 0.0; double former_Z = 0.0;
	double X = pointInOS.X; double Y = pointInOS.Y; double Z = pointInOS.Z;
	while(true)
	{
		double l_x = left_Point.x; double r_x = right_Point.x;
		double l_y = left_Point.y; double r_y = right_Point.y;
		double l_a = l_l9 * X + l_l10 * Y + l_l11 * Z + 1; double r_a = r_l9 * X + r_l10 * Y + r_l11 * Z + 1;
		//左片
		A.at<double>(0, 0) = -1.0 * (l_l1 + l_l9 * l_x) / l_a;
		A.at<double>(0, 1) = -1.0 * (l_l2 + l_l10 * l_x) / l_a;
		A.at<double>(0, 2) = -1.0 * (l_l3 + l_l11 * l_x) / l_a;
		A.at<double>(1, 0) = -1.0 * (l_l5 + l_l9 * l_y) / l_a;
		A.at<double>(1, 1) = -1.0 * (l_l6 + l_l10 * l_y) / l_a;
		A.at<double>(1, 2) = -1.0 * (l_l7 + l_l11 * l_y) / l_a;
		//右片
		A.at<double>(2, 0) = -1.0 * (r_l1 + r_l9 * r_x) / r_a;
		A.at<double>(2, 1) = -1.0 * (r_l2 + r_l10 * r_x) / r_a;
		A.at<double>(2, 2) = -1.0 * (r_l3 + r_l11 * r_x) / r_a;
		A.at<double>(3, 0) = -1.0 * (r_l5 + r_l9 * r_y) / r_a;
		A.at<double>(3, 1) = -1.0 * (r_l6 + r_l10 * r_y) / r_a;
		A.at<double>(3, 2) = -1.0 * (r_l7 + r_l11 * r_y) / r_a;

		l.at<double>(0, 0) = (l_l4 + l_x) / l_a;
		l.at<double>(1, 0) = (l_l8 + l_y) / l_a;
		l.at<double>(2, 0) = (r_l4 + r_x) / r_a;
		l.at<double>(3, 0) = (r_l8 + r_y) / r_a;

		Mat unpt_AcValue = (A.t() * A).inv() * A.t() * l;
		if (abs(former_X - X < 0.01) && abs(former_Y - Y) < 0.01 && abs(former_Z - Z) < 0.01)
		{
			pointInOS.X = unpt_AcValue.at<double>(0, 0);
			pointInOS.Y = unpt_AcValue.at<double>(1, 0);
			pointInOS.Z = unpt_AcValue.at<double>(2, 0);
			break;
		}
		former_X = X; former_Y = Y; former_Z = Z;
		X = unpt_AcValue.at<double>(0, 0);
		Y = unpt_AcValue.at<double>(1, 0);
		Z = unpt_AcValue.at<double>(2, 0);
	}
	return pointInOS;
}

int main()
{
	//读取控制点数据和左右片标志点的像素坐标
	vector<ControlPoint> ControlPoints;
	vector<imgPoint> left_imgPoints;
	vector<imgPoint> right_imgPoints;
	vector<imgPoint> left_unPoints;
	vector<imgPoint> right_unPoints;
	vector<imgPoint> left_ckpts;
	vector<imgPoint> right_ckpts;

	char clptfile[] = "./data/clpts.txt";
	char left_file[] = "./data/left.txt";
	char right_file[] = "./data/right.txt";
	char l_unknown_file[] = "./data/left_unknown.txt";
	char r_unknown_file[] = "./data/right_unknown.txt";
	char l_ckptfile[] = "./data/left_ckpts.txt";
	char r_ckptfile[] = "./data/right_ckpts.txt";
	readClptData(clptfile, ControlPoints);
	readImgData(left_file, left_imgPoints);
	readImgData(right_file, right_imgPoints);
	readImgData(l_unknown_file, left_unPoints);
	readImgData(r_unknown_file, right_unPoints);
	readImgData(l_ckptfile, left_ckpts);
	readImgData(r_ckptfile, right_ckpts);
	//将像素坐标(pixel)转换为图像坐标(mm)
	cvtPix2Img(left_imgPoints);
	cvtPix2Img(right_imgPoints);
	cvtPix2Img(left_unPoints);
	cvtPix2Img(right_unPoints);
	cvtPix2Img(left_ckpts);
	cvtPix2Img(right_ckpts);

	//生成点对
	vector<PointPair> left_PointPairs;
	vector<PointPair> right_PointPairs;
	vector<PointPair> left_ckpt_Pairs;
	vector<PointPair> right_ckpt_Pairs;
 	GeneratePairs(left_imgPoints, ControlPoints, left_PointPairs);
	GeneratePairs(right_imgPoints, ControlPoints, right_PointPairs);
	GeneratePairs(left_ckpts, ControlPoints, left_ckpt_Pairs);
	GeneratePairs(right_ckpts, ControlPoints, right_ckpt_Pairs);
	//for (int i = 0; i < left_PointPairs.size(); i++)
	//{
	//	cout << left_PointPairs[i].flag << " " << setprecision(8) << left_PointPairs[i].x << " " << left_PointPairs[i].y << " " << -1.0 * left_PointPairs[i].Z << " " << left_PointPairs[i].X << " " << left_PointPairs[i].Y << endl;
	//}

	////查找左右片都成像了的点对
	//vector<PointPair> New_left_Pairs;
	//vector<PointPair> New_right_Pairs;
	//for (int i = 0; i < left_imgPoints.size(); i++)
	//{
	//	for (int j = 0; j < right_imgPoints.size(); j++)
	//	{
	//		if (left_PointPairs[i].flag == right_PointPairs[j].flag)
	//		{
	//			New_left_Pairs.push_back(left_PointPairs[i]);
	//			New_right_Pairs.push_back(right_PointPairs[j]);
	//		}
	//	}
	//}

	//解算l系数初值
	Mat left_L_RoughValue = cal_L_RoughValue(left_PointPairs);
	Mat right_L_RoughValue = cal_L_RoughValue(right_PointPairs);
	cout << "左片Li系数初值：" << endl;
	//遍历输出
	for (int i = 0; i < left_L_RoughValue.rows; i++)
	{
		cout << "L" << i + 1 << "\t" << left_L_RoughValue.at<double>(i, 0) << " " << endl;
	}
	cout << "----------------------------------" << endl;
	cout << "右片Li系数初值：" << endl;
	for (int i = 0; i < right_L_RoughValue.rows; i++)
	{
		cout << "L" << i + 1 << "\t" << right_L_RoughValue.at<double>(i, 0) << " " << endl;
	}

	//解算内方位元素初值 --> 计算l精确值要用
	orienParam left_orien; left_orien.x0 = 0; left_orien.y0 = 0; left_orien.f = 0;
	left_orien.Xs = 0; left_orien.Ys = 0; left_orien.Zs = 0;
	left_orien.Kappa = 0; left_orien.Omega = 0; left_orien.Phi = 0;
	left_orien.k1 = 0; left_orien.k2 = 0; left_orien.p1 = 0; left_orien.p2 = 0;
	orienParam right_orien;
	right_orien.x0 = 0; right_orien.y0 = 0; right_orien.f = 0;
	right_orien.Xs = 0; right_orien.Ys = 0; right_orien.Zs = 0;
	right_orien.Kappa = 0; right_orien.Omega = 0; right_orien.Phi = 0;
	right_orien.k1 = 0;  right_orien.k2 = 0; right_orien.p1 = 0; right_orien.p2 = 0;
	double left_x0 = 0.0; double left_y0 = 0.0; double left_f = 0.0;
	double right_x0 = 0.0; double right_y0 = 0.0; double right_f = 0.0;
	cal_InteriorParams(left_L_RoughValue, left_orien);
	cal_InteriorParams(right_L_RoughValue, right_orien);


	//解算l系数精确值
	Mat left_A = Mat::zeros(2 * left_PointPairs.size(), 15, CV_64FC1);
	Mat left_l = Mat::zeros(2 * left_PointPairs.size(), 1, CV_64FC1);
	Mat right_A = Mat::zeros(2 * right_PointPairs.size(), 15, CV_64FC1);
	Mat right_l = Mat::zeros(2 * right_PointPairs.size(), 1, CV_64FC1);
	double former_left_f = left_orien.f;
	double former_right_f = right_orien.f;
	double former_left_x0 = left_orien.x0;
	double former_left_y0 = left_orien.y0;
	Mat left_L_AccurateValue;
	Mat right_L_AccurateValue;
	while (true)
	{
		cal_L_AccurateValue(left_L_RoughValue, left_A, left_l, left_orien, left_PointPairs);
		left_L_AccurateValue = (left_A.t() * left_A).inv() * left_A.t() * left_l;
		left_orien.k1 = left_L_AccurateValue.at<double>(11, 0);
		left_orien.k2 = left_L_AccurateValue.at<double>(12, 0);
		left_orien.p1 = left_L_AccurateValue.at<double>(13, 0);
		left_orien.p2 = left_L_AccurateValue.at<double>(14, 0);
		//tmd没给初始值更新赋值，找了那么久
		left_L_RoughValue.at<double>(0, 0) = left_L_AccurateValue.at<double>(0, 0);
		left_L_RoughValue.at<double>(1, 0) = left_L_AccurateValue.at<double>(1, 0);
		left_L_RoughValue.at<double>(2, 0) = left_L_AccurateValue.at<double>(2, 0);
		left_L_RoughValue.at<double>(3, 0) = left_L_AccurateValue.at<double>(3, 0);
		left_L_RoughValue.at<double>(4, 0) = left_L_AccurateValue.at<double>(4, 0);
		left_L_RoughValue.at<double>(5, 0) = left_L_AccurateValue.at<double>(5, 0);
		left_L_RoughValue.at<double>(6, 0) = left_L_AccurateValue.at<double>(6, 0);
		left_L_RoughValue.at<double>(7, 0) = left_L_AccurateValue.at<double>(7, 0);
		left_L_RoughValue.at<double>(8, 0) = left_L_AccurateValue.at<double>(8, 0);
		left_L_RoughValue.at<double>(9, 0) = left_L_AccurateValue.at<double>(9, 0);
		left_L_RoughValue.at<double>(10, 0) = left_L_AccurateValue.at<double>(10, 0);
		cal_InteriorParams(left_L_AccurateValue, left_orien);
		if (abs(left_orien.f - former_left_f) < 0.001 && abs(left_orien.x0-former_left_x0<0.001) && abs(left_orien.y0 - former_left_y0 < 0.001))
		{
			//精度统计
			Mat V = left_A * left_L_AccurateValue - left_l;
			Mat sigma0 = (V.t() * V / (2 * left_PointPairs.size() - 15));
			cout << "------------------------------" << endl;
			cout << "左片像点坐标观测值的单位权中误差/mm：" << sigma0.at<double>(0, 0) << endl;
			cout << "------------------------------" << endl;
			int num = 0;
			cout << "左片像点坐标观测值的残差/mm：" << endl;
			cout << "点号" << "\t" << "Vx" << "\t\t" << "Vy" << endl;

			for (int i = 0; i < V.rows; i += 2)
			{
				cout << left_PointPairs[num].flag << scientific <<"\t" << V.at<double>(i, 0) << "\t" << V.at<double>(i + 1, 0) << endl;
				num++;
			}
			cout.unsetf(ios::scientific);
			cout << "------------------------------" << endl;
			cout << "左片Li精确值为：" << endl;
			for (int k = 0; k < left_L_AccurateValue.rows - 4; k++)
				cout << "L" << k + 1 << "\t" << left_L_AccurateValue.at<double>(k, 0) << endl;
			break;
		}
		former_left_f = left_orien.f;
		former_left_x0 = left_orien.x0;
		former_left_y0 = left_orien.y0;
	}
	while (true)
	{
		cal_L_AccurateValue(right_L_RoughValue, right_A, right_l, right_orien, right_PointPairs);
		right_L_AccurateValue = (right_A.t() * right_A).inv() * right_A.t() * right_l;
		right_orien.k1 = right_L_AccurateValue.at<double>(11, 0);
		right_orien.k2 = right_L_AccurateValue.at<double>(12, 0);
		right_orien.p1 = right_L_AccurateValue.at<double>(13, 0);
		right_orien.p2 = right_L_AccurateValue.at<double>(14, 0);
		//tmd没给初始值更新赋值，找了那么久
		right_L_RoughValue.at<double>(0, 0) = right_L_AccurateValue.at<double>(0, 0);
		right_L_RoughValue.at<double>(1, 0) = right_L_AccurateValue.at<double>(1, 0);
		right_L_RoughValue.at<double>(2, 0) = right_L_AccurateValue.at<double>(2, 0);
		right_L_RoughValue.at<double>(3, 0) = right_L_AccurateValue.at<double>(3, 0);
		right_L_RoughValue.at<double>(4, 0) = right_L_AccurateValue.at<double>(4, 0);
		right_L_RoughValue.at<double>(5, 0) = right_L_AccurateValue.at<double>(5, 0);
		right_L_RoughValue.at<double>(6, 0) = right_L_AccurateValue.at<double>(6, 0);
		right_L_RoughValue.at<double>(7, 0) = right_L_AccurateValue.at<double>(7, 0);
		right_L_RoughValue.at<double>(8, 0) = right_L_AccurateValue.at<double>(8, 0);
		right_L_RoughValue.at<double>(9, 0) = right_L_AccurateValue.at<double>(9, 0);
		right_L_RoughValue.at<double>(10, 0) = right_L_AccurateValue.at<double>(10, 0);
		cal_InteriorParams(right_L_AccurateValue, right_orien);
		if (abs(right_orien.f - former_right_f) < 0.01)
		{
			//精度统计
			Mat V = right_A * right_L_AccurateValue - right_l;
			Mat sigma0 = (V.t() * V / (2 * right_PointPairs.size() - 15));
			cout << "------------------------------" << endl;
			cout << "右片像点坐标观测值的单位权中误差/mm：" << sigma0.at<double>(0, 0) << endl;
			cout << "------------------------------" << endl;
			int num = 0;
			cout << "右片像点坐标观测值的残差/mm：" << endl;
			cout << "点号" << "\t" << "Vx" << "\t\t" << "Vy" << endl;
			for (int i = 0; i < V.rows; i += 2)
			{
				cout << right_PointPairs[num].flag << scientific << "\t" << V.at<double>(i, 0) << "\t" << V.at<double>(i + 1, 0) << endl;
				num++;
			}
			cout.unsetf(ios::scientific);
			cout << "------------------------------" << endl;
			cout << "右片Li精确值为：" << endl;
			for (int k = 0; k < right_L_AccurateValue.rows - 4; k++)
				cout << "L" << k + 1 << "\t" << right_L_AccurateValue.at<double>(k, 0) << endl;
			break;
		}
		former_right_f = right_orien.f;
	}

	//算一下外方位元素看看结果
	cal_ExteriorParams(left_L_AccurateValue, left_orien, left_PointPairs);
	cal_ExteriorParams(right_L_AccurateValue, right_orien, right_PointPairs);
	cout << "------------------------------" << endl;
	cout << "左相机内外方位元素和畸变系数：" << endl;
	cout << "Xs:" << left_orien.Xs << " Ys:" << left_orien.Ys << " Zs:" << left_orien.Zs << endl;
	cout << "Phi:" << left_orien.Phi << " Omega:" << left_orien.Omega << " Kappa:" << left_orien.Kappa << endl;
	cout << "x0:" << left_orien.x0 << " y0:" << left_orien.y0 << " f:" << left_orien.f << endl;
	cout << "k1:" << left_orien.k1 << " k2:" << left_orien.k2 << " p1:" << left_orien.p1 << " p2:" << left_orien.p2 << endl;
	cout << "ds: " << left_orien.ds << "dp: " << left_orien.dp << endl;
	cout << "------------------------------" << endl;
	cout << "右相机内外方位元素和畸变系数"<< endl;
	cout << "Xs:" << right_orien.Xs << " Ys:" << right_orien.Ys << " Zs:" << right_orien.Zs << endl;
	cout << "Phi" << right_orien.Phi << " Omega:" << right_orien.Omega << " Kappa:" << right_orien.Kappa << endl;
	cout << "x0:" << right_orien.x0 << " y0:" << right_orien.y0 << " f:" << right_orien.f << endl;
	cout << "k1:" << right_orien.k1 << " k2:" << right_orien.k2 << " p1:" << right_orien.p1 << " p2:" << right_orien.p2 << endl;
	cout << "ds: " << right_orien.ds << "dp: " << right_orien.dp << endl;

	//解算控制点物方空间坐标精确值
	//先改正像点坐标 --> 内方位元素的精确值已经在上一步用Li的精确值更新了
	/*for (int i = 0; i < left_unPoints.size(); i++)
		cout << left_unPoints[i].flag << " " << left_unPoints[i].x << " " << left_unPoints[i].y << endl;*/
	modifyImgPoints(left_unPoints, left_orien);
	modifyImgPoints(right_unPoints, right_orien);
	//for (int i = 0; i < left_unPoints.size(); i++)
	//	cout << left_unPoints[i].flag << " " << left_unPoints[i].x << " " << left_unPoints[i].y << endl;
	

	//解算未知点的坐标初值
	vector<ControlPoint> unPoints;
	for (int i = 0; i < left_unPoints.size(); i++)
	{
		Mat un_res = cal_unpt_RoughValue(left_unPoints[i], right_unPoints[i], left_L_AccurateValue, right_L_AccurateValue);
		ControlPoint p;
		p.flag = left_unPoints[i].flag;
		p.X = un_res.at<double>(0, 0);
		p.Y = un_res.at<double>(1, 0);
		p.Z = un_res.at<double>(2, 0);
		unPoints.push_back(p);
		//cout << p.flag << " " << p.X << " " << p.Y << " " << p.Z << endl;
	}
	//解算待定点物方空间坐标精确值
	for (int i = 0; i < left_unPoints.size(); i++)
	{
		//cout << left_unPoints[i].flag << " " << unPoints[i].X << " " << unPoints[i].Y << " " << unPoints[i].Z << endl;
		unPoints[i] = cal_unpt_AccurateValue(left_L_AccurateValue, right_L_AccurateValue, unPoints[i], left_unPoints[i], right_unPoints[i], left_orien, right_orien);
		//cout << left_unPoints[i].flag << " " << unPoints[i].X << " " << unPoints[i].Y << " " << unPoints[i].Z << endl;
	}
	//计算各点相对于点52的欧式距离
	ControlPoint p52;
	for (int i = 0; i < unPoints.size(); i++)
	{
		if (unPoints[i].flag == 52)
		{
			p52 = unPoints[i];
			break;
		}
	}
	vector<double> distance;
	cout << "------------------------------" << endl;
	cout << "待定点相对于点52的欧式距离/mm：" << endl;
	for (int i = 0; i < unPoints.size(); i++)
	{
		if(unPoints[i].flag == 52)
			continue;
		double dis = sqrt(pow(unPoints[i].X - p52.X, 2) + pow(unPoints[i].Y - p52.Y, 2) + pow(unPoints[i].Z - p52.Z, 2));
		distance.push_back(dis);
		cout << unPoints[i].flag << " " << dis << endl;
	}

	//解算检查点物方空间坐标初始值
	//先改正像点坐标 --> 内方位元素的精确值已经在上一步用Li的精确值更新了
	modifyImgPoints(left_ckpts, left_orien);
	modifyImgPoints(right_ckpts, right_orien);
	vector<ControlPoint> ckPoints;
	for (int i = 0; i < left_ckpts.size(); i++)
	{
		Mat ck_res = cal_unpt_RoughValue(left_ckpts[i], right_ckpts[i], left_L_RoughValue, right_L_RoughValue);
		ControlPoint p;
		p.flag = left_ckpts[i].flag;
		p.X = ck_res.at<double>(0, 0);
		p.Y = ck_res.at<double>(1, 0);
		p.Z = ck_res.at<double>(2, 0);
		ckPoints.push_back(p);
		//cout << p.flag << " " << p.X << " " << p.Y << " " << p.Z << endl;
	}
	//解算检查点物方空间坐标精确值
	for (int i = 0; i < left_ckpts.size(); i++)
	{
		//cout << left_ckpts[i].flag << " " << ckPoints[i].X << " " << ckPoints[i].Y << " " << ckPoints[i].Z << endl;
		ckPoints[i] = cal_unpt_AccurateValue(left_L_AccurateValue, right_L_AccurateValue, ckPoints[i], left_ckpts[i], right_ckpts[i], left_orien, right_orien);
		//cout << left_ckpts[i].flag << " " << ckPoints[i].X << " " << ckPoints[i].Y << " " << ckPoints[i].Z << endl;
	}
	cout << "-------------------" << endl;
	cout << "检查点物方空间坐标残差/mm" << endl;
	cout << "点号\t" << "X\t" << "Y\t" << "Z\t" << endl;
	for (int i = 0; i < left_ckpt_Pairs.size(); i++)
	{
		for (int j = 0; j < ckPoints.size(); j++)
		{
			if (left_ckpt_Pairs[i].flag == ckPoints[j].flag)
			{
				double dx = left_ckpt_Pairs[i].X - ckPoints[j].X;
				double dy = left_ckpt_Pairs[i].Y - ckPoints[j].Y;
				double dz = left_ckpt_Pairs[i].Z - ckPoints[j].Z;
				cout << left_ckpt_Pairs[i].flag << "\t" << dx << "\t" << dy << "\t" << dz << endl;
			}
		}
	}
	return 0;
}