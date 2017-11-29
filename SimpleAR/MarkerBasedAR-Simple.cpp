#include<opencv2\opencv.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

class MarkerBasedARProcessor
{
	Mat Image, ImageGray, ImageAdaptiveBinary;
	vector<vector<Point>> ImageContours;
	vector<vector<Point2f>> ImageQuads, ImageMarkers;
	vector<Point2f> FlatMarkerCorners;
	Size FlatMarkerSize;

	uchar CorrectMarker[7 * 7] =
	{
		0,0,0,0,0,0,0,
		0,0,0,0,0,255,0,
		0,0,255,255,255,0,0,
		0,255,255,255,0,255,0,
		0,255,255,255,0,255,0,
		0,255,255,255,0,255,0,
		0,0,0,0,0,0,0
	};

	void Clean()
	{
		ImageContours.clear();
		ImageQuads.clear();
		ImageMarkers.clear();
	}
	void ConvertColor()
	{
		cvtColor(Image, ImageGray, CV_BGR2GRAY);
		adaptiveThreshold(ImageGray, ImageAdaptiveBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	}
	void GetContours(int ContourCountThreshold)
	{
		vector<vector<Point>> AllContours;
		findContours(ImageAdaptiveBinary, AllContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		for (size_t i = 0;i < AllContours.size();++i)
		{
			int contourSize = AllContours[i].size();
			if (contourSize > ContourCountThreshold)
			{
				ImageContours.push_back(AllContours[i]);
			}
		}
	}
	void FindQuads(int ContourLengthThreshold)
	{
		vector<vector<Point2f>> PossibleQuads;
		for (int i = 0;i < ImageContours.size();++i)    //检测并只保留是4个顶点的多边形，按逆时针储存顶点
		{
			vector<Point2f> InDetectPoly;
			approxPolyDP(ImageContours[i], InDetectPoly, ImageContours[i].size() * 0.05, true);
			if (InDetectPoly.size() != 4) continue;
			if (!isContourConvex(InDetectPoly))	continue;
			float MinDistance = 1e10;
			for (int j = 0;j < 4;++j)
			{
				Point2f Side = InDetectPoly[j] - InDetectPoly[(j + 1) % 4];
				float SquaredSideLength = Side.dot(Side);
				MinDistance = min(MinDistance, SquaredSideLength);
			}
			if (MinDistance < ContourLengthThreshold) continue;
			vector<Point2f> TargetPoints;
			for (int j = 0;j < 4;++j)  //储存四个点
			{
				TargetPoints.push_back(Point2f(InDetectPoly[j].x, InDetectPoly[j].y));
			}
			Point2f Vector1 = TargetPoints[1] - TargetPoints[0];    //保证逆时针储存顶点
			Point2f Vector2 = TargetPoints[2] - TargetPoints[0];
			if (Vector2.cross(Vector1) < 0.0) swap(TargetPoints[1], TargetPoints[3]);
			PossibleQuads.push_back(TargetPoints);
		}
		vector<pair<int, int>> TooNearQuads;             //删除几组靠的太近的多边形
		for (int i = 0;i < PossibleQuads.size();++i)
		{
			vector<Point2f>& Quad1 = PossibleQuads[i]; //计算两个maker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近,把这一对四边形放入移除队列。		                  
			for (int j = i + 1;j < PossibleQuads.size();++j)
			{
				vector<Point2f>& Quad2 = PossibleQuads[j];
				float distSquared = 0;
				float x1Sum = 0.0, x2Sum = 0.0, y1Sum = 0.0, y2Sum = 0.0, dx = 0.0, dy = 0.0;
				for (int c = 0;c < 4;++c)
				{
					x1Sum += Quad1[c].x;
					x2Sum += Quad2[c].x;
					y1Sum += Quad1[c].y;
					y2Sum += Quad2[c].y;
				}
				x1Sum /= 4;	x2Sum /= 4;	y1Sum /= 4;	y2Sum /= 4;
				dx = x1Sum - x2Sum;
				dy = y1Sum - y2Sum;
				distSquared = sqrt(dx*dx + dy*dy);
				if (distSquared < 50)
				{
					TooNearQuads.push_back(pair<int, int>(i, j));
				}
			}//移除了相邻的元素对的标识
		}//计算距离相近的两个marker内部，四个点的距离和，将距离和较小的，在removlaMask内做标记，即不作为最终的detectedMarkers 
		vector<bool> RemovalMask(PossibleQuads.size(), false);//创建Vector对象，并设置容量。第一个参数是容量，第二个是元素。
		for (int i = 0;i < TooNearQuads.size();++i)
		{
			float p1 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].first]);  //求这一对相邻四边形的周长
			float p2 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].second]);
			int removalIndex;  //谁周长小，移除谁
			if (p1 > p2) removalIndex = TooNearQuads[i].second;
			else removalIndex = TooNearQuads[i].first;
			RemovalMask[removalIndex] = true;
		}
		//返回候选，移除相邻四边形中周长较小的那个，放入待检测的四边形的队列中。//返回可能的对象
		for (size_t i = 0;i < PossibleQuads.size();++i)
		{
			if (!RemovalMask[i]) ImageQuads.push_back(PossibleQuads[i]);
		}
	}
	void TransformVerifyQuads()
	{
		//为了得到这些矩形的标记图像，我们不得不使用透视变换去恢复(unwarp)输入的图像。这个矩阵应该使用cv::getPerspectiveTransform函数，
		//它首先根据四个对应的点找到透视变换，第一个参数是标记的坐标，第二个是正方形标记图像的坐标。估算的变换将会把标记转换成方形，从而方便我们分析。
		Mat FlatQuad;
		for (size_t i = 0;i < ImageQuads.size();++i)
		{
			vector<Point2f>& Quad = ImageQuads[i];	//找到透视转换矩阵，获得矩形区域的正面视图// 找到透视投影，并把标记转换成矩形，输入图像四边形顶点坐标，输出图像的相应的四边形顶点坐标 
			Mat TransformMartix = getPerspectiveTransform(Quad, FlatMarkerCorners);//输入原始图像和变换之后的图像的对应4个点，便可以得到变换矩阵 (输入变换后的四角，标准的方形四角）
			warpPerspective(ImageGray, FlatQuad, TransformMartix, FlatMarkerSize);
			threshold(FlatQuad, FlatQuad, 0, 255, THRESH_OTSU);
			if (MatchQuadWithMarker(FlatQuad))
			{
				ImageMarkers.push_back(ImageQuads[i]);
			}
			else
			{
				for (int j = 0;j < 3;++j)
				{
					rotate(FlatQuad, FlatQuad, ROTATE_90_CLOCKWISE);
					if (MatchQuadWithMarker(FlatQuad))
					{
						ImageMarkers.push_back(ImageQuads[i]);
						break;
					}
				}
			}
		}
	}
	void DrawMarkerBorder(Scalar Color)
	{
		for (vector<Point2f> Marker : ImageMarkers)
		{
			line(Image, Marker[0], Marker[1], Color, 2, CV_AA);
			line(Image, Marker[1], Marker[2], Color, 2, CV_AA);
			line(Image, Marker[2], Marker[3], Color, 2, CV_AA);
			line(Image, Marker[3], Marker[0], Color, 2, CV_AA);//CV_AA是抗锯齿*/
		}
	}
	void DrawImageAboveMarker()
	{
		if (ImageToDraw.empty())return;
		vector<Point2f> ImageCorners = { Point2f(0,0),Point2f(ImageToDraw.cols - 1,0),Point2f(ImageToDraw.cols - 1,ImageToDraw.rows - 1),Point2f(0,ImageToDraw.rows - 1) };
		Mat_<Vec3b> ImageWarp = Image;
		for (vector<Point2f> Marker : ImageMarkers)
		{
			Mat TransformMartix = getPerspectiveTransform(ImageCorners, Marker);//输入原始图像和变换之后的图像的对应4个点，便可以得到变换矩阵 (输入变换后的四角，标准的方形四角）
			Mat_<Vec3b> Result(Size(Image.cols, Image.rows), CV_8UC3);
			warpPerspective(ImageToDraw, Result, TransformMartix, Size(Image.cols, Image.rows));
			for (int r = 0;r < Image.rows;++r)
			{
				for (int c = 0;c < Image.cols;++c)
				{
					if (Result(r, c) != Vec3b(0, 0, 0))
					{
						ImageWarp(r, c) = Result(r, c);
					}
				}
			}
		}
	}

	bool MatchQuadWithMarker(Mat & Quad)
	{
		int  Pos = 0;
		for (int r = 2;r < 33;r += 5)
		{
			for (int c = 2;c < 33;c += 5)
			{
				uchar V = Quad.at<uchar>(r, c);
				uchar K = CorrectMarker[Pos];
				if (K != V)
					return false;
				Pos++;
			}
		}
		return true;
	}
	float CalculatePerimeter(const vector<Point2f> &Points)  //求多边形周长。
	{
		float sum = 0, dx, dy;
		for (size_t i = 0;i < Points.size();++i)
		{
			size_t i2 = (i + 1) % Points.size();
			dx = Points[i].x - Points[i2].x;
			dy = Points[i].y - Points[i2].y;
			sum += sqrt(dx*dx + dy*dy);
		}
		return sum;
	}
public:
	Mat ImageToDraw;
	MarkerBasedARProcessor()
	{
		FlatMarkerSize = Size(35, 35);
		FlatMarkerCorners = { Point2f(0,0),Point2f(FlatMarkerSize.width - 1,0),Point2f(FlatMarkerSize.width - 1,FlatMarkerSize.height - 1),Point2f(0,FlatMarkerSize.height - 1) };
	}
	Mat Process(Mat& Image)
	{
		Clean();
		Image.copyTo(this->Image);
		ConvertColor();
		GetContours(50);
		FindQuads(100);
		TransformVerifyQuads();
		DrawMarkerBorder(Scalar(255, 255, 255));
		DrawImageAboveMarker();
		return this->Image;
	}
};

int main()
{
	Mat Frame, ProceedFrame;
	VideoCapture Camera(0);
	while (!Camera.isOpened());
	MarkerBasedARProcessor Processor;
	Processor.ImageToDraw = imread("ImageToDraw.jpg");
	while (waitKey(1))
	{
		Camera >> Frame;
		imshow("Frame", Frame);
		ProceedFrame = Processor.Process(Frame);
		imshow("ProceedFrame", ProceedFrame);
	}
}