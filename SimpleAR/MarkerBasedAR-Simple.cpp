#include<opencv2\opencv.hpp>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

class MarkerBasedARProcessor
{
	Mat Image, ImageGray, ImageAdaptiveBinary; //分别是 原图像 灰度图像 自适应阈值化图像
	vector<vector<Point>> ImageContours; //图像所有边界信息
	vector<vector<Point2f>> ImageQuads, ImageMarkers; //图像所有四边形 与 验证成功的四边形

	vector<Point2f> FlatMarkerCorners; //正方形化标记时用到的信息
	Size FlatMarkerSize; //正方形化标记时用到的信息

	//7x7黑白标记的颜色信息
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

	void Clean() // 用于新一帧处理前的初始化
	{
		ImageContours.clear();
		ImageQuads.clear();
		ImageMarkers.clear();
	}
	void ConvertColor() //转换图片颜色
	{
		cvtColor(Image, ImageGray, CV_BGR2GRAY);
		adaptiveThreshold(ImageGray, ImageAdaptiveBinary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);
	}
	void GetContours(int ContourCountThreshold) //获取图片所有边界
	{
		vector<vector<Point>> AllContours; // 所有边界信息
		findContours(ImageAdaptiveBinary, AllContours,
			CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // 用自适应阈值化图像寻找边界
		for (size_t i = 0;i < AllContours.size();++i) // 只储存低于阈值的边界
		{
			int contourSize = AllContours[i].size();
			if (contourSize > ContourCountThreshold)
			{
				ImageContours.push_back(AllContours[i]);
			}
		}
	}
	void FindQuads(int ContourLengthThreshold) //寻找所有四边形
	{
		vector<vector<Point2f>> PossibleQuads;
		for (int i = 0;i < ImageContours.size();++i)
		{
			vector<Point2f> InDetectPoly;
			approxPolyDP(ImageContours[i], InDetectPoly,
				ImageContours[i].size() * 0.05, true); // 对边界进行多边形拟合
			if (InDetectPoly.size() != 4) continue;// 只对四边形感兴趣
			if (!isContourConvex(InDetectPoly)) continue; // 只对凸四边形感兴趣
			float MinDistance = 1e10; // 寻找最短边
			for (int j = 0;j < 4;++j)
			{
				Point2f Side = InDetectPoly[j] - InDetectPoly[(j + 1) % 4];
				float SquaredSideLength = Side.dot(Side);
				MinDistance = min(MinDistance, SquaredSideLength);
			}
			if (MinDistance < ContourLengthThreshold) continue; // 最短边必须大于阈值
			vector<Point2f> TargetPoints;
			for (int j = 0;j < 4;++j) // 储存四个点
			{
				TargetPoints.push_back(Point2f(InDetectPoly[j].x, InDetectPoly[j].y));
			}
			Point2f Vector1 = TargetPoints[1] - TargetPoints[0]; // 获取一个边的向量
			Point2f Vector2 = TargetPoints[2] - TargetPoints[0]; // 获取一个斜边的向量
			if (Vector2.cross(Vector1) < 0.0) // 计算两向量的叉乘 判断点是否为逆时针储存
				swap(TargetPoints[1], TargetPoints[3]); // 如果大于0则为顺时针，需要交替
			PossibleQuads.push_back(TargetPoints); // 保存进可能的四边形，进行进一步判断
		}
		vector<pair<int, int>> TooNearQuads; // 准备删除几组靠太近的多边形
		for (int i = 0;i < PossibleQuads.size();++i)
		{
			vector<Point2f>& Quad1 = PossibleQuads[i]; // 第一个             
			for (int j = i + 1;j < PossibleQuads.size();++j)
			{
				vector<Point2f>& Quad2 = PossibleQuads[j]; // 第二个
				float distSquared = 0;
				float x1Sum = 0.0, x2Sum = 0.0, y1Sum = 0.0, y2Sum = 0.0, dx = 0.0, dy = 0.0;
				for (int c = 0;c < 4;++c)
				{
					x1Sum += Quad1[c].x;
					x2Sum += Quad2[c].x;
					y1Sum += Quad1[c].y;
					y2Sum += Quad2[c].y;
				}
				x1Sum /= 4; x2Sum /= 4; y1Sum /= 4; y2Sum /= 4; // 计算平均值（中点）
				dx = x1Sum - x2Sum;
				dy = y1Sum - y2Sum;
				distSquared = sqrt(dx*dx + dy*dy); // 计算两多边形距离
				if (distSquared < 50)
				{
					TooNearQuads.push_back(pair<int, int>(i, j)); // 过近则准备剔除
				}
			}
		}
		vector<bool> RemovalMask(PossibleQuads.size(), false); // 移除标记列表
		for (int i = 0;i < TooNearQuads.size();++i)
		{
			float p1 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].first]);  //求周长
			float p2 = CalculatePerimeter(PossibleQuads[TooNearQuads[i].second]);
			int removalIndex;  //移除周长小的多边形
			if (p1 > p2) removalIndex = TooNearQuads[i].second;
			else removalIndex = TooNearQuads[i].first;
			RemovalMask[removalIndex] = true;
		}
		for (size_t i = 0;i < PossibleQuads.size();++i)
		{
			// 只录入没被剔除的多边形
			if (!RemovalMask[i]) ImageQuads.push_back(PossibleQuads[i]);
		}
	}
	void TransformVerifyQuads() //变换为正方形并验证是否为标记
	{
		Mat FlatQuad;
		for (size_t i = 0;i < ImageQuads.size();++i)
		{
			vector<Point2f>& Quad = ImageQuads[i];
			Mat TransformMartix = getPerspectiveTransform(Quad, FlatMarkerCorners);
			warpPerspective(ImageGray, FlatQuad, TransformMartix, FlatMarkerSize);
			threshold(FlatQuad, FlatQuad, 0, 255, THRESH_OTSU); // 变为二值化图像
			if (MatchQuadWithMarker(FlatQuad)) // 与正确标记比对
			{
				ImageMarkers.push_back(ImageQuads[i]); // 成功则记录
			}
			else // 如果失败，则旋转，每次90度进行比对
			{
				for (int j = 0;j < 3;++j)
				{
					rotate(FlatQuad, FlatQuad, ROTATE_90_CLOCKWISE);
					if (MatchQuadWithMarker(FlatQuad))
					{
						ImageMarkers.push_back(ImageQuads[i]); // 成功则记录
						break;
					}
				}
			}
		}
	} //变换为正方形并验证是否为标记

	void DrawMarkerBorder(Scalar Color) //绘制标记边界
	{
		for (vector<Point2f> Marker : ImageMarkers)
		{
			line(Image, Marker[0], Marker[1], Color, 2, CV_AA);
			line(Image, Marker[1], Marker[2], Color, 2, CV_AA);
			line(Image, Marker[2], Marker[3], Color, 2, CV_AA);
			line(Image, Marker[3], Marker[0], Color, 2, CV_AA);//CV_AA是抗锯齿
		}
	}
	void DrawImageAboveMarker() //在标记上绘图
	{
		if (ImageToDraw.empty())return;
		vector<Point2f> ImageCorners = { Point2f(0,0),Point2f(ImageToDraw.cols - 1,0),Point2f(ImageToDraw.cols - 1,ImageToDraw.rows - 1),Point2f(0,ImageToDraw.rows - 1) };
		Mat_<Vec3b> ImageWarp = Image;
		for (vector<Point2f> Marker : ImageMarkers)
		{
			Mat TransformMartix = getPerspectiveTransform(ImageCorners, Marker);
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

	bool MatchQuadWithMarker(Mat & Quad) // 检验正方形是否为标记
	{
		int  Pos = 0;
		for (int r = 2;r < 33;r += 5) // 正方形图像大小为(35,35)
		{
			for (int c = 2;c < 33;c += 5)// 读取每块图像中心点
			{
				uchar V = Quad.at<uchar>(r, c);
				uchar K = CorrectMarker[Pos];
				if (K != V) // 与正确标记颜色信息比对
					return false;
				Pos++;
			}
		}
		return true;
	}
	float CalculatePerimeter(const vector<Point2f> &Points)  // 计算周长
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
	Mat ImageToDraw; // 要在标记上绘制的图像
	MarkerBasedARProcessor() // 构造函数
	{
		FlatMarkerSize = Size(35, 35);
		FlatMarkerCorners = { Point2f(0,0),Point2f(FlatMarkerSize.width - 1,0),Point2f(FlatMarkerSize.width - 1,FlatMarkerSize.height - 1),Point2f(0,FlatMarkerSize.height - 1) };
	}
	Mat Process(Mat& Image)// 处理一帧图像
	{
		Clean(); // 新一帧初始化
		Image.copyTo(this->Image); // 复制原始图像到Image中
		ConvertColor(); // 转换颜色
		GetContours(50); // 获取边界
		FindQuads(100); // 寻找四边形
		TransformVerifyQuads(); // 变形并校验四边形
		DrawMarkerBorder(Scalar(255, 255, 255)); // 在得到的标记周围画边界
		DrawImageAboveMarker(); // 在标记上画图
		return this->Image; // 返回结果图案
	}
};

int main()
{
	Mat Frame, ProceedFrame;
	VideoCapture Camera(0); // 初始化相机
	while (!Camera.isOpened()); // 等待相机加载完成
	MarkerBasedARProcessor Processor; // 构造一个AR处理类
	Processor.ImageToDraw = imread("ImageToDraw.jpg"); // 读入绘制图像
	while (waitKey(1)) // 每次循环延迟1ms
	{
		Camera >> Frame; // 读一帧
		imshow("Frame", Frame); // 显示原始图像
		ProceedFrame = Processor.Process(Frame); // 处理图像
		imshow("ProceedFrame", ProceedFrame); // 显示结果图像
	}
}