
#ifndef MY_IMAGEPROCESSFUNCTIONS
#define MY_IMAGEPROCESSFUNCTIONS

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "BlobDetector.h"

using namespace cv;
using namespace std;

class ImageProcessFunctions {

private:
	// ImageProcessFunctions class member variables
	const Scalar SCALAR_BLACK  = Scalar(0.0, 0.0, 0.0);
	const Scalar SCALAR_WHITE  = Scalar(255.0, 255.0, 255.0);
	const Scalar SCALAR_YELLOW = Scalar(0.0, 255.0, 255.0);
    const Scalar SCALAR_BLUE   = Scalar(255.0, 0.0, 0.0);
	const Scalar SCALAR_GREEN  = Scalar(0.0, 200.0, 0.0);
	const Scalar SCALAR_RED    = Scalar(0.0, 0.0, 255.0);

public:
	// small object filter
	int objectRectParameter = 370;
	int objectSize = 25;
	double diagonalSize = 35;
	// define object ratio
	double farAspectRatio = 0.4;
	double nearAspectRatio = 4.0;
	double contourAndBoundingRectRatio = 0.48;

	
public:	
	// Image processing functions
	int OtsuThreshold(Mat &imgGrey);
	void matchCurrentFrameBlobsToExistingBlobs(vector<BlobDetector> &existingBlobs, vector<BlobDetector> &currentFrameBlobs);

	void addBlobToExistingBlobs(BlobDetector &currentFrameBlob, vector<BlobDetector> &existingBlobs, int &intIndex);
	void addNewBlob(BlobDetector &currentFrameBlob, vector<BlobDetector> &existingBlobs);
	
	double distanceBetweenPoints(Point point1, Point point2);
	
	void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName);
	void drawAndShowContours(Size imageSize, vector<BlobDetector> blobs, string strImageName);
	
	void drawBlobInfoOnImage(vector<BlobDetector> &blobs, Mat &imgFrame2Copy);
	
	void writeCarCountAndColourOnImage(int &carCountRight, int &carCountLeft, Scalar &centroidColourLeft, Scalar &centroidColourRight, Mat &imgFrame2Copy);
	void writeCarCountAndColourOnImage(int &carCountRight, int &carCountLeft, String colourTextRight, String colourTextLeft, Mat &imgFrame2Copy);

	void checkIfBlobsCrossedTheLineRight(vector<BlobDetector> &blobs, int &intHorizontalLinePosition, int &carCountRight);
	void checkIfBlobsCrossedTheLineLeft(vector<BlobDetector> &blobs, int &intHorizontalLinePositionLeft, int &carCountLeft);

	void findblobs(vector<vector<Point> > convexHulls, vector<BlobDetector> currentFrameBlobs);
	
	vector<Point3f> findColourInfoOnImage(Mat &imgFrameblob, Mat &imgFrameColour, int centralLine, int countNum);
	Vec3b findCentroidColourAverageOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid, vector<vector<Point> > contours);
	Vec3b findCentroidColourInfoOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid, vector<vector<Point> > contours);
	Vec3b findCentroidColourInfoOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid);

	Scalar findCentroidColourOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid);
	Scalar findCentroidColourAverageOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid);
	String findColourNameFromColourValue(Mat &imgFrameColour, vector<Point2f> muCentroid);
};

#endif    


