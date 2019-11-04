/*
Based on Ahmet Özlü's code

*/
#ifndef MY_BLOBDETECTOR
#define MY_BLOBDETECTOR
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class BlobDetector {
public:
	// BlobDetector class member variables
	vector<Point> currentContour;	
	vector<Point> centerPositions;
	Point predictedNextPosition;
	Rect currentBoundingRect;

	double CurrentDiagonalSize;
	double CurrentAspectRatio;
    int  NumOfSubsequentNoMatch;
	bool CurrentMatchFoundOrNewBlob;
	bool StillBeingTracked;	
	
	// Blob detector functions
	BlobDetector(vector<Point> blob_contour);

	void predictNextBlobPosition( );
};

#endif  


