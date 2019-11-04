//Vehicle detecting, tracking and counting system
//Based on Ahmet �zl�'s code


#include <fstream>
#include <string>
#include <iomanip>
#include<iostream>
#include<conio.h>  

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include<opencv2/video/background_segm.hpp>

#include "BlobDetector.h"
#include "ImageProcessFunctions.h"

using namespace cv;
using namespace std;

int main() 
{				    
	char checkForEscKey = 0;
	bool firstFrame = true;
	Vec3b centroidColourLeftVec3b, centroidColourRightVec3b;
	Scalar centroidColourLeftScalar, centroidColourRightScalar;
	String colourNameLeft, colourNameRight;
	int frameCount = 2, i = 0, k = 0, carCountLeft = 0, carCountRight = 0, carColourLeft = 1, carColourRight = 1;

	// Define the regions of interest as a vector of rectangles (video 2: 100% counting, left:21, right:18)
	int Start_xL = 90, Start_yL = 220, Size_xL = 460, Size_yL = 290;
	int Start_xR = 650, Start_yR = 220, Size_xR = 350, Size_yR = 400;
	Rect RectangleToSelectLeft(Start_xL, Start_yL, Size_xL, Size_yL);
	Rect RectangleToSelectRight(Start_xR, Start_yR, Size_xR, Size_yR);

    VideoCapture m_capVideo;
	Mat imgFrame1, imgFrame2;
	Mat imgFrameCropLeft, imgFrameCropRight;
    Point crossingLine[2];
	Point crossingLineLeft[2];	

	vector<BlobDetector> m_blobDetectorLeft, m_blobDetectorRight;
	ImageProcessFunctions m_ImageProcessFunctions;

	// read video frames
	m_capVideo.open("testVideo.mp4");

    if (!m_capVideo.isOpened()) 
	{    
		// show error message if unable to open video file.	                                             
        cout << "error reading video file" << endl;                              								
        return(0);                                                              
    }

    if (m_capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) 
	{
        cout << "error: video file must have at least two frames" << endl;
        return(0);
    }

	m_capVideo.read(imgFrame1);
	m_capVideo.read(imgFrame2);
	
	imgFrameCropLeft  = imgFrame1(RectangleToSelectLeft);// Crop left region in image
	imgFrameCropRight = imgFrame1(RectangleToSelectRight);// Crop right region in image

	//Control line for vehicle counting (RIGHT WAY)
	int intHorizontalLinePositionRight = (int)round((double)imgFrameCropRight.rows * 0.48);
	
	//Control line for vehicle counting (LEFT WAY)
	int intHorizontalLinePositionLeft = (int)round((double)imgFrameCropLeft.rows * 0.48);

	// execute detecting, tracking, counting and colour indexing tasks 
    while (m_capVideo.isOpened() && checkForEscKey != 27)
	{
        vector<BlobDetector> currentFrameBlobsLeft, currentFrameBlobsRight;
        Mat imgFrame1Copy = imgFrame1.clone();
        Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgFrameCrop1CopyLeft = imgFrame1Copy(RectangleToSelectLeft);// Crop Left image
		Mat imgFrameCrop2CopyLeft = imgFrame2Copy(RectangleToSelectLeft);
		Mat imgFrameCrop1CopyLeftG = imgFrame1Copy(RectangleToSelectLeft);// Crop Left image
		Mat imgFrameCrop2CopyLeftG = imgFrame2Copy(RectangleToSelectLeft);
		Mat imgFrameCrop1CopyLeftM = imgFrame1Copy(RectangleToSelectLeft);// Crop Left image
		Mat imgFrameCrop2CopyLeftM = imgFrame2Copy(RectangleToSelectLeft);
		Mat imgFrameCrop1CopyLeftGS = imgFrame1Copy(RectangleToSelectLeft);// Crop Left image
		Mat imgFrameCrop2CopyLeftGS = imgFrame2Copy(RectangleToSelectLeft);
		Mat imgDifferenceLeft;
		Mat imgThresholdLeft;

		Mat imgFrameCrop1CopyRight = imgFrame1Copy(RectangleToSelectRight);// Crop right image
		Mat imgFrameCrop2CopyRight = imgFrame2Copy(RectangleToSelectRight);
        Mat imgDifferenceRight;
        Mat imgThresholdRight;

		// 1) left region of interest
        cvtColor(imgFrameCrop1CopyLeft, imgFrameCrop1CopyLeftG, CV_BGR2GRAY);
        cvtColor(imgFrameCrop2CopyLeft, imgFrameCrop2CopyLeftG, CV_BGR2GRAY);

		// 2) 7 X 7 median filter
		medianBlur(imgFrameCrop1CopyLeftG, imgFrameCrop1CopyLeftM, 7);
		medianBlur(imgFrameCrop2CopyLeftG, imgFrameCrop2CopyLeftM, 7);	

		//3) difference of neighbour frames
		absdiff(imgFrameCrop1CopyLeftM, imgFrameCrop2CopyLeftM, imgDifferenceLeft);
    
		// 4) thresholding for segment edges of vehicles
		threshold(imgDifferenceLeft, imgThresholdLeft, 27, 255, CV_THRESH_BINARY); 

		//right region of interest Step 1,2,3,4
		cvtColor(imgFrameCrop1CopyRight, imgFrameCrop1CopyRight, CV_BGR2GRAY);
		cvtColor(imgFrameCrop2CopyRight, imgFrameCrop2CopyRight, CV_BGR2GRAY);
		medianBlur(imgFrameCrop1CopyRight, imgFrameCrop1CopyRight, 7);
		medianBlur(imgFrameCrop2CopyRight, imgFrameCrop2CopyRight, 7);
		absdiff(imgFrameCrop1CopyRight, imgFrameCrop2CopyRight, imgDifferenceRight);
		adaptiveThreshold(imgDifferenceRight, imgThresholdRight, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, -12);
        
        Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
   	
		// 5) morphological operations
		for (i = 0; i < 2; i++)
		{	
			erode(imgThresholdLeft, imgThresholdLeft, structuringElement3x3);		
			dilate(imgThresholdLeft, imgThresholdLeft, structuringElement5x5);
			dilate(imgThresholdLeft, imgThresholdLeft, structuringElement5x5);
			
		}
		// display morphological results
		imshow("Segment Left", imgThresholdLeft);
		
		for (i = 0; i < 2; i++)
		{	
			erode(imgThresholdRight, imgThresholdRight, structuringElement3x3);					
			dilate(imgThresholdRight, imgThresholdRight, structuringElement5x5);
			dilate(imgThresholdRight, imgThresholdRight, structuringElement5x5);
			
		}
		imshow("Segment Right", imgThresholdRight);	
	
		Mat imgThresholdCopyLeft = imgThresholdLeft.clone();
        vector<vector<Point> > contoursLeft;
		Mat imgThresholdCopyRight = imgThresholdRight.clone();
		vector<vector<Point> > contoursRight;
		//6) obtain contours from morphological objects
        findContours(imgThresholdCopyLeft, contoursLeft, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		// 7) calculate centroid positions from contours
		// 7.1)get the moments
		vector<Moments> muLeft(contoursLeft.size());
		for (int i = 0; i<contoursLeft.size(); i++)
		{
			muLeft[i] = moments(contoursLeft[i], false);
		}
		// 7.2)get the centroid of blobs.
		vector<Point2f> mcLeft(contoursLeft.size());
		for (int i = 0; i<contoursLeft.size(); i++)
		{
			mcLeft[i] = Point2f(muLeft[i].m10 / muLeft[i].m00, muLeft[i].m01 / muLeft[i].m00);
		}
		// 7.3)draw contours
		Mat drawingCentroidLeft(imgThresholdCopyLeft.size(), CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i<contoursLeft.size(); i++)
		{
			Scalar color = Scalar(167, 151, 0); // B G R values
			drawContours(drawingCentroidLeft, contoursLeft, i, color, 2, 8, RETR_EXTERNAL, 0, Point());
			circle(drawingCentroidLeft, mcLeft[i], 4, color, -1, 8, 0);
		}
		namedWindow("Left Contours", WINDOW_AUTOSIZE);
		imshow("Left Contours", drawingCentroidLeft);

		// 8) find convexHull points from contour points
		vector<vector<Point> > convexHullsLeft(contoursLeft.size());
		for (i = 0; i < contoursLeft.size(); i++)
		{
			convexHull(contoursLeft[i], convexHullsLeft[i]);
		}		
		
		findContours(imgThresholdCopyRight, contoursRight, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > convexHullsRight(contoursRight.size());
        for (i = 0; i < contoursRight.size(); i++)
		{
            convexHull(contoursRight[i], convexHullsRight[i]);
        }
		// 7.1)get the moments
		vector<Moments> muRight(contoursRight.size());
		for (int i = 0; i<contoursRight.size(); i++)
		{
			muRight[i] = moments(contoursRight[i], false);
		}
		// 7.2)get the centroid of blobs.
		vector<Point2f> mcRight(contoursRight.size());
		for (int i = 0; i<contoursRight.size(); i++)
		{
			mcRight[i] = Point2f(muRight[i].m10 / muRight[i].m00, muRight[i].m01 / muRight[i].m00);
		}
		// 7.3)draw Right contour centroids
		Mat drawingCentroidRight(imgThresholdCopyRight.size(), CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i<contoursRight.size(); i++)
		{
			Scalar color = Scalar(167, 151, 0); // B G R values
			drawContours(drawingCentroidRight, contoursRight, i, color, 2, 8, RETR_EXTERNAL, 0, Point());
			circle(drawingCentroidRight, mcRight[i], 4, color, -1, 8, 0);
		}
		namedWindow("Right Contours", WINDOW_AUTOSIZE);
		imshow("Right Contours", drawingCentroidRight);

		// 9) apply small object filter to detect blobs from convexhull points for left region of interest
		for (auto &convexHull : convexHullsLeft)
		{
			BlobDetector possibleBlob(convexHull);

			if (possibleBlob.currentBoundingRect.area() > m_ImageProcessFunctions.objectRectParameter &&
				possibleBlob.CurrentAspectRatio > m_ImageProcessFunctions.farAspectRatio &&
				possibleBlob.CurrentAspectRatio < m_ImageProcessFunctions.nearAspectRatio &&
				possibleBlob.currentBoundingRect.width > m_ImageProcessFunctions.objectSize &&
				possibleBlob.currentBoundingRect.height > m_ImageProcessFunctions.objectSize &&
				possibleBlob.CurrentDiagonalSize > m_ImageProcessFunctions.diagonalSize &&
				(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > m_ImageProcessFunctions.contourAndBoundingRectRatio)
			{	
				currentFrameBlobsLeft.push_back(possibleBlob);
			}
		}

		// right region of interest
		for (auto &convexHull : convexHullsRight)
		{
			BlobDetector possibleBlob(convexHull);

			if (possibleBlob.currentBoundingRect.area() > m_ImageProcessFunctions.objectRectParameter &&
				possibleBlob.CurrentAspectRatio > m_ImageProcessFunctions.farAspectRatio &&
				possibleBlob.CurrentAspectRatio < m_ImageProcessFunctions.nearAspectRatio &&
				possibleBlob.currentBoundingRect.width > m_ImageProcessFunctions.objectSize &&
				possibleBlob.currentBoundingRect.height > m_ImageProcessFunctions.objectSize &&
				possibleBlob.CurrentDiagonalSize > m_ImageProcessFunctions.diagonalSize &&
				(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > m_ImageProcessFunctions.contourAndBoundingRectRatio)
			{
				currentFrameBlobsRight.push_back(possibleBlob);
			}
		}
		
		// 10) update frames for tracking in left and right regions
        if (firstFrame == true) 
		{ 
			for (auto &currentFrameBlob : currentFrameBlobsLeft) 
			{
				m_blobDetectorLeft.push_back(currentFrameBlob);
            }
			for (auto &currentFrameBlob : currentFrameBlobsRight) 
			{
				m_blobDetectorRight.push_back(currentFrameBlob);
			}  			
		}
		else 
		{
			m_ImageProcessFunctions.matchCurrentFrameBlobsToExistingBlobs(m_blobDetectorLeft, currentFrameBlobsLeft);
			m_ImageProcessFunctions.matchCurrentFrameBlobsToExistingBlobs(m_blobDetectorRight, currentFrameBlobsRight);
        }   
		
		// update current frame!!
		imgFrame2Copy = imgFrame2.clone();	// get another copy of frame 2 since we changed the previous frame 2 copy in the processing above			
	    // display regions of interest in tracking
		imgFrameCrop2CopyLeft = imgFrame2Copy(RectangleToSelectLeft);
		m_ImageProcessFunctions.drawBlobInfoOnImage(m_blobDetectorLeft, imgFrameCrop2CopyLeft); 
		
		imshow("Crop Left", imgFrameCrop2CopyLeft);
		imgFrameCrop2CopyRight = imgFrame2Copy(RectangleToSelectRight);
		m_ImageProcessFunctions.drawBlobInfoOnImage(m_blobDetectorRight, imgFrameCrop2CopyRight); 
		imshow("Crop Right", imgFrameCrop2CopyRight);

		// 11) counting vehicles from the left region of interest
		m_ImageProcessFunctions.checkIfBlobsCrossedTheLineLeft(m_blobDetectorLeft, intHorizontalLinePositionLeft, carCountLeft);
		// 11) counting vehicles from the right region of interest
		m_ImageProcessFunctions.checkIfBlobsCrossedTheLineRight(m_blobDetectorRight, intHorizontalLinePositionRight, carCountRight);
	
		// 12) obtain the colour information from the tracked vehicle's centroid for left way!
		if (carCountLeft == carColourLeft)
		{	
			colourNameLeft = m_ImageProcessFunctions.findColourNameFromColourValue(imgFrameCrop2CopyLeft, mcLeft);
			carColourLeft++;
		}
		// 12) obtain the colour information from the tracked vehicle's centroid for right way!
		if (carCountRight == carColourRight)
		{
			colourNameRight = m_ImageProcessFunctions.findColourNameFromColourValue(imgFrameCrop2CopyRight, mcRight);
			carColourRight++;
		}

		// 13) Display final interface for counting results!
		m_ImageProcessFunctions.writeCarCountAndColourOnImage(carCountRight, carCountLeft, colourNameRight, colourNameLeft, imgFrame2Copy);

		imshow("imageFrame2Copy", imgFrame2Copy);
	
        // prepare for the next iteration
        currentFrameBlobsLeft.clear();
		currentFrameBlobsRight.clear();
        imgFrame1 = imgFrame2.clone();	// move frame 1 up to where frame 2 is
		
        if ((m_capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < m_capVideo.get(CV_CAP_PROP_FRAME_COUNT))
		{
			m_capVideo.read(imgFrame2);
        }
        else 
		{
            cout << "end of video\n" <<endl;
            break;
        }

        firstFrame = false;
        frameCount++;
		// the video is 25fps, needing a 0.04 ms delay between each frame
        checkForEscKey = waitKey(40);
    }

	// hold the windows open to allow the "end of video" message to show
    if (checkForEscKey != 27) 
	{               
        waitKey(0);         
    }

    return(0);
}

