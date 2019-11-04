
#include "ImageProcessFunctions.h"
#include<iostream>
#include <stdio.h>

int ImageProcessFunctions::OtsuThreshold(Mat &imgGrey)
{
	int ihist[256];   // Array to store Histogram values
	float hist_val[256];  // Array to store Normalised Histogram values
	float prbn;   // First order cumulative
	float meanitr;   // Second order cumulative
	float meanglb;   // Global mean level
	int OTSU_THRESH_VAL;             // Optimum threshold value
	float param1, param2;  // Parameters required to work out OTSU threshold algorithm
	double param3;
	int h, w, i,j;   // Variables to store Image Height and Width
	
	memset(ihist, 0, 256);

	for (i = 0; i < imgGrey.rows; i++)for (j = 0; j < imgGrey.cols; j++) // Use Histogram values from Gray image row = y, col = x
	{
		
		Scalar intensity = imgGrey.at<uchar>(i, j);
        int hist = (int)intensity.val[0];
		ihist[hist] += 1; // Use the pixel value as the position/"Weight"
	}
	//Parameters required to calculate threshold using OTSU Method
	prbn = 0.0;                   // First order cumulative
	meanitr = 0.0;                // Second order cumulative
	meanglb = 0.0;                // Global mean level
	OTSU_THRESH_VAL = 0;           // Optimum threshold value
	param1, param2;                // Parameters required to work out OTSU threshold algorithm
	param3 = 0.0;
	h = imgGrey.rows;  // Height and width of the Image
	w = imgGrey.cols;
	//Normalise histogram values and calculate global mean level
	for (i = 0; i < 256; ++i)
	{
		hist_val[i] = ihist[i] / (float)(w * h);
		meanglb += ((float)i * hist_val[i]);
	}

	// Implementation of OTSU algorithm
	for (i = 0; i < 255; i++)
	{
		prbn += (float)hist_val[i];
		meanitr += ((float)i * hist_val[i]);

		param1 = (float)((meanglb * prbn) - meanitr);
		param2 = (float)(param1 * param1) / (float)(prbn * (1.0f - prbn));

		if (param2 > param3)
		{
			param3 = param2;
			OTSU_THRESH_VAL = i;     // Update the "Weight/Value" as Optimum Threshold value
		}
	}

	return OTSU_THRESH_VAL;
}

void ImageProcessFunctions::matchCurrentFrameBlobsToExistingBlobs(vector<BlobDetector> &existingBlobs, vector<BlobDetector> &currentFrameBlobs)
{
	for (auto &existingBlob : existingBlobs)
	{
		existingBlob.CurrentMatchFoundOrNewBlob = false;
		existingBlob.predictNextBlobPosition();
	}

	for (auto &currentFrameBlob : currentFrameBlobs)
	{
		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 1000000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++)
		{

			if (existingBlobs[i].StillBeingTracked == true)
			{
				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance)
				{
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.CurrentDiagonalSize * 0.5)
		{
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else
		{
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	for (auto &existingBlob : existingBlobs)
	{
		if (existingBlob.CurrentMatchFoundOrNewBlob == false)
		{
			existingBlob.NumOfSubsequentNoMatch++;
		}
		if (existingBlob.NumOfSubsequentNoMatch >= 5)
		{
			existingBlob.StillBeingTracked = false;
		}
	}
}

void ImageProcessFunctions::addBlobToExistingBlobs(BlobDetector &currentFrameBlob, vector<BlobDetector> &existingBlobs, int &intIndex)
{
	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
	existingBlobs[intIndex].CurrentDiagonalSize = currentFrameBlob.CurrentDiagonalSize;
	existingBlobs[intIndex].CurrentAspectRatio = currentFrameBlob.CurrentAspectRatio;
	existingBlobs[intIndex].StillBeingTracked = true;
	existingBlobs[intIndex].CurrentMatchFoundOrNewBlob = true;
}

void ImageProcessFunctions::addNewBlob(BlobDetector &currentFrameBlob, vector<BlobDetector> &existingBlobs)
{
	currentFrameBlob.CurrentMatchFoundOrNewBlob = true;
	existingBlobs.push_back(currentFrameBlob);
}

double ImageProcessFunctions::distanceBetweenPoints(Point point1, Point point2)
{
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

void ImageProcessFunctions::drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName)
{
	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
	drawContours(image, contours, -1, SCALAR_WHITE, -1);
	imshow(strImageName, image);
}

void ImageProcessFunctions::drawAndShowContours(Size imageSize, vector<BlobDetector> blobs, string strImageName)
{
	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
	vector<vector<Point> > contours;

	for (auto &blob : blobs)
	{
		if (blob.StillBeingTracked == true)
		{
			contours.push_back(blob.currentContour);
		}
	}

	drawContours(image, contours, -1, SCALAR_WHITE, -1);
	imshow(strImageName, image);

}

void ImageProcessFunctions::checkIfBlobsCrossedTheLineLeft(vector<BlobDetector> &blobs, int &intHorizontalLinePosition, int &carCountRight)
{
	for (auto blob : blobs)
	{
		if (blob.StillBeingTracked == true && blob.centerPositions.size() >= 2)
		{
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			// away Left region of interest for RoadsTestCounting.mp4
			if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition &&
				blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition &&
				 blob.centerPositions[currFrameIndex].x > 0) 
			{
				carCountRight++;
			}
		}
	}

}

void ImageProcessFunctions::checkIfBlobsCrossedTheLineRight(vector<BlobDetector> &blobs, int &intHorizontalLinePosition, int &carCountLeft)
{

	for (auto blob : blobs)
	{
		if (blob.StillBeingTracked == true && blob.centerPositions.size() >= 2)
		{
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			// come in Right region of interest for RoadsTestCounting.mp4
			if (blob.centerPositions[prevFrameIndex].y <= intHorizontalLinePosition &&
				blob.centerPositions[currFrameIndex].y > intHorizontalLinePosition &&
				blob.centerPositions[currFrameIndex].x < 350 &&  //350
				blob.centerPositions[currFrameIndex].x > 0) //0
			{
				carCountLeft++;
			}
		}
	}

}

void ImageProcessFunctions::drawBlobInfoOnImage(vector<BlobDetector> &blobs, Mat &imgFrame2Copy)
{
	for (int i = 0; i < blobs.size(); i++)
	{
		if (blobs[i].StillBeingTracked == true)
		{
			rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

			int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
			int intFontThickness = (int)round(dblFontScale * 1.0);

		}
	}
}

void ImageProcessFunctions::findblobs(vector<vector<Point> > convexHulls, vector<BlobDetector> currentFrameBlobs)
{
	for (auto &convexHull : convexHulls)
    {
	    BlobDetector possibleBlob(convexHull);

	    if (possibleBlob.currentBoundingRect.area() > objectRectParameter &&
		    possibleBlob.CurrentAspectRatio > farAspectRatio &&
		    possibleBlob.CurrentAspectRatio < nearAspectRatio &&
		    possibleBlob.currentBoundingRect.width > objectSize &&
		    possibleBlob.currentBoundingRect.height > objectSize &&
		    possibleBlob.CurrentDiagonalSize > diagonalSize &&
		    (contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > contourAndBoundingRectRatio)
	    {
		    currentFrameBlobs.push_back(possibleBlob);
	    }
    }

}

vector<Point3f> ImageProcessFunctions::findColourInfoOnImage(Mat &imgFrameblob, Mat &imgFrameColour, int centralLine, int countNum)
{
	vector<Point3f> colourInfo(countNum);
	Scalar intensityGrey1, intensityGrey2;
	Vec3b intensity1, intensity2, intensityTotal;
	int colWidth = 0;
	
	colWidth = imgFrameblob.cols - 1;
	
	for (int x = 1; x < colWidth; x++)
	{
		intensityGrey1 = imgFrameblob.at<uchar>(Point(x, centralLine));
		intensityGrey2 = imgFrameblob.at<uchar>(Point((x +1), centralLine));
		if (intensityGrey1.val[0] == 255 && intensityGrey1.val[0] == 255)
		{
			intensity1 = imgFrameColour.at<Vec3b>(Point(x, centralLine));
			intensity2 = imgFrameColour.at<Vec3b>(Point((x + 1), centralLine));
			intensityTotal = (intensityTotal + intensity1 + intensity2) / 3;	
				
		}		
	}
	colourInfo[countNum].x = intensityTotal.val[0];
	colourInfo[countNum].y = intensityTotal.val[1];
	colourInfo[countNum].z = intensityTotal.val[2];
	return  colourInfo;
}

Vec3b ImageProcessFunctions::findCentroidColourAverageOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid, vector<vector<Point> > contours)
{
	Vec3b intensityCentroid, intensityCentroid1, intensityCentroid2, intensityCentroid3, intensityCentroid4, intensityCentroid5;
	int row, col;
	// obtain the colour from the vehicle's centroid!
	intensityCentroid.val[0] = intensityCentroid.val[1] = intensityCentroid.val[2] = 0;

	for (int i = 0; i < contours.size(); i++)
	{
		row = (int) muCentroid[i].y;
		col = (int) muCentroid[i].x;
		
		intensityCentroid = imgFrameColour.at<Vec3b>(Point(row, col));

		if (row > 2 && col > 2 && row < 574 && col < 1020)
		{
			intensityCentroid1 = imgFrameColour.at<Vec3b>(Point(row, col));
			intensityCentroid2 = imgFrameColour.at<Vec3b>(Point(row + 1, col + 1));
			intensityCentroid3 = imgFrameColour.at<Vec3b>(Point(row + 1, col - 1));
			intensityCentroid4 = imgFrameColour.at<Vec3b>(Point(row - 1, col + 1));
			intensityCentroid5 = imgFrameColour.at<Vec3b>(Point(row - 1, col - 1));
			intensityCentroid = (Vec3b) (intensityCentroid1 + intensityCentroid2 + intensityCentroid3 + intensityCentroid4 + intensityCentroid5) / 5;
		}
		else
			intensityCentroid = imgFrameColour.at<Vec3b>(Point(row, col));

	}
	
	cout << "intensityCentroid[0] = " << (int) intensityCentroid[0] << "  ;" << (int)intensityCentroid[1] << "  ;" << (int)intensityCentroid[2] << endl;
	//cout << "intensityCentroid1[0] = " << (int)intensityCentroid1[0] << "  ;" << (int)intensityCentroid1[1] << "  ;" << (int)intensityCentroid1[2] << endl;
		
	return  intensityCentroid;
}

Vec3b ImageProcessFunctions::findCentroidColourInfoOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid)
{
	Vec3b intensityCentroid;
	int row, col;
	// obtain the colour from the vehicle's centroid!
	//intensityCentroid.val[0] = intensityCentroid.val[1] = intensityCentroid.val[2] = 0;
		row = (int)muCentroid[0].y;
		col = (int)muCentroid[0].x;
		intensityCentroid = imgFrameColour.at<Vec3b>(Point(row, col));

	return  intensityCentroid;
}

Vec3b ImageProcessFunctions::findCentroidColourInfoOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid, vector<vector<Point> > contours)
{
	Vec3b intensityCentroid;
	int row, col;
	// obtain the colour from the vehicle's centroid!
	intensityCentroid.val[0] = intensityCentroid.val[1] = intensityCentroid.val[2] = 0;

	for (int i = 0; i < contours.size(); i++)
	{
	row = (int)muCentroid[i].y-12;
	col = (int)muCentroid[i].x;
	intensityCentroid = imgFrameColour.at<Vec3b>(Point(row, col));
	}

	return  intensityCentroid;
}

Scalar ImageProcessFunctions::findCentroidColourOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid)
{
	Scalar intensityCentroid;
	int row, col;
	// obtain the colour from the vehicle's centroid!
	row = (int)muCentroid[0].y;
	col = (int)muCentroid[0].x;
	Vec3b intensity = imgFrameColour.at<Vec3b>(row, col);
	uchar blue = intensity.val[0];
	uchar green = intensity.val[1];
	uchar red = intensity.val[2];
	intensityCentroid = Scalar(blue, green, red);
	return  intensityCentroid;
}

Scalar ImageProcessFunctions::findCentroidColourAverageOnImage(Mat &imgFrameColour, vector<Point2f> muCentroid)
{
	Scalar intensityCentroid;
	float blueAverage = 0, greenAverage = 0, redAverage = 0;
	int row, col, i, j, k = 0;
	// obtain the average colour value around of the vehicle's centroid!
	row = (int)muCentroid[0].y;
	col = (int)muCentroid[0].x;
	for (i = -1; i < 2; i++)for (j = -1; j < 2; j++)
	{
		Vec3b intensity = imgFrameColour.at<Vec3b>(row + i, col + j);
		float blue = intensity.val[0];
		float green = intensity.val[1];
		float red = intensity.val[2];
		blueAverage += blue;
		greenAverage += green;
		redAverage += red;
	}
	blueAverage = blueAverage / 9;
	greenAverage = greenAverage / 9;
	redAverage = redAverage / 9;
	intensityCentroid = Scalar(blueAverage, greenAverage, redAverage);

	return  intensityCentroid;
}

void ImageProcessFunctions::writeCarCountAndColourOnImage(int &carCountRight, int &carCountLeft, Scalar &centroidColourLeft, Scalar &centroidColourRight, Mat &imgFrame2Copy)
{
	int FontFace = CV_FONT_HERSHEY_SIMPLEX;
	double FontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 950000.0;
	int FontThickness = (int)round(FontScale * 2.5);

	// Left region og interest
	putText(imgFrame2Copy, "Left Vehicle count: " + to_string(carCountLeft), Point(10, 25), FontFace, FontScale, SCALAR_GREEN, FontThickness);
	putText(imgFrame2Copy, "Left Vehicle Colour: ", Point(10, 65), FontFace, FontScale, SCALAR_GREEN, FontThickness);
	circle(imgFrame2Copy, Point(330, 62), 8, centroidColourLeft, -2, 8, 0);

	// Right region of interest
	putText(imgFrame2Copy, "Right Vehicle count: " + to_string(carCountRight), Point(640, 25), FontFace, FontScale, SCALAR_RED, FontThickness);
	putText(imgFrame2Copy, "Right Vehicle Colour: ", Point(640, 65), FontFace, FontScale, SCALAR_RED, FontThickness);
	circle(imgFrame2Copy, Point(980, 62), 8, centroidColourRight, -2, 8, 0);
	
}

void ImageProcessFunctions::writeCarCountAndColourOnImage(int &carCountRight, int &carCountLeft, String colourTextRight, String colourTextLeft, Mat &imgFrame2Copy)
{
	int FontFace = CV_FONT_HERSHEY_SIMPLEX;
	double FontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 950000.0;
	int FontThickness = (int)round(FontScale * 2.5);

	// Left region og interest
	putText(imgFrame2Copy, "Left Vehicle count: " + to_string(carCountLeft), Point(10, 25), FontFace, FontScale, SCALAR_GREEN, FontThickness);
	putText(imgFrame2Copy, "Left Vehicle Colour: " + colourTextLeft, Point(10, 65), FontFace, FontScale, SCALAR_GREEN, FontThickness);


	// Right region of interest
	putText(imgFrame2Copy, "Right Vehicle count: " + to_string(carCountRight), Point(640, 25), FontFace, FontScale, SCALAR_RED, FontThickness);
	putText(imgFrame2Copy, "Right Vehicle Colour: " + colourTextRight, Point(640, 65), FontFace, FontScale, SCALAR_RED, FontThickness);

}

String ImageProcessFunctions::findColourNameFromColourValue(Mat &imgFrameColour, vector<Point2f> muCentroid)
{

	// colour lookup table
	float red[3] = { 0, 0, 255 };
	float green[3] = { 0, 255,0 };
	float blue[3] = { 255, 0,0 };
	float orange[3] = { 0, 165, 255 };
	float yellow[3] = { 0, 255, 255 };
	float indigo[3] = { 130, 0, 75 };
	float violet[3] = { 238, 130, 238 };
	float grey[3] = { 127,127,127 };
	float white[3] = { 255, 255, 255 };
	float black[3] = { 0, 0, 0 };
	String labels[10] = { "Red","Green","Blue","Orange", "Yellow","Indigo","Violet","Grey","White","Black" };

	int row, col, index = 0;
	float blueDifference = 0.0, greenDifference = 0.0, redDifference = 0.0, colourMin = 800.0, colourDifferenceBGR = 0.0;
	// obtain the colour from the vehicle's centroid!
	row = (int)muCentroid[0].y;
	col = (int)muCentroid[0].x;
	Vec3b intensity = imgFrameColour.at<Vec3b>(row, col);
	float blueValue = intensity.val[0];
	float greenValue = intensity.val[1];
	float redValue = intensity.val[2];

	// red
	blueDifference = red[0] - blueValue;
	greenDifference = red[1] - greenValue;
	redDifference = red[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);

	if (colourDifferenceBGR < colourMin)
	{
		index = 0;
		colourMin = colourDifferenceBGR;
	}
	//green
	blueDifference = green[0] - blueValue;
	greenDifference = green[1] - greenValue;
	redDifference = green[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 1;
		colourMin = colourDifferenceBGR;
	}
	//blue
	blueDifference = blue[0] - blueValue;
	greenDifference = blue[1] - greenValue;
	redDifference = blue[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 2;
		colourMin = colourDifferenceBGR;
	}
	//orange
	blueDifference = orange[0] - blueValue;
	greenDifference = orange[1] - greenValue;
	redDifference = orange[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 3;
		colourMin = colourDifferenceBGR;
	}
	//yellow
	blueDifference = yellow[0] - blueValue;
	greenDifference = yellow[1] - greenValue;
	redDifference = yellow[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 4;
		colourMin = colourDifferenceBGR;
	}
	//indigo
	blueDifference = indigo[0] - blueValue;
	greenDifference = indigo[1] - greenValue;
	redDifference = indigo[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 5;
		colourMin = colourDifferenceBGR;
	}
	//violet
	blueDifference = violet[0] - blueValue;
	greenDifference = violet[1] - greenValue;
	redDifference = violet[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 6;
		colourMin = colourDifferenceBGR;
	}
	//grey
	blueDifference = grey[0] - blueValue;
	greenDifference = grey[1] - greenValue;
	redDifference = grey[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 7;
		colourMin = colourDifferenceBGR;
	}

	//white
	blueDifference = white[0] - blueValue;
	greenDifference = white[1] - greenValue;
	redDifference = white[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 8;
		colourMin = colourDifferenceBGR;
	}
	//black
	blueDifference = black[0] - blueValue;
	greenDifference = black[1] - greenValue;
	redDifference = black[2] - redValue;
	colourDifferenceBGR = fabs(blueDifference) + fabs(greenDifference) + fabs(redDifference);
	if (colourDifferenceBGR < colourMin)
	{
		index = 9;
		colourMin = colourDifferenceBGR;
	}

	return labels[index];

}