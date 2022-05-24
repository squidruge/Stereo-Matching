#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <omp.h>
#include"config.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
	char Left_Origin_String[100];
	char Right_Origin_String[100];
	char Left_Disparity_String[100];
	char Right_Disparity_String[100];


	strcpy_s(Left_Origin_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Right_Origin_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Left_Disparity_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Right_Disparity_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");


	strcat_s(Left_Origin_String, "2003\\cones\\im2.png");
	strcat_s(Right_Origin_String, "2003\\cones\\im6.png");
	strcat_s(Left_Disparity_String, "2003\\cones\\disp2.png");
	strcat_s(Right_Disparity_String, "2003\\cones\\disp6.png");



	cv::Mat left_img = cv::imread(Left_Origin_String);
	if (!left_img.data) {
		std::cerr << "Couldn't read the file " << Left_Origin_String << std::endl;
		return EXIT_FAILURE;
	}

	cv::Mat right_img = cv::imread(Right_Origin_String);
	if (!right_img.data) {
		std::cerr << "Couldn't read the file " << Right_Origin_String << std::endl;
		return EXIT_FAILURE;
	}

#if FIGS
	cv::imshow("left_img", left_img);
	waitKey(0);
	cv::imshow("right_img", right_img);
	waitKey(0);

#endif // FIGS


	// Convert images to grayscale
	if (left_img.channels() > 1) {
		cv::cvtColor(left_img, left_img, cv::COLOR_RGB2GRAY);
	}

	if (right_img.channels() > 1) {
		cv::cvtColor(right_img, right_img, cv::COLOR_RGB2GRAY);
	}

#if FIGS
	cv::imshow("left_img", left_img);
	waitKey(0);
	cv::imshow("right_img", right_img);
	waitKey(0);

#endif // FIGS

	return 0;
}