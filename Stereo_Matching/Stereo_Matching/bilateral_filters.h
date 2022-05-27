#pragma once
//#include "stdafx.h"
#include "opencv2/opencv.hpp"
//#include <vector>

void getColorMask(std::vector<double>& colorMask, double colorSigma);
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma);
void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma);

