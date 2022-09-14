#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "ADCensusStereo.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

const float32 delta_d = 2.0f;
const float32 para[4] = { 0.0f, 8.0f, 0.0f, 4.0f };

/**
 * Loads a PFM image stored in little endian and returns the image as an OpenCV Mat.
 * @brief loadPFM
 * @param filePath
 * @return
 */
cv::Mat LoadPFM(const std::string filePath);

/**
 * Saves the image as a PFM file.
 * @brief savePFM
 * @param image
 * @param filePath
 * @return
 */
bool savePFM(const cv::Mat image, const std::string filePath);

/**
 * @brief 评估匹配准确度
 * @param 
 * @return
 */
/*评估匹配准确度*/
void PerformanceEval(const float32* disp_estimated, const sint32& width, const sint32& height, const std::string& GT_path, const float32& time);

void PerformanceEval0103(const float32* disp_estimated, const sint32& width, const sint32& height, const std::string& GT_path, const float64& time, const sint32 type);

