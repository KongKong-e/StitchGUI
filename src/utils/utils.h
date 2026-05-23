#pragma once


#include <string>
#include <vector>

#include "dlldefine.h"

#include <opencv2/opencv.hpp>


std::wstring  SUPERSTITCH_EXPORTS str2wstr(std::string str);


std::vector<cv::Mat>  SUPERSTITCH_EXPORTS readImages(
	std::string dir, std::string ext, bool isDivide = false);


