#pragma warning(disable:4996)

#include "utils.h"
#include "file_utils.h"
#include <QImage>
#include <QDebug>


std::wstring str2wstr(std::string str)
{

	std::wstring wstr = std::wstring(str.begin(), str.end());


	return wstr;

}


std::vector<cv::Mat> SUPERSTITCH_EXPORTS readImages(
	std::string dir, std::string ext, bool isDivide /*= false*/)
{

	std::cout << "[DEBUG] 读取图像：" << std::endl;
	std::cout << "[DEBUG] 目录：" << dir << std::endl;
	std::cout << "[DEBUG] 扩展名：" << ext << std::endl;

	std::vector<std::string> fileNames;

	libpano::get_filenames_with_absolute_path(dir, fileNames, ext);

	std::cout << "[DEBUG] 找到 " << fileNames.size() << " 个文件" << std::endl;
	for (size_t i = 0; i < fileNames.size(); i++) {
		std::cout << "[DEBUG] 文件 " << i << ": " << fileNames[i] << std::endl;
	}

	std::vector<cv::Mat> imgs;

	for (int j = 0; j < fileNames.size(); j++)
	{
		QString qFileName = QString::fromUtf8(fileNames[j].c_str());
		QImage qImage(qFileName);
		
		if (qImage.isNull())
		{
			std::cout << "Can't read image '" << fileNames[j] << "'\n";
			continue;
		}
		
		cv::Mat img;
		cv::Mat temp(qImage.height(), qImage.width(), CV_8UC4, const_cast<uchar*>(qImage.bits()), qImage.bytesPerLine());
		cv::cvtColor(temp, img, cv::COLOR_BGRA2BGR);

		if (img.empty())
		{
			std::cout << "Can't convert image '" << fileNames[j] << "'\n";
			continue;
		}

		if (isDivide)
		{
			cv::Rect rect(0, 0, img.cols / 2, img.rows);
			imgs.push_back(img(rect).clone());
			rect.x = img.cols / 3;
			imgs.push_back(img(rect).clone());
			rect.x = img.cols / 2;
			imgs.push_back(img(rect).clone());
		}
		else
		{
			imgs.push_back(img);
		}

	}

	std::cout << "[DEBUG] 成功读取 " << imgs.size() << " 张图像" << std::endl;

	return imgs;
}
