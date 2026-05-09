#pragma warning(disable:4996)

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <map>
#include <onnxruntime_cxx_api.h>

#include "superpoint.h"



SuperPoint::SuperPoint(std::wstring modelPath)
{
	this->mModelPath = modelPath;
}


std::vector<float> SuperPoint::ApplyTransform(const cv::Mat& image, float& mean, float& std)
{
	cv::Mat resized, floatImage;
	image.convertTo(floatImage, CV_32FC1);

	std::vector<float> imgData;
	for (int h = 0; h < image.rows; h++)
	{
		for (int w = 0; w < image.cols; w++)
		{
			/*imgData.push_back((floatImage.at<float>(h, w) - mean) / std);*/
			imgData.push_back(floatImage.at<float>(h, w) / 255.0f);
		}
	}

	return imgData;
}

void SuperPoint::detectAndCompute(
	cv::InputArray image, cv::InputArray mask,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors,
	bool useProvidedKeypoints)
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	static Ort::SessionOptions sessionOptions;
	static std::map<std::wstring, Ort::Session*> sessions;
	
	if (sessions.find(this->mModelPath) == sessions.end())
	{
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		
		// 在 Windows 上直接使用宽字符路径，避免编码转换导致的乱码问题
		sessions[this->mModelPath] = new Ort::Session(env, this->mModelPath.c_str(), sessionOptions);
	}

	Ort::Session* extractorSession = sessions[this->mModelPath];

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;
	std::vector<float> imgData = this->ApplyTransform(grayImg, mean, std);

	std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = extractorSession->Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();

	cv::Mat desmat = descriptors.getMat();
	desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);
	for (int h = 0; h < desshape[1]; h++)
	{
		for (int w = 0; w < desshape[2]; w++)
		{
			int index = h * desshape[2] + w;
			desmat.at<float>(h, w) = des[index];
		}
	}

	desmat.copyTo(descriptors);

}

void SuperPoint::detect(
	cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::InputArray mask)
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	static Ort::SessionOptions sessionOptions;
	static std::map<std::wstring, Ort::Session*> sessions;
	
	if (sessions.find(this->mModelPath) == sessions.end())
	{
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		
		// 在 Windows 上直接使用宽字符路径，避免编码转换导致的乱码问题
		sessions[this->mModelPath] = new Ort::Session(env, this->mModelPath.c_str(), sessionOptions);
	}

	Ort::Session* extractorSession = sessions[this->mModelPath];

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;
	std::vector<float> imgData = this->ApplyTransform(grayImg, mean, std);

	std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = extractorSession->Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();

	int keypntcounts = kpshape[1];

	keypoints.resize(keypntcounts);

	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}
}


void SuperPoint::compute(
	cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors)
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	static Ort::SessionOptions sessionOptions;
	static std::map<std::wstring, Ort::Session*> sessions;
	
	if (sessions.find(this->mModelPath) == sessions.end())
	{
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		
		// 在 Windows 上直接使用宽字符路径，避免编码转换导致的乱码问题
		sessions[this->mModelPath] = new Ort::Session(env, this->mModelPath.c_str(), sessionOptions);
	}

	Ort::Session* extractorSession = sessions[this->mModelPath];

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;

	std::vector<float> imgData = this->ApplyTransform(grayImg, mean, std);

	std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;

	std::vector<Ort::Value> outputs = extractorSession->Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();
	cv::Mat desmat = descriptors.getMat();
	desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);

	for (int h = 0; h < desshape[1]; h++)
	{
		for (int w = 0; w < desshape[2]; w++)
		{
			int index = h * desshape[2] + w;
			desmat.at<float>(h, w) = des[index];
		}
	}

	desmat.copyTo(descriptors);
}
