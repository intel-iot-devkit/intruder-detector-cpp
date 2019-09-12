/*
 * Copyright (c) 2018 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <string>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace std;

static string conf_targetDevice;
static string conf_modelPath;
static string conf_binFilePath;
static string conf_labelsFilePath;
static const string conf_file = "../resources/config.json";
static const size_t conf_batchSize = 1;
static const int conf_windowColumns = 2; // OpenCV windows per each row
const int displayWindowWidth = 768;
const int displayWindowHeight = 432;
bool loopVideos = false;

static const int conf_fourcc = 0x00000021; 

static const double conf_thresholdValue = 0.55;
static const int conf_candidateConfidence = 4;
static std::vector<std::string> acceptedDevices{"CPU", "GPU", "MYRIAD", "HETERO:FPGA,CPU", "HDDL"};

typedef struct {
	char time[25];
	string intruder;
	int count;
	int frame;
} event;

class VideoCap {
public:
	size_t inputWidth;
	size_t inputHeight;
	const string inputVideo;
	cv::Mat frame;
	int noLabels; // Number of labels
	vector<int> lastCorrectCount;
	vector<int> totalCount;
	vector<int> currentCount;
	vector<bool> changedCount;

	vector<int> candidateCount;
	vector<int> candidateConfidence;

	vector<string> labelName;
	vector<event> events;
	cv::VideoCapture vc;
	cv::VideoWriter vw;

	int frameCount = 0;
	int loopFrames = 0;
	bool isCam = false;

	const string camName;
	const string videoName;

	VideoCap(size_t inputWidth,
			 size_t inputHeight,
			 const string inputVideo,
			 const string camName,
			 int number)
		: inputWidth(inputWidth)
		, inputHeight(inputHeight)
		, inputVideo(inputVideo)
		, vc(inputVideo.c_str())
		, frame()
		, camName(camName)
		, videoName("../UI/resources/videos/video" + to_string(number+1) + ".mp4") {
			if (!vc.isOpened())
			{
				std::cout << "Couldn't open video " << inputVideo << std::endl;
				exit(1);
			}
		}

	VideoCap(size_t inputWidth,
			 size_t inputHeight,
			 const int inputVideo,
			 const string camName,
			 int number)
		: inputWidth(inputWidth)
		, inputHeight(inputHeight)
		, inputVideo("stream")
		, vc(inputVideo)
		, camName(camName)
		, videoName("../UI/resources/videos/video" + to_string(number+1) + ".mp4") {
			if (!vc.isOpened())
			{
				std::cout << "Couldn't open video " << inputVideo << std::endl;
				exit(1);
			}
			isCam = true;
		}

	void init (int size)
	{
		noLabels = size;
		lastCorrectCount = vector<int>(size);
		totalCount = vector<int>(size);
		currentCount = vector<int>(size);
		changedCount = vector<bool>(size);
		candidateCount = vector<int>(size);
		candidateConfidence = vector<int>(size);
		labelName = vector<string>(size);
	}

	int initVW(int height, int width)
	{
		vw.open(videoName, conf_fourcc, vc.get(cv::CAP_PROP_FPS), cv::Size(width, height), true);
		if (!vw.isOpened())
		{
			return 1;
		}
	}
};
