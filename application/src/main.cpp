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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"

#include <ie_icnn_net_reader.h>
#include <ie_device.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include <ie_extension.h>
#include <ext_list.hpp>

#include <videocap.hpp>

using namespace cv;
using namespace InferenceEngine::details;
using namespace InferenceEngine;


// Parse the environmental variables
void parseEnv()
{
	if (const char *env_d = std::getenv("DEVICE"))
	{
		conf_targetDevice = std::string(env_d);
	}

	if (const char *env_d = std::getenv("LOOP"))
	{
		if (std::string(env_d) == "true")
		{
			loopVideos = true;
		}
	}

}


// Parse the command line argument
void parseArgs(int argc, char **argv)
{
	if ("-h" == std::string(argv[1]) || "--help" == std::string(argv[1]))
	{
		std::cout << argv[0] << " -m MODEL -l LABELS [OPTIONS]\n\n"
					"-m, --model	Path to .xml file containing model layers\n"
					"-l, --labels	Path to labels file\n"
					"-d, --device	Device to run the inference (CPU, GPU, MYRIAD, FPGA or HDDL only). Default option is CPU\n"
					"-lp, --loop	Loop video to mimic continuous input\n";
		exit(0);
	}

	for (int i = 1; i < argc; i += 2)
	{
		if ("-m" == std::string(argv[i]) || "--model" == std::string(argv[i]))
		{
			conf_modelPath = std::string(argv[i + 1]);
			int pos = conf_modelPath.rfind(".");
			conf_binFilePath = conf_modelPath.substr(0, pos) + ".bin";
		}
		if ("-l" == std::string(argv[i]) || "--labels" == std::string(argv[i]))
		{
			conf_labelsFilePath = std::string(argv[i + 1]);
		}
		if ("-d" == std::string(argv[i]) || "--device" == std::string(argv[i]))
		{
			conf_targetDevice = std::string(argv[i + 1]);
		}
		if ("-lp" == std::string(argv[i]) || "--loop" == std::string(argv[i]))
		{
			if (std::string(argv[i + 1]) == "true")
			{
				loopVideos = true;
			}
			if (std::string(argv[i + 1]) == "false")
			{
				loopVideos = false;
			}
		}
	}
}


// Validate the command line arguments
void checkArgs()
{
	if (conf_modelPath.empty())
	{
		std::cout << "You need to specify the path to the .xml file\n";
		std::cout << "Use -m MODEL or --model MODEL\n";
		exit(11);
	}

	if (conf_labelsFilePath.empty())
	{
		std::cout << "You need to specify the path to the labels file\n";
		std::cout << "Use -l LABELS or --labels LABELS\n";
		exit(12);
	}

	if (conf_targetDevice.empty())
	{
		conf_targetDevice = "CPU";
	}
	else if (!(std::find(acceptedDevices.begin(), acceptedDevices.end(), conf_targetDevice) != acceptedDevices.end()))
	{
		std::cout << "Unsupported device " << conf_targetDevice << std::endl;
		exit(13);
	}
}


static InferenceEngine::InferenceEnginePluginPtr loadPlugin(TargetDevice myTargetDevice)
{
	InferenceEngine::PluginDispatcher dispatcher({""});

	return static_cast<InferenceEngine::InferenceEnginePluginPtr>(dispatcher.getPluginByDevice(
		conf_targetDevice));
}


static void configureNetwork(InferenceEngine::CNNNetReader &network, TargetDevice myTargetDevice)
{
	try
	{
		network.ReadNetwork(conf_modelPath);
	}
	catch (InferenceEngineException ex)
	{
		std::cerr << "Failed to load network: " << std::endl;
	}

	network.ReadWeights(conf_binFilePath);

	// Set batch size
	network.getNetwork().setBatchSize(conf_batchSize);
}


// Read the model's label file and get the position of labels required by the application
static std::vector<bool> getUsedLabels(std::vector<string> *reqLabels, std::vector<int> *labelPos, std::vector<string> *labelNames)
{
	std::vector<bool> usedLabels;

	std::ifstream labelsFile(conf_labelsFilePath);

	if (!labelsFile.is_open())
	{
		std::cout << "Could not open labels file" << std::endl;
		return usedLabels;
	}

	std::string label;
	int i = 0;
	while (getline(labelsFile, label))
	{
		if (std::find((*reqLabels).begin(), (*reqLabels).end(), label) != (*reqLabels).end())
		{
			usedLabels.push_back(true);
			(*labelPos).push_back(i);
			(*labelNames).push_back(label);
			++i;
		}
		else
		{
			usedLabels.push_back(false);
			(*labelPos).push_back(0);
		}
	}

	labelsFile.close();

	return usedLabels;
}


// Parse the configuration file conf.txt
std::vector<VideoCap> getInput(std::ifstream *file, size_t width, size_t height, vector<string> *usedLabels)
{
	std::vector<VideoCap> streams;
	std::string str;
	int cams = 0;
	char camName[20];

	while (std::getline(*file, str))
	{
		int delim = str.find(':');
		if (str.substr(0, delim) == string("video"))
		{
			++cams;
			sprintf(camName, "Cam %d", cams);
			std::string path = str.substr(delim + 2);
			if (path.size() == 1 && *(path.c_str()) >= '0' && *(path.c_str()) <= '9') // Get cam ID
			{
				streams.push_back(VideoCap(width, height, 0, camName, cams));
			}
			else
			{
				streams.push_back(VideoCap(width, height, path, camName, cams));
			}
		}
		else if (str.substr(0, delim) == string("intruder"))
		{
			(*usedLabels).push_back(str.substr(delim + 2));
		}
		else
		{
			cout << "Unrecognized option; Ignoring\n";
		}
	}

	for (int i = 0; i < streams.size(); ++i)
	{
		streams[i].init((*usedLabels).size());
	}

	return streams;
}


// Get the minimum fps of the videos
int get_minFPS(std::vector<VideoCap> &vidCaps)
{
	int minFPS = 240;

	for (auto &&i : vidCaps)
	{
		minFPS = std::min(minFPS, (int)round(i.vc.get(CAP_PROP_FPS)));
	}

	return minFPS;
}


// Arranges the windows so that they are not overlapping
void arrangeWindows(vector<VideoCap> *vidCaps, size_t width, size_t height)
{
	int spacer = 470;
	int rowSpacer = 250;
	int cols = 0;
	int rows = 0;

	namedWindow("Intruder Log", WINDOW_AUTOSIZE);
	moveWindow("Intruder Log", 0, 0);

	for (int i = 0; i < (*vidCaps).size(); ++i)
	{
		namedWindow((*vidCaps)[i].camName, WINDOW_NORMAL);
		resizeWindow((*vidCaps)[i].camName, displayWindowWidth, displayWindowHeight);

		if (cols == conf_windowColumns)
		{
			cols = 1;
			++rows;
			moveWindow((*vidCaps)[i].camName, spacer * cols, rowSpacer * rows);
		}
		else
		{
			++cols;
			moveWindow((*vidCaps)[i].camName, spacer * cols, rowSpacer * rows);
		}
	}
}


// Write the video results to json files
void saveJSON(vector<event> events, VideoCap vcap)
{

	ofstream evtJson("../../UI/resources/video_data/events.json");
	if (!evtJson.is_open())
	{
		cout << "Could not create JSON file" << endl;
		return;
	}

	ofstream dataJson("../../UI/resources/video_data/data.json");
	if (!dataJson.is_open())
	{
		cout << "Could not create JSON file" << endl;
		return;
	}

	int total;

	evtJson << "{\n\t\"video1\": {\n";
	dataJson << "{\n\t\"video1\": {\n";
	if (!events.empty())
	{
		int fps = vcap.vc.get(cv::CAP_PROP_FPS);
		int evts = static_cast<int>(events.size());
		int i = 0;
		for (; i < evts - 1; ++i)
		{
			evtJson << "\t\t\"" << i << "\":{\n";
			evtJson << "\t\t\t\"time\":\"" << events[i].time << "\",\n";
			evtJson << "\t\t\t\"content\":\"" << events[i].intruder << "\",\n";
			evtJson << "\t\t\t\"videoTime\":\"" << (float)events[i].frame / fps << "\"\n";
			evtJson << "\t\t},\n";

			dataJson << "\t\t\"" << (float)events[i].frame / fps << "\": \"" << events[i].count << "\",\n";
		}

		evtJson << "\t\t\"" << i << "\":{\n";
		evtJson << "\t\t\t\"time\":\"" << events[i].time << "\",\n";
		evtJson << "\t\t\t\"content\":\"" << events[i].intruder << "\",\n";
		evtJson << "\t\t\t\"videoTime\":\"" << (float)events[i].frame / fps << "\"\n";
		evtJson << "\t\t}\n";

		dataJson << "\t\t\"" << (float)events[i].frame / fps << "\": \"" << events[i].count << "\"\n";
		total = events[i].count;
	}
	evtJson << "\t}\n";
	evtJson << "}";

	dataJson << "\t},\n";
	dataJson << "\t\"totals\":{\n";
	dataJson << "\t\t\"video1\": \"" << total << "\"\n";
	dataJson << "\t}\n";
	dataJson << "}";
}



int main(int argc, char **argv)
{

	int logWinHeight = 432;
	int logWinWidth = 410;
	std::vector<bool> noMoreData;
	parseEnv();
	parseArgs(argc, argv);
	checkArgs();

	std::ifstream confFile(conf_file);
	if (!confFile.is_open())
	{
		cout << "Could not open config file" << endl;
		return 2;
	}

	// Inference engine initialization
	// Set the Target Device
	TargetDevice myTargetDevice = TargetDeviceInfo::fromStr(conf_targetDevice);

	// Load the IE plugin for the target device
	InferenceEngine::InferenceEnginePluginPtr p_plugin = loadPlugin(
		myTargetDevice);

	// Configure the network
	InferenceEngine::CNNNetReader network;
	configureNetwork(network, myTargetDevice);

	// Set input configuration
	InputsDataMap inputs;
	inputs = network.getNetwork().getInputsInfo();

	if (inputs.size() != 1)
	{
		std::cout << "This sample accepts networks having only one input."
				  << std::endl;
		return 1;
	}

	InferenceEngine::SizeVector inputDims = inputs.begin()->second->getDims();

	if (inputDims.size() != 4)
	{
		std::cout << "Not supported input dimensions size, expected 4, got "
				  << inputDims.size() << std::endl;
	}

	std::string imageInputName = inputs.begin()->first;
	DataPtr image = inputs[imageInputName]->getInputData();
	inputs[imageInputName]->setInputPrecision(Precision::FP32);

	// Allocate input blobs
	InferenceEngine::BlobMap inputBlobs;
	InferenceEngine::TBlob<float>::Ptr pInputBlobData = InferenceEngine::make_shared_blob<float,
							   const InferenceEngine::SizeVector>(Precision::FP32, inputDims);
	pInputBlobData->allocate();
	inputBlobs[imageInputName] = pInputBlobData;

	// Add CPU Extension
	InferencePlugin plugin(p_plugin);
	if ((conf_targetDevice.find("CPU") != std::string::npos))
	{
		// Required for support of certain layers in CPU
		plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
	}

	// Load model into plugin
	InferenceEngine::ResponseDesc dsc;
	InferenceEngine::StatusCode sts = p_plugin->LoadNetwork(network.getNetwork(), &dsc);
	if (sts != 0)
	{
		std::cout << "Error loading model into plugin: " << dsc.msg
				  << std::endl;
		return 1;
	}

	//  Inference engine output setup

	// ----------------------
	// get output dimensions
	// ----------------------
	InferenceEngine::OutputsDataMap outputsDataMap;
	outputsDataMap = network.getNetwork().getOutputsInfo();
	InferenceEngine::BlobMap outputBlobs;

	std::string outputName = outputsDataMap.begin()->first;

	int maxProposalCount = -1;

	for (auto &&item : outputsDataMap)
	{
		InferenceEngine::SizeVector outputDims = item.second->dims;

		InferenceEngine::TBlob<float>::Ptr output;
		output = InferenceEngine::make_shared_blob<float, 
						const InferenceEngine::SizeVector>(Precision::FP32, outputDims);
		output->allocate();
		outputBlobs[item.first] = output;
		maxProposalCount = outputDims[1];
	}

	InferenceEngine::SizeVector outputDims = outputBlobs.cbegin()->second->dims();
	size_t outputSize = outputBlobs.cbegin()->second->size() / conf_batchSize;

	// Create VideoCap objects for all the videos and camera
	std::vector<VideoCap> vidCaps;
	// Requested labels 
	std::vector<string> reqLabels;
	vidCaps = getInput(&confFile, inputDims[1], inputDims[0], &reqLabels);

	const size_t output_width = inputDims[1];
	const size_t output_height = inputDims[0];

	// Initializing VideoWriter for each source 
	for (auto &vidCapObj : vidCaps)
	{
		vidCapObj.inputWidth = vidCapObj.vc.get(cv::CAP_PROP_FRAME_WIDTH);
		vidCapObj.inputHeight = vidCapObj.vc.get(cv::CAP_PROP_FRAME_HEIGHT);
		if (!(vidCapObj.initVW(vidCapObj.inputHeight, vidCapObj.inputWidth)))
		{
			cout << "Could not open " << vidCapObj.videoName << " for writing\n";
			return 4;
		}
		noMoreData.push_back(false);
	}

	Mat logs;
	arrangeWindows(&vidCaps, output_width, output_height);

	Mat frameInfer;
	Mat *frame = new Mat[conf_batchSize];
	Mat *output_frames = new Mat[conf_batchSize];

	auto input_channels = inputDims[2]; // Channels for color format, RGB=4
	auto channel_size = output_width * output_height;
	auto input_size = channel_size * input_channels;

	// Read class names
	vector<int> labelPos; // used label position in labels file
	vector<string> labelNames;
	std::vector<bool> usedLabels = getUsedLabels(&reqLabels, &labelPos, &labelNames);
	if (usedLabels.empty())
	{
		std::cout
			<< "Error: No labels currently in use. Please edit conf.txt file"
			<< std::endl;
		return 1;
	}

	ofstream logFile("intruders.log");
	if (!logFile.is_open())
	{
		cout << "Could not create log file\n";
		return 3;
	}

	list<string> logList;
	int rollingLogSize = (logWinHeight - 15) / 20;
	int index = 0;
	int totalCount = 0;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	int minFPS = get_minFPS(vidCaps);
	int waitTime = (int)(round(1000 / minFPS / vidCaps.size()));

	// Main loop starts here
	for (;;)
	{
		index = 0;
		for (auto &vidCapObj : vidCaps)
		{
			for (size_t mb = 0; mb < conf_batchSize; mb++)
			{
				float *inputPtr = pInputBlobData.get()->data() + input_size * mb;

				//---------------------------
				// Get a new frame
				//---------------------------
				int vfps = (int)round(vidCapObj.vc.get(CAP_PROP_FPS));
				for (int i = 0; i < round(vfps / minFPS); ++i)
				{
					vidCapObj.vc.read(frame[mb]);
					vidCapObj.loopFrames++;
				}

				if (!frame[mb].data)
				{
					noMoreData[index] = true;
					break;
				}

				//---------------------------------------------
				// Resize to expected size (in model .xml file)
				//---------------------------------------------

				// Input frame is resized to infer resolution
				resize(frame[mb], output_frames[mb],
					   Size(output_width, output_height));
				frameInfer = output_frames[mb];

				//----------------------------------------------------
				// PREPROCESS STAGE:
				// convert image to format expected by inference engine
				// IE expects planar, convert from packed
				//----------------------------------------------------
				size_t framesize = frameInfer.rows * frameInfer.step1();

				if (framesize != input_size)
				{
					std::cout << "input pixels mismatch, expecting "
							  << input_size << " bytes, got: " << framesize
							  << endl;
					return 1;
				}

				// Store as planar BGR for Inference Engine
				// imgIdx -> image pixel counter
				// channel_size -> size of a channel, computed as image size in bytes divided by number of channels, or image width * image height
				// input_channels -> 3 for RGB image
				// inputPtr -> a pointer to pre-allocated inout buffer
				for (size_t i = 0, imgIdx = 0, idx = 0; i < channel_size;
					 i++, idx++)
				{
					for (size_t ch = 0; ch < input_channels; ch++, imgIdx++)
					{
						inputPtr[idx + ch * channel_size] =
							frameInfer.data[imgIdx];
					}
				}
			}

			if (noMoreData[index])
			{
				++index;
				Mat messageWindow = Mat(displayWindowHeight, displayWindowWidth, CV_8UC1, Scalar(0));
				std::string message = "Video stream from " + vidCapObj.camName + " has ended!";
				cv::putText(messageWindow, message, Point((250), displayWindowHeight/2), 
						cv::FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1);
				imshow(vidCapObj.camName, messageWindow);
				continue;
			}

			//---------------------------
			// INFER STAGE
			//---------------------------
			std::chrono::high_resolution_clock::time_point infer_start_time = std::chrono::high_resolution_clock::now();
			sts = p_plugin->Infer(inputBlobs, outputBlobs, &dsc);
			if (sts != 0)
			{
				std::cout << "An infer error occurred: " << dsc.msg << std::endl;
				return 1;
			}
			std::chrono::high_resolution_clock::time_point infer_stop_time = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> infer_time = std::chrono::duration_cast<std::chrono::duration<float>>(infer_stop_time - infer_start_time);
			//---------------------------
			// POSTPROCESS STAGE:
			// Parse output
			//---------------------------
			InferenceEngine::Blob::Ptr detectionOutBlob = outputBlobs[outputName];
			const InferenceEngine::TBlob<float>::Ptr detectionOutArray =
				std::dynamic_pointer_cast<InferenceEngine::TBlob<float>>(detectionOutBlob);

			for (size_t mb = 0; mb < conf_batchSize; mb++)
			{
				float *box = detectionOutArray->data() + outputSize * mb;
				for (int i = 0; i < vidCapObj.noLabels; ++i)
				{
					vidCapObj.currentCount[i] = 0;
					vidCapObj.changedCount[i] = false;
				}

				//---------------------------
				// Parse SSD output
				//---------------------------
				for (int c = 0; c < maxProposalCount; c++)
				{
					float *localbox = &box[c * 7];
					float image_id = localbox[0];
					float label = localbox[1] - 1;
					float confidence = localbox[2];

					int labelnum = (int)label;
					if ((confidence > conf_thresholdValue) && usedLabels[labelnum])
					{

						int pos = labelPos[labelnum];
						vidCapObj.currentCount[pos]++;

						float xmin = localbox[3] * vidCapObj.inputWidth;
						float ymin = localbox[4] * vidCapObj.inputHeight;
						float xmax = localbox[5] * vidCapObj.inputWidth;
						float ymax = localbox[6] * vidCapObj.inputHeight;

						char tmplabel[32];
						rectangle(frame[mb],
								  Point((int)xmin, (int)ymin),
								  Point((int)xmax, (int)ymax),
								  Scalar(0, 255, 0), 4, LINE_AA, 0);
					}
				}

				for (int i = 0; i < vidCapObj.noLabels; ++i)
				{
					if (vidCapObj.candidateCount[i] == vidCapObj.currentCount[i])
						vidCapObj.candidateConfidence[i]++;
					else
					{
						vidCapObj.candidateConfidence[i] = 0;
						vidCapObj.candidateCount[i] = vidCapObj.currentCount[i];
					}

					if (vidCapObj.candidateConfidence[i] == conf_candidateConfidence)
					{
						vidCapObj.candidateConfidence[i] = 0;
						vidCapObj.changedCount[i] = true;
					}
					else
						continue;

					if (vidCapObj.currentCount[i] > vidCapObj.lastCorrectCount[i])
					{
						vidCapObj.totalCount[i] += vidCapObj.currentCount[i] - vidCapObj.lastCorrectCount[i];
						time_t t = time(nullptr);
						tm *currTime = localtime(&t);
						int detObj = vidCapObj.currentCount[i] - vidCapObj.lastCorrectCount[i];
						char str[50];
						for (int j = 0; j < detObj; ++j)
						{
							totalCount = 0;
							for(auto cnt : vidCapObj.totalCount)
								totalCount += cnt;
							sprintf(str, "%02d:%02d:%02d - Intruder %s detected on %s", currTime->tm_hour, 
								currTime->tm_min, currTime->tm_sec, labelNames[i].c_str(), 
								vidCapObj.camName.c_str());
							logList.emplace_back(str);
							sprintf(str, "%s\n", str);
							cout << str;
							logFile << str;
							if (logList.size() > rollingLogSize)
							{
								logList.pop_front();
							}
							event evt;
							sprintf(evt.time, "%02d:%02d:%02d", currTime->tm_hour, currTime->tm_min, 
								currTime->tm_sec);
							evt.intruder = labelNames[i];
							evt.frame = vidCapObj.frameCount;
							evt.count = totalCount;
							vidCapObj.events.push_back(evt);

						}
						// Saving image when detection occurs
						sprintf(str, "./caps/%d%d_%s.jpg", currTime->tm_hour, currTime->tm_min, labelNames[i].c_str());
						imwrite(str, frame[mb]);
					}

					vidCapObj.lastCorrectCount[i] = vidCapObj.currentCount[i];
				}
				++vidCapObj.frameCount;
			}

			//----------------------------------------
			// Display the video result and log window
			//----------------------------------------

			for (int mb = 0; mb < conf_batchSize; mb++)
			{
				vidCapObj.vw.write(frame[mb]);

				int i = 0;
				logs = Mat(logWinHeight, logWinWidth, CV_8UC1, Scalar(0));
				for (list<string>::iterator it = logList.begin(); it != logList.end(); ++it)
				{
					putText(logs, *it, Point(10, 15 + 20 * i), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
					++i;
				}
				std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float> frame_time = std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time);
				char vid_fps[20];
				sprintf(vid_fps, "FPS: %.2f", 1 / frame_time.count());
				cv::putText(frame[mb], string(vid_fps), cv::Point(10, vidCapObj.inputHeight - 10), cv::FONT_HERSHEY_SIMPLEX, 
					    	0.5, cv::Scalar(255, 255, 255), 1, 8, false);
				char infTm[20];
				sprintf(infTm, "Infer time: %.3f", infer_time.count());
				cv::putText(frame[mb], string(infTm), cv::Point(10, vidCapObj.inputHeight - 30), cv::FONT_HERSHEY_SIMPLEX, 
						0.5, cv::Scalar(255, 255, 255), 1, 8, false);
				cv::imshow(vidCapObj.camName, frame[mb]);
				cv::imshow("Intruder Log", logs);
				start_time = std::chrono::high_resolution_clock::now();

				if (loopVideos && !vidCapObj.isCam)
				{
					int vfps = (int)round(vidCapObj.vc.get(CAP_PROP_FPS));
					if (vidCapObj.loopFrames > vidCapObj.vc.get(cv::CAP_PROP_FRAME_COUNT) - round(vfps / minFPS))
					{
						vidCapObj.loopFrames = 0;
						vidCapObj.vc.set(cv::CAP_PROP_POS_FRAMES, 0);
					}
				}
			}
			++index;
		}
		// Press Esc to exit the application 
		if (waitKey(1) == 27)
		{
			break;
		}

		// Check if all the videos have ended
		if (find(noMoreData.begin(), noMoreData.end(), false) == noMoreData.end())
			break;
	}

	// Save the JSON output
	saveJSON(vidCaps[0].events, vidCaps[0]);
	delete[] output_frames;
	delete[] frame;
	destroyAllWindows();
	return 0;
}
