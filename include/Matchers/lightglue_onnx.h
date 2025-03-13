#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "transform.h"
#include "Configuration.h"


class LightGlueDecoupleOnnxRunner
{
public:
	const unsigned int num_threads;

    Ort::Env env0 , env1;
    Ort::SessionOptions session_options0 , session_options1;
    //Ort::Session* ExtractorSession;
    Ort::Session* MatcherSession;
    Ort::AllocatorWithDefaultOptions allocator;

    // std::vector<char*> ExtractorInputNodeNames;
    // std::vector<std::vector<int64_t>> ExtractorInputNodeShapes;
    // std::vector<char*> ExtractorOutputNodeNames;
    // std::vector<std::vector<int64_t>> ExtractorOutputNodeShapes;

    std::vector<char*> MatcherInputNodeNames;
    std::vector<std::vector<int64_t>> MatcherInputNodeShapes;
    std::vector<char*> MatcherOutputNodeNames;
    std::vector<std::vector<int64_t>> MatcherOutputNodeShapes;

    float matchThresh = 0.0f;
    long long extractor_timer = 0.0f;
    long long matcher_timer = 0.0f;

    std::vector<float> scales = {1.0f , 1.0f};

    // std::vector<std::vector<Ort::Value>> extractor_outputtensors; // 因为要存src和dest的两个结果，所以用嵌套vector
    std::vector<Ort::Value> matcher_outputtensors;

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
    
public:
    // cv::Mat Extractor_PreProcess(Configuration cfg , const cv::Mat& srcImage , float& scale);
    int Extractor_Inference(Configuration cfg , const cv::Mat& image);
    std::pair<std::vector<cv::Point2f> , float*> Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor);
    std::vector<cv::Point2f> Matcher_PreProcess(std::vector<cv::KeyPoint> kpts, int h , int w);
    std::vector<cv::Point2f> Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h , int w);
    std::vector<Ort::Value> Matcher_Inference(std::vector<cv::KeyPoint> kpts0 , std::vector<cv::KeyPoint> kpts1 , float* desc0 , float* desc1);
    std::vector<Ort::Value> Matcher_Inference(std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1 , float* desc0 , float* desc1);
    int Matcher_PostProcess(Configuration cfg , std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1);
    int Matcher_PostProcess_fused(std::vector<Ort::Value>& output, std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1,std::vector<int>& vnMatches1);
    //int Matcher_PostProcess_fused(std::vector<Ort::Value>& output, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, std::vector<int>& vnMatches12)
public:
    explicit LightGlueDecoupleOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueDecoupleOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);
    double GetTimer(std::string name);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult();

    int InitOrtEnv(Configuration cfg);
    
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg , \
            const cv::Mat& srcImage, const cv::Mat& destImage);
};
