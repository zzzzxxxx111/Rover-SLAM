#include <thread>
#include "Matchers/lightglue_onnx.h"

int LightGlueDecoupleOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        // env0 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");
        env1 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Matcher");

        // session_options0 = Ort::SessionOptions();
        // session_options0.SetInterOpNumThreads(std::thread::hardware_concurrency());
        // session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_options1 = Ort::SessionOptions();
        session_options1.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0; 
            cuda_options.arena_extend_strategy = 1; // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            // session_options0.AppendExecutionProvider_CUDA(cuda_options);
            // session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_options1.AppendExecutionProvider_CUDA(cuda_options);
            session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

        cfg.lightgluePath = "onnxmodel/lightglue_sim.onnx";
        //std::string extractor_modelPath = cfg.extractorPath;
        std::string matcher_modelPath = cfg.lightgluePath;


        
        //ExtractorSession = new Ort::Session(env0 , extractor_modelPath.c_str(), session_options0);
        MatcherSession = new Ort::Session(env1 , matcher_modelPath.c_str() , session_options1);

        // Initial Extractor 
        // size_t numInputNodes = ExtractorSession->GetInputCount();
        // ExtractorInputNodeNames.reserve(numInputNodes);
        // for (size_t i = 0 ; i < numInputNodes ; i++)
        // {
        //     ExtractorInputNodeNames.emplace_back(ExtractorSession->GetInputNameAllocated(i , allocator).get());
        //     ExtractorInputNodeShapes.emplace_back(ExtractorSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        //     std::string str = ExtractorSession->GetInputNameAllocated(i , allocator).get();
        //     std::cout << "Converted string: " << str << std::endl;
        // }

        // size_t numOutputNodes = ExtractorSession->GetOutputCount();
        // ExtractorOutputNodeNames.reserve(numOutputNodes);
        // for (size_t i = 0 ; i < numOutputNodes ; i++)
        // {
        //     ExtractorOutputNodeNames.emplace_back(ExtractorSession->GetOutputNameAllocated(i , allocator).get()); 
        //     std::string str = ExtractorSession->GetOutputNameAllocated(i , allocator).get();
        //     std::cout << "Converted string: " << str << std::endl;
        //     ExtractorOutputNodeShapes.emplace_back(ExtractorSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        // }

        size_t numInputNodes = 0;
        size_t numOutputNodes = 0;
        
        // Initial Matcher 
        numInputNodes = MatcherSession->GetInputCount();
        MatcherInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            MatcherInputNodeNames.emplace_back(MatcherSession->GetInputNameAllocated(i , allocator).get());
            MatcherInputNodeShapes.emplace_back(MatcherSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numOutputNodes = MatcherSession->GetOutputCount();
        MatcherOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            MatcherOutputNodeNames.emplace_back(MatcherSession->GetOutputNameAllocated(i , allocator).get());
            MatcherOutputNodeShapes.emplace_back(MatcherSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }


        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f> , float*> LightGlueDecoupleOnnxRunner::Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f> , float*> extractor_result;
    try{
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts = (int64_t*)tensor[0].GetTensorMutableData<void>();
        // for (int i = 0 ; i < kpts_Shape[1] ; i++)
        // {
        //     std::cout << kpts[i] << " ";
        // }
        printf("[RESULT INFO] kpts Shape : (%lld , %lld , %lld)\n" , kpts_Shape[0] , kpts_Shape[1] , kpts_Shape[2]);

        std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float* scores = (float*)tensor[1].GetTensorMutableData<void>();

        std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float* desc = (float*)tensor[2].GetTensorMutableData<void>();
        printf("[RESULT INFO] desc Shape : (%lld , %lld , %lld)\n" , descriptors_Shape[0] , descriptors_Shape[1] , descriptors_Shape[2]);

        // Process kpts and descriptors
        std::vector<cv::Point2f> kpts_f;
        for (int i = 0; i < kpts_Shape[1] * 2; i += 2) 
        {
            kpts_f.emplace_back(cv::Point2f(kpts[i] , kpts[i + 1]));
        }


        extractor_result.first = kpts_f;
        extractor_result.second = desc;

        //std::cout << "[INFO] Extractor postprocessing operation completed successfully" << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }

    return extractor_result;
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h , int w)
{
    return NormalizeKeypoints(kpts , h , w);
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::KeyPoint> kpts, int h , int w)
{
    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    for (const cv::KeyPoint& kpt : kpts) {
        cv::Point2f normalizedKpt = (kpt.pt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
    //return NormalizeKeypoints(kpts , h , w);
}


std::vector<Ort::Value> LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::Point2f> kpts0 , \
            std::vector<cv::Point2f> kpts1 , float* desc0 , float* desc1)
{
    //std::cout << "< - * -------- Matcher Inference START -------- * ->"<< std::endl;
    try
    {
        MatcherInputNodeShapes[0] = {1 , static_cast<int>(kpts0.size()) , 2};
        MatcherInputNodeShapes[1] = {1 , static_cast<int>(kpts1.size()) , 2};

        MatcherInputNodeShapes[2] = {1 , static_cast<int>(kpts0.size()) , 256};
        MatcherInputNodeShapes[3] = {1 , static_cast<int>(kpts1.size()) , 256};
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        float* kpts0_data = new float[kpts0.size() * 2];
        float* kpts1_data = new float[kpts1.size() * 2];

        for (size_t i = 0; i < kpts0.size(); ++i) {
            kpts0_data[i * 2] = kpts0[i].x;
            kpts0_data[i * 2 + 1] = kpts0[i].y;
        }
        for (size_t i = 0; i < kpts1.size(); ++i) {
            kpts1_data[i * 2] = kpts1[i].x;
            kpts1_data[i * 2 + 1] = kpts1[i].y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts0_data , kpts0.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[0].data() , MatcherInputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts1_data , kpts1.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[1].data() , MatcherInputNodeShapes[1].size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc0 , kpts0.size() * 256 * sizeof(float), \
            MatcherInputNodeShapes[2].data() , MatcherInputNodeShapes[2].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc1 , kpts1.size() * 256 * sizeof(float) , \
            MatcherInputNodeShapes[3].data() , MatcherInputNodeShapes[3].size()
        ));


        auto time_start = std::chrono::high_resolution_clock::now();
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0","mscores0"};

        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr} , input_names , input_tensors.data() , \
                    input_tensors.size() , output_names , MatcherOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        matcher_timer += diff;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        //std::cout << "[INFO] LightGlueDecoupleOnnxRunner Matcher inference finish ..." << std::endl;
        //std::cout << "[INFO] Matcher inference cost time : " << diff << "ms" << std::endl;
        return output_tensor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Matcher inference failed : " << ex.what() << std::endl;
        std::vector<Ort::Value> ss;
        return ss;
    }
    
    
}
std::vector<Ort::Value> LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::KeyPoint> kpts0 , \
            std::vector<cv::KeyPoint> kpts1 , float* desc0 , float* desc1)
{
    // std::vector<cv::Point2f> kpts_pf0,kpts_pf1;
    // for(int i = 0 ; i < kpts0.size(); i++)
    // {
    //     kpts_pf0.push_back(kpts0[i].pt);
    // }

    // for(int i = 0 ; i < kpts1.size(); i++)
    // {
    //     kpts_pf1.push_back(kpts1[i].pt);
    // }
    
    //std::cout << "< - * -------- Matcher Inference START -------- * ->"<< std::endl;
    try
    {
        MatcherInputNodeShapes[0] = {1 , static_cast<int>(kpts0.size()) , 2};
        MatcherInputNodeShapes[1] = {1 , static_cast<int>(kpts1.size()) , 2};

        MatcherInputNodeShapes[2] = {1 , static_cast<int>(kpts0.size()) , 256};
        MatcherInputNodeShapes[3] = {1 , static_cast<int>(kpts1.size()) , 256};
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        float* kpts0_data = new float[kpts0.size() * 2];
        float* kpts1_data = new float[kpts1.size() * 2];

        for (size_t i = 0; i < kpts0.size(); ++i) {
            kpts0_data[i * 2] = kpts0[i].pt.x;
            kpts0_data[i * 2 + 1] = kpts0[i].pt.y;
        }
        for (size_t i = 0; i < kpts1.size(); ++i) {
            kpts1_data[i * 2] = kpts1[i].pt.x;
            kpts1_data[i * 2 + 1] = kpts1[i].pt.y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts0_data , kpts0.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[0].data() , MatcherInputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts1_data , kpts1.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[1].data() , MatcherInputNodeShapes[1].size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc0 , kpts0.size() * 256 * sizeof(float), \
            MatcherInputNodeShapes[2].data() , MatcherInputNodeShapes[2].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc1 , kpts1.size() * 256 * sizeof(float) , \
            MatcherInputNodeShapes[3].data() , MatcherInputNodeShapes[3].size()
        ));


        auto time_start = std::chrono::high_resolution_clock::now();
        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0","mscores0"};

        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr} , input_names , input_tensors.data() , \
                    input_tensors.size() , output_names , MatcherOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        matcher_timer += diff;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        //std::cout << "[INFO] LightGlueDecoupleOnnxRunner Matcher inference finish ..." << std::endl;
        std::cout << "[INFO] Matcher inference cost time : " << diff << "ms" << std::endl;
        return output_tensor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Matcher inference failed : " << ex.what() << std::endl;
        std::vector<Ort::Value> ss;
        return ss;
    }
    
    
}
int LightGlueDecoupleOnnxRunner::Matcher_PostProcess(Configuration cfg , std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1)
{
    try{
        std::vector<int64_t> matches0_Shape = matcher_outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches0 = (int64_t*)matcher_outputtensors[0].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches0 Shape : (%lld , %lld)\n" , matches0_Shape[0] , matches0_Shape[1]);

        std::vector<int64_t> matches1_Shape = matcher_outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches1 = (int64_t*)matcher_outputtensors[1].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches1 Shape : (%lld , %lld)\n" , matches1_Shape[0] , matches1_Shape[1]);

        std::vector<int64_t> mscore0_Shape = matcher_outputtensors[2].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores0 = (float*)matcher_outputtensors[2].GetTensorMutableData<void>();
        std::vector<int64_t> mscore1_Shape = matcher_outputtensors[3].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores1 = (float*)matcher_outputtensors[3].GetTensorMutableData<void>();

        // Process kpts0 and kpts1
        std::vector<cv::Point2f> kpts0_f , kpts1_f;

        for (int i = 0; i < kpts0.size(); i++) 
        {
            kpts0_f.emplace_back(cv::Point2f(
                (kpts0[i].x + 0.5) / scales[0] - 0.5 , (kpts0[i].y + 0.5) / scales[0] - 0.5));
        }
        for (int i = 0; i < kpts1.size(); i++) 
        {
            kpts1_f.emplace_back(cv::Point2f(
                (kpts1[i].x + 0.5) / scales[0] - 0.5 , (kpts1[i].y + 0.5) / scales[0] - 0.5));
        }

        // Create match indices
        std::vector<int64_t> validIndices;
        for (int i = 0; i < matches0_Shape[1]; ++i) {
            if (matches0[i] > -1 && mscores0[i] > this->matchThresh && matches1[matches0[i]] == i) { 
                validIndices.emplace_back(i);
            }
        }

        std::set<std::pair<int, int> > matches;
        std::vector<cv::Point2f> m_kpts0 , m_kpts1;
        for (int i : validIndices) {
            matches.insert(std::make_pair(i, matches0[i]));
        }

        //std::cout << "[RESULT INFO] matches Size : " << matches.size() << std::endl;

        for (const auto& match : matches) {
            m_kpts0.emplace_back(kpts0_f[match.first]);
            m_kpts1.emplace_back(kpts1_f[match.second]);
        }

        keypoints_result.first = m_kpts0;
        keypoints_result.second = m_kpts1;
        
        //std::cout << "[INFO] Postprocessing operation completed successfully" << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess_fused(std::vector<Ort::Value>& output, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, std::vector<int>& vnMatches12)
{
    bool outlier_rejection = false;
    //std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> result;
    int size = 0;
    try
    {
        // load date from tensor
        std::vector<int64_t> matches_Shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t *matches = (int64_t *)output[0].GetTensorMutableData<void>();
        //printf("[RESULT INFO] matches Shape : (%ld , %ld)\n", matches_Shape[0], matches_Shape[1]);

        std::vector<int64_t> mscore_Shape = output[1].GetTensorTypeAndShapeInfo().GetShape();
        float *mscores = (float *)output[1].GetTensorMutableData<void>();
        //printf("[RESULT INFO] mscore Shape : (%ld )\n", mscore_Shape[0]);

        // Process kpts0 and kpts1
        // std::vector<cv::Point2f> kpts0_f, kpts1_f;
        // kpts0_f.reserve(kpts0.size());
        // kpts1_f.reserve(kpts1.size());
        // //scales[0] =1;
        // // scales[1] = 1;
        // for (int i = 0; i < kpts0.size(); i++)
        // {
        //     kpts0_f.emplace_back(cv::Point2f(
        //         (kpts0[i].x + 0.5) / scales[0] - 0.5, (kpts0[i].y + 0.5) / scales[0] - 0.5));
        // }
        // for (int i = 0; i < kpts1.size(); i++)
        // {
        //     kpts1_f.emplace_back(cv::Point2f(
        //         (kpts1[i].x + 0.5) / scales[1] - 0.5, (kpts1[i].y + 0.5) / scales[1] - 0.5));
        // }

        // // get the good match
        // std::vector<cv::Point2f> m_kpts0, m_kpts1;
        // m_kpts0.reserve(matches_Shape[0]);
        // m_kpts1.reserve(matches_Shape[0]);

        // std::vector<cv::DMatch> matchesD;
        // matchesD.clear();

        for (int i = 0; i < matches_Shape[0]; i++)
        {
            // if (mscores[i] > this->matchThresh)
            // {
            //     m_kpts0.emplace_back(kpts0_f[matches[i * 2]]);
            //     m_kpts1.emplace_back(kpts1_f[matches[i * 2 + 1]]);
            // }
                        
            if (mscores[i] > this->matchThresh)
            {
                size++;
                //matchesD.emplace_back(matches[i*2], matches[i*2+1], mscores[i]);
                vnMatches12[matches[i * 2]] = matches[i * 2 +1];//本来是vnMatches[i]，错了！！！！
                //m_kpts0.emplace_back(kpts0[matches[i * 2]]);//emplace_back() 更直接的方式向容器中添加元素，并且在添加元素时会自动调用构造函数
                //m_kpts1.emplace_back(kpts1[matches[i * 2 + 1]]);
            }
        }

    //     if(outlier_rejection){
    //         std::vector<uchar> inliers;
    //         cv::findFundamentalMat(m_kpts0, m_kpts1, cv::FM_RANSAC, 3, 0.99, inliers);
    //         int j = 0;
    //         for(int i = 0; i < matchesD.size(); i++){
    //             if(inliers[i]){
    //                 matchesD[j++] = matchesD[i];
    //             }
    //         }
    //         matchesD.resize(j);
        
    // }


        //std::cout << "[RESULT INFO] matches Size : " << m_kpts1.size() << std::endl;
        
        // result.first = m_kpts0;
        // result.second = m_kpts1;

        //std::cout << "[INFO] Postprocessing operation completed successfully" << std::endl;
        return size;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return size;
    }
}

float LightGlueDecoupleOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueDecoupleOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

double LightGlueDecoupleOnnxRunner::GetTimer(std::string name)
{
    if (name == "extractor")
    {
        return this->extractor_timer;
    }else
    {
        return this->matcher_timer;
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(unsigned int threads) : \
    num_threads(threads)
{
}

LightGlueDecoupleOnnxRunner::~LightGlueDecoupleOnnxRunner()
{
}
