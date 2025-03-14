// //
// // Created by haoyuefan on 2021/9/22.
// //

// #ifndef SUPER_POINT_H_
// #define SUPER_POINT_H_

// #include <string>
// #include <memory>
// #include <iostream>
// #include <Eigen/Core>
// #include <NvInfer.h>
// #include <NvOnnxParser.h>
// #include <opencv2/opencv.hpp>

// #include "../Thirdparty/TensorRTBuffer/include/buffers.h"
// // #include "read_configs.h"

// using tensorrt_common::TensorRTUniquePtr;

// struct SuperPointConfig {
//   int max_keypoints;
//   double keypoint_threshold;
//   int remove_borders;
//   int dla_core;
//   std::vector<std::string> input_tensor_names;
//   std::vector<std::string> output_tensor_names;
//   std::string onnx_file;
//   std::string engine_file;
// };


// class SuperPoint {
// public:
//     explicit SuperPoint(const SuperPointConfig &super_point_config);

//     explicit SuperPoint();
//     bool build();

//     bool infer(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &Descriptors, int nKeyPointNum);

//     void visualization(const std::string &image_name, const cv::Mat &image);

//     void save_engine();

//     bool deserialize_engine();

//     SuperPointConfig super_point_config_;

// private:
    
//     nvinfer1::Dims input_dims_{};
//     nvinfer1::Dims semi_dims_{};
//     nvinfer1::Dims desc_dims_{};
//     std::shared_ptr<nvinfer1::ICudaEngine> engine_;
//     std::shared_ptr<nvinfer1::IExecutionContext> context_;
//     std::vector<std::vector<int>> keypoints_;
//     std::vector<std::vector<double>> descriptors_;

//     bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
//                            TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
//                            TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
//                            TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

//     bool process_input(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

//     bool process_output(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<double, Eigen::Dynamic, 259> &features);

//     void remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border, int height,
//                         int width);

//     std::vector<size_t> sort_indexes(std::vector<float> &data);

//     void top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k);

//     void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints, int h, int w,
//                                double threshold);

//     void sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
//                             std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s = 8);
// };

// typedef std::shared_ptr<SuperPoint> SuperPointPtr;

// #endif //SUPER_POINT_H_
