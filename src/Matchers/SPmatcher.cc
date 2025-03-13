
#include "iostream"
#include <opencv2/core/core.hpp>
#include <onnxruntime_cxx_api.h>
#include "Matchers/Configuration.h"
#include "Matchers/SPmatcher.h"
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;

namespace ORB_SLAM3
{
    const float SPmatcher::TH_HIGH = 1.4;
    const float SPmatcher::TH_LOW = 1.2;
    const int SPmatcher::HISTO_LENGTH = 30;

SPmatcher::SPmatcher(float thre)
{
    Configuration cfg;
    cfg.device = "cuda";
    cfg.extractorPath = "";
    cfg.extractorType = "";
    featureMatcher = new LightGlueDecoupleOnnxRunner();
    featureMatcher->InitOrtEnv(cfg);
    featureMatcher->SetMatchThresh(thre);
    std::string mode = "LightGlueDecoupleOnnxRunner";
}

void SPmatcher::plotspmatch(cv::Mat frame1,cv::Mat frame2, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2,  std::vector<int> vmatches12){
    vector<cv::DMatch> vmatches;
    for(int i=0 ; i < vmatches12.size(); i++)
    {
        int idx = vmatches12[i];
        if(idx > 0)
        {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = idx;
            vmatches.push_back(match);
        }
    }
    cv::Mat matched_img;
    cv::drawMatches(frame1,kpts1,frame2,kpts2,vmatches,matched_img, cv::Scalar(0, 255, 0));
    cv::imshow("Matched Features", matched_img);
    cv::waitKey(0);

}

int SPmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    GeometricCamera* pCamera;
    Sophus::SE3f Tcw;
    Eigen::Vector3f Ow;

    if(bRight){
        Tcw = pKF->GetRightPose();
        Ow = pKF->GetRightCameraCenter();
        pCamera = pKF->mpCamera2;
    }
    else{
        Tcw = pKF->GetPose();
        Ow = pKF->GetCameraCenter();
        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused = 0;
    const int nMPs = vpMapPoints.size();
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
    for(int i = 0; i < nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
        {
            count_notMP++;
            continue;
        }
        if(pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if(pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw*p3Dw;

        if(p3Dc(2)<0.0f)
        {
            count_negdepth++;
            continue;
        }
        const float invz = 1/p3Dc(2);

        const Eigen::Vector2f uv = pCamera->project(p3Dc);

        if(!pKF->IsInImage(uv(0),uv(1)))
        {
            count_notinim++;
            continue;
        }

        const float ur = uv(0) - bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        if(dist3D < minDistance || dist3D > maxDistance)
        {
            count_dist++;
            continue;
        }

        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn) < 0.5*dist3D)
        {
            count_normal++;
            continue;
        }

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

        if(vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 10;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            size_t idx = *vit;
            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                            : (!bRight) ? pKF -> mvKeys[idx]
                                                                        : pKF -> mvKeysRight[idx];
            const int &kpLevel = kp.octave;
            if(kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;
            
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                // 双目情况
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv(0)-kpx;
                const float ey = uv(1)-kpy;
                // 右目数据的偏差也要考虑进去
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                //自由度为3, 误差小于1个像素,这种事情95%发生的概率对应卡方检验阈值为7.82
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv(0) - kpx;
                const float ey = uv(1) - kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            if(bRight) idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP, dKF);

            if(dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;

    }

    return nFused;

}

int SPmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
 {
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;
    // 与当前帧闭环匹配上的关键帧及其共视关键帧组成的地图点
    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    // 遍历所有的地图点
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        // 地图点无效 或 已经是该帧的地图点（无需融合），跳过
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        // Step 2 地图点变换到当前相机坐标系下
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f)
            continue;

        // Project into Image
        // Step 3 得到地图点投影到当前帧的图像坐标
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        // 投影点必须在图像范围内
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        // Step 4 根据距离是否在图像合理金字塔尺度范围内和观测角度是否小于60度判断该地图点是否有效
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        // 计算搜索范围
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        // Step 5 在当前帧内搜索匹配候选点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        // Step 6 寻找最佳匹配点（没有用到次佳匹配的比例）
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            // const int &kpLevel = pKF->mvKeysUn[idx].octave;

            // if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
            //     continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        // Step 7 替换或新增地图点
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                // 如果这个地图点已经存在，则记录要替换信息
                // 这里不能直接替换，原因是需要对地图点加锁后才能替换，否则可能会crash。所以先记录，在加锁后替换
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                // 如果这个地图点不存在，直接添加
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }
    // Frame f3; Frame f4;
    // vector<int> vnMatches12;
    // MatchingPoints_onnx(f3,f4,vnMatches12);
    // 融合（替换和新增）的地图点数目
    return nFused;
 }

int SPmatcher::MatchingPoints_onnx(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float* desc0,float* desc1){
    int rows = 300;
    int cols = 400;
    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);
    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, desc0, desc1);
    int size;
    //std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> output_end;
    std::vector<int> vnMatches12;
    vnMatches12 = std::vector<int>(normal_kpts0.size(), -1);
    size = featureMatcher->Matcher_PostProcess_fused(output, kpts0 , kpts1, vnMatches12);
    return size;
}

int SPmatcher::MatchingPoints_onnx(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, cv::Mat desc0,cv::Mat desc1, std::vector<int>& vnMatches12){
    vnMatches12.resize(kpts0.size(),-1);
    int rows = 300;//需要修改！
    int cols = 400;

    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);

    int rows0 = desc0.rows;
    int cols0 = desc0.cols;
    float* descriptors_data0 = new float[rows0 * cols0];
    for (int i = 0 ; i < rows0; i++){
        const float* row_data = desc0.ptr<float>(i);
        for (int j = 0; j < cols0; j++){
            descriptors_data0[i*cols0 + j] = row_data[j];
        }
    }

    int rows1 = desc1.rows;
    int cols1 = desc1.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = desc1.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }

    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, descriptors_data0, descriptors_data1);
    //std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> output_end;
    int size;
    // std::vector<int> vnMatches12;
    // vnMatches12 = std::vector<int>(normal_kpts0.size(), -1);
    size = featureMatcher->Matcher_PostProcess_fused(output, kpts0 , kpts1, vnMatches12);
    return size;
}

int SPmatcher::MatchingPoints_onnx(std::vector<cv::KeyPoint> kpts0, const std::vector<cv::KeyPoint> kpts1, cv::Mat desc0, const cv::Mat desc1, std::vector<int>& vnMatches12){
    vnMatches12.resize(kpts0.size(),-1);
    int rows = 300;//需要修改！
    int cols = 400;
    std::vector<cv::Point2f> kpts_pf0, kpts_pf1;
    for(const cv::KeyPoint& keypoint : kpts0){
        kpts_pf0.emplace_back(keypoint.pt);
    }

    for(const cv::KeyPoint& keypoint : kpts1){
        kpts_pf1.emplace_back(keypoint.pt);
    }
    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);

    int rows0 = desc0.rows;
    int cols0 = desc0.cols;
    float* descriptors_data0 = new float[rows0 * cols0];
    for (int i = 0 ; i < rows0; i++){
        const float* row_data = desc0.ptr<float>(i);
        for (int j = 0; j < cols0; j++){
            descriptors_data0[i*cols0 + j] = row_data[j];
        }
    }

    int rows1 = desc1.rows;
    int cols1 = desc1.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = desc1.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }

    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, descriptors_data0, descriptors_data1);
    //std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> output_end;
    // std::vector<int> vnMatches12;
    // vnMatches12 = std::vector<int>(normal_kpts0.size(), -1);
    int size  = featureMatcher->Matcher_PostProcess_fused(output, kpts_pf0 , kpts_pf1, vnMatches12);
    return size;
}


int SPmatcher::MatchingPoints_onnx(Frame &f1, Frame &f2, vector<int> &vnMatches12)//在C++中，默认参数值只能在函数的声明或定义中的一处给定，而不能同时在两者中都给定。通常，我们在函数的声明中给定默认参数，然后在函数定义中省略默认参数值。
{   
    bool outlier_rejection=false;
    vnMatches12.resize(f1.mvKeys.size(),-1);
    // int rows = 240;//改
    // int cols = 320;//改
    int rows = f2.imgLeft.rows;
    int cols = f2.imgLeft.cols;
    // Frame f3; Frame f4;
    // MatchingPoints_onnx(f3,f4,vnMatches12);
    std::vector<cv::Point2f> kpts1, kpts2;

    for(const cv::KeyPoint& keypoint : f1.mvKeys){
        kpts1.emplace_back(keypoint.pt);
    }

    for(const cv::KeyPoint& keypoint : f2.mvKeys){
        kpts2.emplace_back(keypoint.pt);
    }

    // for (const cv::KeyPoint& point : f1.mvKeys) {
    //     cv::circle(f1.imgLeft, point.pt, 2,cv::Scalar(0, 0, 255), 0.5);
    // }
    // for(const cv::KeyPoint& point : f2.mvKeys) {
    //     cv::circle(f2.imgLeft, point.pt, 2, cv::Scalar(0, 0, 255), 0.5);
    // }
    // int total_width = f1.imgLeft.cols + f2.imgLeft.cols;
    // int max_height = std::max(f1.imgLeft.rows, f2.imgLeft.rows);
    // cv::Mat combined_image(max_height, total_width, f2.imgLeft.type());
    // f1.imgLeft.copyTo(combined_image(cv::Rect(0, 0, f1.imgLeft.cols, f1.imgLeft.rows)));
    // f2.imgLeft.copyTo(combined_image(cv::Rect(f2.imgLeft.cols, 0, f2.imgLeft.cols, f2.imgLeft.rows)));
    // cv::imshow("Feature Points", combined_image);
    // cv::waitKey(0);


    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);
    auto normal_kpts2 = featureMatcher->Matcher_PreProcess(kpts2 , rows , cols);

    // std::vector<cv::DMatch>  matches;
    // matches.clear();
    
    int rows1 = f1.mDescriptors.rows;
    int cols1 = f1.mDescriptors.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = f1.mDescriptors.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }

    // int length = sizeof(descriptors_data1)/sizeof(float);
    // for(int i = 0 ; i < length+100; i++)
    // {
    //     cout<<descriptors_data1[i]<<std::endl;
    // }
    // for(int i = 0 ; i < 256 ; i++)
    // {
    //     cout<<f1.mDescriptors.at<float>(0,i)<<" ";
    // }
    // std::cout<<endl;

    int rows2 = f2.mDescriptors.rows;
    int cols2 = f2.mDescriptors.cols;
    float* descriptors_data2 = new float[rows2 * cols2];
    for (int i = 0 ; i < rows2; i++){
        const float* row_data = f2.mDescriptors.ptr<float>(i);
        for (int j = 0; j < cols2; j++){
            descriptors_data2[i*cols2 + j] = row_data[j];
        }
    }
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts1, normal_kpts2, descriptors_data1, descriptors_data2);
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> output_end;
    int size  = featureMatcher->Matcher_PostProcess_fused(output, kpts1 , kpts2, vnMatches12);
    // if(outlier_rejection){
    //     std::vector<uchar> inliers;
    //     cv::findFundamentalMat(output_end.first, output_end.second, cv::FM_RANSAC, 3, 0.99, inliers);
    //     int j = 0;
    //     for(int i = 0; i < output_end.first.size(); i++){
    //         if(inliers[i]){

    //         }
    //     }
    // }
    return size;
}

// int SPmatcher::MatchingPoints(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
//     const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches, bool outlier_rejection=false)
// {
//     std::vector<cv::DMatch> matches;
//     matches.clear();

//     Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 = NormalizeKeypoints(features0, _superglue_config.image_width, _superglue_config.image_height);
//     Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 = NormalizeKeypoints(features1, _superglue_config.image_width, _superglue_config.image_height);
//     Eigen::VectorXi indices0, indices1;
//     Eigen::VectorXd mscores0, mscores1;
//     superglue.infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

//     int num_match = 0;
//     std::vector<cv::Point2f> points0, points1;
//     std::vector<int> point_indexes;
//     for(size_t i = 0; i < indices0.size(); i++){
//         if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
//             double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
//             matches.emplace_back(i, indices0[i], d);
//             points0.emplace_back(features0(1, i),features0(2, i));
//             points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
//             num_match++;
//         }
//     }

//     if(outlier_rejection){
//         std::vector<uchar> inliers;
//         cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
//         int j = 0;
//         for(int i = 0; i < matches.size(); i++){
//             if(inliers[i]){
//                 matches[j++] = matches[i];
//             }
//         }
//         matches.resize(j);
        
//     }
//     return matches.size();
// }


// int SPmatcher::MatchingPoints(Frame &f1, Frame &f2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, bool outlier_rejection=false)
// {
//     std::vector<cv::DMatch>  matches;
//     matches.clear();
//     // std::vector<cv::KeyPoint> f1vKey = f1.mvKeys;
//     // cv::Mat f1Des = f1.mDescriptors;

//     // std::vector<cv::KeyPoint> f2vKey = f2.mvKeys;
//     // cv::Mat f2Des = f2.mDescriptors;

//     Eigen::Matrix<double, 259, Eigen::Dynamic> features0 = ConvertToEigenMatrix(f1.mvKeys, f1.mDescriptors);
//     Eigen::Matrix<double, 259, Eigen::Dynamic> features1 = ConvertToEigenMatrix(f2.mvKeys,f2.mDescriptors);
//     Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 = NormalizeKeypoints(features0, _superglue_config.image_width, _superglue_config.image_height);
//     Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 = NormalizeKeypoints(features1, _superglue_config.image_width, _superglue_config.image_height);
//     Eigen::VectorXi indices0, indices1;
//     Eigen::VectorXd mscores0, mscores1;
//     superglue.infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

//     int num_match = 0;
//     std::vector<cv::Point2f> points0, points1;
//     std::vector<int> point_indexes;
//     for(size_t i = 0; i < indices0.size(); i++){
//         if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
//             double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
//             matches.emplace_back(i, indices0[i], d);
//             points0.emplace_back(features0(1, i),features0(2, i));
//             points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
//             num_match++;
//         }
//     }

//     vbPrevMatched = points0;
//     if(outlier_rejection){
//         std::vector<uchar> inliers;
//         cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
//         int j = 0;
//         for(int i = 0; i < matches.size(); i++){
//             if(inliers[i]){
//                 matches[j++] = matches[i];
//             }
//         }
//         matches.resize(j);
//     }
    
//     std::vector<int> vnMatches12;
//     ConvertMatchesToVector(matches,vnMatches12);
    
//     return matches.size();
// }

Eigen::Matrix<double, 259, Eigen::Dynamic> SPmatcher::NormalizeKeypoints(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height)
{
    Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
    norm_features.resize(259, features.cols());
    norm_features = features;
    for (int col = 0; col <features.cols(); ++col) {
        norm_features(1, col) = 
            (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
        norm_features(2, col) = 
            (features(2, col) - height / 2) /(std::max(width, height) * 0.7);
    }
    return norm_features;
}

void SPmatcher::ConvertMatchesToVector(const std::vector<cv::DMatch>& matches, std::vector<int>& vnMatches12){
            // 初始化vnMatches12，将所有元素置为-1，表示没有匹配
            vnMatches12 = std::vector<int>(matches.size(), -1);
            
            // 将matches中的匹配点对信息存储到vnMatches12中
            for (size_t i = 0; i < matches.size(); ++i){
                int idxF1 = matches[i].queryIdx;
                int idxF2 = matches[i].trainIdx;

                vnMatches12[idxF1] = idxF2;
            }
}

Eigen::Matrix<double, 259, Eigen::Dynamic> SPmatcher::ConvertToEigenMatrix(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
{
    int numPoints = static_cast<int>(keypoints.size());
    Eigen::Matrix<double, 259, Eigen::Dynamic> pointsAndDescriptors(259, numPoints);

    for(int i = 0; i < numPoints; ++i)
    {
        pointsAndDescriptors(0, i) = keypoints[i].pt.x;
        pointsAndDescriptors(1, i) = keypoints[i].pt.y;
        pointsAndDescriptors(2, i) = keypoints[i].response;

        cv::Mat descriptor = descriptors.row(i);
        for(int j = 0; j < descriptors.cols; ++j){
            pointsAndDescriptors(3 + j, i) = static_cast<double>(descriptor.at<uchar>(0, j));
        }
    }
    return pointsAndDescriptors;
}

int SPmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeys.size(), -1);
    
    vector<int> vMatchedDistance(F2.mvKeys.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeys.size(), -1);

    nmatches =  MatchingPoints_onnx(F1, F2, vnMatches12);
    
    return nmatches;
    
}

int SPmatcher::SearchByProjection(Frame &CurrentFrame, Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;
    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();
    
    vector<int> vmatches(CurrentFrame.N,-1);

    const Sophus::SE3f Tlw = LastFrame.GetPose();
    const Eigen::Vector3f tlc = Tlw * twc; 
    
    // 判断前进还是后退
    const bool bForward = tlc(2)>CurrentFrame.mb && !bMono;     // 非单目情况，如果Z大于基线，则表示相机明显前进
    const bool bBackward = -tlc(2)>CurrentFrame.mb && !bMono;   // 非单目情况，如果-Z小于基线，则表示相机明显后退

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                    // 对上一帧有效的MapPoints投影到当前帧坐标系
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0/x3Dc(2);

                if(invzc<0)
                    continue;

                    // 投影到当前帧中
                Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                    continue;
                if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                    continue;
                    // 认为投影前后地图点的尺度信息不变
                int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    // 单目：th = 7，双目：th = 15
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; // 尺度越大，搜索范围越大

                    // 记录候选匹配点的id
                vector<size_t> vIndices2;

                    // Step 4 根据相机的前后前进方向来判断搜索尺度范围。
                    // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
                    // 当相机前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
                    // 当相机后退时，圆点的面积减小，在某个尺度m下它是一个特征点，由于面积减小，则需要在更低的尺度下才能检测出来
                if(bForward)  // 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                else if(bBackward)  // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                else  // 在[nLastOctave-1, nLastOctave+1]中搜索
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = 256;
                int bestIdx2 = -1;

                    // Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点 
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;

                        // 如果该特征点已经有对应的MapPoint了,则退出该次循环
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                    {
                            // 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
                        const float ur = uv(0) - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const float dist = DescriptorDistance_sp(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                    // 最佳匹配距离要小于设定阈值
                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    //cout<<vmatches.size()<<" "<<i<<" "<<bestIdx2<<endl;
                    vmatches[bestIdx2] = i;
                    nmatches++;

                        
                }
                if(CurrentFrame.Nleft != -1){
                    Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                        : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    float bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                        const float dist = DescriptorDistance_sp(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                        nmatches++;
                    }
                }
            }
        }
    }
    // if(CurrentFrame.mnId%100 ==0)
    // {
    //     plotspmatch(CurrentFrame.imgLeft, LastFrame.imgLeft, CurrentFrame.mvKeys, LastFrame.mvKeys, vmatches);
    // }
    // const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    // nmatches = MatchingPoints_onnx(CurrentFrame, LastFrame, vnMatches12);
    return nmatches;

}

int SPmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist)
{
    int nmatches = 0;
    std::vector<int> vnMatches12;
    nmatches = MatchingPoints_onnx(CurrentFrame.mvKeys, pKF->mvKeys, CurrentFrame.mDescriptors, pKF->mDescriptors, vnMatches12);
    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {   
        MapPoint* pMP = vpMPs[vnMatches12[i]];
        if(!pMP||pMP->isBad()||sAlreadyFound.count(pMP))
        {
            nmatches--;
            continue;
        }
        if(!CurrentFrame.mvpMapPoints[i]){
            CurrentFrame.mvpMapPoints[i]=pMP;
        }
    }
    // for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    //     {
    //         MapPoint* pMP = vpMPs[i];

    //         if(pMP)
    //         {
    //             // 地图点存在 并且不在已有地图点集合里
    //             if(!pMP->isBad() && !sAlreadyFound.count(pMP))
    //             {
    //                 //Project
    //                 Eigen::Vector3f x3Dw = pMP->GetWorldPos();
    //                 Eigen::Vector3f x3Dc = Tcw * x3Dw;

    //                 const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

    //                 if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
    //                     continue;
    //                 if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
    //                     continue;

    //                 // Compute predicted scale level
    //                 Eigen::Vector3f PO = x3Dw-Ow;
    //                 float dist3D = PO.norm();

    //                 const float maxDistance = pMP->GetMaxDistanceInvariance();
    //                 const float minDistance = pMP->GetMinDistanceInvariance();

    //                 // Depth must be inside the scale pyramid of the image
    //                 if(dist3D<minDistance || dist3D>maxDistance)
    //                     continue;

    //                 //预测尺度
    //                 int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

    //                 // Search in a window
    //                 // 搜索半径和尺度相关
    //                 const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

    //                 //  Step 3 搜索候选匹配点
    //                 const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

    //                 if(vIndices2.empty())
    //                     continue;

    //                 const cv::Mat dMP = pMP->GetDescriptor();

    //                 int bestDist = 256;
    //                 int bestIdx2 = -1;
    //                 // Step 4 遍历候选匹配点，寻找距离最小的最佳匹配点 
    //                 for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
    //                 {
    //                     const size_t i2 = *vit;
    //                     if(CurrentFrame.mvpMapPoints[i2])
    //                         continue;

    //                     const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

    //                     const int dist = DescriptorDistance(dMP,d);

    //                     if(dist<bestDist)
    //                     {
    //                         bestDist=dist;
    //                         bestIdx2=i2;
    //                     }
    //                 }

    //                 if(bestDist<=ORBdist)
    //                 {
    //                     CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
    //                     nmatches++;
    //                 }

    //             }
    //         }
    //     }
        return nmatches;
    
}

int SPmatcher::SearchBySP(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches)
{
    const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = std::vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    std::vector<cv::DMatch> matches;
    vector<int> vnMatches12;
    MatchingPoints_onnx(F,F, vnMatches12);//需要修改！
    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        int realIdxKF = matches[i].queryIdx;
        int bestIdxF = matches[i].trainIdx;

        if(matches[i].distance > TH_HIGH)
            continue;
        
        MapPoint* pMP = vpMapPointsKF[realIdxKF];
        if(!pMP)
            continue;
        if(pMP->isBad())
            continue;
        vpMapPointMatches[bestIdxF] = pMP;
        nmatches++;
    }

    return nmatches;
}

int SPmatcher::SearchBySP(Frame &F, const std::vector<MapPoint*> &vpMapPoints)
{
    std::cout << vpMapPoints.size() <<std::endl;
    std::cout << F.mDescriptors.rows << std::endl;

    std::vector<cv::Mat> MPdescriptorAll;
    std::vector<int> select_indice;
    for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];

        if(!pMP)
            continue;
        if(!pMP->mbTrackInView)
            continue;
        if(pMP->isBad())
            continue;
        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        MPdescriptorAll.push_back(MPdescriptor);
        select_indice.push_back(iMP);
    }
    cv::Mat MPdescriptors;
    MPdescriptors.create(MPdescriptorAll.size(), 32, CV_8U);

    for (int i=0; i<static_cast<int>(MPdescriptorAll.size()); i++)
    {
        for(int j=0; j<32; j++)
        {
            MPdescriptors.at<unsigned char>(i, j) = MPdescriptorAll[i].at<unsigned char>(j);
        }
    }

    std::vector<cv::DMatch> matches;
    cv::BFMatcher desc_matcher(cv::NORM_HAMMING, true);
    desc_matcher.match(MPdescriptors, F.mDescriptors, matches, cv::Mat());

    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        int realIdxMap = select_indice[matches[i].queryIdx];
        int bestIdxF = matches[i].trainIdx;

        if(matches[i].distance > TH_HIGH)
            continue;
        if(F.mvpMapPoints[bestIdxF])
            if(F.mvpMapPoints[bestIdxF]->Observations()>0)
                continue;
        
        MapPoint* pMP = vpMapPoints[realIdxMap];
        F.mvpMapPoints[bestIdxF] = pMP;
        nmatches++;
    }
}

int SPmatcher::SearchBySP(Frame &CurrentFrame, Frame &LastFrame)
{
    vector<cv::Point2f> vbPrevMatched;
    vector<int> vnMatches1;
    int size = MatchingPoints_onnx(CurrentFrame, LastFrame, vnMatches1);
    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(vnMatches1.size());++i)
    {
        int IdxLF = vnMatches1[i];
        int IdxCF = i;
        // if(matches[i].distance > TH_LOW)
        //     continue;
        if(IdxLF != -1){
            MapPoint* pMP = LastFrame.mvpMapPoints[IdxLF];
            if(!pMP)
                continue;
            if(pMP->isBad())
                continue;
            if(!LastFrame.mvbOutlier[IdxLF])
                CurrentFrame.mvpMapPoints[IdxCF] = pMP;
             nmatches++;
             vnMatches1[i]=-1;
        }
        // if(CurrentFrame.mnId%100 == 0){
        //     plotspmatch(CurrentFrame.imgLeft, LastFrame.imgLeft, CurrentFrame.mvKeys, LastFrame.mvKeys, vnMatches1);
        // }
        
    }
    
    return nmatches;
}

int SPmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const int th){
    int nmatches = 0;
    int nullnum = 0;
    int numObservations = 0;
    const bool bFactor = th != 1.0;
    for(size_t iMP=0;  iMP<vpMapPoints.size(); iMP++){
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;
        const int &nPredictedLevel = 0;
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        //r = 4;
        //cout<<"r: "<<r<<endl;
        if(bFactor)
            r*=th;
        int num = 0;
        const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);
        //cout<<"vIndices.size()"<<vIndices.size()<<endl;
        if(vIndices.empty())
        {
            nullnum++;
            continue;
        }
           
        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        float bestDist=256;
        int bestLevel= -1;
        float bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            
            const size_t idx = *vit;
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0){

                    if(num == 0){
                        numObservations++;
                    }
                    num++;
                    continue;
                }

                    
            
            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);
            const float dist = DescriptorDistance_sp(MPdescriptor,d);

            if(dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx = idx;
            }
            else if(dist < bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2 = dist;
            }
        }

        if(bestDist <= TH_HIGH)
        {
            //cout<<bestDist<<endl;
            //if(bestLevel == bestLevel2 && bestDist>mfNNratio*bestDist2)
            // if(bestLevel == bestLevel2 && bestDist>0.8*bestDist2)
            //     continue;
            F.mvpMapPoints[bestIdx] = pMP;
            //cout<<"search by projection bestDist: "<<bestDist<<" besDist2:"<<bestDist2<<endl;
            nmatches++;
        }
    }
    //cout<<"numnull: "<<nullnum<<" Observation0: "<<numObservations<<endl;
    return nmatches;
}
int SPmatcher::SearchByProjection1(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
        int nmatches=0, left = 0, right = 0;

        // 如果 th！=1 (RGBD 相机或者刚刚进行过重定位), 需要扩大范围搜索
        const bool bFactor = th!=1.0;

        // Step 1 遍历有效的局部地图点
        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView)
            {
                // 通过距离预测的金字塔层数，该层数相对于当前的帧
                const int &nPredictedLevel = 0;

                // The size of the window will depend on the viewing direction
                // Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                // 如果需要扩大范围搜索，则乘以阈值th
                if(bFactor)
                    r*=th;

                // Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,      // 该地图点投影到一帧上的坐标
                                            r*F.mvScaleFactors[nPredictedLevel],    // 认为搜索窗口的大小和该特征点被追踪到时所处的尺度也有关系
                                            nPredictedLevel-1,nPredictedLevel);     // 搜索的图层范围
                //cout<<vIndices.size()<<"vIndices size"<<endl;
                // 没找到候选的,就放弃对当前点的匹配
                if(!vIndices.empty()){
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    // 最优的次优的描述子距离和index
                    float bestDist=256;
                    int bestLevel= -1;
                    float bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    // Step 4 寻找候选匹配点中的最佳和次佳匹配点
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        // 如果Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
                        if(F.mvpMapPoints[idx])
                            if(F.mvpMapPoints[idx]->Observations()>0)
                                continue;

                        

                        const cv::Mat &d = F.mDescriptors.row(idx);

                        // 计算地图点和候选投影点的描述子距离
                        const float dist = DescriptorDistance_sp(MPdescriptor,d);

                        // 寻找描述子距离最小和次小的特征点和索引
                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysUn[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysUn[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    // Step 5 筛选最佳匹配点
                    // 最佳匹配距离还需要满足在设定阈值内
                    if(bestDist<=TH_HIGH)
                    {
                        // 条件1：bestLevel==bestLevel2 表示 最佳和次佳在同一金字塔层级
                        // 条件2：bestDist>mfNNratio*bestDist2 表示最佳和次佳距离不满足阈值比例。理论来说 bestDist/bestDist2 越小越好
                        // if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2){
                        //     cout<<"mfNNratio: "<<mfNNratio<<endl;
                        //     continue;
                            
                        // }
                            
                        F.mvpMapPoints[bestIdx] = pMP;
                        nmatches++;
                        // if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                        //     F.mvpMapPoints[bestIdx]=pMP;
                        //     cout<<"bestLevel!=bestLevel2"<<endl;
                        //     if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                        //         F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                        //         nmatches++;
                        //         right++;
                        //     }

                        //     nmatches++;
                        //     left++;
                        // }
                    }
                }
            }

            if(F.Nleft != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    float bestDist=256;
                    int bestLevel= -1;
                    float bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;


                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                        const float dist = DescriptorDistance_sp(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;
                            left++;
                        }


                        F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                        nmatches++;
                        right++;
                    }
                }
            }
        }
        return nmatches;
}
int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{
    int nmatches = 0;
    vector<int> vnMatches12 = vector<int>((*pKF1).mvKeys.size(), -1);
    // vector<int> vnMatches12_res = vector<int>((*pKF1).mvKeys.size(),-1);
    // vector<int> vMatchedDistance((*pKF1).mvKeys.size(), INT_MAX);
    // vector<int> vnMatches21((*pKF2).mvKeys.size(), -1);
    
    // //cv::Mat Cw = pKF1->GetCameraCenter();
    // cv::Mat Cw = (cv::Mat_<float>(3, 1) << pKF1->GetCameraCenter()(0), pKF1->GetCameraCenter()(1), pKF1->GetCameraCenter()(2));
    // //cv::Mat R2w = pKF2->GetRotation();
    // cv::Mat R2w(3, 3, CV_32F);
    // cv::eigen2cv(pKF2->GetRotation(), R2w);
    // cv::Mat t2w = (cv::Mat_<float>(3, 1) << pKF2->GetTranslation()(0), pKF2->GetTranslation()(1), pKF2->GetTranslation()(2));
    // cv::Mat C2 = R2w*Cw+t2w;
    // const float invz = 1.0f/C2.at<float>(2);
    // const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    // const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;



    nmatches = MatchingPoints_onnx(pKF1->mvKeys, pKF2->mvKeys, pKF1->mDescriptors, pKF2->mDescriptors,  vnMatches12);
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {
        if(vnMatches12[i]<0)
        
            continue;
        MapPoint* pMP1 = pKF1->GetMapPoint(i);
        if(pMP1)
        {   //vnMatches12[i] = -1;
            continue;
        }
        MapPoint* pMP2 = pKF2->GetMapPoint(vnMatches12[i]);
        if(pMP2)
        {
            //vnMatches12[i]=-1;
            continue;
        }
        vMatchedPairs.push_back(make_pair(i,vnMatches12[i]));
    }
    //if(pKF1->mnId%10 == 0)
    //   plotspmatch(pKF1->mImg,pKF2->mImg,pKF1->mvKeys,pKF2->mvKeys,vnMatches12);
    return nmatches;
}

int SPmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,vector<cv::DMatch>& vmatches)
    {

        // Step 1 分别取出两个关键帧的特征点、BoW 向量、地图点、描述子
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeat3Vec;
        //cout<<"DBoW3::FeatureVector:"<<pKF1->mFeat3Vec;
        //cout<<"pKF1->mFeat3Vec: "<<pKF1->mFeat3Vec<<endl;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeat3Vec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        // 保存匹配结果
        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码   
        // const float factor = HISTO_LENGTH/360.0f;

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            // Step 3 开始遍历，分别取出属于同一node的特征点(只有属于同一node，才有可能是匹配点)
            if(f1it->first == f2it->first)
            {
                // 遍历KF中属于该node的特征点
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    float bestDist1=256;
                    int bestIdx2 =-1 ;
                    float bestDist2=256;

                    // Step 4 遍历KF2中属于该node的特征点，找到了最优及次优匹配点
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        // 如果已经有匹配的点，或者遍历到的特征点对应的地图点无效
                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        float dist = DescriptorDistance_sp(d1,d2);
                        //cout<<"dist: "<<dist<<" bestDist1: "<<bestDist1<<endl;
                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    // Step 5 对匹配结果进行检查，满足阈值、最优/次优比例，记录旋转直方图信息
                    if(bestDist1<TH_LOW)
                    {
                        //cout<<"bestDist1: "<<bestDist1<<" bestDist2: "<<bestDist2<<" mfNNratio: "<<mfNNratio<<endl;
                        if(static_cast<float>(bestDist1)<0.8*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;
                            nmatches++;
                            cv::DMatch match;
                            match.queryIdx = idx1;
                            match.trainIdx = bestIdx2;
                            match.distance = bestDist1;
                            vmatches.push_back(match);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        return nmatches;
    }

int SPmatcher::SearchByBoWSP(KeyFrame* pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12, vector<cv::DMatch> & vmatches)
{
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    int nmatches = 0;
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<int> vnMatches12 = vector<int>((*pKF1).mvKeys.size(), -1);
    nmatches = MatchingPoints_onnx(pKF1->mvKeys, pKF2->mvKeys, pKF1->mDescriptors, pKF2->mDescriptors,  vnMatches12);
    //cout<<"vnMatches12.size():  "<<vnMatches12.size();
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {
        if(vnMatches12[i]<0)
            continue;
        
        MapPoint* pMP1 = pKF1->GetMapPoint(i);
        if(!pMP1||pMP1->isBad())
        {
            continue;
        }
        MapPoint* pMP2 = pKF2->GetMapPoint(vnMatches12[i]);
        if(!pMP2||pMP2->isBad())
        {
            continue;
        }
        //cout<<"22 ";
        vpMatches12[i] = pMP2;
        //cout<<"vpMatches"<<i<<" "<<pMP2<<endl;
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = vnMatches12[i];
        vmatches.push_back(match);
    }
    return vmatches.size();
}
int SPmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            KeyFrame* pKFi = vpPointsKFs[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const float invz = 1/p3Dc(2);
            const float x = p3Dc(0)*invz;
            const float y = p3Dc(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF->IsInImage(u,v))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                // const int &kpLevel= pKF->mvKeysUn[idx].octave;

                // if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const float dist = DescriptorDistance_sp(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                vpMatched[bestIdx] = pMP;
                vpMatchedKF[bestIdx] = pKFi;
                nmatches++;
            }

        }

        return nmatches;
}
int SPmatcher::SearchByBoWSP(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = std::vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    std::vector<cv::DMatch> matches;
    // cv::BFMatcher desc_matcher(cv::NORM_HAMMING, true);
    // desc_matcher.match(pKF->mDescriptors, F.mDescriptors, matches, cv::Mat());
    std::vector<cv::KeyPoint> kpts0 = F.mvKeys;
    std::vector<cv::KeyPoint> kpts1 = pKF->mvKeys;
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    for(int i = 0 ; i < kpts0.size(); i++)
    {
        pts0.push_back(kpts0[i].pt);
    }
    for(int i = 0 ; i < kpts1.size(); i++)
    {
        pts1.push_back(kpts1[i].pt);
    }
    cv::Mat desc0 = F.mDescriptors;
    cv::Mat desc1 = pKF->mDescriptors;
    std::vector<int> vnMatches01;
    int nmatches_ori = MatchingPoints_onnx(pts0, pts1, desc0, desc1, vnMatches01);

    int nmatches =0;
    for(int i = 0 ; i < vnMatches01.size(); i++){
        int IdxKF, IdxF;
        if(vnMatches01[i] != -1){
            IdxKF = vnMatches01[i];
            IdxF = i;
            MapPoint* pMP = vpMapPointsKF[IdxKF];
            if(!pMP)
                continue;
            if(pMP->isBad())
                continue;
            vpMapPointMatches[IdxF]=pMP;
            nmatches++;
        }
        else{
            continue;
        }
    }
    
    // for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
    //     int realIdxKF = matches[i].queryIdx;
    //     int bestIdxF  = matches[i].trainIdx;

    //     if(matches[i].distance > TH_HIGH)
    //         continue;

    //     MapPoint* pMP = vpMapPointsKF[realIdxKF];

    //     if(!pMP)
    //         continue;

    //     if(pMP->isBad())
    //         continue;  
        
    //     vpMapPointMatches[bestIdxF]=pMP;
    //     nmatches++;
    // }
    // std::cout << nmatches << std::endl;

    return nmatches;
}

float SPmatcher::RadiusByViewingCos(const float &viewCos)
{
    // 当视角相差小于3.6°，对应cos(3.6°)=0.998，搜索范围是2.5，否则是4
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}

//没用到
int SPmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // 从world到camera1的变换
    // 从world到camera2的变换
    // Camera 1 & 2 from world
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();

    //Transformation between cameras
    //sim3的逆
    Sophus::Sim3f S21 = S12.inverse();
    //Camera 2 from world
    // cv::Mat R2w = pKF2->GetRotation();
    // cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    // cv::Mat sR12 = s12*R12;
    // cv::Mat sR21 = (1.0/s12)*R12.t();
    // cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if(idx2>=0 && idx2<N2)
                //pKF2中第idx2个特征点在pKF1中有匹配
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        // Step 3.1：通过Sim变换，将pKF1的地图点投影到pKF2中的图像坐标
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        // 把pKF1的地图点从world坐标系变换到camera1坐标系
        Eigen::Vector3f p3Dc1 = T1w * p3Dw;
        // 再通过Sim3将该地图点从camera1变换到camera2坐标系
        Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

        // Depth must be positive
        if(p3Dc2(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2(2);
        const float x = p3Dc2(0)*invz;
        const float y = p3Dc2(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc2 = T2w * p3Dw;
        Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

        // Depth must be positive
        if(p3Dc1(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1(2);
        const float x = p3Dc1(0)*invz;
        const float y = p3Dc1(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

// int SPmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
// {
//     const float &fx = pKF->fx;
//     const float &fy = pKF->fy;
//     const float &cx = pKF->cx;
//     const float &cy = pKF->cy;

//     Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation()/Scw.scale());
//     Eigen::Vector3f Ow = Tcw.inverse().translation();

//     const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

//     int nFused = 0;
//     const int nPoints = vpPoints.size();

//     for(int iMP = 0; iMP < nPoints; iMP++)
//     {
//         MapPoint* pMP = vpPoints[iMP];

//         if(pMP->isBad() || spAlreadyFound.count(pMP))
//             continue;
        
//         Eigen::Vector3f p3Dw = pMP->GetWorldPos();

//         Eigen::Vector3f p3Dc = Tcw * p3Dw;

//         if(p3Dc(2)<0.0f)
//             continue;
        
//         const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

//         if(!pKF->IsInImage(uv(0),uv(1)))
//             continue;
        
//         const float maxDistance = pMP->GetMaxDistanceInvariance();
//         const float minDistance = pMP->GetMinDistanceInvariance();
//         Eigen::Vector3f PO = p3Dw-Ow;
//         const float dist3D = PO.norm();

//         if(dist3D<minDistance || dist3D>maxDistance)
//             continue;
        
//         Eigen::Vector3f Pn = pMP->GetNormal();

//         if(PO.dot(Pn)<0.5*dist3D)
//             continue;

//         const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);
//         const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
//         const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius);

//         if(vIndices.empty())
//             continue;
        
//         const cv::Mat dMP = pMP->GetDescriptor();

//         int bestDist = INT_MAX;
//         int bestIdx = -1;
//         for(vector<size_t>::const_iterator vit = vIndices.begin(); vit!=vIndices.end(); vit++)
//         {
//             const size_t idx = *vit;
//             const int &kpLevel = pKF->mvKeysUn[idx].octave;

//             if(kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
//                 continue;
            
//             const cv::Mat &dKF = pKF->mDescriptors.row(idx);

//             int dist = DescriptorDistance_sp(dMP, dKF);

//             if(dist < bestDist)
//             {
//                 bestDist = dist;
//                 bestIdx = idx;
//             }
//         }

//         if(bestDist <= TH_LOW)
//         {
//             MapPoint* pMPinKF = pKF->GetMapPoint(bestDist);
//             if(pMPinKF)
//             {
//                 if(!pMPinKF->isBad())
//                     vpReplacePoint[iMP] = pMPinKF;
//             }
//             else
//             {
//                 pMP->AddObservation(pKF, bestIdx);
//                 pKF->AddMapPoint(pMP, bestIdx);
//             }
//             nFused++;
//         }
//     }
//     return nFused;
// }

int SPmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // // Project into Image

            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            //观察角度小于60度
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;

                // const int &kpLevel= pKF->mvKeysUn[idx].octave;

                // if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                //     continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance_sp(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx] = pMP;
                nmatches++;
            }

        }

        return nmatches;  
}

float SPmatcher::DescriptorDistance_sp(const cv::Mat &a, const cv::Mat &b)
{
    float dist = (float)cv::norm(a, b, cv::NORM_L2);

    return dist;
}

}

