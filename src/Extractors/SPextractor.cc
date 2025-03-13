/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#include "Matchers/Configuration.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "Extractors/SPextractor.h"
#include "ORBextractor.h"
// #include "SuperPoint.h"
#include <opencv2/core/eigen.hpp>

#include "Extractors/super_point.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


const float factorPI = (float)(CV_PI/180.f);

SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         float _iniThFAST, float _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
   
    if(mModelstr == "onnx")
    {   Configuration cfg;
        cfg.device = "cuda";
        cfg.extractorPath = "onnxmodel/superpoint.onnx";
        cfg.extractorType = "superpoint";
        featureExtractor = new SuperPointOnnxRunner();
        featureExtractor->InitOrtEnv(cfg);

    }
    // else{
    //     mModel = _superpoint;
    //     if(!(mModel)->build()){
    //         std::cout << "Error in SuperPoint building" << std::endl;
    //     exit(0);
    //     }
    // }

    // model = make_shared<SuperPoint>();
    // torch::load(model, "/media/xiao/data3/learning-slam/ORB-SLAM2-SP/Examples/Monocular/superpoint.pt");
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }
    for(int i = 0; i < mvLevelSigma2.size(); i++)
    {
        cout<<mvLevelSigma2[i]<<" assd";
    }
    cout<<endl;
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

}


// vector<cv::KeyPoint> SPextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
//                                        const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
// {
//     // Compute how many initial nodes   
//     const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

//     const float hX = static_cast<float>(maxX-minX)/nIni;

//     list<ExtractorNode> lNodes;

//     vector<ExtractorNode*> vpIniNodes;
//     vpIniNodes.resize(nIni);

//     for(int i=0; i<nIni; i++)
//     {
//         ExtractorNode ni;
//         ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
//         ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
//         ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
//         ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
//         ni.vKeys.reserve(vToDistributeKeys.size());

//         lNodes.push_back(ni);
//         vpIniNodes[i] = &lNodes.back();
//     }

//     //Associate points to childs
//     for(size_t i=0;i<vToDistributeKeys.size();i++)
//     {
//         const cv::KeyPoint &kp = vToDistributeKeys[i];
//         vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);//出现段错误，索引越界了
//     }

//     list<ExtractorNode>::iterator lit = lNodes.begin();

//     while(lit!=lNodes.end())
//     {
//         if(lit->vKeys.size()==1)
//         {
//             lit->bNoMore=true;
//             lit++;
//         }
//         else if(lit->vKeys.empty())
//             lit = lNodes.erase(lit);
//         else
//             lit++;
//     }

//     bool bFinish = false;

//     int iteration = 0;

//     vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
//     vSizeAndPointerToNode.reserve(lNodes.size()*4);

//     while(!bFinish)
//     {
//         iteration++;

//         int prevSize = lNodes.size();

//         lit = lNodes.begin();

//         int nToExpand = 0;

//         vSizeAndPointerToNode.clear();

//         while(lit!=lNodes.end())
//         {
//             if(lit->bNoMore)
//             {
//                 // If node only contains one point do not subdivide and continue
//                 lit++;
//                 continue;
//             }
//             else
//             {
//                 // If more than one point, subdivide
//                 ExtractorNode n1,n2,n3,n4;
//                 lit->DivideNode(n1,n2,n3,n4);

//                 // Add childs if they contain points
//                 if(n1.vKeys.size()>0)
//                 {
//                     lNodes.push_front(n1);                    
//                     if(n1.vKeys.size()>1)
//                     {
//                         nToExpand++;
//                         vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
//                         lNodes.front().lit = lNodes.begin();
//                     }
//                 }
//                 if(n2.vKeys.size()>0)
//                 {
//                     lNodes.push_front(n2);
//                     if(n2.vKeys.size()>1)
//                     {
//                         nToExpand++;
//                         vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
//                         lNodes.front().lit = lNodes.begin();
//                     }
//                 }
//                 if(n3.vKeys.size()>0)
//                 {
//                     lNodes.push_front(n3);
//                     if(n3.vKeys.size()>1)
//                     {
//                         nToExpand++;
//                         vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
//                         lNodes.front().lit = lNodes.begin();
//                     }
//                 }
//                 if(n4.vKeys.size()>0)
//                 {
//                     lNodes.push_front(n4);
//                     if(n4.vKeys.size()>1)
//                     {
//                         nToExpand++;
//                         vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
//                         lNodes.front().lit = lNodes.begin();
//                     }
//                 }

//                 lit=lNodes.erase(lit);
//                 continue;
//             }
//         }       

//         // Finish if there are more nodes than required features
//         // or all nodes contain just one point
//         if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
//         {
//             bFinish = true;
//         }
//         else if(((int)lNodes.size()+nToExpand*3)>N)
//         {

//             while(!bFinish)
//             {

//                 prevSize = lNodes.size();

//                 vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
//                 vSizeAndPointerToNode.clear();

//                 sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
//                 for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
//                 {
//                     ExtractorNode n1,n2,n3,n4;
//                     vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

//                     // Add childs if they contain points
//                     if(n1.vKeys.size()>0)
//                     {
//                         lNodes.push_front(n1);
//                         if(n1.vKeys.size()>1)
//                         {
//                             vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
//                             lNodes.front().lit = lNodes.begin();
//                         }
//                     }
//                     if(n2.vKeys.size()>0)
//                     {
//                         lNodes.push_front(n2);
//                         if(n2.vKeys.size()>1)
//                         {
//                             vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
//                             lNodes.front().lit = lNodes.begin();
//                         }
//                     }
//                     if(n3.vKeys.size()>0)
//                     {
//                         lNodes.push_front(n3);
//                         if(n3.vKeys.size()>1)
//                         {
//                             vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
//                             lNodes.front().lit = lNodes.begin();
//                         }
//                     }
//                     if(n4.vKeys.size()>0)
//                     {
//                         lNodes.push_front(n4);
//                         if(n4.vKeys.size()>1)
//                         {
//                             vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
//                             lNodes.front().lit = lNodes.begin();
//                         }
//                     }

//                     lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

//                     if((int)lNodes.size()>=N)
//                         break;
//                 }

//                 if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
//                     bFinish = true;

//             }
//         }
//     }

//     // Retain the best point in each node
//     vector<cv::KeyPoint> vResultKeys;
//     vResultKeys.reserve(nfeatures);
//     for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
//     {
//         vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
//         cv::KeyPoint* pKP = &vNodeKeys[0];
//         float maxResponse = pKP->response;

//         for(size_t k=1;k<vNodeKeys.size();k++)
//         {
//             if(vNodeKeys[k].response>maxResponse)
//             {
//                 pKP = &vNodeKeys[k];
//                 maxResponse = vNodeKeys[k].response;
//             }
//         }

//         vResultKeys.push_back(*pKP);
//     }

//     return vResultKeys;
// }


// void SPextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints, cv::Mat &_desc)
// {
//     allKeypoints.resize(nlevels);

//     vector<cv::Mat> vDesc;

//     const float W = 30;
//     // torch::Tensor x[nlevels];
//     // for (int level = 0; level < nlevels; ++level){
//     //     x[level] = torch::from_blob(mvImagePyramid[level].clone().data, {1, 1, mvImagePyramid[level].rows, mvImagePyramid[level].cols}, torch::kByte).cuda();
//     // }
//     for (int level = 0; level < nlevels; ++level)
//     {
        
//         // TODO gpu
//         // detector.detect(x[level], true);
//         Eigen::Matrix<double, 259, Eigen::Dynamic>  points1;
//         mModel->infer(mvImagePyramid[level], points1);
//         std::cout << "points1 dimensions: " << points1.rows() << " x " << points1.cols() << std::endl;
//         Eigen::MatrixXd descriptors = points1.block(3, 0, points1.rows() - 3, points1.cols());
//         cv::Mat desc_mat(descriptors.rows(), descriptors.cols(), CV_64FC1, descriptors.data());
//         vDesc.push_back(desc_mat);
//         cv::KeyPoint keypoint;
//         // std::vector<cv::KeyPoint> &vKeyPoints;
//         vector<cv::KeyPoint> vToDistributeKeys;
//         vToDistributeKeys.reserve(nfeatures*10);
//         for (int i = 0 ; i < points1.cols(); i++)
//         {
//             keypoint.pt.x = static_cast<float>(points1(1, i)); // 第二行是 x 坐标
//             keypoint.pt.y = static_cast<float>(points1(2, i)); // 第三行是 y 坐标
//             keypoint.response = static_cast<float>(points1(0, i));// 第一行是 响应值
//             vToDistributeKeys.emplace_back(keypoint);
//         }

//         const int minBorderX = EDGE_THRESHOLD-3;
//         const int minBorderY = minBorderX;
//         const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
//         const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

//         // vector<cv::KeyPoint> vToDistributeKeys;
//         // vToDistributeKeys.reserve(nfeatures*10);

//         const float width = (maxBorderX-minBorderX);
//         const float height = (maxBorderY-minBorderY);

//         const int nCols = width/W;
//         const int nRows = height/W;
//         const int wCell = ceil(width/nCols);
//         const int hCell = ceil(height/nRows);

//         // for(int i=0; i<nRows; i++)
//         // {
//         //     const float iniY =minBorderY+i*hCell;
//         //     float maxY = iniY+hCell+6;

//         //     if(iniY>=maxBorderY-3)
//         //         continue;
//         //     if(maxY>maxBorderY)
//         //         maxY = maxBorderY;

//         //     for(int j=0; j<nCols; j++)
//         //     {
//         //         const float iniX =minBorderX+j*wCell;
//         //         float maxX = iniX+wCell+6;
//         //         if(iniX>=maxBorderX-6)
//         //             continue;
//         //         if(maxX>maxBorderX)
//         //             maxX = maxBorderX;

//         //         vector<cv::KeyPoint> vKeysCell;
//         //         // FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
//         //         //      vKeysCell,iniThFAST,true);
//         //         detector.getKeyPoints(iniThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
                
//         //         if(vKeysCell.empty())
//         //         {
//         //             // FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
//         //             //      vKeysCell,minThFAST,true);
//         //             detector.getKeyPoints(minThFAST, iniX, maxX, iniY, maxY, vKeysCell, true);
//         //         }

//         //         if(!vKeysCell.empty())
//         //         {
//         //             for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
//         //             {
//         //                 (*vit).pt.x+=j*wCell;
//         //                 (*vit).pt.y+=i*hCell;
//         //                 vToDistributeKeys.push_back(*vit);
//         //             }
//         //         }

//         //     }
//         // }


//         vector<KeyPoint> & keypoints = allKeypoints[level];
//         keypoints.reserve(nfeatures);
        
        
//         keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
//                                       minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

//         const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

//         // Add border to coordinates and scale information
//         const int nkps = keypoints.size();
//         for(int i=0; i<nkps ; i++)
//         {
//             keypoints[i].pt.x+=minBorderX;
//             keypoints[i].pt.y+=minBorderY;
//             keypoints[i].octave=level;
//             keypoints[i].size = scaledPatchSize;
//         }

//         // cv::Mat desc;
//         // detector.computeDescriptors(keypoints, desc);
//         // vDesc.push_back(desc);
        
//     }

//     cv::vconcat(vDesc, _desc);//什么意思????应该不能拼接阿

//     //示例:
// //     std::vector<cv::Mat> matrices = {cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
// //                                     cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
// //                                     cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};
// // cv::Mat out;
// // cv::vconcat(matrices, out);
// // cout << out << endl;
// // // out
// // [  1,   1,   1,   1;
// //    2,   2,   2,   2;
// //    3,   3,   3,   3]

//     // // compute orientations
//     // for (int level = 0; level < nlevels; ++level)
//     //     computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
// }


int SPextractor::operator()( InputArray _image,  vector<KeyPoint>& _keypoints,
                      cv::Mat& _descriptors)
{
    if(_image.empty())
        return 0;

    Mat image = _image.getMat();
    
    //cout << typeToString(image.type());
    assert(image.type() == CV_8UC1 );

    Mat descriptors;
    //两种模式，一种单层图像提取特征，一种多层金字塔提取特征
    int res = -1;
    if (nlevels == 1) 
        res = ExtractSingleLayer(image, _keypoints, _descriptors);
    else{
        ComputePyramid(image);
        res = ExtractMultiLayers(image, _keypoints, descriptors);
    }
    // Pre-compute the scale pyramid

    // ComputePyramid(image);

    // vector < vector<KeyPoint> > allKeypoints;
    // ComputeKeyPointsOctTree(allKeypoints, descriptors);
    // cout <<"descriptors.rows:"<< descriptors.rows <<"  descriptors.cols:"<<descriptors.cols<<endl;


    // int nkeypoints = 0;
    // for (int level = 0; level < nlevels; ++level)
    //     nkeypoints += (int)allKeypoints[level].size();
    // if( nkeypoints == 0 )
    //     _descriptors.release();
    // else
    // {
    //     _descriptors.create(nkeypoints, 256, CV_32F);
    //     descriptors.copyTo(_descriptors.getMat());
    // }

    // _keypoints.clear();
    // _keypoints.reserve(nkeypoints);

    // int offset = 0;
    // for (int level = 0; level < nlevels; ++level)
    // {
    //     vector<KeyPoint>& keypoints = allKeypoints[level];
    //     int nkeypointsLevel = (int)keypoints.size();

    //     if(nkeypointsLevel==0)
    //         continue;

    //     // // preprocess the resized image
    //     // Mat workingMat = mvImagePyramid[level].clone();
    //     // GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    //     // // Compute the descriptors
    //     // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    //     // computeDescriptors(workingMat, keypoints, desc, pattern);

    //     // offset += nkeypointsLevel;

    //     // Scale keypoint coordinates
    //     if (level != 0)
    //     {
    //         float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
    //         for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
    //              keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    //             keypoint->pt *= scale;
    //     }
    //     // And add the keypoints to the output
    //     _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    // }
    return res;
}

int SPextractor::ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors)
{
    if(mModelstr == "onnx"){
        Configuration cfg;
        cv::Mat image_copy = image.clone();
        cv::Mat inputImage = NormalizeImage(image_copy);
        featureExtractor->lastmatch = lastmatchnum;
        featureExtractor->Extractor_Inference(cfg , inputImage);
        featureExtractor->Extractor_PostProcess(cfg , std::move(featureExtractor->extractor_outputtensors[0]),vKeyPoints,Descriptors);
    }
    else{
        // if(!mModel->infer(image,  vKeyPoints, Descriptors, nfeatures))
        // cerr <<"Error while detecting keypoints"<<endl;
    }

    //std::cout << "[INFO] keyPoints size:" <<vKeyPoints.size()<<std::endl;


    // cv::Mat imagewithKeyPoints;
    // cv::drawKeypoints(image, vKeyPoints, imagewithKeyPoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::imshow("Image with KeyPoints", imagewithKeyPoints);
    // cv::waitKey(0);


    return vKeyPoints.size();
}

int SPextractor::ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors)
{
    ComputePyramid(image);
    int nKeyPoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        if(level == 0)
        {
            // if (!mModel->infer(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], nfeatures))
            // cerr << "Error while detecting keypoints" << endl;
        }
        else
        {
            // if (!mModel->infer(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], nfeatures))//toDo 构建模型Vector
            // cerr << "Error while detecting keypoints" << endl;
        }
        nKeyPoints +=allKeypoints[level].size();
    }
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeyPoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for(auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), Descriptors);
    
    return vKeyPoints.size();
}
// void SPextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
//                       OutputArray _descriptors)
// { 
//     if(_image.empty())
//         return;

//     Mat image = _image.getMat();
//     assert(image.type() == CV_8UC1 );

//     vector<KeyPoint> keypoints;

//     Mat desc = SPdetect(model, image, _keypoints, iniThFAST, true, false);

//     // Mat kpt_mat(keypoints.size(), 2, CV_32F);
//     // for (size_t i = 0; i < keypoints.size(); i++) {
//     //     kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.x;
//     //     kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.y;
//     // }
//     // Mat descriptors;
//     // int border = 8;
//     // int dist_thresh = 4;
//     // int height = image.rows;
//     // int width = image.cols;
//     // nms(kpt_mat, desc, _keypoints, descriptors, border, dist_thresh, width, height);
//     // cout << "hihihi" << endl;

//     int nkeypoints = _keypoints.size();
//     _descriptors.create(nkeypoints, 256, CV_32F);
//     desc.copyTo(_descriptors.getMat());

// }

void SPextractor::ComputePyramid(cv::Mat image)
{
    std::cout<<" compute pyramid !"<<std::endl;
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);            
        }
    }

}

} //namespace ORB_SLAM
