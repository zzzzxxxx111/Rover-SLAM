#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>
#include <random>
struct Configuration
{
    std::string lightgluePath;
    std::string extractorPath;
    
    std::string extractorType;
    bool isEndtoEnd = true;
    bool grayScale = false;

    unsigned int image_size = 512; 
    float threshold = 0.0f;

    std::string device;
    bool viz = false;
};
#endif // CONFIGURATION_H