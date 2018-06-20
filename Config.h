#ifndef CV_CONFIG_H
#define CV_CONFIG_H
#endif //CV_CONFIG_H

#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>


extern const std::string ImgTestPath;
extern const std::string ImgTrainPath;
extern const std::string ImgRetrainPath;
extern const std::string ImgAnnotationPath;
extern const std::string RetrainTmpStorePath;
extern const std::string RetrainPosPath;
extern const int Imgwidth;
extern const int Imgheight;
extern const int block_x;
extern const int block_y;
extern const int Window_x;
extern const int Window_y;
extern const int block_stride_x;
extern const int block_stride_y;
extern const int cell_x;
extern const int cell_y;
extern const int Window_stride_x;
extern const int Window_stride_y;
extern int dimension;
extern bool debug;
extern float confidence;
extern float nms_threshold;


int stringToNum(const std::string& str);

int TurnConstChar2Num(const char* str);

char* TurnStringToCharArray(std::string str);