#include "Config.h"

const int Imgwidth = 64;
const int Imgheight = 64;



const int block_x = 16;
const int block_y = 16;
const int Window_x = 64;
const int Window_y = 64;
const int block_stride_x = 8;
const int block_stride_y = 8;
const int cell_x = 4;
const int cell_y = 4;
const int Window_stride_x = 8;
const int Window_stride_y = 8;
int dimension = 9 * (block_x/cell_x) * (block_y/cell_y) *
                (1+(Window_x-block_x)/block_stride_x) * (1+(Window_y-block_y)/block_stride_y);


float confidence = 0.6;
float nms_threshold = 0.4;
bool debug= true;

//first train
const std::string ImgTestPath = "/home/oliver/data/ProData/";
const std::string ImgTrainPath = "/home/oliver/data/ProData/";

//second train
const std::string ImgRetrainPath = "/home/oliver/data/ProData6/JEPGImages/";
const std::string ImgAnnotationPath = "/home/oliver/data/ProData6/Annotations/";
const std::string RetrainTmpStorePath = "/home/oliver/data/TempData/";
const std::string RetrainPosPath = "/home/oliver/data/ProData7/";
//const std::string ImgRetrainPath = "../img/";
//const std::string ImgAnnotationPath = "../img/";
//const std::string RetrainTmpStorePath = "../img/tmp/";

int stringToNum(const std::string &str) {
    std::istringstream iss(str);
    int num;
    iss >> num;
    return num;
}

int TurnConstChar2Num(const char* str){
    int y;
    std::stringstream s(str);
    s >> y;
    return y;
}

char* TurnStringToCharArray(std::string str){
    char* tmp = new char[str.length()];
    std::strcpy(tmp, str.c_str());
    return tmp;
}
