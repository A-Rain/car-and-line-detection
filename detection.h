#include "opencvLib.h"
#include <opencv2/ximgproc.hpp>
#include "GetFeature.h"
#include <set>

using namespace cv::ximgproc;
using namespace cv::ximgproc::segmentation;



void adaboostTest(string model_path, string img_name);
void HOGdetect(string filename);
void MultiScaleDetect(string model_hog, string model_cascade, string filename);
void LineDetect(string imgname);
void FinalDetect(string filename, string model_cascade, string model_hog,int dataset, bool IsLine);

class bbox_info{
public:
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    float score;
    bbox_info(int x1, int y1, int x2, int y2, float s):xmin(x1), ymin(y1), xmax(x2), ymax(y2), score(s){}
    int area(){
        return (xmax-xmin) * (ymax-ymin);
    }
    void print(){
        cout << "xmin: " << xmin << ", ymin: " << ymin << ", xmax: " << xmax << ", ymax: " << ymax << endl;
    }
};

