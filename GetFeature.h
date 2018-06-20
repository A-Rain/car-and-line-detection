#ifndef CV_GETFEATURE_H
#define CV_GETFEATURE_H
#endif


#include "opencvLib.h"
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "Config.h"



Mat GetSIFTfeature(string imgname);
Mat GetHOGfeatureFromMat(Mat img);
Mat GetAllImgSIFTfeature(string Trainlist, string img_dir_path_preced);
Mat GetHOGfeature(string imgname);
void GetAllImgHOGfeature(Mat& AllImgDescriptor, Mat& label, string Trainlist, string img_dir_path_preced);





