#ifndef CV_TRAIN_H
#define CV_TRAIN_H
#endif

#include "GetFeature.h"
#include <opencv2/ml.hpp>
#include "Config.h"
#include <time.h>
#include <sys/time.h>
#include <tinyxml.h>
#include <tinystr.h>
#include <fstream>

void HOGSVMtest(string testlist);
void HOGSVMtrainAuto(string trainlist);
void FindHardExampleAndRetrain(string filename);



