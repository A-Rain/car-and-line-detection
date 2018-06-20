#include <iostream>
#include "opencvLib.h"
#include <opencv2/ml.hpp>
#include "detection.h"
#include "train.h"



int main() {
//    HOGSVMtrainAuto("../img/train.txt");
    FinalDetect("/home/oliver/data/out2.avi",
                "../model/adaboost/cascade.xml",
                "../model/svm_hog_classifier.xml",
                2,
                false);
    return 0;
}

