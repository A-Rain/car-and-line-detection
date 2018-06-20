#include "train.h"

bool IoU_train(int* gt_box, Rect RoI){
    double ratio_x = 448./1920.;
    double ratio_y = 448./1200.;
    double gt_xmin = double(gt_box[0]) * ratio_x;
    double gt_ymin = double(gt_box[1]) * ratio_y;
    double gt_xmax = double(gt_box[2]) * ratio_x;
    double gt_ymax = double(gt_box[3]) * ratio_y;
    int xx1 = gt_xmin < RoI.x ? RoI.x : gt_xmin;
    int xx2 = gt_xmax < RoI.br().x ? gt_xmax : RoI.br().x;
    int yy1 = gt_ymin < RoI.y ? RoI.y : gt_ymin;
    int yy2 = gt_ymax < RoI.br().y ? gt_ymax : RoI.br().y;
    int w = xx2-xx1 > 0 ? xx2-xx1 : 0;
    int h = yy2-yy1 > 0 ? yy2-yy1 : 0;
    float area1 = w * h;
    float area2 = (gt_xmax-gt_xmin)*(gt_ymax-gt_ymin) + RoI.area() - area1;
    //in gt_box or include gt_box is not right
    if (gt_xmin > RoI.x && gt_xmax < RoI.x + RoI.width &&
        gt_ymin > RoI.y && gt_ymax < RoI.y + RoI.height)
        return false;
    if (gt_xmin < RoI.x && gt_xmax > RoI.x + RoI.width &&
        gt_ymin < RoI.y && gt_ymax > RoI.y + RoI.height)
        return false;
    //cout << area1 / area2 << endl;
    return area1 / area2 < 0.4;
}

string ImgName(int No) {
    if (No<10)
        return "IMG_0000" + to_string(No) + ".jpg";
    if (No>=10 && No<100)
        return "IMG_000" + to_string(No) + ".jpg";
    if (No>=100 && No<1000)
        return "IMG_00" + to_string(No) + ".jpg";
    if (No>=1000 && No<10000)
        return "IMG_0" + to_string(No) + ".jpg";
    if (No>=10000)
        return "IMG_" + to_string(No) + ".jpg";
}

void HOGSVMtest(string testlist){
    Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>("../model/svm_hog_classifier.xml");
    Mat test;
    int label, allNum, correct;
    string line, imgname;
    allNum = 0; correct = 0;
    fstream in(testlist);
    while (getline(in, line)) {
        test.release();
        vector<string> fields;
        boost::split(fields, line, boost::is_any_of(" "));
        imgname = fields[0];
        label = stringToNum(fields[1]);
        test.push_back(GetHOGfeature(ImgTestPath + imgname));

        if (allNum % 32 == 0)
            cout << "test at " << allNum << endl;
        int predict_result = model->predict(test);
        float distance = model->predict(test, noArray(), true);
        float probability = 1/(1 + exp(-abs(model->predict(test, noArray(), true))));
        cout << "test at " << imgname << " " << predict_result << ", correct?-> " << (distance < 0 ? 1 : -1) << ", probability:  " << probability << endl;
        if (label == predict_result)
            correct++;
        allNum++;
    }
    cout << "the result of testing is " << float(correct)/allNum << endl;
}

void HOGSVMtrainAuto(string trainlist){
    Mat Data4Train(0, dimension, CV_32FC1), labels(0, 1, CV_32SC1);
    GetAllImgHOGfeature(Data4Train, labels, trainlist, ImgTrainPath);

    struct timeval pre, after;
    gettimeofday(&pre, NULL);

    Ptr<ml::SVM> model = ml::SVM::create();
    model->setKernel(ml::SVM::KernelTypes::LINEAR);
    model->setType(ml::SVM::C_SVC);
    model->setP(1e-2);
    model->setC(1);
    model->setGamma(1e-2);
    model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.000001));

    if(debug){
        cout << "height: "<<Data4Train.rows << ", width: " << Data4Train.cols << endl;
        cout << "trainingdata depth: " << Data4Train.depth() << endl;
        cout << "label depth: " << labels.depth() << endl;
        cout << "trainingdata type " << Data4Train.type() << endl;
        cout << "label type " << labels.type() << endl;
    }

    assert(Data4Train.type() == CV_32FC1);
    assert(labels.type() == CV_32SC1);

    Ptr<ml::TrainData> data = ml::TrainData::create(Data4Train, ml::ROW_SAMPLE, labels);
    cout << "start training ..." << endl;
    model->trainAuto(data, 10);
    cout << "finish training ..." << endl;
    gettimeofday(&after, NULL);
    cout << "training time: " << after.tv_sec - pre.tv_sec << "s"<< endl;
    model->save("../model/svm_hog_classifier.xml");
    cout << "model saving fininshed ..." << endl;

}

vector<int*> ParseXMLFILE(char *filename){
    vector<int*> data;
    int* objbbox;
    TiXmlDocument doc;
    doc.LoadFile(filename);
    TiXmlElement* root = doc.RootElement();
    TiXmlElement* objNode = root->FirstChildElement();
    objNode = objNode->NextSiblingElement();
    for(;objNode!=NULL;objNode = objNode->NextSiblingElement()) {
        const char *name = objNode->FirstChildElement("name")->GetText();
        if (strcmp(name, "Pedestrian")) {
            objbbox = new int[4];
            objbbox[0] = (TurnConstChar2Num(
                    objNode->FirstChildElement("bndbox")->
                            FirstChildElement("xmin")->GetText()));
            objbbox[1] = (TurnConstChar2Num(
                    objNode->FirstChildElement("bndbox")->
                            FirstChildElement("ymin")->GetText()));
            objbbox[2] = (TurnConstChar2Num(
                    objNode->FirstChildElement("bndbox")->
                            FirstChildElement("xmax")->GetText()));
            objbbox[3] = (TurnConstChar2Num(
                    objNode->FirstChildElement("bndbox")->
                            FirstChildElement("ymax")->GetText()));
            data.push_back(objbbox);
        }
    }
    return data;
}


void FindHardExample(string filename, vector<Rect> bbox, Mat& img, int& ID, vector<string>& HExlist) {
    vector<int*> gt_box = ParseXMLFILE(TurnStringToCharArray(filename));
    //double ratio_x = 448./1920.;
    //double ratio_y = 448./1200.;
    for (size_t i=0; i<bbox.size(); i++) {
        bool flag = false;
        for (size_t j=0; j< gt_box.size(); j++) {
            if (!IoU_train(gt_box[j], bbox[i]))
            {flag = true;break;}
        }
        if (!flag){
            imwrite(RetrainTmpStorePath + ImgName(ID), img(bbox[i]));
            HExlist.push_back(ImgName(ID));
            ID++;
        }
    }

}

void FindHardExampleAndRetrain(string filename){
    HOGDescriptor my_hog(Size(Window_y, Window_x), Size(block_y, block_x), Size(block_stride_y, block_stride_x), Size(cell_y, cell_x), 9);
    vector<string> HardExampleList;
    int iter_time = 0;

    while (iter_time < 20){
        HardExampleList.clear();

        //first: get support vector from last trained model
        cout << "+--------------------------------------------+" << endl <<">>>load last trained model" << endl;
        string model_name = "../model/hog/svm_hog_classifier" + to_string(iter_time) + ".xml";
        Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>(model_name);
        Mat sv = model->getSupportVectors();
        vector<float> hog_detector;
        const int sv_total = sv.cols;
        Mat alpha, svidx;
        double rho = model->getDecisionFunction(0, alpha, svidx);
        cout << "rho: " << rho << endl;
        Mat alpha2;
        alpha.convertTo(alpha2, CV_32FC1);
        Mat result(1, sv_total, CV_32FC1);
        result = alpha2*sv;
        for (int i = 0; i < sv_total; ++i)
            hog_detector.push_back(-1*result.at<float>(0, i));
        hog_detector.push_back((float)rho);
        cout << hog_detector.size() << endl;
        my_hog.setSVMDetector(hog_detector);

        //second: using hog detector to find hard example
        cout << ">>>finish loading, start finding hard examples: " << endl;
        vector<Rect> detections;
        vector<double> foundWeights;
        vector<Rect> det_bbox;
        fstream in(filename);
        string line;
        int ID = 0;
        int imgNo = 0;
        while (getline(in, line)) {

            Mat img = imread(ImgRetrainPath+line+".jpg");
            resize(img, img, Size(448, 448));
            assert(img.depth() == CV_8U || img.depth() == CV_8UC3);

            detections.clear();
            foundWeights.clear();
            det_bbox.clear();

            my_hog.detectMultiScale(img, detections, foundWeights, 0, Size(8, 8), Size(), 1.2);
            for (size_t i = 0; i<detections.size(); i++) {
                if (foundWeights[i]>1.) {
                    det_bbox.push_back(detections[i]);
                }
            }
            FindHardExample(ImgAnnotationPath+line+".xml", det_bbox, img, ID, HardExampleList);
            if (imgNo % 1000 == 0)
                cout << ">>>Find Total Hard Examples: " << ID << " in img from 0 to " << imgNo << endl;
            imgNo++;
        }

        //third: using those hard example to retrain
        cout << ">>>finish finding hard examples, begin retraining models... " << endl;
        Mat Data4Train(0, dimension, CV_32FC1), labels(0, 1, CV_32SC1);
        for (size_t i=0; i<HardExampleList.size(); i++){
            Data4Train.push_back(GetHOGfeature(RetrainTmpStorePath+HardExampleList[i]));
            labels.push_back(-1);
        }

        GetAllImgHOGfeature(Data4Train, labels, "../img/retrain_pos.txt", RetrainPosPath);
        model->setKernel(ml::SVM::KernelTypes::LINEAR);
        model->setType(ml::SVM::C_SVC);
        model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001));
        Ptr<ml::TrainData> data = ml::TrainData::create(Data4Train, ml::ROW_SAMPLE, labels);

        struct timeval pre, after;
        gettimeofday(&pre, NULL);
        cout << ">>>finish extract hog feature, start training ..." << endl;
        model->trainAuto(data);
        cout << ">>>finish training ..." << endl;
        gettimeofday(&after, NULL);
        cout << ">>>training time: " << after.tv_sec - pre.tv_sec << "s"<< endl;
        iter_time++;
        model->save("../model/hog/svm_hog_classifier" + to_string(iter_time) + ".xml");
        cout << ">>>model saving fininshed ..." << endl;
        cout << "+--------------------------------------------+" << endl;
    }
}

