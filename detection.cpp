#include "detection.h"

bool comp(const bbox_info &a, const bbox_info &b) {
    return a.score > b.score;
}

bool IoU(bbox_info &a, bbox_info &b) {
    int xx1 = a.xmin < b.xmin ? b.xmin : a.xmin;
    int xx2 = a.xmax < b.xmax ? a.xmax : b.xmax;
    int yy1 = a.ymin < b.ymin ? b.ymin : a.ymin;
    int yy2 = a.ymax < b.ymax ? a.ymax : b.ymax;
    int w = xx2 - xx1 > 0 ? xx2 - xx1 + 1 : 0;
    int h = yy2 - yy1 > 0 ? yy2 - yy1 + 1 : 0;
    float area1 = w * h;
    float area2 = a.area() + b.area() - area1;
    if (area1 / area2 > nms_threshold)
        return true;
    else
        return false;
}

vector<bbox_info> nms(vector<bbox_info> &dets) {
    vector<bbox_info> keep;
    sort(dets.begin(), dets.end(), comp);
    int num = 0;
    vector<bbox_info>::iterator it;
    while (dets.size() > 0) {
        bbox_info temp = dets[0];
        keep.push_back(temp);
        //dets.pop_back();
        it = dets.begin();
        for (; it < dets.end(); it++) {
            if (IoU(temp, *it))
                it = dets.erase(it);
        }
        num++;
    }
    for (it = keep.begin(); it < keep.end(); it++)
        if ((*it).ymin < 100)
            it = keep.erase(it);
    return keep;
}

void detect_and_display(CascadeClassifier car_classifier, Mat car, string video_windows_name) {
    std::vector<Rect> cars;
    Mat car_gray, small_img;
    vector<bbox_info> dets;
    vector<bbox_info> keep;
    cvtColor(car, car_gray, CV_BGR2GRAY);  //rgb类型转换为灰度类型
    resize(car_gray, small_img, Size(448, 448), 0, 0, CV_INTER_LINEAR);
    equalizeHist(small_img, small_img);   //直方图均衡化
    vector<int> rejLevel;
    vector<double> levelW;
    car_classifier.detectMultiScale(small_img, cars, rejLevel, levelW, 1.1, 3, 0, Size(), Size(), true);
    for (int i = 0; i < cars.size(); i++) {
        if (rejLevel[i] < 20 || levelW[i] < 1.)
            continue;
        bbox_info tmp(cars[i].x, cars[i].y, cars[i].br().x, cars[i].br().y, levelW[i]);
        dets.push_back(tmp);
    }
    keep = nms(dets);
    for (size_t i = 0; i < keep.size(); i++) {
        Point p1(keep[i].xmin, keep[i].ymin), p2(keep[i].xmax, keep[i].ymax);
        Scalar color(0, 255, 0);
        rectangle(car, p1, p2, color, 2);
    }
    imshow(video_windows_name, car);
    cout << "detect" << endl;
    waitKey(1);
}

void adaboostTest(string model_path, string vedio_name) {
    CascadeClassifier car_classifier;
    car_classifier.load(model_path);
    fstream in(vedio_name);
    string line;
    while (getline(in, line)) {
        Mat img = imread(ImgRetrainPath + line + ".jpg");
        resize(img, img, Size(448, 448));
        detect_and_display(car_classifier, img, "detect_and_display");
    }

}

void HOGdetect(string filename) {
    HOGDescriptor my_hog(Size(Window_y, Window_x), Size(block_y, block_x), Size(block_stride_y, block_stride_x),
                         Size(cell_y, cell_x), 9);

    //get support vector from model
    Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>("../model/svm_hog_classifier.xml");
    Mat sv = model->getSupportVectors();
    vector<float> hog_detector;
    const int sv_total = sv.cols;
    Mat alpha, svidx;
    double rho = model->getDecisionFunction(0, alpha, svidx);
    Mat alpha2;
    alpha.convertTo(alpha2, CV_32FC1);
    Mat result(1, sv_total, CV_32FC1);
    result = alpha2 * sv;
    for (int i = 0; i < sv_total; ++i)
        hog_detector.push_back(-1 * result.at<float>(0, i));
    hog_detector.push_back((float) rho);
    //load vector to hog detector
    my_hog.setSVMDetector(hog_detector);

    vector<Rect> detections;
    vector<double> foundWeights;
    vector<bbox_info> dets;
    vector<bbox_info> keep;
    fstream in(filename);
    string line;
    while (getline(in, line)) {
        Mat img = imread(ImgRetrainPath + line + ".jpg");
        resize(img, img, Size(448, 448));
        assert(img.depth() == CV_8U || img.depth() == CV_8UC3);
        detections.clear();
        foundWeights.clear();
        dets.clear();
        keep.clear();
        my_hog.detectMultiScale(img, detections, foundWeights, 0, Size(8, 8), Size(), 1.1, 2., true);
        cout << "large scale: " << detections.size() << endl;
        for (size_t i = 0; i < detections.size(); i++) {
            //cout << foundWeights[i] << " ";
            if (foundWeights[i] > 1.1) {
                //rectangle(img, detections[i], Scalar(0, 255, 0), 2);
                bbox_info tmp_bbox(detections[i].x, detections[i].y, detections[i].br().x, detections[i].br().y,
                                   foundWeights[i]);
                dets.push_back(tmp_bbox);
            }
        }
        keep = nms(dets);
        for (size_t i = 0; i < keep.size(); i++) {
            Point p1(keep[i].xmin, keep[i].ymin), p2(keep[i].xmax, keep[i].ymax);
            Scalar color(0, 255, 0);
            rectangle(img, p1, p2, color, 2);
        }
        cout << endl;
        imshow("detect", img);
        waitKey(50);
    }
}

void mag_threshold(const Mat img, Mat &out, int sobel_kernel, int min_thres, int max_thres) {

    cvtColor(img, out, CV_BGR2GRAY);
    Sobel(out, out, CV_8UC1, 1, 0, sobel_kernel);
    //imshow("sobel y", out);
    normalize(out, out, 0, 255, NORM_MINMAX);
    threshold(out, out, min_thres, 0, THRESH_TOZERO);
    threshold(out, out, max_thres, 255, THRESH_BINARY);
    //imshow("thres", img);
    //waitKey(0);
}

void yellow_white_threshold(Mat origin, Mat &out1) {
    int y_lower[] = {10, 0, 100};
    int y_upper[] = {40, 255, 255};
    int w_lower[] = {0, 200, 0};
    int w_upper[] = {180, 255, 255};
    Mat HLS, y_mask, w_mask, mask;
    cvtColor(origin, HLS, CV_BGR2HLS);

    vector<int> yellow_lower(y_lower, y_lower + 3);
    vector<int> yellow_upper(y_upper, y_upper + 3);
    vector<int> white_lower(w_lower, w_lower + 3);
    vector<int> white_upper(w_upper, w_upper + 3);
    inRange(HLS, yellow_lower, yellow_upper, y_mask);
    inRange(HLS, white_lower, white_upper, w_mask);

    bitwise_or(y_mask, w_mask, mask);
    bitwise_and(origin, origin, out1, mask);
    cvtColor(out1, out1, CV_HLS2BGR);
    cvtColor(out1, out1, CV_BGR2GRAY);
    threshold(out1, out1, 130, 255, THRESH_BINARY);
}

void MultiScaleDetect(string model_hog, string model_cascade, string filename) {
    HOGDescriptor my_hog(Size(Window_y, Window_x), Size(block_y, block_x), Size(block_stride_y, block_stride_x),
                         Size(cell_y, cell_x), 9);
    CascadeClassifier car_classifier;
    car_classifier.load(model_cascade);
    //get support vector from model
    Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>(model_hog);
    Mat sv = model->getSupportVectors();
    vector<float> hog_detector;
    const int sv_total = sv.cols;
    Mat alpha, svidx;
    double rho = model->getDecisionFunction(0, alpha, svidx);
    Mat alpha2;
    alpha.convertTo(alpha2, CV_32FC1);
    Mat result(1, sv_total, CV_32FC1);
    result = alpha2 * sv;
    for (int i = 0; i < sv_total; ++i)
        hog_detector.push_back(-1 * result.at<float>(0, i));
    hog_detector.push_back((float) rho);
    //load vector to hog detector
    my_hog.setSVMDetector(hog_detector);

    vector<Rect> detections;
    vector<double> foundWeights;
    vector<int> rejLevel;
    vector<bbox_info> dets;
    vector<bbox_info> keep;
    fstream in(filename);
    string line;
    while (getline(in, line)) {
        Mat img = imread(ImgRetrainPath + line + ".jpg");
        resize(img, img, Size(448, 448));

        detections.clear();
        foundWeights.clear();
        rejLevel.clear();
        dets.clear();
        keep.clear();

        my_hog.detectMultiScale(img, detections, foundWeights, 0, Size(8, 8), Size(), 1.1, 2., true);
        cout << "hog detect object: " << detections.size() << endl;
        for (size_t i = 0; i < detections.size(); i++) {
            if (foundWeights[i] > 1.3) {
                bbox_info tmp_bbox(detections[i].x, detections[i].y, detections[i].br().x, detections[i].br().y,
                                   foundWeights[i]);
                dets.push_back(tmp_bbox);
            }
        }
        detections.clear();
        foundWeights.clear();

        car_classifier.detectMultiScale(img, detections, rejLevel, foundWeights, 1.1, 3, 0, Size(), Size(), true);
        cout << "cascade detect object: " << detections.size() << endl;
        for (int i = 0; i < detections.size(); i++) {
            if (rejLevel[i] < 20 || foundWeights[i] < 1.)
                continue;
            bbox_info tmp(detections[i].x, detections[i].y, detections[i].br().x, detections[i].br().y,
                          foundWeights[i]);
            dets.push_back(tmp);
        }

        keep = nms(dets);
        for (size_t i = 0; i < keep.size(); i++) {
            Point p1(keep[i].xmin, keep[i].ymin), p2(keep[i].xmax, keep[i].ymax);
            Scalar color(0, 255, 0);
            rectangle(img, p1, p2, color, 2);
        }
        cout << "final object: " << keep.size() << endl;
        imshow("detect", img);
        waitKey(24);
    }

}

void LineDetect2(Mat& img_o, int dataset = 1) {
    /*
     * first: doing perspective
     */
    Mat trans, inverse, img;;
    if (dataset == 1) {
        Point2f origin[] = {Point2f(204, 286), Point2f(71, 448), Point2f(394, 448), Point2f(243, 286)};
        Point2f dst[] = {Point2f(112, 0), Point2f(112, 448), Point2f(336, 448), Point2f(336, 0)};
        trans = getPerspectiveTransform(origin, dst);
    } else {
        Point2f origin[] = {Point2f(249, 255), Point2f(191, 251), Point2f(393, 448), Point2f(20, 448)};
        Point2f dst[] = {Point2f(112, 0), Point2f(336, 0), Point2f(112, 448), Point2f(336, 448)};
        trans = getPerspectiveTransform(origin, dst);
    }

    invert(trans, inverse);
    warpPerspective(img_o ,img, trans, img.size());
    //imshow("perspective", img);

    /*
     * second: doing threshold filtering
     */
    Mat out1, out2;
    mag_threshold(img, out1, 5, 30, 150);
    yellow_white_threshold(img, out2);
    //imshow("out2", out2);
    out1 = out1 + out2;

    /*
     * third: using hough transform to get some lines and extract basic right lines
     */
    vector<Vec4i> lines;
    vector<Point2f> leftlines;
    vector<Point2f> rightlines;
    HoughLinesP(out1, lines, 1, CV_PI / 180, 50, 30, 10);
    cout << lines.size() << endl;
    for (size_t i = 0; i < lines.size(); i++) {
        //abandon horizontal line.
        if (lines[i][1] == lines[i][3])
            continue;
        //get left lines
        if (lines[i][0] <= 224 && lines[i][2] <=224){
            float k = 1.5;
            //if not verticle line
            if (lines[i][0] != lines[i][2])
                k = fabs(float(lines[i][3]-lines[i][1])/float(lines[i][2]-lines[i][0]));
            if (k>=1.5) {
                leftlines.push_back(Point2f(lines[i][0], lines[i][1]));
                leftlines.push_back(Point2f(lines[i][2], lines[i][3]));
            }

        }
        //get right lines
        if (lines[i][0] >= 224 && lines[i][2] >=224){
            float k = 1.5;
            //if not verticle line
            if (lines[i][0] != lines[i][2])
                k = fabs(float(lines[i][3]-lines[i][1])/float(lines[i][2]-lines[i][0]));
            if (k>=1.5) {
                rightlines.push_back(Point2f(lines[i][0], lines[i][1]));
                rightlines.push_back(Point2f(lines[i][2], lines[i][3]));
            }
        }
    }

    /*
     * fourth: doing linear regression
     */
    cout << "left point: " << leftlines.size() << " right point: " << rightlines.size() << endl;
    if (leftlines.empty()){
        leftlines.push_back(Point2f(120.0, 0.0));
        leftlines.push_back(Point2f(110.0, 448.0));
    }
    if (rightlines.empty()){
        rightlines.push_back(Point2f(336.0, 0.0));
        rightlines.push_back(Point2f(346.0, 448.0));
    }
    Vec4f line_left, line_right;
    fitLine(leftlines, line_left, DIST_L1, 0, 0.01, 0.01);
    fitLine(rightlines, line_right, DIST_L1, 0, 0.01, 0.01);
    float k_left = line_left[1]/line_left[0];
    float k_right = line_right[1]/line_right[0];
    float x_ll = line_left[2]-line_left[3]/k_left;
    float x_lh= line_left[2]+(448-line_left[3])/k_left;
    float x_rl = line_right[2]-line_right[3]/k_right;
    float x_rh = line_right[2]+(448-line_right[3])/k_right;

    //line(img, Point(x_ll, 0), Point(x_lh, 448), Scalar(112, 25, 25), 6, 8);
    //line(img, Point(x_rl, 0), Point(x_rh, 448), Scalar(112, 25, 25), 6, 8);
    Point pts[4] = {Point(x_lh, 448), Point(x_ll, 0), Point(x_rl, 0), Point(x_rh, 448)};
    vector<Point> p(pts, pts+4);
    vector<vector<Point>> content;
    content.push_back(p);
    fillPoly(img, content, Scalar(112, 25, 25));
    /*
     * fifth: transform back and add to original img
     */
    warpPerspective(img, img, inverse, img.size());
    addWeighted(img_o, 1, img, 0.3, 0, img_o);
    //imshow("img", img_o);
    //waitKey(50);
}

void LineDetect(string filename) {
    VideoCapture cap;
    cap.open(filename);
    while (true) {
        Mat img;
        cap >> img;
        if (!img.data)
            break;
        resize(img, img, Size(448, 448));
        LineDetect2(img);
        imshow("detect lines", img);
        waitKey(24);
    }
}

void FinalDetect(string filename, string model_cascade, string model_hog, int dataset = 1, bool IsLine = false) {
    setUseOptimized(true);
    setNumThreads(8);

    HOGDescriptor my_hog(Size(Window_y, Window_x), Size(block_y, block_x), Size(block_stride_y, block_stride_x),
                         Size(cell_y, cell_x), 9);
    CascadeClassifier car_classifier;
    car_classifier.load(model_cascade);
    //get support vector from model
    Ptr<ml::SVM> model = ml::StatModel::load<ml::SVM>(model_hog);
    Mat sv = model->getSupportVectors();
    vector<float> hog_detector;
    const int sv_total = sv.cols;
    Mat alpha, svidx;
    double rho = model->getDecisionFunction(0, alpha, svidx);
    Mat alpha2;
    alpha.convertTo(alpha2, CV_32FC1);
    Mat result(1, sv_total, CV_32FC1);
    result = alpha2 * sv;
    for (int i = 0; i < sv_total; ++i)
        hog_detector.push_back(-1 * result.at<float>(0, i));
    hog_detector.push_back((float) rho);
    //load vector to hog detector
    my_hog.setSVMDetector(hog_detector);

    vector<Rect> detections;
    vector<double> foundWeights;
    vector<int> rejLevel;
    vector<bbox_info> dets;
    vector<bbox_info> keep;
    VideoCapture cap;
    cap.open(filename);

    int num = 0;
    while (true) {
        Mat img;
        cap >> img;
        num++;
        if (!img.data)
            break;
        resize(img, img, Size(448, 448));
        cout << img.size() << endl;
        if (IsLine)
            LineDetect2(img, dataset);

        detections.clear();
        foundWeights.clear();
        rejLevel.clear();
        dets.clear();
        keep.clear();

        my_hog.detectMultiScale(img, detections, foundWeights, 0, Size(8, 8), Size(), 1.1, 2., true);
        cout << "hog detect object: " << detections.size() << endl;
        for (size_t i = 0; i < detections.size(); i++) {
            if (foundWeights[i] > 1.3) {
                bbox_info tmp_bbox(detections[i].x, detections[i].y, detections[i].br().x, detections[i].br().y,
                                   foundWeights[i]);
                dets.push_back(tmp_bbox);
            }
        }

        car_classifier.detectMultiScale(img, detections, rejLevel, foundWeights, 1.1, 3, 0, Size(), Size(), true);
        cout << "cascade detect object: " << detections.size() << endl;
        for (int i = 0; i < detections.size(); i++) {
            if (rejLevel[i] < 20 || foundWeights[i] < 1.)
                continue;
            bbox_info tmp(detections[i].x, detections[i].y, detections[i].br().x, detections[i].br().y,
                          foundWeights[i]);
            dets.push_back(tmp);
        }

        keep = nms(dets);
        for (size_t i = 0; i < keep.size(); i++) {
            Point p1(keep[i].xmin, keep[i].ymin), p2(keep[i].xmax, keep[i].ymax);
            Scalar color(0, 255, 0);
            rectangle(img, p1, p2, color, 2);
        }
        imshow("detect", img);
        string name = "/home/oliver/data/tmp/IMG_" + to_string(num) + ".jpg";
        imwrite(name, img);
        waitKey(5);

    }
}