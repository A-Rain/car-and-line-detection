#include "GetFeature.h"

Mat GetSIFTfeature(string imgname){
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    Mat src = imread(imgname, 0);
    resize(src, src, Size(448, 448));
    //store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //store the SIFT descriptor of img
    Mat featureDescriptor;
    //detect feature points
    f2d->detect(src, keypoints);
    //compute the descriptor for each keypoint
    f2d->compute(src, keypoints, featureDescriptor);
    return featureDescriptor;
}

Mat GetAllImgSIFTfeature(string Trainlist, string img_dir_path_preced){
    fstream in(Trainlist);
    string line, imgname;
    Mat AllImgDescriptor;
    while (getline(in, line)) {
        vector<string> fields;
        boost::split(fields, line, boost::is_any_of(" "));
        imgname = fields[0];
        Mat descriptor = GetSIFTfeature(img_dir_path_preced + imgname);
        AllImgDescriptor.push_back(descriptor);
    }
    return AllImgDescriptor;
}

Mat GetHOGfeature(string imgname){
    Mat img = imread(imgname);
    resize(img, img, Size(Imgheight, Imgwidth));
    Ptr<HOGDescriptor> hog = new HOGDescriptor(Size(Window_y, Window_x),
                                               Size(block_y, block_x),
                                               Size(block_stride_y, block_stride_x),
                                               Size(cell_y, cell_x), 9);
    assert(hog->getDescriptorSize() == dimension);
    vector<float> descriptor;
    hog->compute(img, descriptor, Size(Window_stride_y, Window_stride_x), Size(0, 0));
    assert(descriptor.size() == dimension);
    Mat s(descriptor);
    transpose(s, s);
    //cout << s.size() << endl;
    return s;
}

Mat GetHOGfeatureFromMat(Mat img){
    resize(img, img, Size(Imgheight, Imgwidth));
    Ptr<HOGDescriptor> hog = new HOGDescriptor(Size(Window_y, Window_x),
                                               Size(block_y, block_x),
                                               Size(block_stride_y, block_stride_x),
                                               Size(cell_y, cell_x), 9);
    //cout << "hog dimension: " << hog->getDescriptorSize() << endl;
    vector<float> descriptor;
    hog->compute(img, descriptor, Size(Window_stride_y, Window_stride_x), Size(0, 0));
    //cout << "descriptor size: " << descriptor.size() << endl;
    Mat s(descriptor);
    transpose(s, s);
    return s;
}

void GetAllImgHOGfeature(Mat& AllImgDescriptor, Mat& label, string Trainlist, string img_dir_path_preced){
    cout << "training txt: "<<Trainlist << endl;
    cout << "training path: " << img_dir_path_preced << endl;
    fstream in(Trainlist);
    string line, imgname;
    //Mat AllImgDescriptor;
    int i = 0;
    while (getline(in, line)) {
        vector<string> fields;
        boost::split(fields, line, boost::is_any_of(" "));
        imgname = fields[0];
        //cout << imgname << endl;
        Mat descriptor = GetHOGfeature(img_dir_path_preced + imgname);
        AllImgDescriptor.push_back(descriptor);
        label.push_back(stringToNum(fields[1]));
        if (i%1000 == 0)
            cout << i << endl;
        i++;
    }
    //return AllImgDescriptor;
}