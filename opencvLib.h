#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)
#pragma warning(disable: 4819)

#include <opencv2/opencv.hpp>
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif


#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)

#if CV_MAJOR_VERSION == 3
#pragma comment(lib, cvLIB("world"))
#else
#pragma comment(lib, cvLIB("core"))
#pragma comment(lib, cvLIB("imgproc"))
#pragma comment(lib, cvLIB("highgui"))
#pragma comment(lib, cvLIB("contrib"))
#endif

using namespace std;
using namespace cv;
