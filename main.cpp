#include <bits/stdc++.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ArmorDetector.hpp"
using namespace cv;
using namespace std;
int main()
{
    ArmorDetector armor_detector;
    Mat frame;
    string video_filename="/home/wenmingbang_2019152030/dataset/2019-5-16-13-27-4.avi";
    VideoCapture cap(video_filename);
    int i=1;
    bool stop=true;
    RotatedRect target_armor;
    Point2f p1,p2,p3,p4;
    while (true) {
        cap.read(frame);
        if(frame.empty()){
            cout<<"video ended"<<endl;
            break;
        }
        cout<<"frame: "<<i++<<endl;
//        armor_detector.setImage(frame);
//        vector<armor::matched_rect> match_rects;
//        armor_detector.findTargetInContours(match_rects);
//        armor_detector.chooseTarget(match_rects,target_armor);
//        armor::drawRotateRect(frame,target_armor);

        armor_detector.getTarget(frame,target_armor,p1,p2,p3,p4);
        armor::drawRotateRect(frame,target_armor);
        armor_detector.debugTest(frame);
        if(target_armor.size.width!=0&&target_armor.size.height!=0){
            circle(frame,p1,5,Scalar(0,0,255),2);
            circle(frame,p2,5,Scalar(0,0,255),2);
            circle(frame,p3,5,Scalar(0,0,255),2);
            circle(frame,p4,5,Scalar(0,0,255),2);
        }
        imshow("video",frame);
        int key;
        if(stop){
            key=waitKey(0);
        }else{
            key=waitKey(14);
        }
        switch (key) {
            case ' ':
                stop=!stop;
                break;
            case 'n':
                stop=true;
            break;
        }
    }
}
