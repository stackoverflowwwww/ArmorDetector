#ifndef ARMORDETECTOR_HPP
#define ARMORDETECTOR_HPP

#include <bits/stdc++.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
#define POINT_DIST(p1,p2) std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y))

//#define setImage_debug
//#define findTargetInContours_debug
//#define chooseTarget_debug
namespace armor {
enum EnermyColor{RED,BLUE};
// 匹配灯条的结构体
struct matched_rect{
    cv::RotatedRect rect;
    //下面两个用来计算候选装甲板的分数
    float lr_rate;
    float angle_abs;
    Point2f left_top_point;
    Point2f left_bottom_point;
    Point2f right_top_point;
    Point2f right_bottom_point;
};
void drawRotateRect(Mat &img,RotatedRect &r_rect);
}
namespace  {
struct ArmorParam{
    armor::EnermyColor enermy_color=armor::BLUE;
    int blue_sentry_gray_thres=60;
    int blue_sentry_color_thres=46;
    double light_left_angle=45.0;// \这中情形
    double light_right_angle=60.0;// /这种情形
    int min_light_height=8;
    double min_h_w_rate=1.1;
    double max_h_w_rate=15;
    double min_light_delta_h=12;
    double max_light_delta_h=450;
    double max_light_delta_angle=25;
    double max_lr_rate = 1.5;
    double max_wh_ratio = 4.0;
    double min_wh_ratio = 1.25;
    double small_big_armor_heght_threshold=113;
    double big_armor_max_delta_v=166;
    double big_armor_max_delta_v_rate=1.66;
    double small_armor_max_delta_v=50;
    double small_armor_max_delta_v_rate=1.2;
    double max_angle_abs=12;
};
struct CandidateTarget{
    int armor_height;
    float armor_angle;
    int index;
    bool is_small_armor;
    float bar_lr_rate;
    float bar_angle_abs;
};

Mat element=getStructuringElement(MORPH_ELLIPSE,cv::Size(7,7));
Mat element2=getStructuringElement(MORPH_ELLIPSE,cv::Size(3,3));
auto ptAngle=[](const Point2f &p1,const Point2f &p2){
    return std::atan2(p2.y-p1.y,p2.x-p1.x)*180.0/CV_PI;
};
}
class ArmorDetector
{
public:
    ArmorDetector();
    void set_enermy_color(armor::EnermyColor enermy_color);
private:
    struct SetImageParam{
        float roi_scale_w=1.6;
        float roi_scale_h=1.6;
    } _set_image_param;
public:
    void debugTest(Mat &src);
    void setImage(const Mat &src);
    void findTargetInContours(vector<armor::matched_rect> &match_rects);
private:
    struct chooseTargetParam{
        float armor_angle_threshold=26;
        float brightness_threshold=66.66;
        float best_armor_height_rate=1.1;
        float small_wh_rate_max=3.6;
        float enhance_digit_scale=8;
        float blob_digit_scale=0.00390625f;
    } _choose_target_param;
public:
    void chooseTarget(const vector<armor::matched_rect> &match_rects,RotatedRect &target_armor,Point2f &p1_,Point2f &p2_,Point2f &p3_,Point2f &p4_);
private:
    vector<armor::matched_rect> match_rrects_;
public:
    void getTarget(const Mat &src,RotatedRect &target_armor,Point2f &p1,Point2f &p2,Point2f &p3,Point2f &p4);
private:
    void boundingRRect(const RotatedRect &left,const RotatedRect &right,RotatedRect &bound_rrect);
    bool makeRectSafe(Rect &roi_rect,const cv::Size size_);
private:
    ArmorParam _param;
    Mat _src;
    Mat _max_color;
    bool _is_lost=false;
    bool _is_small_armor=false;
    dnn::Net _net;
    Rect _dect_rect;
    RotatedRect _target_armor;
    int _lost_cnt=0;//连续丢失目标的次数
};

#endif // ARMORDETECTOR_HPP
