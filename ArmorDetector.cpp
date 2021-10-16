#include "ArmorDetector.hpp"


//#define setImage_debug
//#define findTargetInContours_debug
//#define chooseTarget_debug
void armor::drawRotateRect(Mat &img, RotatedRect &r_rect){
    Point2f pt[4];
    r_rect.points(pt);
    for(int i=0;i<4;i++){
        line(img,pt[i],pt[(i+1)%4],Scalar(0,255,0));
    }
}

ArmorDetector::ArmorDetector(){
    string proto_file="/home/wenmingbang_2019152030/opencv/robotmaster/armorDetector/extraFile/caffemodel/lenet_train_test_deploy.prototxt";
    string caffe_model_file="/home/wenmingbang_2019152030/opencv/robotmaster/armorDetector/extraFile/caffemodel/lenet_iter_200000.caffemodel";
    _net=dnn::readNetFromCaffe(proto_file,caffe_model_file);
}

void ArmorDetector::set_enermy_color(armor::EnermyColor enermy_color){
    _param.enermy_color=enermy_color;
}

void ArmorDetector::debugTest(Mat &src){
    rectangle(src,_dect_rect,Scalar(128,128,0),2);
}

void ArmorDetector::setImage(const Mat &src){
    if(_target_armor.size.width==0 || _target_armor.size.height==0){
        _src=src;
        _dect_rect=Rect(Point(0,0),cv::Size(src.cols,src.rows));
    }else{
        Rect rect=_target_armor.boundingRect();
        // 截图的区域大小
        double scale_w =_set_image_param.roi_scale_w;
        double scale_h = _set_image_param.roi_scale_h;

        int w = int(rect.width * scale_w);
        int h = int(rect.height * scale_h);
        Point center = _target_armor.center;
        int x = std::max(center.x - w, 0);
        int y = std::max(center.y - h, 0);
        Point lu = Point(x, y); /* point left up */
        x = std::min(center.x + w, src.cols);
        y = std::min(center.y + h, src.rows);
        Point rd = Point(x, y); /* point right down */
        // 构造出矩形找到了搜索的ROI区域
        _dect_rect = Rect(lu, rd);
        if(makeRectSafe(_dect_rect,src.size())){
            _src=src(_dect_rect);
        }else{
            _src=src;
        }
    }
//#ifdef setImage_debug
//    Mat tmp_src;

//    src.copyTo(tmp_src);
//    //            tmp_src=src;

//    rectangle(tmp_src,_dect_rect,Scalar(128,128,0),2);
//    imshow("roi_rect",tmp_src);
//#endif
    if(_param.enermy_color==armor::RED){
        Mat thres_whole;
        vector<Mat> splited;
        split(_src, splited);
        cvtColor(_src, thres_whole, CV_BGR2GRAY);
        threshold(thres_whole, thres_whole, _param.red_sentry_gray_thres, 255, THRESH_BINARY); //gray threshold
//#ifdef setImage_debug
//        imshow("thres_whole",thres_whole);
//#endif
        subtract(splited[2],splited[0],_max_color);
        threshold(_max_color,_max_color,_param.red_sentry_color_thres,255,THRESH_BINARY);
        dilate(_max_color,_max_color,element);
        _max_color=_max_color&thres_whole;
        dilate(_max_color, _max_color, element2);
#ifdef setImage_debug
        imshow("_max_color",_max_color);
#endif

    }else{
        Mat thres_whole;
        vector<Mat> splited;
        split(_src, splited);
        cvtColor(_src, thres_whole, CV_BGR2GRAY);
        threshold(thres_whole, thres_whole, _param.blue_sentry_gray_thres, 255, THRESH_BINARY); //gray threshold
#ifdef setImage_debug
        imshow("thres_whole",thres_whole);
#endif
        subtract(splited[0],splited[2],_max_color);
        threshold(_max_color,_max_color,_param.blue_sentry_color_thres,255,THRESH_BINARY);
        dilate(_max_color,_max_color,element);
        _max_color=_max_color&thres_whole;
        dilate(_max_color, _max_color, element2);
#ifdef setImage_debug
        imshow("_max_color",_max_color);
#endif
    }
}

void ArmorDetector::findTargetInContours(vector<armor::matched_rect> &match_rects){
    match_rects.clear();
    vector<vector<Point2i>> contours_max;
    vector<Vec4i> hierarchy;
    findContours(_max_color, contours_max, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    vector<RotatedRect> first_result_r_rect;
    //灯条筛选
    //竖直程度
    //高度（设置下限）//todo：上限？
    //宽高比

    //for debug
#ifdef findTargetInContours_debug
    Mat filter_rrect;
    _src.copyTo(filter_rrect);
#endif
    for(int i=0;i<contours_max.size();i++){
        RotatedRect rotate_rect=minAreaRect(contours_max[i]);
        double max_r_rect_len=max(rotate_rect.size.height,rotate_rect.size.width);
        double min_r_rect_len=min(rotate_rect.size.height,rotate_rect.size.width);
        bool if1=fabs(rotate_rect.angle)<_param.light_left_angle&&(rotate_rect.size.height>rotate_rect.size.width);
        bool if2=fabs(rotate_rect.angle)>_param.light_right_angle&&(rotate_rect.size.height<rotate_rect.size.width);
        bool if3=max_r_rect_len>_param.min_light_height;
        bool if4=(max_r_rect_len/min_r_rect_len>=_param.min_h_w_rate)&&(max_r_rect_len/min_r_rect_len<_param.max_h_w_rate);
        if((if1||if2)&&if3&&if4){
            first_result_r_rect.push_back(rotate_rect);

#ifdef findTargetInContours_debug
            armor::drawRotateRect(filter_rrect,rotate_rect);//for debug
#endif
        }
    }
#ifdef findTargetInContours_debug
    imshow("filtered: rotate_rect",filter_rrect);
#endif
    if(first_result_r_rect.size()<2){
        return;
    }
    sort(first_result_r_rect.begin(),first_result_r_rect.end(),[](RotatedRect &r1,RotatedRect &r2){
        return r1.center.x<=r2.center.x;
    });
    //装甲板筛选条件：
    //水平间距
    //竖直间距（用相对值，可能是因为有大小装甲板）
    //角度差
    //左右高度比值(也是因为大小装甲板)

    //debug
#ifdef findTargetInContours_debug
    Mat filter_armor_rrect;
    _src.copyTo(filter_armor_rrect);
#endif

    RotatedRect armor_rrect;
    Point2f p1,p2,p3,p4;
    Point2f pt[4];
    for(int i=0;i<first_result_r_rect.size()-1;i++){
        const RotatedRect& r_rect_i=first_result_r_rect[i];
        double xi=r_rect_i.center.x;
        double yi=r_rect_i.center.y;
        double leni=max(r_rect_i.size.height,r_rect_i.size.width);
        double angle_i=fabs(r_rect_i.angle);
        float angle_abs;
        r_rect_i.points(pt);
        if(angle_i<_param.light_left_angle){
            p1=(pt[0]+pt[3])/2;
            p2=(pt[1]+pt[2])/2;
        }else{
            p1=(pt[0]+pt[1])/2;
            p2=(pt[2]+pt[3])/2;
        }
        for(int j=i+1;j<first_result_r_rect.size();j++){
            const RotatedRect& r_rect_j=first_result_r_rect[j];
            double xj=r_rect_j.center.x;
            double yj=r_rect_j.center.y;
            double lenj=max(r_rect_j.size.height,r_rect_j.size.width);
            double angle_j=fabs(r_rect_j.angle);
            double delta_h=xj-xi;
            double delta_v=fabs(yi-yj);
            if (angle_i > _param.light_left_angle && angle_j < _param.light_left_angle)
            { // 八字 / \   //
                angle_abs = 90.0 - angle_i + angle_j;
            }
            else if (angle_i <= _param.light_left_angle && angle_j >= _param.light_left_angle)
            { // 倒八字 \ /
                angle_abs = 90.0 - angle_j + angle_i;
            }
            else
            {
                if (angle_i > angle_j)
                    angle_abs = angle_i - angle_j; // 在同一边
                else
                    angle_abs = angle_j - angle_i;
            }
            bool if1=delta_h>_param.min_light_delta_h&&delta_h<_param.max_light_delta_h;
            double max_len_ij=max(leni,lenj);
            bool if2;
            if(max_len_ij>=_param.small_big_armor_heght_threshold){
                if2=delta_v<_param.big_armor_max_delta_v_rate*max_len_ij && delta_v<_param.big_armor_max_delta_v;
            }else{
                if2=delta_v<_param.small_armor_max_delta_v_rate*max_len_ij && delta_v<_param.small_armor_max_delta_v;
            }
            float lr_height_rate=leni>lenj?leni/lenj:lenj/leni;
            bool if3=lr_height_rate<_param.max_lr_rate;
            bool if4=angle_abs<_param.max_angle_abs;
            if(if1&&if2&&if3&&if4){
                boundingRRect(r_rect_i,r_rect_j,armor_rrect);
                double wh_rate=armor_rrect.size.width/armor_rrect.size.height;
                if(wh_rate>_param.max_wh_ratio||wh_rate<_param.min_wh_ratio){ //区域的宽高比
                    continue;
                }
                r_rect_j.points(pt);
                if(angle_j<_param.light_left_angle){
                    p3=(pt[0]+pt[3])/2;
                    p4=(pt[1]+pt[2])/2;
                }else{
                    p3=(pt[0]+pt[1])/2;
                    p4=(pt[2]+pt[3])/2;
                }
                match_rects.push_back({armor_rrect,lr_height_rate,angle_abs,p1,p2,p3,p4});
                //到这一步误识别还是挺高的，把两个装甲板相邻的灯条的区域都识别了
#ifdef findTargetInContours_debug
                armor::drawRotateRect(filter_armor_rrect,armor_rrect);//debug
#endif
            }
        }
    }
#ifdef findTargetInContours_debug
    Mat tmp;
    _src.copyTo(tmp);
    circle(tmp,p1,5,Scalar(0,0,255),2);
    circle(tmp,p2,5,Scalar(0,0,255),2);
    circle(tmp,p3,5,Scalar(0,0,255),2);
    circle(tmp,p4,5,Scalar(0,0,255),2);
    imshow("1.jpg",tmp);
#endif
#ifdef findTargetInContours_debug
    imshow("armor_rrect",filter_armor_rrect);
#endif
}

void ArmorDetector::chooseTarget(const vector<armor::matched_rect> &match_rects, RotatedRect &target_armor, Point2f &p1_, Point2f &p2_, Point2f &p3_, Point2f &p4_){
    // 如果没有两条矩形围成一个目标装甲板就返回一个空的旋转矩形
    if (match_rects.size() < 1)
    {
        _is_lost = true;
        target_armor=RotatedRect();
    }
    //选择最终打击目标
    //根据rrect区域的亮度进行第一步筛选
    //识别数字，进行筛选
    //利用高度决定远近，近的优先，然后是rrect的倾斜度，接着是左右灯条比例
    //todo: 考虑利用数字识别概率计算最终权重
    bool is_small=false;
    int ret_idx=-1;
    Point p1,p2;
    Mat roi;
    Mat mean,stdDev;
    RotatedRect screen_rrect;
    Rect roi_rect;
    Mat input_sample;
    vector<CandidateTarget> candidates;

    //一些基本筛选
    for(int i=0;i<match_rects.size();i++){
        const RotatedRect &rrect=match_rects[i].rect;
        float cur_angle=fabs(rrect.angle);
        //            float cur_angle = match_rects[i].rect.size.width > match_rects[i].rect.size.height ? abs(match_rects[i].rect.angle) : 90 - abs(match_rects[i].rect.angle);
        //            cout<<cur_angle1<<" "<<cur_angle<<endl;
        if(cur_angle>_choose_target_param.armor_angle_threshold){
            continue;
        }
        double w=rrect.size.width;
        double h=rrect.size.height;
        double wh_rate=w/h;
        screen_rrect=RotatedRect(rrect.center,cv::Size(0.88*w,h),rrect.angle); //因为后面要把旋转矩形变成平行矩形，因此宽度要小点，避免涉及到灯条
        if(wh_rate<_choose_target_param.small_wh_rate_max){
            is_small=true;
        }else{
            is_small=false;
        }
        int x,y;
        x=screen_rrect.center.x-screen_rrect.size.width/2;
        y=screen_rrect.center.y-screen_rrect.size.height/2;
        p1=Point(x,y);
        x=screen_rrect.center.x+screen_rrect.size.width/2;
        y=screen_rrect.center.y+screen_rrect.size.height/2;
        p2=Point(x,y);
        roi_rect=Rect(p1,p2);
        cv::Size src_size=cv::Size(_src.cols,_src.rows);
        if(makeRectSafe(roi_rect,src_size)){
            _src(roi_rect).copyTo(roi);
            meanStdDev(roi,mean,stdDev);
            double avg=mean.ptr<double>(0)[0];
            if(avg>_choose_target_param.brightness_threshold){
                continue;
            }
        }else{
            continue;
        }
        screen_rrect=RotatedRect(rrect.center,cv::Size(0.9*w,2.4*h),rrect.angle); //要做数字识别，因此要调大点
        x=screen_rrect.center.x-screen_rrect.size.width/2;
        y=screen_rrect.center.y-screen_rrect.size.height/2;
        p1=Point(x,y);
        x=screen_rrect.center.x+screen_rrect.size.width/2;
        y=screen_rrect.center.y+screen_rrect.size.height/2;
        p2=Point(x,y);
        roi_rect=Rect(p1,p2);
        if(makeRectSafe(roi_rect,src_size)){
            _src(roi_rect).copyTo(input_sample);
        }else{
            continue;
        }
        resize(input_sample,input_sample,cv::Size(28,28));
        convertScaleAbs(input_sample,input_sample,_choose_target_param.enhance_digit_scale);
#ifdef chooseTarget_debug
        imshow("input_sample",input_sample);
#endif
        cvtColor(input_sample,input_sample,CV_BGR2GRAY);
        Mat input_blob=dnn::blobFromImage(input_sample,_choose_target_param.blob_digit_scale); //返回4通道数据，第一个通道维度为1，代表数据数量
        _net.setInput(input_blob,"data");
        Mat prob=_net.forward("loss");
        Mat probMat = prob.reshape(1, 1);

        //            cout<<"prob:"<<endl;
        //            cout<<probMat.size()<<endl;
        //            cout<<probMat<<endl;
        Point pos;
        double classProb;
        minMaxLoc(probMat,nullptr,&classProb,nullptr,&pos);
        int classId=pos.x;
        if(classId==0){
            continue;
        }
        int cur_height=min(w,h);
        candidates.push_back({cur_height,cur_angle,i,is_small,match_rects[i].lr_rate,match_rects[i].angle_abs});
        ret_idx=1;
    }
    //再次筛选
    int final_index=0;
    if(candidates.size()>1){
        sort(candidates.begin(),candidates.end(),[](const CandidateTarget &t1,const CandidateTarget &t2){
            return t1.armor_height>=t2.armor_height;
        });
        float tmp_armor_angle=candidates[0].armor_angle;
        float tmp_bar_angle_abs=candidates[0].bar_angle_abs;
        float tmp_bar_lr_rate=candidates[0].bar_lr_rate;
        for(int i=1;i<candidates.size();i++){
            if(candidates[0].armor_height/candidates[i].armor_height<_choose_target_param.best_armor_height_rate){
                if(candidates[i].armor_angle<tmp_armor_angle){
                    tmp_armor_angle=candidates[i].armor_angle;
                    if(candidates[i].bar_lr_rate<=tmp_bar_lr_rate){
                        tmp_bar_lr_rate=candidates[i].bar_lr_rate;
                        if(candidates[i].bar_angle_abs<=tmp_bar_angle_abs){
                            tmp_bar_angle_abs=candidates[i].bar_angle_abs;
                            final_index=i;
                        }
                    }
                }
            }else{
                break;
            }
        }
    }
    if(ret_idx==-1){
        _is_lost=true;
        target_armor=RotatedRect();
        return;
    }
    _is_lost=false;
    _is_small_armor=candidates[final_index].is_small_armor;
    int index=candidates[final_index].index;
    target_armor=match_rects[index].rect;
    Point2f b=Point2f(_dect_rect.tl());
    p1_=match_rects[index].left_top_point+b;
    p2_=match_rects[index].left_bottom_point+b;
    p3_=match_rects[index].right_top_point+b;
    p4_=match_rects[index].right_bottom_point+b;
}

void ArmorDetector::getTarget(const Mat &src, RotatedRect &target_armor, Point2f &p1, Point2f &p2, Point2f &p3, Point2f &p4){
    setImage(src);
    findTargetInContours(match_rrects_);
    chooseTarget(match_rrects_,target_armor,p1,p2,p3,p4);
    if(target_armor.size.width!=0){
        target_armor.center.x+=_dect_rect.x;
        target_armor.center.y+=_dect_rect.y;
        _target_armor=target_armor;
        _lost_cnt=0;
    }else{
        _lost_cnt++;
        _target_armor=RotatedRect();
    }
}

void ArmorDetector::boundingRRect(const RotatedRect &left, const RotatedRect &right, RotatedRect &bound_rrect){
    Point2f center=(left.center+right.center)/2;
    double angle=ptAngle(left.center,right.center);
    double left_w=min(left.size.width,left.size.height);
    double left_h=max(left.size.width,left.size.height);
    double right_w=min(right.size.width,right.size.height);
    double right_h=max(right.size.width,right.size.height);
    double width=POINT_DIST(left.center,right.center)-(left_w+right_w)/2;
    double height=max(left_h,right_h);
    bound_rrect.size=cv::Size(width,height);
    bound_rrect.center=center;
    bound_rrect.angle=angle;
}

bool ArmorDetector::makeRectSafe(Rect &roi_rect, const Size size_){
    if(roi_rect.x<0){
        roi_rect.x=0;
    }
    if(roi_rect.y<0){
        roi_rect.y=0;
    }
    if(roi_rect.x+roi_rect.width>size_.width){
        roi_rect.width=size_.width-roi_rect.x;
    }
    if(roi_rect.y+roi_rect.height>size_.height){
        roi_rect.height=size_.height-roi_rect.y;
    }
    if(roi_rect.width<=0||roi_rect.height<=0){
        return false;
    }
    return true;
}
