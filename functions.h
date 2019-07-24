#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include<stdlib.h>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <fstream>
#include <Eigen/StdVector>

#include <unordered_set>
#include <stdint.h>
#include <algorithm>
#include <iterator> 
#include <vector>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class tracking_store
{
    //map for 3d points projections
    unordered_map<int,vector<Point2f>> pt_2d;

    //map for image index of 3d points projections;
    unordered_map<int,vector<int>> img_index;

    //map for match index
    unordered_map<int,vector<int>> match_index;

    //map for initialization of 3d_points
    unordered_map<int,Eigen::Vector3d> init_guess;

    //map for correspondences between vertex identificator and point identificator
    unordered_map<int,int> map_to_custom_struct;

    public:

    //function to add the projections of the new 3d points for two consecutive frames
    void add_new_points_proyections(vector<KeyPoint> _features1,vector<KeyPoint> _features2,vector<int> _left_matches,vector<int> _right_matches,int &_point_identifier,int _frame,Mat used_features)
    {
        for (int i=0;i<_left_matches.size();i++)
        {
            if(used_features.at<double>(i)==0)
            {
                pt_2d[_point_identifier]=vector<Point2f>();
                pt_2d[_point_identifier].push_back(_features1[_left_matches[i]].pt);
                img_index[_point_identifier]=vector<int>();
                img_index[_point_identifier].push_back(_frame-1);
                match_index[_point_identifier]=vector<int>();
                match_index[_point_identifier].push_back(_left_matches[i]);
                pt_2d[_point_identifier].push_back(_features2[_right_matches[i]].pt);
                img_index[_point_identifier].push_back(_frame);
                match_index[_point_identifier].push_back(_right_matches[i]);
                _point_identifier++;
            }
        }
    }

    //function to add a new projection for an existent point
    void add_new_projection_for_existent_point(int _point_ident,int _frame,vector<KeyPoint> _features1,vector<KeyPoint> _features2,vector<int> _left_matches,vector<int> _right_matches,Mat &used_points)
    {
        for(int j=0;j<_point_ident;j++)
        {
            auto search_match=match_index.find(j);
            auto search_img=img_index.find(j);
            if(search_match!=match_index.end() && search_img!=img_index.end())
            {
                auto it_match=search_match->second.end();
                it_match--;
                auto it_img=search_img->second.end();
                it_img--;
                int last_match=*it_match;
                int last_img=*it_img;
                int flag=0;
                for(int k=0;k<_left_matches.size() && !flag;k++)
                {
                    if(_left_matches[k]==last_match && last_img==_frame-1)
                    {
                            //we add the new projection for the same 3d point
                            pt_2d[j].push_back(_features2[_right_matches[k]].pt);
                            img_index[j].push_back(_frame);
                            match_index[j].push_back(_right_matches[k]);
                            used_points.at<double>(k)=1;
                            flag=1;
                    }
                }
            }
        }
    }

    int extract_values(int ident,Eigen::Vector3d &xyz_coordinates)
    {
        auto search_value=init_guess.find(ident);
        if(search_value !=init_guess.end())
        {
            xyz_coordinates=init_guess[ident];
            return 1;
        }
        else
        {
            return 0;
        }
        
    }

    int extract_values(int ident,vector<Point2f> &projections,vector<int> &imgs)
    {
        auto search_value=pt_2d.find(ident);
        if(search_value !=pt_2d.end())
        {
            projections=pt_2d[ident];
            imgs=img_index[ident];
            return 1;
        }
        else
        {
            return 0;
        }
        
    }
    void delete_invalid_points(int img_threshold,int total_points,int &remaining_points,vector<int> &valid_points)
    {
        for(int i=0;i<total_points;i++)
        {
            auto search_point=pt_2d.find(i);
            if(search_point!=pt_2d.end())
            {
                int dimension= search_point->second.size();
                if(dimension>=img_threshold)
                {
                    valid_points.push_back(1);
                    remaining_points++;
                }
                else
                {
                    valid_points.push_back(0);
                }
            }
        }
    }

    void initial_guess_for_3d_points(int total_points,vector<int> valid_points,Eigen::Vector2d principal_point,double focal_length)
    {
        for(int i=0;i<total_points;i++)
        {
            auto search_point=pt_2d.find(i);
            if(search_point!=pt_2d.end())
            {
                if(valid_points[i]==1)
                {
                    vector<Point2f> aux=pt_2d[i];
                    double dimension=aux.size();
                    double  z=0.5; //initial z invented
                    Eigen::Vector3d init_guess_aux;
                    init_guess_aux << 0.,0.,0.;
                    Eigen::Vector3d value;
                    for (int j=0;j<aux.size();j++)
                    {
                        init_guess_aux[0]+=(((double)search_point->second[j].x - principal_point[0])/focal_length)*z;
                        init_guess_aux[1]+=(((double)search_point->second[j].y - principal_point[1])/focal_length)*z;
                        init_guess_aux[2]+=z;
                    }
                    value[0]=init_guess_aux[0]/dimension;
                    value[1]=init_guess_aux[1]/dimension;
                    value[2]=init_guess_aux[2]/dimension;
                    init_guess[i]=value;
                }  
            }
        }
    }
    void set_correspondence(int map_ident,int structure_ident)
    {
        map_to_custom_struct[map_ident]=structure_ident;
    }

    void fill_new_3d_points(int map_vertex,Eigen::Vector3d pt)
    {
        int custom_structure_id;
        custom_structure_id=map_to_custom_struct[map_vertex];
        auto it=init_guess.find(custom_structure_id);
        if(it!=init_guess.end())
        {
            init_guess[custom_structure_id]=pt;
        }
    }
    //default destructor
    ~tracking_store(){}
};

void matchFeatures(	vector<KeyPoint> &_features1, cv::Mat &_desc1, 
					vector<KeyPoint> &_features2, cv::Mat &_desc2,
					vector<int> &_ifKeypoints, vector<int> &_jfKeypoints,double dst_ratio,double confidence,double reproject_err){

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<vector<DMatch>> matches;
	vector<Point2d> source, destination;
	vector<uchar> mask;
	vector<int> i_keypoint, j_keypoint;
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{
			source.push_back(_features1[matches[k][0].queryIdx].pt);
			destination.push_back(_features2[matches[k][0].trainIdx].pt);
			i_keypoint.push_back(matches[k][0].queryIdx);
			j_keypoint.push_back(matches[k][0].trainIdx);
		}
	}

	//aplicamos filtro ransac
	findFundamentalMat(source, destination, FM_RANSAC,reproject_err, confidence, mask);
	for (int m = 0; m < mask.size(); m++)
	{
		if (mask[m])
		{
			_ifKeypoints.push_back(i_keypoint[m]);
			_jfKeypoints.push_back(j_keypoint[m]);
		}
	}
}

void displayMatches(	cv::Mat &_img1, std::vector<cv::KeyPoint> &_features1, std::vector<int> &_filtered1,
						cv::Mat &_img2, std::vector<cv::KeyPoint> &_features2, std::vector<int> &_filtered2){
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);

	for(unsigned i = 0; i < _filtered1.size(); i++){
		auto p1 = _features1[_filtered1[i]].pt;
		auto p2 = _features2[_filtered2[i]].pt + cv::Point2f(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}

	cv::imshow("display", display);
	cv::waitKey(3);
}
cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs) {
	stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss << _folder << "/left_" << _number << ".png";
	std::cout << "Loading image: " << ss.str() << std::endl;
	Mat image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
	cvtColor(image, image, COLOR_BGR2GRAY);
	cv::Mat image_u;
	undistort(image, image_u, _intrinsics, _coeffs);
	return image_u;
}