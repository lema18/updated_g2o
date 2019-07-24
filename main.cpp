#include "functions.h"
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

int main(int argc,char ** argv)
{
    if(argc< 13)
    {
        cout<< "bad input\n";
        cout << "please enter:\n";
        cout << "argv[1]= path to rgb images\n";
        cout << "argv[2]= number of images to compose the initial map\n";
        cout << "argv[3]= threshold for points tha are seen in more than i images\n";
        cout << "argv[4]= distance ratio to reject features\n";
        cout << "argv[5]= confidence on ransac\n";
        cout << "argv[6]= threshold for error_reprojection on ransac\n";
        cout << "argv[7]= g2o iterations\n";
        cout << "argv[8]= use dense solver(1 to set or 0 to disable)\n";
        cout << "argv[9]= use robust Kernel(1 to set or 0 to disable)\n";
        cout << "argv[10]= Nº of known poses\n";
        cout << "argv[11]= Nº of vertices to get fixed in g2o solver\n";
        cout << "argv[12]= first img index of dataset\n";
        exit(-1);
    }
    //init calibration matrices
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinseca = (Mat_<float>(3, 3) << 517.3, 0., 318.6, 0., 516.5, 255.3, 0., 0., 1.);
	intrinseca.convertTo(intrinseca, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    double focal_length=516.9;
    Eigen::Vector2d principal_point(318.6,255.3); 
    //number of images to compose the initial map
    int nImages =  atoi(argv[2]);
    //threshold for points tha are seen in more than i images
    int img_threshold;
    sscanf(argv[3],"%d",&img_threshold);
    //distance ratio to reject features
    double dst_ratio;
    sscanf(argv[4],"%lf",&dst_ratio);
    //confidence on ransac
    double confidence;
    sscanf(argv[5],"%lf",&confidence);
    //threshold for error_reprojection on ransac
    double reproject_err;
    sscanf(argv[6],"%lf",&reproject_err);
    //g2o iterations
    int niter;
    sscanf(argv[7],"%d",&niter);
    //use dense solver
    int dense;
    sscanf(argv[8],"%d",&dense);
    //use robust Kernel
    int robust_kernel;
    sscanf(argv[9],"%d",&robust_kernel);
    //known poses
    int known_poses;
    sscanf(argv[10],"%d",&known_poses);
    int cam_fixed;
    sscanf(argv[11],"%d",&cam_fixed);
    int offset;
    sscanf(argv[12],"%d",&offset);
    //iterator to iterate through images
    int l;
    //identifier for 3d_point
    int ident=0;
    //first index of dataset
    l=offset;
    //we need variables to store the last image and the last features & descriptors
    auto pt =SURF::create();
    Mat foto1_u;
    vector<KeyPoint> features1;
    Mat descriptors1;
    //custom class to store matching between images
    tracking_store tracks;
    //load first image
    foto1_u = loadImage(argv[1], l, intrinseca, distcoef);
    pt->detectAndCompute(foto1_u, Mat(), features1, descriptors1);
    l++;
    while(l<nImages+offset)
    {   
        //load new image
        Mat foto2_u = loadImage(argv[1], l, intrinseca, distcoef);
        //create pair of features
        vector<KeyPoint> features2;
	    Mat descriptors2;
	    pt->detectAndCompute(foto2_u, Mat(), features2, descriptors2);
        //match features
        vector<int> left_index_matches, right_index_matches;
        matchFeatures(features1, descriptors1, features2, descriptors2, left_index_matches, right_index_matches,dst_ratio,confidence,reproject_err);
        displayMatches(foto1_u, features1, left_index_matches,foto2_u, features2, right_index_matches);
        Mat used_features=Mat::zeros(1,int(left_index_matches.size()),CV_64F);//to differentiate the features that
        if(ident>0)
        {
            tracks.add_new_projection_for_existent_point(ident,l,features1,features2,left_index_matches,right_index_matches,used_features);
        }
        tracks.add_new_points_proyections(features1,features2,left_index_matches,right_index_matches,ident,l,used_features);
        foto1_u=foto2_u;
        features1=features2;
        descriptors1=descriptors2;
        used_features.release();
        l++;
    }
    //prepare varibles for g2o optimization
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (dense)
    {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }
    else
    {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    //filter to reject points that are not visible in more than 3 images
    vector<int> valid_points;
    int remaining_points=0;
    tracks.delete_invalid_points(img_threshold,ident,remaining_points,valid_points);
    //we add camera vertices to optimizator
    vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > camera_poses;
    g2o::CameraParameters * cam_params = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);
    int vertex_id=0;
    double aux;
    string line;
    Eigen::Matrix4d initPose = Eigen::Matrix4d::Identity();
    //we read the known poses from groundtruth
    ifstream myfile ("/home/angel/new_g2o/build/lectura_datos/first_50.txt");
    if (myfile.is_open())
    {
        for(int i=0;i<nImages;i++)
        {
            g2o::VertexSE3Expmap * v_se3=new g2o::VertexSE3Expmap();
            Eigen::Quaterniond qa;
            g2o::SE3Quat pose;
            Eigen::Vector3d trans;
            Eigen::Matrix4d poseMatrix=Eigen::Matrix4d::Identity();
            //Eigen::Matrix4d inter;
            if(i<known_poses)
            {
                getline(myfile,line);
                std::istringstream in(line);
                for(int j=0;j<8;j++)
                {
                    in >> aux;
                    if(j==1) trans[0]=aux;
                    if(j==2) trans[1]= aux;
                    if(j==3) trans[2]= aux;
                    if(j==4) qa.x()= aux;
                    if(j==5) qa.y()= aux;
                    if(j==6) qa.z()= aux;
                    if(j==7) qa.w()= aux;
                }
                poseMatrix.block<3,3>(0,0)=qa.normalized().toRotationMatrix();
                poseMatrix.block<3,1>(0,3)=trans;
                /*
                inter=poseMatrix.inverse();
                qa=Eigen::Quaterniond(inter.block<3,3>(0,0));
                trans=inter.block<3,1>(0,3);
                */
                pose=g2o::SE3Quat(qa,trans);
                v_se3->setId(vertex_id);
                v_se3->setEstimate(pose);
                if(i<cam_fixed)
                {
                    v_se3->setFixed(true);
                }
                optimizer.addVertex(v_se3);
                camera_poses.push_back(pose);
                if(i==0)
                {
                    initPose=poseMatrix;
                }
            }
            else
            {
                pose=camera_poses[known_poses-1];
                v_se3->setId(vertex_id);
                v_se3->setEstimate(pose);
                optimizer.addVertex(v_se3);
                camera_poses.push_back(pose);
            }
            vertex_id++;
        }
        myfile.close();
    } 

    //calculation of initial guess for 3d points
    int point_id=vertex_id;
    tracks.initial_guess_for_3d_points(ident,valid_points,principal_point,focal_length);

    //we add 3dpoints vertices to optimizator and the edges conecting cameras and points
    
    for(int j=0;j<ident;j++)
    {
        if(valid_points[j]==1)
        {
            Eigen::Vector3d init_guess;
            tracks.extract_values(j,init_guess);
            tracks.set_correspondence(point_id,j);
            g2o::VertexSBAPointXYZ * v_p= new g2o::VertexSBAPointXYZ();
            v_p->setId(point_id);
            v_p->setMarginalized(true);
            v_p->setEstimate(init_guess);
            optimizer.addVertex(v_p);
            vector<Point2f> aux_pt;
            vector<int> aux_im;
            tracks.extract_values(j,aux_pt,aux_im);
            //we search point j on image i
            for(int p=0;p<aux_im.size();p++)
            {
                //we add the edge connecting the vertex of camera position and the vertex point
                Eigen::Vector2d measurement(aux_pt[p].x,aux_pt[p].y);
                g2o::EdgeProjectXYZ2UV * e= new g2o::EdgeProjectXYZ2UV();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(aux_im[p])->second));
                e->setMeasurement(measurement);
                e->information() = Eigen::Matrix2d::Identity();
                if (robust_kernel)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                }
                e->setParameterId(0, 0);
                optimizer.addEdge(e);
            }
            point_id++;
        }
    }
    cout << endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(niter);
    optimizer.save("test.g2o");
    /* Graphical representation of camera's position and 3d points*/
	pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35); 

    mkdir("./lectura_datos", 0777);
    std::ofstream file("./lectura_datos/odometry.txt");
    if (!file.is_open()) return -1;

	for (int i = 0; i < nImages; i++)
    {
		stringstream sss;
		string name;
		sss << i;
		name = sss.str();
		Eigen::Affine3f cam_pos;
        g2o::SE3Quat updated_pose;
        Eigen::Matrix4f eig_cam_pos=Eigen::Matrix4f::Identity();
        Eigen::Quaterniond cam_quat;
        Eigen::Vector3d cam_translation;
        g2o::HyperGraph::VertexIDMap::iterator pose_it= optimizer.vertices().find(i);
        g2o::VertexSE3Expmap * v_se3= dynamic_cast< g2o::VertexSE3Expmap * >(pose_it->second);
        updated_pose=v_se3->estimate();
        cam_translation=updated_pose.translation();
        cam_quat=updated_pose.rotation();
        eig_cam_pos.block<3,3>(0,0) = cam_quat.matrix().cast<float>();
        eig_cam_pos.block<3,1>(0,3) = cam_translation.cast<float>();
        cam_pos=eig_cam_pos.inverse();
        viewer.addCoordinateSystem(0.05, cam_pos, name);
		pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		viewer.addText3D(std::to_string(i), textPoint, 0.01, 1, 1, 1, "text_"+std::to_string(i));
        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

        file << i << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    
    }
    
    file.close();
	pcl::PointCloud<pcl::PointXYZ> cloud;
    Eigen::Matrix4d identidad =Eigen::Matrix4d::Identity();
    for(int j=0;j<remaining_points;j++)
    {
        g2o::HyperGraph::VertexIDMap::iterator point_it= optimizer.vertices().find(vertex_id+j);
        g2o::VertexSBAPointXYZ * v_p= dynamic_cast< g2o::VertexSBAPointXYZ * > (point_it->second);
        Eigen::Vector3d p_aux=v_p->estimate();
        Eigen::Vector4d p_homo;
        p_homo[0]=p_aux[0];
        p_homo[1]=p_aux[1];
        p_homo[2]=p_aux[2];
        p_homo[3]=1;
        p_homo=initPose.inverse()*p_homo;
        p_aux[0]=p_homo[0]/p_homo[3];
        p_aux[1]=p_homo[1]/p_homo[3];
        p_aux[2]=p_homo[2]/p_homo[3];
        pcl::PointXYZ p(p_aux[0], p_aux[1], p_aux[2]);
        cloud.push_back(p);
    }
	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");

	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;
}
