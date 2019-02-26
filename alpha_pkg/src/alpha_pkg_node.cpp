/************************************************************
 * Name: alpha_pkg_node.cpp
 
 * Author:  Shashank Shastry 	scshastr@eng.ucsd.edu
 	    Sumukha Harish 	ssumukha@eng.ucsd.edu
 	    Sai Adithya		schittur@eng.ucsd.edu
 
 * Date: 02/26/2019
 
 * Description: This is the source file of the node in which 
 		the robot wanders, detects and interacts with humans. 
            	The robot is also capable of avoiding obstacles and 
            	carrying on exploring on its own.

		Topics subscribed to:
		1. "/camera/depth/points" 
		2. "/usb_cam/image_raw"

		Topics published:
		1. "cmd_vel_mux/input/teleop"

* Usage:  roscore
	  roslaunch turtlebot_bringup minimal.launch
	  roslaunch astra_launch astra_pro.launch
          rosrun usb_cam usb_cam_node
	  rosrun alpha_pkg alpha_pkg_node
**************************************************************/

#include <kobuki_msgs/BumperEvent.h> 
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <stdio.h>
#include <vector>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <ros/console.h>

#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <cv_bridge/cv_bridge.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

uint16_t state = 0;
bool obstacle_flag = false, human_flag = false;
int rand_dir = 1;
float image_height = 480, image_width = 640;
float linear_speed = 0.2, angular_speed = 0.7, angular_speed_thresh = 0.3;

cv::CascadeClassifier face_cascade,eyes_cascade;


/************************************************************
 * Function Name: PointCloud_Callback

 * Description: This is the callback function of the topic
		"/camera/depth/points". The function also computes
		the number of points that are closer than a threshold
		z_min and raises the obstacle_flag if the number of 
        	points are greater than a threshold (10)
*************************************************************/
void PointCloud_Callback (const PointCloud::ConstPtr& cloud){
	double min_z = 0.7;
	std::vector<double> PCL_closest_points;

  	// Iterate through all the points in the image and populate buffer
  	// if the z coordinate is lesser than threshold (min_z)
  	for(int k = 0; k < 240; k++){
    	for(int i = 0; i < 640; i++){
      		const pcl::PointXYZ & pt=cloud->points[640*(180+k)+(i)];
      		if((pt.z < min_z)){
        		PCL_closest_points.push_back(i); 
      		}
    	}
  	}	
	  
	// Raise obstacle_flag if the size of the buffer is greater than
	// threshold 10
	if(PCL_closest_points.size() > 10){
		if(!obstacle_flag){
			obstacle_flag = true;
			if(rand()%2==0){
				rand_dir*=-1;
			}
		}
	}
	else{
		obstacle_flag = false;
	}
}

/************************************************************
 * Function Name: DetectHuman

 * Description: This is the helper function that uses the 
          cv::cascade classifier to detect faces of humans. The
          detection of eyes are also implemented here, and can
          can be used for enhanced detection and analysis.
*************************************************************/
bool DetectHuman(cv::Mat frame)
{
  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;

  cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
  cv::equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  // face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) ); // Default paramaters
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 4, 0|CV_HAAR_SCALE_IMAGE, cv::Size(15, 15) );    // Set parameters

  // Return true if at least one face is detected
  if(faces.size()>0){
      return true;
  }

  // For enhanced detection the code below which has eye detection
  // can be used in conjunction with face detection
  for( std::size_t i = 0; i < faces.size(); i++ )
  {
    cv::Mat faceROI = frame_gray( faces[i] );
    std::vector<cv::Rect> eyes;

    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    if(eyes.size()>0){
      return true;
    }
  }
  return false;
 }

/************************************************************
 * Function Name: Cam_Callback

 * Description:	This is the callback function for the usb_cam
		node that is publishing camera data from the laptop
		camera. The function uses the DetectHuman function 
		to determine humans in the camera feed.
*************************************************************/
void Cam_Callback(const sensor_msgs::Image::ConstPtr& image_msg){
  // std::cout << "Got cam" << std::endl;

  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  human_flag=DetectHuman(cv_ptr->image);
}

/************************************************************
 * Function Name: rotate

 * Description: Generic function which makes the robot rotate
 		about its z axis at constant angular velocity
*************************************************************/
void rotate(ros::Publisher& velocityPublisher, float angular_speed){
	  geometry_msgs::Twist T;
  	T.linear.x = 0.0; T.linear.y = 0.0; T.linear.z = 0.0;
  	T.angular.x = 0.0; T.angular.y = 0.0; T.angular.z = angular_speed;
  	velocityPublisher.publish(T);
    ros::Duration(0.1).sleep();
}

/************************************************************
 * Function Name: advance

 * Description: Generic function which makes the robot 
 		move forward with constant linear velocity.
*************************************************************/
void advance(ros::Publisher& velocityPublisher){
    geometry_msgs::Twist T;
    T.linear.x = linear_speed; T.linear.y = 0.0; T.linear.z = 0.0;
    T.angular.x = 0.0; T.angular.y = 0.0; T.angular.z = 0.0;
    velocityPublisher.publish(T);
    ros::Duration(0.1).sleep();
}

/************************************************************
 * Function Name: retreat

 * Description: Generic function which makes the robot 
 		move backward with constant linear velocity.
*************************************************************/
void retreat(ros::Publisher& velocityPublisher){
  	geometry_msgs::Twist T;
  	T.linear.x = -linear_speed; T.linear.y = 0.0; T.linear.z = 0.0;
  	T.angular.x = 0.0; T.angular.y = 0.0; T.angular.z = 0.0;
  	velocityPublisher.publish(T);
    ros::Duration(0.1).sleep();
}

/************************************************************
 * Function Name: draw_attention

 * Description: Makes the robot draw attention by spinning to
        	and fro on the spot.
*************************************************************/
void draw_attention(ros::Publisher& velocityPublisher){
  int time=10; // variable parameter
  for(int i=0;i<time;i++){
    rotate(velocityPublisher,angular_speed);
  }
  for(int i=0;i<2*time;i++){
    rotate(velocityPublisher,-1*angular_speed);
  }
  for(int i=0;i<time;i++){
    rotate(velocityPublisher,angular_speed);
  }
}

/************************************************************
 * Function Name: interact

 * Description: Makes the robot interact with the human
        	by spinning to engaging in conversation.
*************************************************************/
void interact(ros::Publisher& velocityPublisher){
  std::cout<<"How is your day going?"<<std::endl;
  ros::Duration(3).sleep(); // Wait for 3 seconds
  int time=10;
  for(int i=0;i<5*time;i++){
    rotate(velocityPublisher,angular_speed);
  }
}

/************************************************************
 * Function Name: not_interact

 * Description: Makes the robot retreat and move away from the 
        	human not engaging in conversation.
*************************************************************/
void not_interact(ros::Publisher& velocityPublisher){
  std::cout<<"Have a good day!"<<std::endl;
  int time=10;
  for(int i=0;i<time;i++){
    retreat(velocityPublisher);  }
  for(int i=0;i<5*time;i++){
    rotate(velocityPublisher,angular_speed);
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "blob");

  // Path to files
  cv::String face_cascade_name = "/home/turtlebot/alpha_ws/src/alpha_pkg/src/haarcascade_frontalface_alt.xml";
  cv::String eyes_cascade_name = "/home/turtlebot/alpha_ws/src/alpha_pkg/src/haarcascade_eye_tree_eyeglasses.xml";
  
  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  ros::NodeHandle nh;
  ros::Publisher velocityPublisher = nh.advertise<geometry_msgs::Twist>("cmd_vel_mux/input/teleop", 10);
  ros::Subscriber PCSubscriber = nh.subscribe<PointCloud>("/camera/depth/points", 1, PointCloud_Callback);
  ros::Subscriber CamSubscriber = nh.subscribe<sensor_msgs::Image>("/usb_cam/image_raw", 1, Cam_Callback);

  ros::Rate loop_rate(10);

  //State variable initialized to 0
  state = 0;
  cv::Mat m;

  while(ros::ok()){
    std::cout<<" state:"<<state<<" obstacle_flag:"<<obstacle_flag<<" human_flag"<<human_flag<<std::endl;
    switch(state){
      // Functionalities of state 0
    	case 0:{
        if(human_flag){
          state=1;
          break;
        }
    		if(!obstacle_flag){
    			advance(velocityPublisher);
    		}
    		else{
    			rotate(velocityPublisher, rand_dir*angular_speed);
    		}
        break;
    	}

      // Functionalities of state 1
      case 1:{
        draw_attention(velocityPublisher);
        // Ask if human wants to interact
        std::cout<<"Would you like to interact? (y/n)"<<std::endl;
        char answer;
        std::cin>>answer;
        // If answered a 'yes'
        if(answer=='y'){
          interact(velocityPublisher);
        }
        // If answered a 'no'
        else{
          not_interact(velocityPublisher);
        }
        human_flag=false; // Reset to default after interaction
        state=0; // Return to explore
      }
    }

    ros::spinOnce();
    loop_rate.sleep();
  }
}
