
#include "ros/ros.h"
#include "geometry_msgs/Twist.h" 
#include "sensor_msgs/LaserScan.h" 
#include "tf/transform_listener.h" 

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stack>

ros::Publisher motor_command_publisher;
ros::Subscriber laser_subscriber;
sensor_msgs::LaserScan laser_msg;
geometry_msgs::Twist motor_command;

typedef enum _ROBOT_MOVEMENT {
    STOP = 0,
    FORWARD,
    BACKWARD,
    TURN_LEFT,
    TURN_RIGHT,
    GO_RIGHT,
    GO_LEFT

} ROBOT_MOVEMENT;


bool robot_move(const ROBOT_MOVEMENT move_type)
{
    if (move_type == STOP) {
        ROS_INFO("[ROBOT] HALT! \n");

        motor_command.angular.z = 0.0;
        motor_command.linear.x = 0.0;
    }

    else if (move_type == FORWARD) {
        ROS_INFO("[ROBOT] Always FORWARD! \n");
        motor_command.angular.z = 0.0;
        motor_command.linear.x = 0.5;
    }

    else if (move_type == BACKWARD) {
        ROS_INFO("[ROBOT] I'm going back! \n");
        motor_command.linear.x = -0.75;
        motor_command.angular.z = 0.0;
    }

    else if (move_type == TURN_LEFT) {
        ROS_INFO("[ROBOT] I'm turning left! \n");
        motor_command.linear.x = 0.0;
        motor_command.angular.z = 1.0;
    }

    else if (move_type == TURN_RIGHT) {
        ROS_INFO("[ROBOT] I'm turning right! \n");
        motor_command.linear.x = 0.0;
        motor_command.angular.z = -1.0;
    }
    else if (move_type == GO_RIGHT) {
        ROS_INFO("[ROBOT] I'm goin right! \n");
        motor_command.linear.x = 0.25;
        motor_command.angular.z = -0.25;
    }
    else if (move_type == GO_LEFT) {
        ROS_INFO("[ROBOT] I'm goin left! \n");
        motor_command.linear.x = 0.25;
        motor_command.angular.z = 0.25;
    }
    else {
        ROS_INFO("[ROBOT_MOVE] Move type wrong! \n");
        return false;
    }

    motor_command_publisher.publish(motor_command);
    usleep(10);
    return true;
}

bool following_wall = false;
bool thats_a_door = false;
bool crashed = false;


void laser_callback(const sensor_msgs::LaserScan::ConstPtr& scan_msg)
{
    laser_msg = *scan_msg;
    std::vector<float> laser_ranges;
    laser_ranges = laser_msg.ranges;
    size_t range_size = laser_ranges.size();
    float left_side = 0.0, right_side = 0.0;
    float range_min = laser_msg.range_max, range_max = laser_msg.range_min;
    int nan_count = 0;
    for (size_t i = 0; i < range_size; i++) {
        if (laser_ranges[i] < range_min) {
            range_min = laser_ranges[i];
        }

        if (std::isnan(laser_ranges[i])) {
            nan_count++;
        }
        if (i < range_size / 4) {
            if (laser_ranges[i] > range_max) {
                range_max = laser_ranges[i];
            }
        }

        if (i > range_size / 2) {
            left_side += laser_ranges[i];
        }
        else {
            right_side += laser_ranges[i];
        }
    }

    if (nan_count > (range_size * 0.9) || laser_ranges[range_size / 2] < 0.25) {
        crashed = true;
    }
    else {
        crashed = false;
    }


    if (!crashed) {

        if (range_min <= 0.5 && !thats_a_door) {
            following_wall = true;
            crashed = false;
            robot_move(STOP);

            if (left_side >= right_side) {
                robot_move(TURN_RIGHT);
            }
            else {
                robot_move(TURN_LEFT);
            }
        }
        else {
            ROS_INFO("[ROBOT] Dam son: %f , %d \n", range_max, following_wall);
            robot_move(STOP);
            if (following_wall) {
                if (range_max >= 2.0) {
                    thats_a_door = true;
                    following_wall = false;
        
                    ROS_INFO("[ROBOT] I am following wall and my max range > 2.0 Range Max: %f \n", range_max);
                }
            }
            if (thats_a_door) {
                if (laser_ranges[0] <= 0.5) {
                    thats_a_door = false;
                }
                else {
                    robot_move(GO_RIGHT);
                }
                ROS_INFO("[ROBOT] I am goin' right!: %d \n", thats_a_door);
            }
            else {
                robot_move(FORWARD);
            }
        }
    }
  
    else {
        robot_move(BACKWARD);
    }
}

int main(int argc, char** argv)
{

    ros::init(argc, argv, "wall_follower");


    ros::NodeHandle n;


    motor_command_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    laser_subscriber = n.subscribe("/scan", 1, laser_callback);

    ros::Duration time_between_ros_wakeups(0.001);
    while (ros::ok()) {
        ros::spinOnce();
        time_between_ros_wakeups.sleep();
    }

    return 0;
}
