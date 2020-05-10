#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

def robot_control(msg):
       
    #left = msg.ranges[270:360]
    #front = msg.ranges[0:180]  
    #right = msg.ranges[181:270]
    #print('left:',min(msg.ranges[0:180]))
    #print(left)
    
    min_range=msg.ranges[0]
    min_range_angle=0
    for i in range(0,360,1):
	if(msg.ranges[i]<min_range):
		min_range=msg.ranges[i]
		min_range_angle=i/2
		print(min_range,min_range_angle)
      
    if(min_range<=1):
	if(min_range_angle<90):
		message = Twist(
            	Vector3(0.0, 0, 0),
           	Vector3(0, 0, 0.3)
		
        )
		print('left')
	else:
		message = Twist(
            	Vector3(0.0, 0, 0),
            	Vector3(0, 0, -0.3)
		
        )
		print('right')
    else:
	message = Twist(
        Vector3(0.3, 0, 0),
        Vector3(0, 0, 0.0)
		
        )	 
        print('straight')	
    pub.publish(message)

# Subscriber node
rospy.init_node('mini_01')

# The topic "/cmd_vel" manages messages of "geometry_msgs/Twist" type.
pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
# The node is subscribed to the topic "/kobuki/laser/scan"
sub = rospy.Subscriber('/scan', LaserScan, robot_control)

rospy.spin()    # Blocks until ROS node is shutdown
