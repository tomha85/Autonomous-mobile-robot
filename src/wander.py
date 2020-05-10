#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
	global g_range_ahead
	g_range_ahead = min(msg.ranges)
	print ("range ahead : %0.1f", g_range_ahead)

g_range_ahead = 1 # anything to start
scan_sub = rospy.Subscriber('scan', LaserScan, scan_callback)
cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
rospy.init_node('wander')
state_change_time = rospy.Time.now()
driving_forward = True
rate = rospy.Rate(10)
twist = Twist()
while not rospy.is_shutdown():
	twist.linear.x = 0.2
	if (g_range_ahead < 0.5):
		for i in range(1000):
			twist.linear.x = 0
		twist.linear.x = 0.1
		
				
	
	cmd_vel_pub.publish(twist)

	rate.sleep()
