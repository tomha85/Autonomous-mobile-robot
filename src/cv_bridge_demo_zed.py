#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class depth_processing():

    def __init__(self):

        rospy.init_node('zed_depth', anonymous=True)
        self.bridge = CvBridge()
        
        rospy.Subscriber("/zed/depth/depth_registered", Image, self.callback)
     
    
    def callback(self, depth_data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        except CvBridgeError, e:
            print e

        depth_array = np.array(depth_image, dtype=np.float32)

        print('Image size: {width}x{height}'.format(width=depth_data.width,height=depth_data.height))

        u = depth_data.width/2
        v = depth_data.height/2

        print('Center depth: {dist} m'.format(dist=depth_array[u,v]))


if __name__ == '__main__': 
    try:
        detector = depth_processing()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Detector node terminated.")
