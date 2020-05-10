#!/usr/bin/env python


import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import numpy as np
from geometry_msgs.msg import Twist
import imutils
global perr, ptime, serr, dt,move, stop_sign,idd
perr = 0
ptime = 0
serr = 0
dt = 0
move = False



class cvBridgeDemo():
    def __init__(self):
        self.node_name = "cv_bridge_demo"
        
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.node_name
	# Create the cv_bridge object
        self.bridge = CvBridge()
        # Subscribe to the camera image and depth topics and set
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
	self.cmd_vel_pub = rospy.Publisher('/cmd_vel',Twist, queue_size=1)
    	self.twist = Twist()
        
        rospy.Timer(rospy.Duration(0.03), self.show_img_cb)
        rospy.loginfo("Waiting for image topics...") 

    def show_img_cb(self,event):
    	try: 


		cv2.namedWindow("RGB_Image", cv2.WINDOW_NORMAL)
		cv2.moveWindow("RGB_Image", 25, 75)

		cv2.namedWindow("Processed_Image", cv2.WINDOW_NORMAL)
		cv2.moveWindow("Processed_Image", 500, 75)

		
		cv2.namedWindow("Processed_Image1", cv2.WINDOW_NORMAL)
		cv2.moveWindow("Processed_Image1", 1000, 75)

        	cv2.imshow("RGB_Image",self.frame)
        	cv2.imshow("Processed_Image",self.display_image)
       		cv2.imshow("Processed_Image1",self.image_line)

      		cv2.waitKey(3)
    	except:
		pass


    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            self.frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e
	    pass

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(self.frame, dtype=np.uint8)
        
        # Process the frame using the process_image() function
        self.display_image = self.process_image(frame)
                       
	          
    def process_image(self, frame):
	global perr, ptime, serr, dt,sub
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     	#lower = (158, 145, 0)
	#upper = (255, 255, 255)
	lower = (0, 0, 180)
	upper = (255, 255, 255)
	mask = cv2.inRange(hsv, lower, upper)
	#mask = cv2.erode(mask, None, iterations=2)
	lane_mask = cv2.dilate(mask, None, iterations=1)
	
    	
    	h, w= lane_mask.shape[:2]
    	#search_ver = 2*w/3
    	#search_bot = h/3 
    	lane_mask[0:h, 0:3*w/4] = 0
	lane_mask[0:h/2, 0:w] = 0
    	lane_mask[3*h/4:h, 0:w] = 0
	#lane_mask[0:w, h/2:h] = 0
	cnts=cv2.findContours(lane_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts=imutils.grab_contours(cnts)
	clone=frame.copy()
    	self.image_line=lane_mask
	center=[]
	for c in cnts:
		cv2.drawContours(clone,[c],-1,(255,0,0),2)
		M=cv2.moments(c)
	   	if M['m00'] > 0:
	        	cxm = int(M['m10']/M['m00'])
	        	cym = int(M['m01']/M['m00'])
      			cv2.circle(clone,(cxm,cym),10,(0,255,0),-1)

        		cx = cxm-240        	
      			center= (cx,cym)
        		cv2.circle(clone, (cx, cym), 5, (255,255,0),-1)
			cv2.line(clone,(w/2,0),(w/2,h),(255,23,245),4)
			err = cx - 1*w/2
      			self.twist.linear.x = 0.2
      			dt = rospy.get_time() - ptime
      			self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 
      			serr = err + serr
      			perr = err
      			ptime = rospy.get_time()
			self.cmd_vel_pub.publish(self.twist)
	

        return clone
    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()   
    
def main(args):       
    try:
        cvBridgeDemo()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    
