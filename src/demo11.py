#!/usr/bin/env python


import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import numpy as np
import imutils
from geometry_msgs.msg import Twist

global perr, ptime, serr, dt,move
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
        # the appropriate callbacks

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

        	
        	cv2.imshow("RGB_Image",self.frame)
        	cv2.imshow("Processed_Image",self.display_image)
        	
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
	global perr, ptime, serr, dt,move
	perr = 0
	ptime = 0
	serr = 0
	dt = 0
	#transformation
    	img = cv2.resize(frame,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    #print img.shape
    	rows, cols, ch = img.shape
    	pts1 = np.float32([[90,122],[313,122],[35,242],[385,242]])
    	pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
    	M = cv2.getPerspectiveTransform(pts1,pts2)
    	img_size = (img.shape[1], img.shape[0])
    	image = cv2.warpPerspective(img,M,(img_size[0]+100,img_size[1]+100))#img_size
    
    	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    	lower_yellow = np.array([ 10,  10,  10])
    	upper_yellow = np.array([255, 255, 250])

    	lower_white = np.array([100,100,200], dtype= "uint8")
    	upper_white = np.array([255,255,255], dtype= "uint8")

    #threshold to get only white
    	maskw = cv2.inRange(image, lower_white, upper_white)
    	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #remove pixels not in this range
    	mask_yw = cv2.bitwise_or(maskw, masky)    
    	rgb_yw = cv2.bitwise_and(image, image, mask = mask_yw).astype(np.uint8)

    	rgb_yw = cv2.cvtColor(rgb_yw, cv2.COLOR_RGB2GRAY)
 
    #filter mask
    	kernel = np.ones((7,7), np.uint8)
    	opening = cv2.morphologyEx(rgb_yw, cv2.MORPH_OPEN, kernel)
    	rgb_yw2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)		
	
    	h, w= rgb_yw2.shape
    	search_top = 7*h/8+20
    	search_bot = 7*h/8 + 800 +20
    	rgb_yw2[0:search_top, 0:w] = 0
    	rgb_yw2[search_bot:h, 0:w] = 0
    	M = cv2.moments(rgb_yw2)
    	c_time = rospy.Time.now()
    	if M['m00'] > 0:
      		cxm = int(M['m10']/M['m00'])
      		cym = int(M['m01']/M['m00'])
      
      		cx = cxm - 110 #120#143 #CW
      		if cxm <= 2*h/8:
      			cx = cxm +(h/2)
      
      		l1=cv2.circle(rgb_yw2, (cxm, cym), 20, (255,0,0), -1)
      		l2=cv2.circle(rgb_yw2, (cx, cym), 20, (255,0,0),-1)
		# BEGIN CONTROL
      		err = cx - 1*w/2
      		self.twist.linear.x = 0.1
      		dt = rospy.get_time() - ptime
      		self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 #+ (serr*dt)*1/20/100 #1 is best, starting 3 unstable
      		serr = err + serr
      		perr = err
      		ptime = rospy.get_time()
		self.cmd_vel_pub.publish(self.twist)
        return l1,l2

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
    
