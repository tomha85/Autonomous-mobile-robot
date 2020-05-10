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

def get_bird_view(img,shrink_ratio,dsize=(640,480)):
    height,width=img.shape[:2]
    k=int(height*0.1)
    roi=img.copy()
    cv2.rectangle(roi,(0,0),(width,k),0,-1)

    dst_width,dst_height=dsize
    src_pts=np.float32([[0,k],[width,k],[0,height],[width,height]])
    dst_pts=np.float32([[0,0],[dst_width,0],[dst_width*shrink_ratio,dst_height],[dst_width*(1-shrink_ratio),dst_height]])
    M=cv2.getPerspectiveTransform(src_pts,dst_pts)
    dst=cv2.warpPerspective(roi,M,dsize)
    return dst


def binary_hsv(img):
    minThreshold = (0, 0, 180)
    maxThreshold = (255, 255, 255)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    out = cv2.inRange(hsv_img, minThreshold, maxThreshold)
    return out

def inv_bird_view(img, stretch_ratio, dsize=(320, 240)):
    """ function for invert bird view transform
    :param img: the 
    """
    height, width = img.shape[:2]
    dst_width, dst_height = dsize
    SKYLINE = int(dst_height*0.4)
    src_pts = np.float32([[0, 0], [width, 0], [width*stretch_ratio, height], [width*(1-stretch_ratio), height]])
    dst_pts = np.float32([[0, SKYLINE], [dst_width, SKYLINE], [0, dst_height], [dst_width, dst_height]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst = cv2.warpPerspective(img, M, dsize)
    return dst



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
	global perr, ptime, serr, dt
    #print img.shape
	#frame = cv2.resize(frame,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    	img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    	bird_view=get_bird_view(img1,0.1)
    	lane_mask1  = binary_hsv(bird_view)
	kernel = np.ones((7,7), np.uint8)
    	lane_mask = cv2.morphologyEx(lane_mask1, cv2.MORPH_OPEN, kernel)
    	
    	h, w= lane_mask.shape[:2]
    	search_top = 2*h/4
    	search_bot = 2*h/4+50 
    	lane_mask[0:search_top, 0:w] = 0
    	lane_mask[search_bot:h, 0:w] = 0
	lane_mask[0:h, 0:w/4] = 0
    	M = cv2.moments(lane_mask)
    	if M['m00'] > 0:
        	cxm = int(M['m10']/M['m00'])
        	cym = int(M['m01']/M['m00'])
      
        	cx = cxm - 190 #120#143 #CW
        	if cxm <= h/4:
			cx = cxm +(h/2)
      
        	cv2.circle(lane_mask, (cxm, cym), 10, (255,0,0), -1)
        	cv2.circle(lane_mask, (cx, cym), 15, (255,255,0),-1)    
      		err = cx - 1*w/2
      		self.twist.linear.x = 0.3
      		dt = rospy.get_time() - ptime
      		self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 #+ (serr*dt)*1/20/100 #1 is best, starting 3 unstable
      		serr = err + serr
      		perr = err
      		ptime = rospy.get_time()
		self.cmd_vel_pub.publish(self.twist)
		print(cx,cxm)
	lane_overlay = inv_bird_view(lane_mask , 0.3)	
        return lane_mask
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
    
