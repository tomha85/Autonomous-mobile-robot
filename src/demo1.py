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
	h,w,c=frame.shape
	decentre=160
	rowtowatch=40
	crop=frame[(h)/2+decentre:(h)/2+(decentre+rowtowatch)][1:w]
	hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    	lower_yellow = np.array([ 20,  100,  100])
    	upper_yellow = np.array([55, 255, 255])
    	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	res=cv2.bitwise_and(crop,crop,mask=mask)
  

    	M = cv2.moments(mask,False)
	try:
        	cx = int(M['m10']/M['m00'])
      		cy = int(M['m01']/M['m00'])
	except ZeroDivisionError:
		cy,cx=h/2,w/2
	res=cv2.bitwise_and(crop,crop,mask=mask)
      	l=cv2.circle(res, (cx, cy), 20, (0,0,255), -1)
	# BEGIN CONTROL
      	err = cx - 1*w/2
      	self.twist.linear.x = 0.1
      	dt = rospy.get_time() - ptime
      	self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 #+ (serr*dt)*1/20/100 #1 is best, starting 3 unstable
      	serr = err + serr
      	perr = err
      	ptime = rospy.get_time()
	self.cmd_vel_pub.publish(self.twist)
        return l

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
	h,w,c=frame.shape
	decentre=160
	rowtowatch=40
	crop=frame[(h)/2+decentre:(h)/2+(decentre+rowtowatch)][1:w]
	hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    	lower_yellow = np.array([ 20,  100,  100])
    	upper_yellow = np.array([55, 255, 255])
    	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	res=cv2.bitwise_and(crop,crop,mask=mask)
  

    	M = cv2.moments(mask,False)
	if ['m00'] > 0:
		cxm = int(M['m10']/M['m00'])
      		cym = int(M['m01']/M['m00'])
		cx = cxm - 110 #120#143 #CW
      		if cxm <= 2*h/8:
      			cx = cxm +(h/2)
		res=cv2.bitwise_and(crop,crop,mask=mask)
      		l=cv2.circle(res, (cxm, cym), 20, (0,0,255), -1)
		l2=cv2.circle(res, (cx, cym), 20, (0,0,250),-1)
		print(cx,cxm)
	# BEGIN CONTROL
      		err = cx - 1*w/2
      		self.twist.linear.x = 0.1
      		dt = rospy.get_time() - ptime
      		self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 #+ (serr*dt)*1/20/100 #1 is best, starting 3 unstable
      		serr = err + serr
      		perr = err
      		ptime = rospy.get_time()
		self.cmd_vel_pub.publish(self.twist)
        return l2

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
    
