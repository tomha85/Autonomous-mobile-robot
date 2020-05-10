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
from sensor_msgs.msg import LaserScan
#from std_msgs.msg import Float32MultiArray

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
        self.g_range_ahead =0
	self.obj =0
        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks
	rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/rgb/image_raw_color', Image, self.image_callback)
	rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback1)
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
        	
		cv2.namedWindow("Processed_Image2", cv2.WINDOW_NORMAL)
		cv2.moveWindow("Processed_Image2", 500, 400)

        	cv2.imshow("RGB_Image",self.frame)
        	cv2.imshow("Processed_Image",self.display_image)
		cv2.imshow("Processed_Image1",self.display)
		cv2.imshow("Processed_Image2",self.display_image1)
        	
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

    #def image_callback1(self, ros_image1):
    #    try:
    #        self.frame1 = self.bridge.imgmsg_to_cv2(ros_image1, "bgr8")
    #    except CvBridgeError, e:
    #        print e
	#    pass
        #frame1 = np.array(self.frame1, dtype=np.uint8)
        #self.display_image1 = self.process_image1(frame1)
                   
    def scan_callback(self, msg):
	self.g_range_ahead = min(msg.ranges)
	#print ("range ahead : %0.1f", g_range_ahead)

    #def obj_callback(self,msg):
    #    self.obj=msg.data
	#if(len(self.obj)>0):
	#	print(self.obj[0])
	#else:
	#	print('nothing')
          
    def process_image(self, frame):
	global perr, ptime, serr, dt
    	img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    	bird_view=get_bird_view(img1,0.1)
    	lane_mask1  = binary_hsv(bird_view)
	kernel = np.ones((7,7), np.uint8)
    	lane_mask = cv2.morphologyEx(lane_mask1, cv2.MORPH_OPEN, kernel)
    	
    	h, w= lane_mask.shape[:2]
    	search_top = 2*h/3
    	search_bot = 2*h/3+50 
    	lane_mask[0:search_top, 0:w] = 0
    	lane_mask[search_bot:h, 0:w] = 0
	lane_mask[0:h, 3*w/4:w] = 0
	lane_mask[0:h, 0:w/4] = 0
	
       	cnts=cv2.findContours(lane_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts=imutils.grab_contours(cnts)
	clone=frame.copy()		
	self.display=lane_mask
	center=[]
	for c in cnts:
		cv2.drawContours(clone,[c],-1,(255,0,0),2)
		M=cv2.moments(c)
		if (M["m00"]>0):
			cX=int(M["m10"]/M["m00"])
			cY=int(M["m01"]/M["m00"])
			cv2.circle(clone,(cX,cY),10,(0,255,255),-1)
			center.append((int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])))
			#print(center)
			xm=[x[0] for x in center]
			ym=[x[1] for y in center]
			cxm=sum(xm)/len(xm)
			cym=sum(ym)/len(ym)
			#print(sum(xm)/len(xm))
			#print(sum(ym)/len(ym))
       			cx = cxm
       			#cv2.circle(clone, center[-1], 10, (255,0,0), -1)
       			cv2.circle(clone, (cxm, cym), 10, (0,255,255),-1)
			cv2.line(clone,(w/2,0),(w/2,h),(255,23,245),4)    
      			err = cx - 1*w/2
      			self.twist.linear.x = 0.1
     			dt = rospy.get_time() - ptime
     			self.twist.angular.z = (-float(err) / 100)*1 + ((err - perr)/(rospy.get_time() - ptime))*1/15/100 
     			serr = err + serr
     			perr = err
     			ptime = rospy.get_time()
		
	#if (self.g_range_ahead <0.4):
		#self.twist.linear.x=0
		#self.twist.angular.z=0
		#if (self.obj>0):
		#	idi=self.obj[0]
		#	if (idi==9):
		#		self.twist.linear.x=0
		#		self.twist.angular.z=0					
		
			self.cmd_vel_pub.publish(self.twist)
	return clone

    def process_image1(self, frame):    
	colorRanges=[
	((24, 59, 130), (91, 220, 255), "green light"),
	((93, 108, 171), (255, 255, 255), "red light")]

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	for (lower, upper, colorName) in colorRanges:
		mask = cv2.inRange(hsv, lower, upper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
				
		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center= (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
 				cv2.putText(frame, colorName, center, cv2.FONT_HERSHEY_SIMPLEX,
					1.0, (50, 255, 125), 2)
	return frame

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
    
