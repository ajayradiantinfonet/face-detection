from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request  # import urllib because of internet 
import json
import cv2
import os
from absl.flags import FLAGS
from django.conf import settings

import uuid
import datetime

# define the path to the face detector
FACE_DETECTOR_PATH = 'cascades/haarcascade_frontalface_default.xml'




@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			# load the image and convert
			image = _grab_image(url=url) # download
		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		rects = detector.detectMultiScale(image1,1.3, 5)
		# construct a list of bounding boxes from the detection
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
		# update the data dictionary with the faces detected
		data.update({"num_faces": len(rects), "faces": rects, "success": True})
         
		 #add manully
		for (startX, startY, endX, endY) in rects:
    			cv2.rectangle(image1, (startX, startY), (endX, endY), (0, 255, 0), 2)
		newimg ='detection/'+format(str(uuid.uuid4().hex))+'.jpg'
		cv2.imshow('URL2Image',image1)
		cv2.imwrite(newimg, image1)
		cv2.waitKey()


	# return a JSON response
	return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
    			
			data = stream.read()
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		# add new line by me for draw faces
		# for (startX, startY, endX, endY) in rects:
    	# 		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
		# cv2.imshow('URL2Image',image)
		# cv2.waitKey()
 
	# return the image
	return image