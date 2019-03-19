import caffe
import cv2
import sys
import numpy as np
caffe.set_mode_cpu()
class_model_def = '/home/dxz/download_github/libfacedetection/models/caffe/yufacedetectnet-open-v1.prototxt'#set your prototxt path
class_model_weights = '/home/dxz/download_github/libfacedetection/models/caffe/yufacedetectnet-open-v1.caffemodel'#set your caffemodel path
net = caffe.Net(class_model_def,class_model_weights,caffe.TEST)
net.blobs['data'].reshape(1,3,240,320)
cap =cv2.VideoCapture(0)
while True:
	ret,img = cap.read()
	if ret:
		h,w,_ = img.shape
		tmp_batch = np.zeros([1, 3,240,320], dtype=np.float32)
		rect = cv2.resize(img, (320,240))
		tmp_batch[0, :, :, :] = rect.transpose(2,0,1)
		tmp_batch[:, 0, :, :] -= 124.16
		tmp_batch[:, 1, :, :] -= 116.736
		tmp_batch[:, 2, :, :] -= 103.936
		net.blobs['data'].data[...] = tmp_batch # transformed_image
 # Forward pass.
		prob = net.forward()['detection_out'][0][0]#(1,39,7)
		index =np.where(prob[:,2]>0.75)
		index = list(index[0])
		if len(index) == 0:
			print("no face")
		else:
			for i in index:
				score = prob[i][2]
				x = int(prob[i][3]*w)
				y = int(prob[i][4]*h)
				x2 = int(prob[i][5]*w)
				y2 = int(prob[i][6]*h)
				cv2.rectangle(img, (x, y), (x2,y2), (0, 255, 0), 2)
				cv2.putText(img, str(score), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		face_num = len(index)
		cv2.putText(img, "face num:%d"%face_num, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
		cv2.imshow("test",img)
                if cv2.waitKey(33) == ord('q'):
                        break

	else:
		cap.release()
		break

