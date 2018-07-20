import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import argparse

#Description: Inference script to generate a file of predictions given an input.
#
#Args:
#    - checkpoint: A filepath to the exported pb (model) file.
#        ie ("saved_model.pb")
#    - chip_size: An integer describing how large chips of test image should be
#    - input: A filepath to a single test chip
#		 ie ("11924.tif")
#	 - output: A filepath where the script will save  its predictions
#		 ie ("predictions.txt")
#
#Outputs:
#    - Saves a file specified by the 'output' parameter containing predictions for the model.
#		Per-line format:  xmin ymin xmax ymax class_prediction score_prediction
#		Note that the variable "num_preds" is dependent on the trained model (some models output only 100 bboxes per iteration)
#
#Author: Darius Lam (dariusl@diux.mil)

#chop image into nxn chips
def chip_image(img, chip_size=(300,300)):
	width,height,_ = img.shape
	wn,hn = chip_size
	images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
	k = 0
	for i in tqdm(range(int(width/wn))):
		for j in range(int(height/hn)):

			chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
			images[k]=chip

			k = k + 1

	return images.astype(np.uint8)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-c","--checkpoint", default='pbs/080417-12-03-164952.pb', help="Path to saved model")
	parser.add_argument("--chip_size", default=300, help="Size in pixels to chip input image")
	parser.add_argument("--input", help="Path to test chip")
	parser.add_argument("--output",default="predictions.txt",help="Filepath of desired output")
	args = parser.parse_args()

	PATH_TO_CKPT = args.checkpoint

	arr = np.array(Image.open(args.input))
	chip_size = (args.chip_size,args.chip_size)
	images = chip_image(arr,chip_size)
	print(images.shape)

	print("Creating Graph...")
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	boxes = []
	scores = []
	classes = []
	k = 0
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			for image_np in tqdm(images):
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				box = detection_graph.get_tensor_by_name('detection_boxes:0')
				score = detection_graph.get_tensor_by_name('detection_scores:0')
				cls = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(box, score, cls, num_detections) = sess.run(
						[box, score, cls, num_detections],
						feed_dict={image_tensor: image_np_expanded})
				if k == 0:
					boxes = box
					scores = score
					classes = cls
				else:
					boxes = np.concatenate((boxes,box))
					scores = np.concatenate((scores,score))
					classes = np.concatenate((classes,cls))

				k = k + 1

	#Process boxes to be in the full nxn chip
	n = int(np.sqrt(boxes.shape[0]))

	width,height,_ = arr.shape
	cwn,chn = (chip_size)
	wn,hn = (int(width/cwn),int(height/chn))

	num_preds = 200
	bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))

	bfull[:,:,:,0] *= cwn
	bfull[:,:,:,2] *= cwn
	bfull[:,:,:,1] *= chn
	bfull[:,:,:,3] *= chn

	for i in range(wn):
		for j in range(hn):
			bfull[i,j,:,0] += i*cwn
			bfull[i,j,:,2] += i*cwn

			bfull[i,j,:,1] += j*chn
			bfull[i,j,:,3] += j*chn

	bfull = bfull.reshape((hn*wn,num_preds,4))

	with open(args.output,'w') as f:
		for i in range(bfull.shape[0]):
			for j in range(bfull[i].shape[0]):
				#box should be xmin ymin xmax ymax
				box = bfull[i,j]
				class_prediction = classes[i,j]
				score_prediction = scores[i,j]
				f.write('%d %d %d %d %d %f \n' % \
					(box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))
