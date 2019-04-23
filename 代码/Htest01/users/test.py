from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from users import model
from django.shortcuts import render

CHECK_POINT_DIR = 'modelsave'
def evaluate_one_image(image_array):
	with tf.Graph().as_default():
		image = tf.cast(image_array, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, 64,64,3])

		logit = model.inference(image, 1, 2)
		logit = tf.nn.softmax(logit)

		#x = tf.placeholder(tf.float32, shape=[64,64,3])

		saver = tf.train.Saver()
		with tf.Session() as sess:
			print ('Reading checkpoints...')
			ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Loading success, global_step is %s' %global_step)
			else:
				print ('No checkpoint file found')
			prediction = sess.run(logit)
			#prediction = sess.run(logit, feed_dict = {x:image_array})
			max_index = np.argmax(prediction)
			print (prediction)
			if max_index == 0:
				result = ('this is cat rate: %.6f, result prediction is [%s]' %(prediction[:,0],','.join(str(i) for i in prediction[0])))
			else:
				result = ('this is dog rate: %.6f, result prediction is [%s]' %(prediction[:,1],','.join(str(i) for i in prediction[0])))

			print(result)
			result_msg = {}

			a_start = result.find(":")
			a_end = result.find(",")
			accuracy = result[a_start + 1:a_end]

			a = result.find('[')
			b = result.find(']')
			numStr = result[a:b+1]
			middle = numStr.find(',')
			first_num = numStr[0+1:middle]
			second_num = numStr[middle+1:len(numStr)-1]
			# print(first_num)
			# print(second_num)
			if(first_num>second_num):
				result = "这是狗"
			else:
				result = "这是猫"


			result_msg['accuracy'] = accuracy
			result_msg["result"] = result
			# result_msg["accuracy"]
			return result_msg


# if __name__ == '__main__':
# 	image = Image.open('../static/data/test_img/7.jpg')
# 	plt.imshow(image)
# 	plt.show()
# 	image = image.resize([64,64])
# 	image = np.array(image)
# 	print(evaluate_one_image(image)["result"])
# 	print(evaluate_one_image(image)["accuracy"])

def predict(img_path):
	answer = {}
	img_path = "static/data/test_img/"+img_path
	image = Image.open(img_path)
	image = image.resize([64, 64])
	image = np.array(image)
	answer["result"] = evaluate_one_image(image)["result"]
	answer["accuracy"] = evaluate_one_image(image)["accuracy"]
	return answer


# predict('img/7.jpg')




