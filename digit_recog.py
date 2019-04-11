import cv2
import matplotlib.pyplot as plt
import numpy as np
from base64 import b64decode
import torch

class prediction:
	def fun(st): 
		data_uri = st
		header, encoded = data_uri.split(",", 1)
		data = b64decode(encoded)

		with open("image.png", "wb") as f:
			f.write(data)

		img=cv2.imread('image.png',0)
		plt.imshow(img,cmap='gray')


		img=cv2.GaussianBlur(img,(5,5),0)
		plt.imshow(img,cmap='gray')


		kernel = np.ones((5,5),np.uint8)
		erosion = cv2.dilate(img,kernel,iterations = 1)

		plt.imshow(erosion,cmap='gray')



		x1=cv2.resize(erosion,(28,28))
		plt.imshow(x1,cmap='gray')


		ret,t1 = cv2.threshold(x1,0,255,cv2.THRESH_BINARY)
		plt.imshow(t1,cmap='gray')


		cv2.imwrite('mod.png',t1)

		from PIL import Image as l
		o=l.open('mod.png')

		#img.shape
		from torchvision import transforms
		m1=transforms.Grayscale().__call__(o)
		m=transforms.ToTensor().__call__(m1)

		#m=torch.from_numpy(img)
		print(m.shape)
		m=m.view(1, -1)
		print(m.shape)
		x=torch.load('model_full.pt')
		#print(x)
		with torch.no_grad():
			output = x(m)
			_, pred = torch.max(output, 1)

		print(pred)

		print(output)

		values=output.numpy()

		val=np.exp(values) / np.sum(np.exp(values), axis=1)
		'''
		index = np.arange(len(val[0]))
		plt.bar(index,val[0])
		plt.xlabel('digits', fontsize=15)
		plt.ylabel('probablity', fontsize=15)
		plt.xticks(index)
		plt.show()
		'''
		return pred.item()
