#!/usr/bin/env python
# coding: utf-8

import argparse, os, math, glob
import numpy as np
from PIL import Image
import piexif
import cv2
import network
# import tf_recoder as tensorboard
import torch
from torch.autograd import Variable
import time
import random
from tqdm import tqdm

def random_x_y_generator(img_width, img_height, target_width, target_height):
	x = (int)(random.random()*1000)%(img_width-target_width)
	y = (int)(random.random()*1000)%(img_height-target_height)
	return x, y
	
def random_crop(img, x, y, target_width, target_height):
	# print('target_width:'+str(target_width))
	# print('target_height:'+str(target_height))

	im_result = img[y:y+target_height,x:x+target_width,0:3]
	# cv2.imshow("img_crop",im_result)
	# cv2.waitKey(0)
	return im_result
class trainer:
	def __init__(self):
		parser = argparse.ArgumentParser(description='')
		parser.add_argument('-i', help='Directory path of training data.', default='./training_samples')
		parser.add_argument('-o', help='Saved path Wof an output model file.', default='./models/downexposure_model/')
		parser.add_argument('-l', help='Learning type. (0:downexposure, 1:upexposure)', default='0')
		parser.add_argument('-gpu', help='GPU device specifier.', default='1')
		args = parser.parse_args()

		#cuda check
		if torch.cuda.is_available():
			self.use_cuda = True
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
		else:
			self.use_cuda = False
			torch.set_default_tensor_type('torch.FloatTensor')
		self.lr = 0.0002
		# self.smoothing = config.smoothing
		# self.max_resl = config.max_resl
		# self.trns_tick = config.trns_tick
		self.TICK = 1000
		self.globalIter = 0
		self.globalTick = time.clock()
		self.kimgs = 0
		self.stack = 0

		# self.renew_everything()
		# self.use_tb = config.use_tb
		# if self.use_tb:
		# 	self.tb = tensorboard.tf_recorder()
		

		# gpu = int(args.gpu)
		#learning_type: 0: downexposure, 1: upexposure
		self.is_upexposure_trained = int(args.l)
		#saved path of an output model file - default: models/downexposures
		self.out_path = args.o
		self.dir_path_list = glob.glob(args.i+'/*')
		# print(self.dir_path_list)
		# self.dir_path_list = self.dir_path_list[:1]
		# print(self.dir_path_list)
		batch_size = 1
		self.maximum_epoch = 200
		self.predicted_window_len = 8

		self.lossmask_list = list()
		img_shape = (3,512,512)
		for i in range(self.predicted_window_len):
			lossmask = np.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])
			for j in range(7,0,-1):
				if i<j:
					append_img = np.ones(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])
				else:
					append_img = np.zeros(img_shape[0]*img_shape[1]*img_shape[2]).reshape(img_shape[:1]+(1,)+img_shape[1:])
				lossmask = np.hstack([lossmask, append_img])
			lossmask = np.broadcast_to(lossmask, (batch_size,)+lossmask.shape).astype(np.float32)
			self.lossmask_list.append(lossmask)

		self.model = network.CNNAE3D512()
		print ('network structure')
		print(self.model)
		self.mse = torch.nn.MSELoss()
		if torch.cuda.is_available():
			self.mse = self.mse.cuda()
			self.model = self.model.cuda()
			torch.cuda.manual_seed((int)(time.time()))
			gpus = []
			for i in range(2):
				gpus.append(i)
			self.model = torch.nn.DataParallel(self.model,device_ids =gpus)

		self.TARGET_WIDTH = 512
		self.TARGET_HEIGHT = 512
		self.IMG_WIDTH = 1920
		self.IMG_HEIGHT = 1080


	def train(self):
		if torch.cuda.is_available():
			self.use_cuda = True
			torch.set_default_tensor_type('torch.cuda.FloatTensor')
		else:
			self.use_cuda = False
			torch.set_default_tensor_type('torch.FloatTensor')
		self.opt = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr = self.lr, betas = (0.5,0.99))
		N = len(self.dir_path_list)
		before_loss = 10000000
		for epoch in range(self.maximum_epoch):
			print epoch
			start_time = time.time()
			self.globalTick = int(time.time())
			loss_sum = 0.
			perm = np.random.permutation(N)
			# print perm
			for i in tqdm(range(N),ascii = True, desc = 'epoch:'+str(epoch)+'/'+str(self.maximum_epoch)):
				dir_path = self.dir_path_list[perm[i]]
				img_path_list = glob.glob(dir_path+'/*.png')
				img_path_list.sort()

				img_list = list()
				if self.is_upexposure_trained:
					img_order = range(len(img_path_list))
					# print('upexposure model')
				else:
					img_order = range(len(img_path_list)-1, -1, -1)
					# print('downexposure model')
				#random crop
				start_x,start_y = random_x_y_generator(self.IMG_WIDTH, self.IMG_HEIGHT, self.TARGET_WIDTH, self.TARGET_HEIGHT)
				
				for j in img_order:
					img_path = img_path_list[j]
					img = cv2.imread(img_path,-1)
					
					# cv2.imshow("image",img)
					# print(img.shape)
					# cv2.waitKey(0)
					#random crop
					img = random_crop(img, start_x, start_y, self.TARGET_WIDTH, self.TARGET_HEIGHT)
					# cv2.imshow("image_crop",img)
					# cv2.waitKey(0)
					img_ = (img.astype(np.float32)/255.).transpose(2,0,1)
					img_list.append(img_)

				# print perm[i]
				img_list = np.array(img_list)

				for input_frame_id in range(len(img_list)-1):
					start_frame_id = input_frame_id +1
					end_frame_id = min(start_frame_id+8, len(img_list))

					x_batch = np.array([img_list[input_frame_id,:,:,:]])
					print x_batch.shape
					#x_batch 8 -> 7th -> 6 -> 5 -> 4 -> 3-> 2-> 1
					y_batch = np.array([img_list[start_frame_id:end_frame_id,:,:,:].transpose(1,0,2,3)])
					#y_batch real value, the number of length -> 8, 7, 6, 5, 4, 3, 2, 1
					dummy_len = self.predicted_window_len-y_batch.shape[2]
					zero_dummy = np.zeros(x_batch.size*dummy_len).reshape(y_batch.shape[:2]+(dummy_len,)+y_batch.shape[3:]).astype(np.float32)
					y_batch = np.concatenate([y_batch, zero_dummy], axis=2)
					# print(x_batch)

					x_batch = Variable(torch.from_numpy(np.array(x_batch))).cuda()
					y_batch = Variable(torch.from_numpy(np.array(y_batch))).cuda()
					lossmask = Variable(torch.from_numpy(np.array(self.lossmask_list[dummy_len]))).cuda()
					# print(x_batch.shape)
					print(x_batch.shape)
					y_hat =(self.model(x_batch))
					y_hat = lossmask * y_hat
					loss_gen = self.mse(y_hat, y_batch)
					self.model.zero_grad()
					loss_gen.backward()
					self.opt.step()
					loss_sum += loss_gen.data*len(x_batch.data)
			if before_loss/torch.sum(loss_sum) >1.1:
				before_loss = torch.sum(loss_sum)
				print("save model " + "!"*10)
				if not os.path.exists(self.out_path):
					os.system('mkdir -p {}'.format(self.out_path))
				w_name = 'epoch:{}_loss:{}.pth'.format(epoch,torch.sum(loss_sum))
				save_path = os.path.join(self.out_path,w_name)
				torch.save(self.model.state_dict(), save_path)

			print('epoch_'+str(epoch)+' loss:{} '.format(torch.sum(loss_sum)/(N))+' time: '+str(time.time()-start_time)+'\n')
				
		        # if self.use_tb:
		        # 	x_test = self.model


	    
	def snapshot(self,path):
		if not os.path.exists(path):
			os.system('mkdir -p {}'.format(path))
		w_name = 'epoch:{}_T{}.pth'.format(epoch,self.globalTick)
		self.globalTick = time.clock()-self.globalTick
		if self.globalTick%50 == 0:
			# if (self.phase == 'gstab' or self.phase =='dstab'):
			save_path = os.path.join(path,w_name)

			if not os.path.exists(save_path):
				torch.save(self.model.state_dict(), save_path)
				print('[snapshot] model saved @ {}'.format(path))



## perform training.
print '----------------- configuration -----------------'
# for k, v in vars(config).items():
#     print('  {}: {}').format(k, v)
print '-------------------------------------------------'
torch.backends.cudnn.benchmark = True           # boost speed.
trainer = trainer()
trainer.train()




