#!/usr/bin/env python
# coding: utf-8

import argparse, os, math
import numpy as np
from PIL import Image
import piexif
import cv2
import torch
import network
from torch.autograd import Variable
from collections import OrderedDict

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='File path of input image.', default='./data/Forest.png')
parser.add_argument('-o', help='Output directory.', default='./results')
parser.add_argument('-gpu', help='GPU device specifier. Two GPU devices must be specified, such as 0,1.', default='1')
parser.add_argument('-dm', help='File path of a downexposure model.', default='./models/downexposure_model.pth')
parser.add_argument('-um', help='File path of a upexposure model.', default='./models/upexposure_model.pth')
args = parser.parse_args()

f_path = args.i
model_path_list = [args.dm, args.um]
base_outdir_path = args.o
gpu_list = []
if args.gpu !='-1':
	for gpu_num in (args.gpu).split(','):
		gpu_list.append(int(gpu_num))

model_list = [network.CNNAE3D512().cuda(), network.CNNAE3D512().cuda()]
print "load_weight"
for i in range(2) :
    state_dict = torch.load(model_path_list[i])
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        # print name
        new_state_dict[name] = v
    model_list[i].load_state_dict(new_state_dict)
    # model_list[i].load_state_dict(torch.load(model_path_list[i]))


def estimate_images(input_img, model):
    model.train_dropout = False
    #input image RGB to BRG
    input_img_ = (input_img.astype(np.float32)/255.).transpose(2,0,1)
    #numpy to torch variable
    input_img_ = Variable(torch.from_numpy(input_img_), volatile = True).cuda()
    input_img_ = input_img_.unsqueeze(0)
    res = model(input_img_).data[0]
    res = res.cpu().numpy()
    out_img_list = list()
    for i in range(res.shape[1]):
    	out_img = (255.*res[:,i,:,:].transpose(1,2,0)).astype(np.uint8)
        # out_img = (255.*res[:,i,:,:].transpose(0,1).transpose(1,2)).astype(np.uint8)
        out_img_list.append(out_img)

    return out_img_list


img = cv2.imread(f_path)
out_img_list = list()
for i in range(2):
    model = model_list[i]
    out_img_list.extend(estimate_images(img, model_list[i]))
    if i == 0:
        out_img_list.reverse()
        out_img_list.append(img)
    # print(len(out_img_list))

print 'Select and Merge'
threshold = 128
stid = 0
prev_img = out_img_list[8].astype(np.float32)
out_img_list.reverse()
# for i in out_img_list:
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
for out_img in out_img_list[9:]:
    img = out_img.astype(np.float32)
    if (img>(prev_img+threshold)).sum() > 0:
        break
    prev_img = img[:,:,:]
    stid+=1

edid = 0
prev_img = out_img_list[8].astype(np.float32)
out_img_list.reverse()
for out_img in out_img_list[9:]:
    img = out_img.astype(np.float32)
    if (img<(prev_img-threshold)).sum() > 0:
        break
    prev_img = img[:,:,:]
    edid+=1
print('edid:'+str(edid))
print('stid:'+str(stid))
out_img_list = out_img_list[8-stid:9+edid]
outdir_path = base_outdir_path+'/'+f_path.split('/')[-1]
os.system('mkdir ' + outdir_path)

exposure_times = list()
lowest_exp_time = 1/1024.
for i in range(len(out_img_list)):
    exposure_times.append(lowest_exp_time*math.pow(math.sqrt(2.),i))
exposure_times = np.array(exposure_times).astype(np.float32)

for i, out_img in enumerate(out_img_list):
    numer, denom = float(exposure_times[i]).as_integer_ratio()
    if int(math.log10(numer)+1)>9:
        numer = int(numer/10*(int(math.log10(numer)+1)-9))
        denom = int(denom/10*(int(math.log10(numer)+1)-9))
    if int(math.log10(denom)+1)>9:
        numer = int(numer/10*(int(math.log10(denom)+1)-9))
        denom = int(denom/10*(int(math.log10(denom)+1)-9))
    exif_ifd = {piexif.ExifIFD.ExposureTime:(numer,denom)}
    exif_dict = {"Exif":exif_ifd}
    exif_bytes = piexif.dump(exif_dict)
    out_img_ = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    out_img_pil = Image.fromarray(out_img_)
    out_img_pil.save(outdir_path+"/exposure_"+str(i)+".jpg", exif=exif_bytes)

merge_debvec = cv2.createMergeDebevec()
hdr_debvec = merge_debvec.process(out_img_list, times=exposure_times.copy())
cv2.imwrite(outdir_path+'/MergeDebevec.hdr', hdr_debvec)

merge_mertens = cv2.createMergeMertens(1.,1.,1.e+38)
res_mertens = merge_mertens.process(out_img_list)
cv2.imwrite(outdir_path+'/MergeMertens.hdr', res_mertens)
