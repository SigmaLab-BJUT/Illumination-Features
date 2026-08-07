import random

import math
from torch import Tensor
from torch.utils import data
import numpy as np

import cv2
import os

class SetData(data.Dataset):
    def __init__(self, folders, lenNum,
                 frames_root='/home/zhangsijia/zhanggq/code/guang/Xcep_Ceit_LMN_Normals/train_data/celeb-df-v1/frames'):
        super(SetData, self).__init__()

        self.folders = folders
        self.frames_root = frames_root
        ll = []
        # for folder in os.listdir('/farm/data/guang/guang_suiji_cross' + '/lmns_unsup'):
        for folder in os.listdir(self.folders + '/lmns_unsup'):
            if 'celeb-fake' in folder:
            # if 'DFDC-fake' in folder:
            # if 'fake' == folder or 'real7' == folder:
                print(folder)
                # folePath = os.path.join('/farm/data/guang/guang_suiji_cross' + '/lmns_unsup', folder)
                folePath = os.path.join(self.folders + '/lmns_unsup', folder)
                l = []
                for f in sorted(os.listdir(folePath)):
                    vi = f.replace('.txt', '')
                    if os.path.exists(os.path.join(self.frames_root, folder, vi)):
                    # if os.path.exists('/farm/data/guang/DFDC-guang-yuan/' + os.path.join(folder, vi)):
                        # if vi in videos_fake:
                        l.append(os.path.join(folder, f))
                print(len(l))
                # random.shuffle(l)
                ll.extend(l)
                # ll.extend(l)

            if 'celeb-real' in folder:
            # if 'DFDC-real' in folder:
                print(folder)
                # folePath = os.path.join('/farm/data/guang/guang_suiji_sec' + '/lmns_unsup', folder)
                folePath = os.path.join(self.folders + '/lmns_unsup', folder)
                l = []
                for f in sorted(os.listdir(folePath)):
                    vi = f.replace('.txt', '')
                    if os.path.exists(os.path.join(self.frames_root, folder, vi)):
                    # if os.path.exists('/farm/data/guang/DFDC-guang-yuan/' + os.path.join(folder, vi)):
                        # if vi in videos_real:
                        l.append(os.path.join(folder, f))
                print(len(l))
                ll.extend(l)
        self.pid = ll
        self.pid = list(map(str, self.pid))
        self.len = len(self.pid)
        self.len_seq = lenNum

    def __len__(self):
        return self.len


    def __getitem__(self, item):
        x0 = np.zeros([15, 3, 224, 224])
        # nameOfVideo = self.pid[item].split('.')[0]
        nameOfVideo = self.pid[item].split('.')[0]

        # if 'real' in nameOfVideo:
        #     folders = '/farm/data/guang/guang_suiji_sec'
        #     pathOfVideo = '/farm/data/guang/celeb-DF-v2-guang/' + nameOfVideo
        # else:
        #     folders = '/farm/data/guang/guang_suiji_cross'
        #     pathOfVideo = '/farm/data/guang/celeb-DF-v2-guang-yuan/' + nameOfVideo
        pathOfVideo = os.path.join(self.frames_root, nameOfVideo)
        # pathOfVideo = '/farm/data/guang/DFDC-guang-yuan/' + nameOfVideo

        # pathOfVideo = os.path.join(self.folders, nameOfVideo)
        frames = os.listdir(pathOfVideo)
        frames.sort(key=lambda x: int(x.split('.')[0]))
        for la in range(0, self.len_seq):
            file = frames[la*5]
            imgPath = pathOfVideo + '/' + file
            img = cv2.imread(imgPath)
            img = cv2.resize(img, [224, 224])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose([2, 0, 1])
            x0[la] = img
        xI = x0.astype('float')
        lmn = np.loadtxt(self.folders + '/lmns_unsup' + '/' + self.pid[item])

        normalsL = np.loadtxt(self.folders + '/NL_unsup' + '/' + self.pid[item])
        normalsR = np.loadtxt(self.folders + '/NR_unsup' + '/' + self.pid[item])
        normalsBL = np.loadtxt(self.folders + '/NBL_unsup' + '/' + self.pid[item])
        normalsBR = np.loadtxt(self.folders + '/NBR_unsup' + '/' + self.pid[item])
        normals = np.concatenate((normalsL, normalsR, normalsBL, normalsBR), axis=1)
        normals = np.reshape(normals, [90, 3, 44])
        # lightL = np.loadtxt(self.folders + '/LL_unsup' + '/' + self.pid[item])
        # lightR = np.loadtxt(self.folders + '/LR_unsup' + '/' + self.pid[item])
        # lightBL = np.loadtxt(self.folders + '/LBL_unsup' + '/' + self.pid[item])
        # lightBR = np.loadtxt(self.folders + '/LBR_unsup' + '/' + self.pid[item])
        # light = np.concatenate((lightL, lightR, lightBL, lightBR), axis=1)
        # light = np.reshape(light, [90, 1, 44])
        # NL = np.concatenate((normals, light), axis=1)
        xNL = Tensor(normals)
        xL = Tensor(lmn)

        datas = (xL, xNL)

        if 'real' in self.pid[item]:
            y = int(1)
        else:
            y = int(0)

        datas = (Tensor(xI), datas)
        #
        return datas, y, nameOfVideo
