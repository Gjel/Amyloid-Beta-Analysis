import time

import matplotlib.pyplot as plt
import bioformats
import javabridge
import h5py
import torch
import numpy as np

from src.segmentation import Unet, brainsec_resnet18
from src.data_access import DataIterator
from src.util import LabelEnum

############
# SETTINGS #
############
data_file = 'F:/sample_46.hdf5'
data_partition = 'validation'
batch_size = 8
patch_size = 256
stride = 128
epochs = 1

###############
# PREPARATION #
###############
javabridge.start_vm(class_path=bioformats.JARS)
logback = javabridge.JClassWrapper("loci.common.LogbackTools")
logback.enableLogging()
logback.setRootLevel("ERROR")

file = h5py.File(data_file, 'r')
it = DataIterator(file[data_partition], patch_size, stride, batch_size, LabelEnum.CLASS_MIDDLE)
it.open()

model = brainsec_resnet18().to('cuda')
# model = Unet(3, 1)
optim = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()  # include weighting per class
batch_loss = []

############
# TRAINING #
############
for e in range(epochs):
    epoch_start = time.time()
    print('epoch', e)
    for i, data in enumerate(it):
        batch_start = time.time()
        patches, labels = data
        patches = torch.from_numpy(patches).to('cuda')
        labels = torch.from_numpy(labels).unsqueeze(1).to('cuda')
        optim.zero_grad()
        out = model(patches)
        loss = criterion(out, labels)
        loss.backward()
        optim.step()
        batch_loss.append(loss.item())
        print('batch time', time.time() - batch_start)
        if i % 100 == 99:
            print(i)
    print('epoch time', time.time() - epoch_start)

#################
# VISUALISATION #
#################
plt.figure()
plt.plot(batch_loss)
plt.show()

###########
# CLOSING #
###########
it.close()
file.close()
javabridge.kill_vm()
