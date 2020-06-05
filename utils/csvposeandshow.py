import csv
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import utils.paramUtil as paramUtil
from utils.plot_script import plot_3d_pose

# matplotlib.use('Qt5Agg')
mpl.rcParams['legend.fontsize'] = 10

filename = "../dataset/pose/chenxiangye_group1_time1/pose.csv"

with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype(float)


pose_orig = data[1][1:]

# poseSeq = np.load('./3d_joints/endecoder/1001.0.npy')
# pose_orig = poseSeq[0]
offset = numpy.matlib.repmat(np.array([pose_orig[0], pose_orig[1], pose_orig[2]]), 1, 24)[0]

# pose = pose_orig - offset
pose = pose_orig
body_entity = paramUtil.SMPLBodyPart()
plot_3d_pose(pose, body_entity)
