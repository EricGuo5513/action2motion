import os
import cv2
import pickle
import numpy as np
import random
import calib_new.utils as utils
import matplotlib.pyplot as plt
from fitting.smpl_webuser.serialization import load_model
from fitting.render_model import render_model


class Camera:
    def __init__(self, t, rt, f, c):
        self.t = t
        self.rt = rt
        self.f = f
        self.c = c


def projection(points, params):
    # points: [N, 3]
    fx, fy, cx, cy, k1, k2, k3 = params

    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    r2 = x ** 2 + y ** 2
    k = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    U = cx + fx * k * x
    V = cy + fy * k * y
    return np.stack((U, V), axis=-1)  # N x 2


def draw_skeleton_opencv(img, uv_pose, kinematic_tree):
    for uv in uv_pose:
        point = tuple(uv.astype(np.int32))
        cv2.circle(img, point, 8, (0, 0, 255), -1)
    lw = 4
    color = (255, 0, 0)
    pose = uv_pose.astype(np.int32)
    for idx1, idx2 in kinematic_tree:
        cv2.line(img, (pose[idx1, 0], pose[idx1, 1]), (pose[idx2, 0], pose[idx2, 1]), color, lw)
    return img


root_dir = '/home/data/data_shihao'
filename = 'zoushihao_demo'
# kinematic_tree = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12],
#                   [9, 14], [9, 13], [12, 15], [13, 16], [14, 17], [16, 18],
#                   [17, 19], [18, 20], [19, 21], [21, 23], [20, 22]]
kinematic_tree = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12],
                  [9, 14], [9, 13], [12, 15], [13, 16], [14, 17], [16, 18],
                  [17, 19], [18, 20], [19, 21]]
c_H, c_W = 1080, 1920

MODEL_DIR = '/home/shihao/MultiCameraDataProcessing/fitting/models/'
MODEL_MALE_PATH = MODEL_DIR + "basicModel_m_lbs_10_207_0_v1.0.0.pkl"
MODEL_FEMALE_PATH = MODEL_DIR + "basicModel_f_lbs_10_207_0_v1.0.0.pkl"
model = load_model(MODEL_MALE_PATH)
# model = load_model(MODEL_FEMALE_PATH)


# load extrinsic params
extrinsic_subset = 'sub0906'
with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
    params0906 = pickle.load(f)
extrinsic_subset = 'sub0909'
with open('../calib_new/calib_multi_data/%s/extrinsic_params.pkl' % extrinsic_subset, 'rb') as f:
    params0909 = pickle.load(f)

pose_3d = {}
with open('%s/pose/%s/pose.txt' % (root_dir, filename), 'r') as f:
    for line in f.readlines():
        tmp = line.split(' ')
        pose = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3]) * 1000
        pose_3d[tmp[0]] = pose

# load correpsonding camera params
name = filename.split('_')[0]
if name in ['lyuxingzheng', 'yanhe', 'zoushihao', 'houpengyue', 'zouting', 'guochuan']:
    T_d1p = utils.convert_param2tranform(params0906['d1p'])
    T_cd1 = utils.convert_param2tranform(params0906['cd1'])
    T_c1p = T_cd1 * T_d1p
    param_c1 = params0906['param_c1']

    T_d2p = utils.convert_param2tranform(params0906['d2p'])
    T_cd2 = utils.convert_param2tranform(params0906['cd2'])
    T_c2p = T_cd2 * T_d2p
    param_c2 = params0906['param_c2']

    T_d3p = utils.convert_param2tranform(params0906['d3p'])
    T_cd3 = utils.convert_param2tranform(params0906['cd3'])
    T_c3p = T_cd1 * T_d3p
    param_c3 = params0906['param_c3']
else:
    T_d1p = utils.convert_param2tranform(params0909['d1p'])
    T_cd1 = utils.convert_param2tranform(params0909['cd1'])
    T_c1p = T_cd1 * T_d1p
    param_c1 = params0909['param_c1']

    T_d2p = utils.convert_param2tranform(params0909['d2p'])
    T_cd2 = utils.convert_param2tranform(params0909['cd2'])
    T_c2p = T_cd2 * T_d2p
    param_c2 = params0909['param_c2']

    T_d3p = utils.convert_param2tranform(params0909['d3p'])
    T_cd3 = utils.convert_param2tranform(params0909['cd3'])
    T_c3p = T_cd1 * T_d3p
    param_c3 = params0909['param_c3']


# from matplotlib.backends.backend_pdf import PdfPages
# with PdfPages('%s/%s.pdf' % (root_dir, filename)) as pdf:
for i in ['700', '1000', '1300']:
    smpl_file = '%s/fitting_results/%s/smpl_sfd_%s.pkl' % (root_dir, filename, i)
    if os.path.exists(smpl_file):
        print(i)
        with open(smpl_file, 'rb') as f:
            param = pickle.load(f)
        pose = pose_3d[i][0:22, :]
    else:
        print('[warning] skip %i' % i)
        continue

    c_file1 = '%s/color/PC1/%s/color_%s.jpg' % (root_dir, filename, i)
    img_c1 = (cv2.cvtColor(cv2.imread(c_file1), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
    pose_c1 = T_c1p.transform(pose)
    uv_c1 = projection(pose_c1, param_c1)
    img_c1 = draw_skeleton_opencv(img_c1, uv_c1, kinematic_tree)
    cam_t = T_c1p.transform(param['cam_t'] * 1000) / 1000
    verts = T_c1p.transform((param['verts'] + param['cam_t']) * 1000) / 1000 - cam_t
    dist = np.abs(cam_t[2] - np.mean(verts, axis=0)[2])
    cam = Camera(cam_t, [0, 0, 0], param_c1[0:2], param_c1[2:4])
    rendered_img_c1 = (render_model(verts, model.f, c_W, c_H, cam, far=20 + dist, img=img_c1) * 255.).astype('uint8')


    c_file2 = '%s/color/PC2/%s/color_%s.jpg' % (root_dir, filename, i)
    img_c2 = (cv2.cvtColor(cv2.imread(c_file2), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
    pose_c2 = T_c2p.transform(pose)
    uv_c2 = projection(pose_c2, param_c2)
    img_c2 = draw_skeleton_opencv(img_c2, uv_c2, kinematic_tree)
    cam_t = T_c2p.transform(param['cam_t'] * 1000) / 1000
    verts = T_c2p.transform((param['verts'] + param['cam_t']) * 1000) / 1000 - cam_t
    dist = np.abs(cam_t[2] - np.mean(verts, axis=0)[2])
    cam = Camera(cam_t, [0, 0, 0], param_c2[0:2], param_c2[2:4])
    rendered_img_c2 = (render_model(verts, model.f, c_W, c_H, cam, far=20 + dist, img=img_c2) * 255.).astype('uint8')


    c_file3 = '%s/color/PC3/%s/color_%s.jpg' % (root_dir, filename, i)
    img_c3 = (cv2.cvtColor(cv2.imread(c_file3), cv2.COLOR_BGR2RGB)[:, ::-1, :]).astype(np.uint8)
    pose_c3 = T_c3p.transform(pose)
    uv_c3 = projection(pose_c3, param_c3)
    img_c3 = draw_skeleton_opencv(img_c3, uv_c3, kinematic_tree)
    cam_t = T_c3p.transform(param['cam_t'] * 1000) / 1000
    verts = T_c3p.transform((param['verts'] + param['cam_t']) * 1000) / 1000 - cam_t
    dist = np.abs(cam_t[2] - np.mean(verts, axis=0)[2])
    cam = Camera(cam_t, [0, 0, 0], param_c3[0:2], param_c3[2:4])
    rendered_img_c3 = (render_model(verts, model.f, c_W, c_H, cam, far=20 + dist, img=img_c3) * 255.).astype('uint8')



    # plt.figure(figsize=(10, 16))
    # plt.subplot(311)
    # plt.title('%s/color_%s.jpg' % (filename, i))
    # plt.imshow(img_c1)
    # plt.axis('off')
    #
    # plt.subplot(312)
    # plt.imshow(img_c2)
    # plt.axis('off')
    #
    # plt.subplot(313)
    # plt.imshow(img_c3)
    # plt.axis('off')
    # # plt.show()
    # plt.tight_layout()
    # # pdf.savefig()
    # plt.close()

    plt.figure(figsize=(12, 9))
    plt.subplot(321)
    plt.imshow(img_c1)
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(img_c2)
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(img_c3)
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(rendered_img_c1)
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(rendered_img_c2)
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(rendered_img_c3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('result_%s.png' % i, dpi=300, bbox_inches='tight')
    plt.show()



'''
i = 1000
pose = pose_3d['1000'][0:22, :]
c_file1 = '%s/color/PC1/%s/color_%s.jpg' % (root_dir, filename, i)
img_c1 = (cv2.imread(c_file1)[:, ::-1, :]).astype(np.uint8)
pose_c1 = T_c1p.transform(pose)
uv_c1 = projection(pose_c1, param_c1)
img_c1 = draw_skeleton_opencv(img_c1, uv_c1, kinematic_tree)
cv2.imwrite('final_PC1.jpg', img_c1)

c_file2 = '%s/color/PC2/%s/color_%s.jpg' % (root_dir, filename, i)
img_c2 = (cv2.imread(c_file2)[:, ::-1, :]).astype(np.uint8)
pose_c2 = T_c2p.transform(pose)
uv_c2 = projection(pose_c2, param_c2)
img_c2 = draw_skeleton_opencv(img_c2, uv_c2, kinematic_tree)
cv2.imwrite('final_PC2.jpg', img_c2)

c_file3 = '%s/color/PC3/%s/color_%s.jpg' % (root_dir, filename, i)
img_c3 = (cv2.imread(c_file3)[:, ::-1, :]).astype(np.uint8)
pose_c3 = T_c3p.transform(pose)
uv_c3 = projection(pose_c3, param_c3)
img_c3 = draw_skeleton_opencv(img_c3, uv_c3, kinematic_tree)
cv2.imwrite('final_PC3.jpg', img_c3)
'''
