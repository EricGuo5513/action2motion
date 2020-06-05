import imageio
import os
from PIL import Image
import images2gif
import numpy as np
import matplotlib.pyplot as plt
import glob


def project_3d_to_2d(verts):
    '''
    project 3d points to original 2d coordinate space.
    Input:
        cam: (1, 3) camera parameters (f, cx, cy) output by model.
        verts: 3d verts output by model.
        proc_param: preprocessing parameters. this is for converting points from crop (model input) to original image.
    Output:
    '''
    cam_f = np.array([1.0591841e+03, 1.0594705e+03])
    cam_c = np.array([9.7817145e+02, 5.6321869e+02])
    fx = cam_f[0]
    fy = cam_f[1]
    tx = cam_c[0]
    ty = cam_c[1]

    verts = verts.reshape(-1, 3)
    verts2d = np.zeros((verts.shape[0], 2))
    # print(verts2d.shape)
    verts2d[:, 0] = fx * verts[:, 0] / verts[:, 2] + tx
    verts2d[:, 1] = fy * verts[:, 1] / verts[:, 2] + ty

    return verts2d


def compute_videocrop_bbox(motion_2d, extend, w_h_r, img_border, thresold=20):
    motion_2d = np.array(motion_2d, dtype=np.float)
    h_min = np.min(motion_2d[:, :, 0])
    h_max = np.max(motion_2d[:, :, 0])
    v_min = np.min(motion_2d[:, :, 1])
    v_max = np.max(motion_2d[:, :, 1])
    v_max = img_border[1] if v_max >= img_border[1] else v_max
    crop_size = [0, 0]
    crop_size[0] = max(h_max - h_min, (v_max - v_min) * w_h_r) + 2 * extend
    crop_size[1] = crop_size[0] / w_h_r
    h_gap = crop_size[0] - h_max + h_min
    v_gap = crop_size[1] - v_max + v_min
    print(crop_size)
    print('h_max:{0:.2f} h_min:{1:.2f} v_max:{2:.2f} v_min:{3:.2f}'.format(h_max, h_min, v_max, v_min))
    print('h_gap:{0}\t\t v_gap:{1}'.format(h_gap, v_gap))
    assert h_gap > thresold
    assert v_gap > thresold

    def get_crop_box(min, max, gap, up_bd):
        if min - gap/2 >= 0 and max + gap/2 <= up_bd:
            l_t = min - gap/2
            r_b = max + gap/2
        elif min - gap/2 < 0:
            l_t = 0
            r_b = max + gap - min
        elif max + gap/2 > up_bd:
            l_t = min + up_bd - max - gap
            r_b = up_bd
        return l_t, r_b
    l_t_x, r_b_x = get_crop_box(h_min, h_max, h_gap, img_border[0])
    l_t_y, r_b_y = get_crop_box(v_min, v_max, v_gap, img_border[1])
    crop_bbox = (l_t_x, l_t_y, r_b_x, r_b_y)
    return crop_bbox, crop_size

def crop_and_resize_motion(motion_2d, crop_bbox, origin_size, final_size, joints_num=24):
    rz_ratio =  final_size[0] / origin_size[0] * 1.0
    l_t_trans = np.array([crop_bbox[0:2]])
    motion_2d = np.array(motion_2d).reshape(-1, joints_num, 2)
    # print(l_t_trans)
    # print(rz_ratio)
    # print(motion_2d[0, :, :])
    motion_2d_t = motion_2d - l_t_trans
    # print(motion_2d_t[0, :, :])
    motion_2d_t_r = motion_2d_t * rz_ratio
    # print(motion_2d_t_r[0, :, :])
    return motion_2d_t_r

def compose_gif_img_dir_2(img_dir, filename, fps=5):
    img_paths = os.listdir(img_dir)
    gif_images = []
    for path in img_paths:
        # gif_images.append(imageio.imread(os.path.join(img_dir, path)))
        gif_images.append(Image.open(os.path.join(img_dir, path)))
        print(path)
    imageio.mimsave(filename, gif_images, fps=fps)


def compose_gif_img_dir(img_dir, filename, fps=5):
    filenames = os.listdir(img_dir)
    frames = []
    for image_name in filenames:
        im = Image.open(os.path.join(img_dir, image_name)).convert('RGB')
        im = np.array(im)
        plt.imshow(im)
        plt.show()
        frames.append(im)
    images2gif.writeGif(filename, frames, duration=0.5, subRectangles=False)


def compose_gif_img_dir_3(fp_in, fp_out, duration):
    s_f = sorted(glob.glob(fp_in))
    print(s_f)
    img, *imgs = [Image.open(f) for f in s_f]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)

def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)

if __name__ == '__main__':
    img_dir = "/home/chuan/motion2video/result/video/unet_with_depth_res_II_l1.2/P10G01R03C02F0832/P10G01R03C02F0832_P10G01R03C02F1219*.jpg"
    save_name = "/home/chuan/motion2video/result/video/unet_with_depth_res_II_l1.2/P10G01R03C02F0832/P10G01R03C02F0832_P10G01R03C02F1219.gif"
    compose_gif_img_dir_3(img_dir, save_name, 0.5)