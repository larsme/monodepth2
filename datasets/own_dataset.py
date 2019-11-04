import glob
import numpy as np
from .mono_dataset import MonoDataset
import os
import PIL.Image as pil
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2


class OwnDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, train_to_val_ratio=0.7, assign_only_true_matches=True, min_depth=0, *args, **kwargs):
        super(OwnDataset, self).__init__(*args, **kwargs)
        self.img_ext = '.png'

        if self.check_depth():
            self.frame_idxs = [0]

        self.depth_dir = "../../../../../DL_Datasets/Leaf_Log_18102019/point_cloud_full_rotation"
        self.rgb_dir = "../../../../../DL_Datasets/Leaf_Log_18102019/Cameras/camera_18497292"
        calib_dir = "../../../../../DL_Datasets/Leaf_Log_18102019"

        self.intrinsics = np.loadtxt(calib_dir+'/UndistortedIntrinsics').astype(np.float32)
        self.K = np.array([[self.intrinsics[0, 0], self.intrinsics[0, 1], self.intrinsics[0, 2], 0],
                           [self.intrinsics[1, 0], self.intrinsics[1, 1], self.intrinsics[1, 2], 0],
                           [self.intrinsics[2, 0], self.intrinsics[2, 1], self.intrinsics[2, 2], 0],
                           [0,                     0,                     0,                     1]], dtype=np.float32)
        self.rvec = np.loadtxt(calib_dir+'/ExtrinsicsRVec')
        self.tvec = np.loadtxt(calib_dir+'/ExtrinsicsTVec')
        self.rgb_delay=6
        self.min_depth = min_depth

        self.full_res_shape = (2048, 1536)

        self.rgb_filenames, self.depth_filenames = self.get_filenames(train_to_val_ratio, assign_only_true_matches)

    def get_color_from_file(self, idx, do_flip):
        color = pil.fromarray(cv2.imread(self.rgb_filenames[idx]))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth_from_file(self, idx, do_flip):

        # X Y Z Intensity Reflectivity Noise Range Ring ... (repeat)
        # to x y z per line

        lidar_scan = np.fromfile(self.depth_filenames[idx], dtype=np.float32).reshape(-1, 9)
        points = lidar_scan[:, :3]

        rot = np.array(cv2.Rodrigues(self.rvec)[0])

        projectedPoints = np.dot(self.intrinsics, (rot.dot(points.transpose()) + np.expand_dims(self.tvec, axis=1)))
        depths = projectedPoints[2, :]
        val = depths > self.min_depth
        depths = depths[val]
        u = np.round(projectedPoints[0, val] / self.full_res_shape[0] * self.width / depths).astype(np.int_)
        v = np.round(projectedPoints[1, val] / self.full_res_shape[1] * self.height / depths).astype(np.int_)
        val = (u >= 0) & (v >= 0) & (u < self.width) & (v < self.height)

        depths = depths[val]
        v = v[val]
        u = u[val]
        if do_flip:
            u = self.width-1-u

        depth_map = np.zeros((self.height, self.width), np.float)
        for i in range(np.array(u).shape[0]):
            d = depth_map[v[i], u[i]]
            if d == 0 or d > depths[i]:
                depth_map[v[i], u[i]] = depths[i]

        return depth_map

    def __len__(self):
        return len(self.rgb_filenames)

    def get_filenames(self, train_to_val_ratio, assign_only_true_matches):
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        inputs['index'] = torch.Tensor([[[[index]]]])

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            if i == "s":
                raise NotImplementedError
            else:
                inputs[("color", i, -1)] = self.get_color_from_file(index + i, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth_from_file(index, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        stop = False
        for l, v in inputs.items():
            if torch.isnan(v).any():
                s = ''
                for var in l:
                    s += str(var)
                s += " is NaN for item "+str(inputs['index'].squeeze().numpy())
                raise Warning(s)
                stop = True
        assert not stop

        return inputs


class OwnUnsupervisedTrainDataset(OwnDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OwnUnsupervisedTrainDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return False

    def __len__(self):
        return len(self.rgb_filenames) - max(self.frame_idxs) + min(self.frame_idxs)

    def __getitem__(self, index):
        return super(OwnUnsupervisedTrainDataset, self).__getitem__(index + min(self.frame_idxs))

    def get_filenames(self, train_to_val_ratio, assign_only_true_matches):
        rgb_paths = list(sorted(glob.iglob(self.rgb_dir + "/*.png")))
        rgb_paths = rgb_paths[430:]
        num = rgb_paths.__len__()
        train_num = round(num * train_to_val_ratio)

        train_rgb_paths = rgb_paths[:train_num]

        return train_rgb_paths, None


class OwnSupervisedTrainDataset(OwnDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OwnSupervisedTrainDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return True

    def get_filenames(self, train_to_val_ratio, assign_only_true_matches):

        rgb_paths = list(sorted(glob.iglob(self.rgb_dir + "/*.png")))
        num = rgb_paths.__len__()
        train_num = round(num * train_to_val_ratio)

        depth_paths = list(sorted(glob.iglob(self.depth_dir + "/*.bin")))

        num_rgb = int(rgb_paths[0].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
        num_depth = int(depth_paths[0].split(self.depth_dir + '/')[1].split('.bin')[0])
        i_rgb = 0
        i_depth = 0

        train_rgb_paths = []
        train_depth_paths = []
        while True:
            if num_depth < num_rgb:
                i_depth += 1
                if i_depth < depth_paths.__len__():
                    num_prev_depth = num_depth
                    num_depth = int(depth_paths[i_depth].split(self.depth_dir + '/')[1].split('.bin')[0])
                    if i_rgb < train_num and num_depth > num_rgb and not assign_only_true_matches:
                        train_rgb_paths.append(rgb_paths[i_rgb])
                        if abs(num_depth - num_rgb) <= abs(num_prev_depth - num_rgb):
                            train_depth_paths.append(depth_paths[i_depth])
                        else:
                            train_depth_paths.append(depth_paths[i_depth - 1])
                else:
                    break
            elif num_rgb < num_depth:
                i_rgb += 1
                if i_rgb < rgb_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
                else:
                    break
            else:
                if i_rgb < train_num:
                    train_rgb_paths.append(rgb_paths[i_rgb])
                    train_depth_paths.append(depth_paths[i_depth])
                i_rgb += 1
                i_depth += 1
                if i_rgb < rgb_paths.__len__() and i_depth < depth_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
                    num_depth = int(depth_paths[i_rgb].split(self.depth_dir + '/')[1].split('.bin')[0])
                else:
                    break

        return train_rgb_paths, train_depth_paths


class OwnSupervisedEvalDataset(OwnDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OwnSupervisedEvalDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return True

    def get_filenames(self, train_to_val_ratio, assign_only_true_matches):

        rgb_paths = list(sorted(glob.iglob(self.rgb_dir + "/*.png")))
        num = rgb_paths.__len__()
        train_num = round(num * train_to_val_ratio)

        depth_paths = list(sorted(glob.iglob(self.depth_dir + "/*.bin")))
        num_depth = int(depth_paths[0].split(self.depth_dir + '/')[1].split('.bin')[0])
        num_rgb = int(rgb_paths[train_num].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
        i_rgb = train_num
        i_depth = 0

        rgb_eval_paths = []
        depth_eval_paths = []
        while True:
            if num_depth < num_rgb:
                i_depth += 1
                if i_depth < depth_paths.__len__():
                    num_prev_depth = num_depth
                    num_depth = int(depth_paths[i_depth].split(self.depth_dir + '/')[1].split('.bin')[0])
                    if num_depth > num_rgb and not assign_only_true_matches:
                        rgb_eval_paths.append(rgb_paths[i_rgb])
                        if abs(num_depth - num_rgb) <= abs(num_prev_depth - num_rgb):
                            depth_eval_paths.append(depth_paths[i_depth])
                        else:
                            depth_eval_paths.append(depth_paths[i_depth - 1])
                else:
                    break
            elif num_rgb < num_depth:
                i_rgb += 1
                if i_rgb < rgb_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
                else:
                    break
            else:
                rgb_eval_paths.append(rgb_paths[i_rgb])
                depth_eval_paths.append(depth_paths[i_depth])
                i_rgb += 1
                i_depth += 1
                if i_rgb < rgb_paths.__len__() and i_depth < depth_paths.__len__():
                    num_rgb = int(rgb_paths[i_rgb].split(self.rgb_dir + '/')[1].split('.png')[0])+self.rgb_delay
                    num_depth = int(depth_paths[i_rgb].split(self.depth_dir + '/')[1].split('.bin')[0])
                else:
                    break

        return rgb_eval_paths, depth_eval_paths
