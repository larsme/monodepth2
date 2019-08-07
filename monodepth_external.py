# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import torch
import PIL.Image as pil
from torchvision import transforms, datasets
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def load_net( model_name=None, use_cuda=True):
    '''
    loads a monocular depth prediction model
    :param model_name: choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"]
    :return:
    model
    '''
    assert model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return MonodepthExternal(feed_width, feed_height, device, encoder, depth_decoder, min_depth=0.1, max_depth=100)


class MonodepthExternal:
    def __init__(self, feed_width, feed_height, device, encoder, depth_decoder, min_depth, max_depth):
        self.feed_width = feed_width
        self.feed_height = feed_height
        self.device = device
        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.min_depth = min_depth
        self.max_depth = max_depth

    def return_one_prediction(self, input_image, post_process):
        with torch.no_grad():
            input_image = pil.fromarray(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            if post_process:
                # Post-processed results require each image to have two forward passes
                input_image = torch.cat((input_image, torch.flip(input_image, [3])), 0)
            output = self.depth_decoder(self.encoder(input_image))

            pred_scaled_disp, _ = disp_to_depth(output[("disp", 0)], self.min_depth, self.max_depth)

            if post_process:
                N = pred_scaled_disp.shape[0] // 2
                pred_scaled_disp = batch_post_process_disparity(pred_scaled_disp[:N], pred_scaled_disp[N:, :, ::-1])

            pred_scaled_disp_resized = torch.nn.functional.interpolate(
                pred_scaled_disp, (original_height, original_width), mode="bilinear", align_corners=False)

            pred_depth = 1 / pred_scaled_disp_resized
            return np.squeeze(pred_depth.cpu()[:, 0].numpy())