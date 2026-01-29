from math import log10

import torch
import torch.nn.functional as F
import torchvision
from skimage.metrics import structural_similarity as ssim


def to_psnr(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]

    ssim_list = []
    for ind in range(len(dehaze_list_np)):
        pred = dehaze_list_np[ind]
        ref = gt_list_np[ind]
        h, w = pred.shape[:2]
        min_side = min(h, w)
        if min_side >= 7:
            win_size = 7
        elif min_side >= 5:
            win_size = 5
        elif min_side >= 3:
            win_size = 3
        else:
            ssim_list.append(1.0)
            continue
        try:
            val = ssim(pred, ref, data_range=1, channel_axis=-1, win_size=win_size)
        except TypeError:
            val = ssim(pred, ref, data_range=1, multichannel=True, win_size=win_size)
        ssim_list.append(val)

    return ssim_list


def predict(gridnet, test_data_loader):
    psnr_list = []
    for batch_idx, (frame1, frame2, frame3) in enumerate(test_data_loader):
        with torch.no_grad():
            frame1 = frame1.to(torch.device('cuda'))
            frame3 = frame3.to(torch.device('cuda'))
            gt = frame2.to(torch.device('cuda'))
                           

            frame_out = gridnet(frame1, frame3)
                              
            frame_debug = torch.cat((frame1, frame_out, gt, frame3), dim=0)
            filepath = "./image" + str(batch_idx) + '.png'
            torchvision.utils.save_image(frame_debug, filepath)
                              
                                                             

                                         

                                              
        psnr_list.extend(to_psnr(frame_out, gt))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    return avr_psnr
