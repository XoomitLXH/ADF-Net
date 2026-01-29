import math
import os
import ssl
import time
import argparse
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image as imwrite
from model_ADF import ADF_Net
from dataset_ADF import DehazeDataset_ADF
from CR import ContrastLoss
from Model import Discriminator
from Model_util import padding_image, Lap_Pyramid_Conv
from make import getTxt
from Loss import SSIMLoss
from utils_test import to_psnr, to_ssim_skimage
ssl._create_default_https_context = ssl._create_unverified_context


def tv_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (diff_h.abs().mean() + diff_w.abs().mean())


LAMBDA_L1 = 1.0           
LAMBDA_SSIM = 0.45        
LAMBDA_ATM = 0.35         
LAMBDA_TV = 0.1           
LAMBDA_ID = 1.0           
LAMBDA_CONTRAST = 0.1     


parser = argparse.ArgumentParser(description='ADF-Net Dehazing Network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=60000, type=int)
parser.add_argument("--type", default=5, type=int, help="choose a type 0-11")
parser.add_argument('--train_dir', type=str, default='D:/Dehaze/UME-NET/UME-Net-train/datasets_train/Outdoor/train/')
parser.add_argument('--train_name', type=str, default='hazy,clean')
parser.add_argument('--test_dir', type=str, default='D:/Dehaze/UME-NET/UME-Net-train/datasets_test/Outdoor/test/')
parser.add_argument('--test_name', type=str, default='hazy,clean')
parser.add_argument('--model_save_dir', type=str, default='./output_result_ADF_ColorHFEM')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument('--restart', action='store_true', help='')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--sep', type=int, default='100', help='')
parser.add_argument('--save_psnr', action='store_true', help='')
parser.add_argument('--seps', action='store_true', help='')
parser.add_argument('--warmup', type=int, default=1000, help='warmup epochs without adv/contrast')
parser.add_argument('--adv_w', type=float, default=0.2, help='adversarial loss weight after warmup')
parser.add_argument('--contrast_w', type=float, default=0.1, help='contrastive loss weight after warmup')
parser.add_argument('--tv_w', type=float, default=1.0, help='TV regularization weight for T (与 λ4 一起控制 TV 强度)')
parser.add_argument('--stage1_epochs', type=int, default=1000, help='stage 1: epochs without adversarial loss')
parser.add_argument('--stage1_contrast_w', type=float, default=0.1, help='contrastive loss weight in stage 1')
parser.add_argument('--stage2_adv_w', type=float, default=0.03, help='adversarial loss weight in stage 2')
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
start_epoch = 0
sep = args.sep
tag = 'else'
if args.type == 0:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'Remote_Thin')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'Remote_Thin')
    args.test_name = 'hazy,clean'
    tag = 'Remote_Thin'
elif args.type == 1:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'Remote_Moderate')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'Remote_Moderate')
    args.test_name = 'hazy,clean'
    tag = 'Remote_Moderate'
elif args.type == 2:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'Remote_Thick')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'Remote_Thick')
    args.test_name = 'hazy,clean'
    tag = 'Remote_Thick'
elif args.type == 3:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'NH-HAZE')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'NH-HAZE')
    args.test_name = 'hazy,clean'
    tag = 'NH-HAZE'
elif args.type == 4:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'Dense_Hazy')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'Dense_Hazy')
    args.test_name = 'hazy,clean'
    tag = 'dense'
elif args.type == 5:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'Middleburry')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'Middleburry')
    args.test_name = 'hazy,clean'
    tag = 'Middleburry'
elif args.type == 6:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'I-HAZY')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'I-HAZY')
    args.test_name = 'hazy,clean'
    tag = 'I-HAZY'
elif args.type == 7:
    args.train_dir = os.path.join(script_dir, 'datasets_train', 'O-HAZY')
    args.train_name = 'hazy,clean'
    args.test_dir = os.path.join(script_dir, 'datasets_test', 'O-HAZY')
    args.test_name = 'hazy,clean'
    tag = 'O-HAZY'
if __name__ == '__main__':
    print('We are training datasets: ', tag)
    getTxt(args.train_dir, args.train_name, args.test_dir, args.test_name)
    test_image_name = args.test_name.split(',')[0] if ',' in args.test_name else args.test_name
    new_model_save_dir = os.path.join('./output_result_ADF_ColorHFEM', f'{tag}_{test_image_name}')
    args.model_save_dir = new_model_save_dir
    print(f'模型保存路径: {args.model_save_dir}')
    predict_result_dir = os.path.join(args.model_save_dir, 'predict_result')
    test_batch_size = args.test_batch_size
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(predict_result_dir):
        os.makedirs(predict_result_dir)
    device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]
    print('use gpus ->', args.gpus)
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print('Initializing ADF-Net...')
    model = ADF_Net(in_channels=3, out_channels=3, base_channels=32)
    print('ADF-Net parameters:', sum(param.numel() for param in model.parameters()))
    discriminator = Discriminator()
    discriminator1 = Discriminator()
    criterionSsim = SSIMLoss()
    criterion = torch.nn.MSELoss()
    criterionP = torch.nn.L1Loss()
    criterionC = ContrastLoss(device=device, ablation=False)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    optimizer_T = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.0001}
    ])
    model.to(device)
    criterion.to(device)
    criterionP.to(device)
    discriminator.to(device)
    discriminator1.to(device)
    criterionSsim.to(device)
    scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_T, T_max=args.train_epoch, eta_min=1e-6, last_epoch=-1)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.train_epoch, eta_min=1e-6, last_epoch=-1)
    dataset = DehazeDataset_ADF(args.train_dir, args.train_name, is_train=True, tag=tag)
    print('trainDataset len: ', len(dataset))
    train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True,
                              num_workers=4)
    test_dataset = DehazeDataset_ADF(args.test_dir, args.test_name, is_train=False, tag=tag)
    print('testDataset len: ', len(test_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)
    print(f'Training on {tag} dataset with 4 workers')
    laplace = Lap_Pyramid_Conv(num_high=1, device=device, color_high_freq=True).to(device)
    writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))
    if args.restart:
        pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
        if len(pkl_list) > 0:
            num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
            name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
            model.load_state_dict(
                torch.load(os.path.join(args.model_save_dir, name),
                           map_location="cuda:{}".format(device_ids[0])))
            print('--- {} epoch weight loaded ---'.format(num))
            start_epoch = int(num) + 1
        else:
            print('--- no weight found for restart ---')
    elif args.num != '9999999':
        pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
        name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i]
        if len(name) > 0:
            name = name[0]
            model.load_state_dict(
                torch.load(os.path.join(args.model_save_dir, name),
                           map_location="cuda:{}".format(device_ids[0])))
            print('--- {} epoch weight loaded ---'.format(args.num))
            start_epoch = int(args.num) + 1
        else:
            print('--- specified weight {} not found ---'.format(args.num))
    else:
        print('--- no weight loaded ---')
    iteration = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    pl = []
    sl = []
    best_psnr = 0
    best_psnr_ssim = 0
    best_ssim = 0
    best_ssim_psnr = 0
    print()
    start_time = time.time()
    for epoch in range(start_epoch, train_epoch):
        if epoch == 0:
            weight_config = f"统一损失权重 (λ1/L1:{LAMBDA_L1}, λ2/SSIM:{LAMBDA_SSIM}, λ3/ATM:{LAMBDA_ATM}, λ4/TV:{LAMBDA_TV}, λ5/Identity:{LAMBDA_ID}, λ6/Contrast:{LAMBDA_CONTRAST})"
            print(f"\n=== 开始训练: 使用 L1 + SSIM + ATM + Contrast + TV + Identity 组合损失 ===")
            print(f"=== 数据集类型 {args.type} ({tag}): {weight_config} ===")
        model.train()
        discriminator.train()
        discriminator1.train()
        with tqdm(total=len(train_loader)) as t:
            for (hazy, clean, hazy_high_freq, _) in train_loader:
                iteration += 1
                hazy = hazy.to(device)
                clean = clean.to(device)
                hazy_high_freq = hazy_high_freq.to(device)
                I_h = laplace.pyramid_decom_color(hazy_high_freq)
                real_label = torch.ones((hazy.size()[0], 1, 30, 30), requires_grad=False).to(device)
                fake_label = torch.zeros((hazy.size()[0], 1, 30, 30), requires_grad=False).to(device)
                optimizer_D.zero_grad()
                real_out = discriminator(clean)
                real_out1 = discriminator1(clean)
                loss_real_D = criterion(real_out, real_label)
                loss_real_D1 = criterion(real_out1, real_label)
                J_E, A_pred, T_pred = model(hazy, I_h)
                fake_out = discriminator(J_E.detach())
                p_de_out = discriminator1(J_E.detach())
                loss_fake_D = criterion(fake_out, fake_label)
                loss_fake_D1 = criterion(p_de_out, fake_label)
                loss_D = (loss_real_D + loss_fake_D + loss_fake_D1) / 3
                loss_D.backward()
                optimizer_D.step()
                optimizer_T.zero_grad()
                J_E, A_pred, T_pred = model(hazy, I_h)
                if iteration % 20 == 0:
                    with torch.no_grad():
                        ch_mean = J_E.detach().mean(dim=[0, 2, 3]).tolist()
                        ch_std = J_E.detach().std(dim=[0, 2, 3]).tolist()
                    print("J_E channel mean/std:", [round(m, 4) for m in ch_mean], [round(s, 4) for s in ch_std])
                output_D_fake = discriminator(J_E)
                output_D1_fake = discriminator1(J_E)
                                           
                use_adv = 0.0
                                              
                use_contrast = LAMBDA_CONTRAST
                loss_G_adv = criterion(output_D_fake, real_label) * use_adv
                loss_G_adv1 = criterion(output_D1_fake, real_label) * use_adv
                loss_L1 = criterionP(J_E, clean)
                loss_ssim = 1 - criterionSsim(J_E, clean)
                if T_pred.shape[-2:] != J_E.shape[-2:]:
                    T_pred = F.interpolate(T_pred, size=J_E.shape[-2:], mode='bilinear', align_corners=False)
                reconstructed_hazy = T_pred * J_E + A_pred.unsqueeze(-1).unsqueeze(-1) * (1 - T_pred)
                loss_atm_model = criterionP(reconstructed_hazy, hazy)
                                                          
                loss_tv = tv_loss(T_pred) * (LAMBDA_TV * args.tv_w)
                loss_contrast = criterionC(J_E, clean, hazy, reconstructed_hazy) * use_contrast
                with torch.no_grad():
                    J_identity, _, _ = model(clean, clean)
                loss_identity = criterionP(J_identity, clean)
                total_loss_G = (loss_G_adv + loss_G_adv1 +
                              LAMBDA_L1 * loss_L1 +
                              LAMBDA_SSIM * loss_ssim +
                              LAMBDA_ATM * loss_atm_model +
                              loss_contrast +                    
                              loss_tv +                                 
                              LAMBDA_ID * loss_identity)
                total_loss_G.backward()
                optimizer_T.step()
                t.set_description(
                    "Epoch[{}] | D_loss: {:.4f} | G_loss: {:.4f} | L1: {:.4f} | SSIM: {:.4f} | ATM: {:.4f} | Contrast: {:.4f} | Identity: {:.4f}".format(
                        epoch, loss_D.item(), total_loss_G.item(), loss_L1.item(), loss_ssim.item(), loss_atm_model.item(), loss_contrast.item(), loss_identity.item()))
                t.update(1)
                writer.add_scalars('training', {'D_loss': loss_D.item(), 'G_loss': total_loss_G.item()},
                                   iteration)
                writer.add_scalars('training_losses', {'L1_loss': loss_L1.item(), 'SSIM_loss': loss_ssim.item(),
                                                        'ATM_loss': loss_atm_model.item(), 'TV_T': (loss_tv.item() if 'loss_tv' in locals() else 0.0), 'Contrast_loss': (loss_contrast / max(use_contrast, 1e-8) if use_contrast>0 else 0)},
                                   iteration)
        scheduler_T.step()
        scheduler_D.step()
        if args.seps:
            torch.save(model.state_dict(),
                       os.path.join(args.model_save_dir,
                                    'epoch_' + str(epoch) + '_' + '.pkl'))
            continue
        if epoch % sep == 0:
            with torch.no_grad():
                psnr_list = []
                ssim_list = []
                model.eval()
                for (hazy, clean, hazy_high_freq, name) in tqdm(test_loader):
                    hazy = hazy.to(device)
                    clean = clean.to(device)
                    hazy_high_freq = hazy_high_freq.to(device)
                    I_h = laplace.pyramid_decom_color(hazy_high_freq)
                    h, w = hazy.shape[2], hazy.shape[3]
                    max_h = int(math.ceil(h / 256)) * 256
                    max_w = int(math.ceil(w / 256)) * 256
                    hazy_padded, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)
                    I_h_padded, _, _, _, _ = padding_image(I_h, max_h, max_w)
                    J_E_padded, A_pred_test, T_pred_test = model(hazy_padded, I_h_padded)
                    if T_pred_test.shape[-2:] != J_E_padded.shape[-2:]:
                        T_pred_test = F.interpolate(T_pred_test, size=J_E_padded.shape[-2:], mode='bilinear', align_corners=False)
                    J_E = J_E_padded.data[:, :, ori_top:ori_down, ori_left:ori_right]
                    T_pred_test = T_pred_test.data[:, :, ori_top:ori_down, ori_left:ori_right]
                    output_img_path = os.path.join(predict_result_dir, name[0])
                    imwrite(J_E, output_img_path, normalize=False)
                    psnr_list.extend(to_psnr(J_E, clean))
                    ssim_list.extend(to_ssim_skimage(J_E, clean))
                    t_path = os.path.join(predict_result_dir, f"T_{name[0]}")
                    imwrite(T_pred_test, t_path, normalize=True, value_range=(0, 1))
                avr_psnr = sum(psnr_list) / len(psnr_list)
                avr_ssim = sum(ssim_list) / len(ssim_list)
                pl.append(avr_psnr)
                sl.append(avr_ssim)
                if avr_psnr >= best_psnr:
                    best_psnr = avr_psnr
                    best_epoch_psnr = epoch
                    best_psnr_ssim = avr_ssim
                if avr_ssim >= best_ssim:
                    best_ssim = avr_ssim
                    best_epoch_ssim = epoch
                    best_ssim_psnr = avr_psnr
                print(epoch, 'dehazed', avr_psnr, avr_ssim)
                print('Best PSNR epoch: {}, PSNR: {:.4f}, SSIM: {:.4f}'.format(best_epoch_psnr, best_psnr, best_psnr_ssim))
                print('Best SSIM epoch: {}, PSNR: {:.4f}, SSIM: {:.4f}'.format(best_epoch_ssim, best_ssim_psnr, best_ssim))
                print()
                writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                               'testing ssim': avr_ssim
                                               }, epoch)
                torch.save(model.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                        round(avr_ssim, 3)) + '.pkl'))
        if epoch % 2000 == 0:
            pass
    writer.close()