import argparse
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from tqdm import tqdm
from model_ADF import ADF_Net
from dataset_ADF import DehazeDataset_ADF
from Model_util import padding_image, Lap_Pyramid_Conv
from make import getTxt
from utils_test import to_psnr, to_ssim_skimage

def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tag = 'default'
    if args.type != 999:
        if args.type == 3:
            args.train_dir = os.path.join(script_dir, 'datasets_train', 'NH-HAZE')
            args.train_name = 'hazy,clean'
            args.test_dir = os.path.join(script_dir, 'datasets_test', 'NH-HAZE')
            args.test_name = 'hazy,clean'
            tag = 'nhhaze'
        elif args.type == 4:
            args.train_dir = os.path.join(script_dir, 'datasets_train', 'Dense_Hazy')
            args.train_name = 'hazy,clean'
            args.test_dir = os.path.join(script_dir, 'datasets_test', 'Dense_Hazy')
            args.test_name = 'hazy,clean'
            tag = 'dense'
    else:
        try:
            tag = os.path.basename(os.path.normpath(args.test_dir))
        except:
            tag = 'gui_test'
    if os.path.isfile(args.model_save_dir):
        model_dir = os.path.dirname(args.model_save_dir)
        model_file = os.path.basename(args.model_save_dir)
        if not hasattr(args, 'predict_result_root_dir'):
            args.predict_result_root_dir = os.path.join(model_dir, tag)
    else:
        model_dir = args.model_save_dir
        model_file = None
        if not hasattr(args, 'predict_result_root_dir'):
            args.predict_result_root_dir = os.path.join(args.model_save_dir, tag)
    predict_img_dir = os.path.join(args.predict_result_root_dir, 'predict')
    tran_dir = os.path.join(args.predict_result_root_dir, 'tran')
    atp_dir = os.path.join(args.predict_result_root_dir, 'atp')
    test_batch_size = args.test_batch_size
    if not os.path.isfile(args.model_save_dir) and not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.predict_result_root_dir):
        os.makedirs(args.predict_result_root_dir)
    if not os.path.exists(predict_img_dir):
        os.makedirs(predict_img_dir)
    if not os.path.exists(tran_dir):
        os.makedirs(tran_dir)
    if not os.path.exists(atp_dir):
        os.makedirs(atp_dir)
    device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]
    print('use gpus ->', args.gpus)
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    print('Initializing ADF-Net for testing...')
    model = ADF_Net(in_channels=3, out_channels=3, base_channels=32)
    print('ADF-Net parameters:', sum(param.numel() for param in model.parameters()))
    model = model.to(device)
    model.eval()
    laplace = Lap_Pyramid_Conv(num_high=1, device=device, color_high_freq=True).to(device)
    print('We are testing datasets: ', tag)
    getTxt(None, None, args.test_dir, args.test_name)
    test_dataset = DehazeDataset_ADF(args.test_dir, args.test_name, is_train=False, tag=tag)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    def load_best_model(model_save_dir):
        pkl_list = [i for i in os.listdir(model_save_dir) if '.pkl' in i]
        if not pkl_list:
            print(f"No model weights found in {model_save_dir}")
            return None
        best_psnr = -1.0
        best_model_name = None
        for pkl_file in pkl_list:
            try:
                parts = pkl_file.split('_')
                current_psnr = float(parts[2])
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_model_name = pkl_file
            except Exception as e:
                print(f"Error parsing pkl file name {pkl_file}: {e}")
                continue
        return best_model_name
    if model_file:
        model_path = args.model_save_dir
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'--- Model {model_file} loaded ---')
        else:
            print(f'--- Model file {model_path} not found ---')
            exit()
    elif args.num == 'best':
        model_name = load_best_model(model_dir)
        if model_name:
            model_path = os.path.join(model_dir, model_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'--- Best model {model_name} loaded ---')
        else:
            print('--- No best model found to load ---')
            exit()
    elif args.num != '9999999':
        pkl_list = [i for i in os.listdir(model_dir) if '.pkl' in i]
        name_candidates = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i]
        if len(name_candidates) > 0:
            model_path = os.path.join(model_dir, name_candidates[0])
            model.load_state_dict(
                torch.load(model_path,
                           map_location=device))
            print('--- {} epoch weight loaded ---'.format(args.num))
        else:
            print('--- Specified weight {} not found ---'.format(args.num))
            exit()
    else:
        print('--- No weight specified, exiting ---')
        exit()
    test_txt = open(os.path.join(args.predict_result_root_dir, 'result.txt'), 'w+')
    with torch.no_grad():
        psnr_list = []
        ssim_list = []
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
            J_E_padded, A_pred, T_pred = model(hazy_padded, I_h_padded)
            J_E = J_E_padded.data[:, :, ori_top:ori_down, ori_left:ori_right]
            A_pred = A_pred.data
            T_pred = T_pred.data[:, :, ori_top:ori_down, ori_left:ori_right]
            imwrite(J_E, os.path.join(predict_img_dir, name[0]), normalize=True, value_range=(0, 1))
            imwrite(T_pred, os.path.join(tran_dir, name[0]), normalize=True, value_range=(0, 1))
            current_psnr = to_psnr(J_E, clean)[0]
            current_ssim = to_ssim_skimage(J_E, clean)[0]
            psnr_list.append(current_psnr)
            ssim_list.append(current_ssim)
            print(f'{name[0]} ->\tpsnr: {current_psnr:.4f}\tssim: {current_ssim:.4f}')
            test_txt.writelines(f'{name[0]} ->\tpsnr: {current_psnr:.4f}\tssim: {current_ssim:.4f}\n')
        avr_psnr = sum(psnr_list) / len(psnr_list)
        avr_ssim = sum(ssim_list) / len(ssim_list)
        test_txt.writelines(f'{tag} datasets ==>>\tpsnr: {avr_psnr:.4f}\tssim: {avr_ssim:.4f}\n')
        print(f'\nFinal Average for {tag} datasets: PSNR: {avr_psnr:.4f}, SSIM: {avr_ssim:.4f}')
    test_txt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADF-Net Testing')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_save_dir', type=str, default='./output_result_ADF_ColorHFEM', help='Model save directory or specific model file path')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
    parser.add_argument('--test_dir', type=str, default='D:/Dehaze/UME-NET/UME-Net-train/datasets_test/Outdoor/test/')
    parser.add_argument('--test_name', type=str, default='hazy,clean')
    parser.add_argument('--num', type=str, default='best', help='Specific epoch to load, or \'best\' for latest best')
    parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
    parser.add_argument("--type", default=5, type=int, help="choose a type 0-11")
    args = parser.parse_args()
    main(args)