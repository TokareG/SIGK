import os
import pandas as pd
from torchvision.io import read_image
from .metrics import *
import numpy as np
from tqdm import tqdm



def mean_std(results_array):
    mean = np.mean(results_array, axis=0)
    std = np.std(results_array, axis=0)
    return mean, std

def compare_all_denoising_results(target_path, noised_path, dncnn_path, bilateral_path):
    assert os.path.exists(target_path)
    assert os.path.exists(noised_path)
    assert os.path.exists(dncnn_path)
    assert os.path.exists(bilateral_path)

    target_files = os.listdir(target_path)
    noised_files = os.listdir(noised_path)
    dncnn_files = os.listdir(dncnn_path)
    bilateral_files = os.listdir(bilateral_path)

    assert len(target_files) == len(noised_files) and len(target_files) == len(bilateral_files) and len(target_files) == len(dncnn_files)
    records = []
    SNE_results = np.zeros((len(target_files), 3), dtype=np.float32)
    PSNR_results = np.zeros((len(target_files), 3), dtype=np.float32)
    SSIM_results = np.zeros((len(target_files), 3), dtype=np.float32)
    LPIPS_results = np.zeros((len(target_files), 3), dtype=np.float32)

    for idx, files in enumerate(tqdm(zip(target_files, noised_files, dncnn_files, bilateral_files), total=len(target_files), desc="Denoising methods comparison")):
        target_file, noised_file, dncnn_file, bilateral_file = files
        target_img_path = os.path.join(target_path, target_file)
        noised_img_path = os.path.join(noised_path, noised_file)
        dncnn_img_path = os.path.join(dncnn_path, dncnn_file)
        bilateral_img_path = os.path.join(bilateral_path, bilateral_file)

        target_img = read_image(target_img_path).float() / 255.
        noised_img = read_image(noised_img_path).float() / 255.
        dncnn_img = read_image(dncnn_img_path).float() / 255.
        bilateral_img = read_image(bilateral_img_path).float() / 255.

        bilateral_sne = SNE(bilateral_img, target_img)
        dncnn_sne = SNE(dncnn_img, target_img)
        noisy_sne = SNE(noised_img, target_img)

        bilateral_psnr = PSNR(bilateral_img, target_img)
        dncnn_psnr = PSNR(dncnn_img, target_img)
        noised_psnr = PSNR(noised_img, target_img)

        bilateral_ssim = SSIM(bilateral_img, target_img)
        dncnn_ssim = SSIM(dncnn_img, target_img)
        noisy_ssim = SSIM(noised_img, target_img)

        bilateral_lpips = LPIPS(bilateral_img, target_img)
        dncnn_lpips = LPIPS(dncnn_img, target_img)
        noisy_lpips = LPIPS(noised_img, target_img)

        SNE_results[idx] = [noisy_sne, dncnn_sne, bilateral_sne]
        PSNR_results[idx] = [noised_psnr, dncnn_psnr, bilateral_psnr]
        SSIM_results[idx] = [noisy_ssim, dncnn_ssim, bilateral_ssim]
        LPIPS_results[idx] = [noisy_lpips, dncnn_lpips, bilateral_lpips]

        if idx < 5:
            records.append({"Target image": target_file,
                            "DnCNN SNE": dncnn_sne.item(),
                            "Bilateral SNE": bilateral_sne.item(),
                            "Nosied PSNR": noised_psnr.item(),
                            "DNCNN PSNR": dncnn_psnr.item(),
                            "Bilateral PSNR": bilateral_psnr.item(),
                            "DnCNN SSIM": dncnn_ssim.item(),
                            "Bilateral SSIM": bilateral_ssim.item(),
                            "DnCNN LPIPS": dncnn_lpips.item(),
                            "Bilateral LPIPS": bilateral_lpips.item()
                            })

    mean_sne, std_sne = mean_std(SNE_results)
    mean_psnr, std_psnr = mean_std(PSNR_results)
    mean_ssim, std_ssim = mean_std(SSIM_results)
    mean_lpips, std_lpips = mean_std(LPIPS_results)

    records_df = pd.DataFrame(records)

    methods = ["Noisy", "DnCNN", "Bilateral"]

    summary_df = pd.DataFrame({
        "Method": methods,
        "Mean SNE": mean_sne,
        "Std SNE": std_sne,
        "Mean PSNR": mean_psnr,
        "Std PSNR": std_psnr,
        "Mean SSIM": mean_ssim,
        "Std SSIM": std_ssim,
        "Mean LPIPS": mean_lpips,
        "Std LPIPS": std_lpips
    })

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(records_df)
    print()
    print(summary_df)

def compare_inpainting(target_path, output_path, compare_path):
    assert os.path.exists(target_path)
    assert os.path.exists(output_path)
    assert os.path.exists(compare_path)

    target_files = os.listdir(target_path)
    output_files = os.listdir(output_path)
    compare_files = os.listdir(compare_path)

    assert len(target_files) == len(output_files) and len(target_files) == len(compare_files)

    sne_results = torch.zeros(len(target_files), 2)
    psnr_results = torch.zeros(len(target_files), 2)
    ssim_results = torch.zeros(len(target_files), 2)
    lpips_results = torch.zeros(len(target_files), 2)

    for i, files in enumerate(tqdm(zip(target_files, output_files, compare_files), total=len(target_files), desc="Inpainting methods comparison")):
        target_file, unet_file, telea_file = files
        target_img = read_image(os.path.join(target_path, target_file)).float() / 255.
        unet_img = read_image(os.path.join(output_path, unet_file)).float() / 255.
        telea_file = read_image(os.path.join(compare_path, telea_file)).float() / 255.

        unet_sne = SNE(unet_img, target_img)
        telea_sne = SNE(telea_file, target_img)

        unet_psnr = PSNR(unet_img, target_img)
        telea_psnr = PSNR(telea_file, target_img)

        unet_ssim = SSIM(unet_img, target_img)
        telea_ssim = SSIM(telea_file, target_img)

        unet_lpips = LPIPS(unet_img, target_img)
        telea_pips = LPIPS(telea_file, target_img)

        sne_results[i] = torch.tensor([unet_sne, telea_sne])
        psnr_results[i] = torch.tensor([unet_psnr, telea_psnr])
        ssim_results[i] = torch.tensor([unet_ssim, telea_ssim])
        lpips_results[i] = torch.tensor([unet_lpips, telea_pips])

    mean_sne = torch.mean(sne_results, dim=0)
    mean_psnr = torch.mean(psnr_results, dim=0)
    mean_ssim = torch.mean(ssim_results, dim=0)
    mean_lpips = torch.mean(lpips_results, dim=0)

    std_sne = torch.std(sne_results, dim=0)
    std_psnr = torch.std(psnr_results, dim=0)
    std_ssim = torch.std(ssim_results, dim=0)
    std_lpips = torch.std(lpips_results, dim=0)

    methods = ["Unet", "Telea"]

    summary_df = pd.DataFrame({
        "Method": methods,
        "Mean SNE": mean_sne,
        "Std SNE": std_sne,
        "Mean PSNR": mean_psnr,
        "Std PSNR": std_psnr,
        "Mean SSIM": mean_ssim,
        "Std SSIM": std_ssim,
        "Mean LPIPS": mean_lpips,
        "Std LPIPS": std_lpips
    })

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(summary_df)