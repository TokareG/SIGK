import os

#from project_2.run_postprocesing import output_path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

import cv2
from numpy import ndarray
import numpy as np
from brisque import BRISQUE
import pandas as pd

def tone_map_mantiuk(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.2)
    result = tonemap_operator.process(src=image)
    return result

def tone_map_reinhard(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapReinhard(gamma=2.2, intensity=0.0, light_adapt=0.0, color_adapt=0.0)
    result = tonemap_operator.process(src=image)
    return result



def reinhard(input_path, output_path):
    assert os.path.exists(input_path)
    assert os.path.exists(output_path)
    input_filenames = os.listdir(input_path)
    for input_filename in input_filenames:
        input_basename = os.path.splitext(input_filename)[0]
        img = cv2.imread(filename=os.path.join(input_path, input_filename), flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img_normalized = img / (np.mean(img) + 1e-8)
        img_reinhard = tone_map_reinhard(img_normalized)
        if np.isnan(img_reinhard).any():
            print("⚠️ Tone mapping returned NaNs!")
            print("Reinhard output min/max:", np.nanmin(img_reinhard), np.nanmax(img_reinhard))
            #img_reinhard = np.zeros_like(img_reinhard)
        img_reinhard = (img_reinhard * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, input_basename + ".png"), img_reinhard)


def mantiuk(input_path, output_path):
    assert os.path.exists(input_path)
    assert os.path.exists(output_path)
    input_filenames = os.listdir(input_path)
    for input_filename in input_filenames:
        input_basename = os.path.splitext(input_filename)[0]
        img = cv2.imread(filename=os.path.join(input_path, input_filename), flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img_normalized = img / (np.mean(img) + 1e-8)
        img_mantiuk = tone_map_mantiuk(img_normalized)
        if np.isnan(img_mantiuk).any():
            print("⚠️ Tone mapping returned NaNs!")
            print("Reinhard output min/max:", np.nanmin(img_mantiuk), np.nanmax(img_mantiuk))
            #img_mantiuk = np.zeros_like(img_mantiuk)
        img_mantiuk = (img_mantiuk * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, input_basename + ".png"), img_mantiuk)

def tm_reinhard(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = img / np.max(img)
    tonemap_operator = cv2.createTonemapReinhard(gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.6)
    result = tonemap_operator.process(src=img)
    result = np.nan_to_num(result, nan=0.0)
    ldr_8bit = np.clip(result * 255, 0, 255).astype(np.uint8)
    return ldr_8bit

def evaluate_image(image: ndarray)-> float:
    metric = BRISQUE(url=False)
    return metric.score(img=image)

def get_mean_std(scores):
    return np.mean(scores), np.std(scores)

def brisque_dir(path):
    assert os.path.exists(path)
    filenames = os.listdir(path)
    assert len(filenames) > 0
    scores = []
    for filename in filenames:
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_UNCHANGED)
        scores.append(evaluate_image(img))
    mean, std = np.mean(scores), np.std(scores)
    return scores, mean, std

def evaluate_brisque(paths: dict):
    scores = {}
    statistics = {}
    for method, path in paths.items():
        score, mean, std = brisque_dir(path)
        scores[method] = score
        statistics[method] = [mean, std]
    filenames = os.listdir(next(iter(paths.values())))
    scores_df = pd.DataFrame(scores, index=filenames)
    stats_df = pd.DataFrame(statistics, index=["Mean", "Std"])
    print(scores_df)
    print()
    print(stats_df)
