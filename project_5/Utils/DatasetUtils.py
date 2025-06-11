from typing import List

import gdown
import os, tarfile, shutil

class IPN_Hand:
    @staticmethod
    def download(target_dir: str) -> None:
        assert os.path.exists(target_dir)
        assert os.path.isdir(target_dir), f"{target_dir} is not a valid directory"

        frames_urls = [
            'https://drive.google.com/file/d/17yn_1n3LrMHLVSCT4zAbN6sbGXKy1GlA/view?usp=sharing',
            'https://drive.google.com/file/d/1OBlvjl-Z0Wr6xXnLaCGGwFUEXJQLCkDw/view?usp=sharing',
            'https://drive.google.com/file/d/1YCDW763mlXQlfuydFHX_EHG_hyBWC48F/view?usp=sharing',
            'https://drive.google.com/file/d/1I90cK_4gyyQQRkLjFn2p90CCIPV4q41Y/view?usp=sharing',
            'https://drive.google.com/file/d/1SY_TEQX80MtjygS8RqASqRMDDPyiENlZ/view?usp=sharing'
            ]
        for i, frames_url in enumerate(frames_urls):
            filename = f'frames_{i}.tgz'
            path = os.path.join(target_dir, filename)
            gdown.download(frames_url, path, quiet=False, fuzzy=True)

    @staticmethod
    def unpack(target_dir: str) -> None:
        exact_path = os.path.join(target_dir, 'temp')
        if not os.path.exists(exact_path):
            os.makedirs(exact_path)
        try:
            tgz_list =[f for f in os.listdir(target_dir) if f.endswith('.tgz')]
            for f in tgz_list:
                with tarfile.open(os.path.join(target_dir, f), 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.name != "frames/" and member.name.endswith('.jpg'):
                            member.name = member.name[len("frames/"):]
                            dir_name, filename = member.name.split('/')
                            dir_name = f"scene_{dir_name[dir_name.find("#")+1:]}"
                            filename = filename.rsplit('_', 1)[1]
                            member.name = os.path.join(dir_name, filename)
                            tar.extract(member, exact_path)

        except Exception as e:
            print(e)

    @staticmethod
    def patch_frames(frames_list, patch_size = 5) -> List[List[str]]:
        for i in range(0, len(frames_list), patch_size):
            yield frames_list[i:i + patch_size]

    @staticmethod
    def create(target_dir: str) -> None:
        path = os.path.join(target_dir, 'temp')
        dataset_dir = os.path.join(target_dir, 'dataset')
        assert os.path.exists(path), f"{path} is not a valid directory"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        scenes_list = os.listdir(path)
        assert len(scenes_list) > 0, f"There are no scenes in {path}"
        for scene in scenes_list:
            os.makedirs(os.path.join(dataset_dir, scene), exist_ok=True)
            scene_path = os.path.join(path, scene)
            frames_list = os.listdir(scene_path)
            for i, frames_patch in enumerate(IPN_Hand.patch_frames(frames_list)):
                if len(frames_patch) < 5:
                    continue
                patch_dir = os.path.join(dataset_dir, scene, f"{i:05d}")
                os.makedirs(patch_dir, exist_ok=True)
                for j, frame in enumerate(frames_patch[::2], start=1):
                    os.replace(os.path.join(scene_path, frame), os.path.join(patch_dir, f"img{j}.png"))
        tar_list =[f for f in os.listdir(target_dir) if f.endswith('.tgz')]
        for tar_file in tar_list:
            os.remove(os.path.join(target_dir, tar_file))
        shutil.rmtree(path)

if __name__ == '__main__':
    target_dir = "TARGET_DIR"
    IPN_Hand.download(target_dir=target_dir)
    IPN_Hand.unpack(target_dir=target_dir)
    IPN_Hand.create(target_dir=target_dir)