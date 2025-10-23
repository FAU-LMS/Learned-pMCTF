
import torch
import argparse
from pathlib import Path
import numpy as np
from skimage import transform
from skimage.util import img_as_float
from utils.DCVCDC.src.models.video_net import ME_Spynet

from tqdm import tqdm
import ffmpeg

def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    flow_dict = {}
    for k, v in ckpt.items():
        if k.startswith('optic_flow'):
            flow_dict[k.replace('optic_flow.', '')] = v
    return flow_dict


def setup_spynet_dcvcdc(device):
    video_model_path = Path("utils/DCVCDC/checkpoints/cvpr2023_video_psnr.pth.tar")
    video_state_dict = get_state_dict(video_model_path)
    spy_net = ME_Spynet()
    spy_net.load_state_dict(video_state_dict)
    spy_net.to(device)
    spy_net.eval()
    return spy_net


def average_optical_flow_spynet(spynet_model, images, width, height, device):
    all_flow = []
    for i in range(len(images) - 1):
        reference_torch = torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0).to(device)
        moving_torch = torch.from_numpy(images[i+1]).permute(2, 0, 1).float().unsqueeze(0).to(device)

        flow = spynet_model(moving_torch, reference_torch)

        flow = flow.detach().cpu().numpy()[0]  # [2, H, W] -> 2: (x, y)
        all_flow.append(flow)

    all_flow = np.array(all_flow)
    flow_height, flow_width = all_flow.shape[-2:]
    all_flow[:, 0] = all_flow[:, 0] * (width / flow_width)
    all_flow[:, 1] = all_flow[:, 1] * (height / flow_height)
    average_flow_magnitude = np.mean(np.sqrt(np.sum(np.square(all_flow), axis=1)), axis=0)
    transform.resize(average_flow_magnitude, (height, width), order=2)
    return average_flow_magnitude


def process_flow_for_video(video_clips_path, model):
    scene_paths = [path for path in video_clips_path.iterdir() if path.is_dir()]
    metadata = ffmpeg.probe(str(scene_paths[0] / "01.png"))['streams'][0]
    width, height = metadata['width'], metadata['height']
    for scene_path in scene_paths:
        flow_path = scene_path / "flow.npy"
        if flow_path.exists():
            continue
        else:
            imgs = []
            for i in range(9):
                img_path = scene_path / f"{i + 1:02d}.png"

                # Read using ffmpeg including downscaling to 2048x1024
                stream, err = ffmpeg.input(str(img_path)).output(
                    "pipe:", format="rawvideo", pix_fmt="rgb24", s="448x256"
                ).run(capture_stdout=True, capture_stderr=True)

                img = img_as_float(np.frombuffer(stream, np.uint8).reshape([256, 448, 3]))
                imgs.append(img)

            average_flow_magnitude = average_optical_flow_spynet(model, imgs, width, height, device)
            np.save(scene_path / "flow.npy", average_flow_magnitude)
            # others = ["flow_preview.jpg", "flow_spynet_preview.jpg", "flow_mask.png", "flow_spynet.npy"]
            # for path in [scene_path / file for file in others]:
            #     if path.exists():
            #         path.unlink()


if __name__ == '__main__':
    clips_basepath = Path("/home/data/vimeo/clips")
    all_clips = [d for d in clips_basepath.iterdir()]

    device = 'cuda:0'
    model = setup_spynet_dcvcdc(device)

    # video_id = 491188944
    # video_clips_path = clips_basepath / f"{video_id}"
    # process_flow_for_video(video_clips_path, model)
    # exit()
    progress_bar = tqdm(total=len(all_clips), desc="Calculate average flow statistics")

    for index, row in enumerate(all_clips):
        video_id = str(row).split('/')[-1]
        video_clips_path = clips_basepath / f"{video_id}"
        process_flow_for_video(video_clips_path, model)
        progress_bar.update()
