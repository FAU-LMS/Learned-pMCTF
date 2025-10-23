import glob

import gist
import numpy as np
from pathlib import Path
from skimage import io
from skimage.util import img_as_ubyte
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import ffmpeg
from PIL import Image, ImageDraw, ImageFont



def gist_descriptors_for_video(video_clips_path):
    video_clips = len([path for path in video_clips_path.iterdir() if path.is_dir()])
    gist_descriptors = []
    for i in range(video_clips):
        clip_path = video_clips_path / f"{i:02d}"
        gist_descriptor_path = clip_path / "gist_descriptor.npy"
        if gist_descriptor_path.exists():
            descriptor = np.load(gist_descriptor_path)
        else:
            img_path = clip_path / "01.png"

            stream, err = ffmpeg.input(str(img_path)).output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s="448x256"
            ).run(capture_stdout=True, capture_stderr=True)
            img = np.frombuffer(stream, np.uint8).reshape([256, 448, 3])

            descriptor = gist.extract(img)
            np.save(gist_descriptor_path, descriptor)

        gist_descriptors.append(descriptor)
    return gist_descriptors


if __name__ == '__main__':

    clips_basepath = Path("/home/data/vimeo/clips")
    all_clips = [d for d in clips_basepath.iterdir()]

    ids = []
    gist_descriptors = []
    progress_bar = tqdm(total=len(all_clips), desc="Calculate gist descriptors")
    for index, row in enumerate(all_clips):
        video_id = str(row).split('/')[-1]
        video_clips_path = clips_basepath / f"{video_id}"
        video_gist_descriptors = gist_descriptors_for_video(video_clips_path)
        ids.extend(map(lambda i: f"{video_id}/{i:02d}", range(len(video_gist_descriptors))))
        gist_descriptors.extend(video_gist_descriptors)
        progress_bar.update()

    gist_descriptors = np.array(gist_descriptors)


    def make_similars_grid(ids, mark_id=None):
        font_size = 40
        padding = 5
        image_size = (512, 256)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSerif.ttf", font_size)
        similars = []
        for id in ids:
            video_id, scene_id = id.split("/")
            img_path = clips_basepath / video_id / scene_id / "01.png"
            img = Image.open(img_path)
            img = img.resize(image_size)

            # Add text
            img_with_text = Image.new('RGB', (image_size[0], image_size[1] + font_size + 2 * padding), "black")
            img_with_text.paste(img, (0, 0))
            draw = ImageDraw.Draw(img_with_text)
            text_width = draw.textlength(id, font=font)
            draw.text(((image_size[0] - text_width) / 2, image_size[1] + padding), id, font=font, fill="white")
            if id == mark_id:
                draw.rectangle((0, 0, image_size[0] - 1, image_size[1] - 1), fill=None, outline='red', width=5)

            similars.append(torch.tensor(np.array(img_with_text)).permute(2, 0, 1))
        return make_grid(similars)


    def is_image_flat(image):
        """
        Determine if image is flat by calculating grayscale pdf and checking
        if one value covers more than 50% of the image.
        """
        pdf, _ = np.histogram(image.reshape(-1), bins=256, range=(0, 255), density=True)
        if np.max(pdf) >= 0.5:
            return True
        return False


    def select_best_shot(similar_shot_ids):
        "Return shot id with highest average flow if it is not flat."
        mean_shot_flows = np.empty(len(similar_shot_ids))
        is_shot_flat = np.empty(len(similar_shot_ids), dtype=bool)
        for i, shot_id in enumerate(similar_shot_ids):
            video_id, scene_id = shot_id.split("/")
            clip_path = clips_basepath / video_id / scene_id
            mean_shot_flows[i] = np.mean(np.load(clip_path / "flow.npy"))
            is_shot_flat[i] = is_image_flat(img_as_ubyte(io.imread(clip_path / "01.png", as_gray=True)))

        best_flow_idx = np.argmax(mean_shot_flows)

        # If shot with best flow is flat, return None
        if is_shot_flat[best_flow_idx]:
            return None

        return similar_shot_ids[best_flow_idx]


    threshold = 0.15  # GIST descriptor L2 distance threshold
    similarity_folder = Path("./similiarities")
    similarity_folder.mkdir(exist_ok=True)
    i = 0
    ids_to_delete = []
    while i < len(gist_descriptors):
        descriptor = gist_descriptors[i]

        distances = np.sqrt(np.sum(np.square(gist_descriptors - descriptor), axis=1))
        idxs_close = np.argwhere(distances < threshold)[:, 0]
        idxs_close_sorted = idxs_close[np.argsort(distances[idxs_close])]

        similar_shot_ids = [ids[i] for i in idxs_close_sorted]
        best_shot_id = select_best_shot(similar_shot_ids)

        if len(similar_shot_ids) > 1:
            similars_grid = make_similars_grid(similar_shot_ids, mark_id=best_shot_id)
            io.imsave(similarity_folder / f"{ids[i].replace('/', '-')}.png", similars_grid.permute(1, 2, 0).numpy())

        for idx_close_sorted in idxs_close_sorted:
            shot_id = ids[idx_close_sorted]
            if shot_id != best_shot_id:
                gist_descriptors = np.delete(gist_descriptors, idx_close_sorted, axis=0)
                ids.remove(shot_id)
                ids_to_delete.append(shot_id)

        i += 1

    print(len(ids_to_delete))

    with open("ids_to_delete.txt", "w") as f:
        for id_to_delete in ids_to_delete:
            f.write(id_to_delete + "\n")
