import glob

from TransNetV2 import TransNetV2
from pathlib import Path
from tqdm import tqdm
import ffmpeg


def make_clips_from_scenes(clips_basepath: Path,
                           video_path: Path,
                           video_id: int,
                           scenes: list,
                           min_scene_length: int,
                           clip_length: int):
    """
    :param clips_basepath: Basepath of where to store all clips
    :param video_path: Location of current video
    :param video_id: ID of current video
    :param scenes: Detected scenes for current video
    :param min_scene_length: Minimum length of scene in frames to extract a clip from it
    :param clip_length: Length of clip in frames
    """
    scene_count = 0
    video_clips_path = clips_basepath / f"{video_id}"
    video_clips_path.mkdir(exist_ok=True)

    fps_components = ffmpeg.probe(str(video_path))['streams'][0]['avg_frame_rate'].split("/")
    if float(fps_components[1]) == 0:
        fps = float(1)
    else:
        fps = float(fps_components[0]) / float(fps_components[1])

    for scene in scenes:
        # Check clip conditions
        scene_length = scene[1] - scene[0]
        if scene_length < min_scene_length:
            continue

        # Prepare video clip path
        video_clip_path = video_clips_path / f"{scene_count:02d}"
        video_clip_path.mkdir(exist_ok=True)

        # Calculate ffmpeg start timecode
        clip_start_idx = scene[0] + (scene_length // 2) - (clip_length // 2)
        time_start = clip_start_idx / fps

        # Extract clip using ffmpeg
        ffmpeg.input(
            str(video_path),
            ss=f"{time_start:.3f}"
        ).filter(
            'scale', 448, 256,
        ).output(
            str(video_clip_path / "%02d.png"),
            **{"frames:v": str(clip_length)}
        ).run(capture_stdout=True, capture_stderr=True)

        scene_count += 1
    return scene_count




download_path = Path('/home/data/vimeo')
clips_basepath = Path('/home/data/vimeo/clips')
all_clips = glob.glob(str(download_path) + "/*.mp4")
# include all downloaded clips
progress_bar = tqdm(total=len(all_clips), desc="Detect and extract clips")
all_clips = enumerate(all_clips)


clips_basepath.mkdir(exist_ok=True)
model = TransNetV2("TransNetV2/transnetv2-weights")

total_clips = 0
for index, row in all_clips:

    video_path = Path(row)
    video_id = row.split('/')[-1][:-4]

    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(str(video_path))
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=0.6)
    total_clips += make_clips_from_scenes(clips_basepath,
                                          video_path,
                                          video_id,
                                          scenes,
                                          min_scene_length=96,
                                          clip_length=32)
    progress_bar.set_description(f"Detect and extract clips - {total_clips}")
    progress_bar.update()
    video_path.unlink()
progress_bar.close()
