# Installation
- Create environment:
```shell
conda env create -f datasetvimeo.yml
pip install tensorflow pandas tqdm ffmpeg-python pillow scikit-image vimeo-downloader einops opencv-python
```
- Follow install instructions in subfolder pygist.
- [Download](utils/DCVCDC/checkpoints/download.py) checkpoint for flow calculation
# Usage

Note: Please adjust the path for storing your data set in every script.
- Run download.py
- Run extract_clips.py (extract PNGs of size 448x256. Detect scenes using TransNetV2. clip_length=32 to extract 32 frames, careful: .mp4 video are deleted after clips have been extracted)
- Run average_flow.py (Calculate average flow in clip using DCVC-DC's SpyNet, flow will be needed for shot exclusion)
- Run gist_shot_exclusion.py
- Run delete_excluded_shots.py

# Acknowledgement

Original video list from https://data.csail.mit.edu/tofu/dataset/original_video_list.txt <br>
Clip extraction - TransNetV2: https://github.com/soCzech/TransNetV2<br>
Flow calculation - optical flow weights from DCVC-DC: https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-DC<br>
Gist shot exclusion - pygist: https://github.com/whitphx/lear-gist-python