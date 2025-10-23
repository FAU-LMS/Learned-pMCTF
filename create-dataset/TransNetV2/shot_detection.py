import torch
from transnetv2 import TransNetV2
import imageio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

model = TransNetV2()
video_frames, single_frame_predictions, all_frame_predictions = model.predict_video("/home/regensky/Resources/dataset360/318728950.mp4")

# print(video_frames.shape, single_frame_predictions.shape, all_frame_predictions.shape)

plt.plot(np.linspace(0, len(video_frames)/30, len(video_frames)), single_frame_predictions)
plt.savefig("pred_single.png")
plt.close()

plt.plot(np.linspace(0, len(video_frames)/30, len(video_frames)), all_frame_predictions)
plt.savefig("pred_all.png")
plt.close()

#frames = np.empty((1, 90, 27, 48, 3), dtype=np.uint8)
#reader = imageio.get_reader("/home/regensky/Development/TransNetV2/417310731.mp4")
#reader.set_image_index(450)
#for i, frame in enumerate(reader):
#    if i >= frames.shape[1]:
#        break
#    print(i)
#    frames[0, i] = resize(frame, (27, 48), order=1, anti_aliasing=True)

