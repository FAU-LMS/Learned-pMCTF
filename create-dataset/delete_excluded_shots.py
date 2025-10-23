from pathlib import Path
import shutil


def get_directory_size_mb(path):
    size = 0
    for file in path.glob("*"):
        if file.is_file():
            print(file)
            size += file.stat().st_size / 1000 / 1000
    return size


clips_basepath = Path("/home/data/vimeo/clips")

ids_to_delete = []

with open('ids_to_delete.txt', 'r') as f:
    for line in f:
        ids_to_delete.append(line.strip())

space_freed = 0
count = 0
for id in ids_to_delete:
    video_id, clip_id = id.split("/")
    clip_path = clips_basepath / video_id / clip_id
    if clip_path.exists():
        space_freed += get_directory_size_mb(clip_path)
        count += 1
        shutil.rmtree(clip_path)
    video_path = clips_basepath / video_id
    if video_path.exists() and len(list(video_path.glob("*"))) == 0:
        video_path.rmdir()

print(f"Freed {space_freed / 1000:.2f}GB disk space by deleting {count} clips.")
