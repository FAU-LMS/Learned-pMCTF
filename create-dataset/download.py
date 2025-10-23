import vimeo_downloader
from vimeo_downloader import Vimeo
import pandas as pd
from pathlib import Path
import time


df = pd.read_fwf("original_vimeo_links.txt")
download_path = Path('/home/data/vimeo')

if not download_path.exists():
    download_path.mkdir()

max_count = df.size

count = 0
for video_data in df.itertuples():
    if count == max_count:
        break
    video_url = video_data[1]
    video_id = video_url.split('/')[-1]
    if (download_path / f"{video_id}.mp4").exists():
        count += 1
        continue
    try:
        v = Vimeo(f"https://vimeo.com/{video_id}")
        if len(v.streams) == 0:
            print("xxxx", v.metadata.id, v.metadata.title, "Cannot download.")
        else:
            count += 1
            print(f"{count:04d}", v.metadata.id, v.metadata.title, "Downloading...")
            best_stream = v.streams[-1]
            best_stream.download(str(download_path), filename=f"{v.metadata.id}")
            time.sleep(10)
    except vimeo_downloader.RequestError:
        print("xxxx", video_id, "Cannot download.")
        continue
    except vimeo_downloader.UnableToParseHtml:
        print("xxxx", video_id, "Cannot download.")
        continue
    except KeyError:
        print("xxxx", video_id, "Cannot download.")
        continue
