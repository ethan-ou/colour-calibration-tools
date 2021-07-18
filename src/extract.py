from utils import scale_range_1
import ffmpeg
from decimal import Decimal
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm


def get_metadata(path):
    try:
        probe = ffmpeg.probe(path)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        if video_stream is None:
            raise Exception('No video stream found.')

    except ffmpeg.Error as e:
        raise e

    except Exception as e:
        raise e

    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': int_or_float(Decimal(video_stream['r_frame_rate'].split(
            '/')[0]) / Decimal(video_stream['r_frame_rate'].split('/')[1]))}


def sample_frames(path, metadata, fps, quantity, interval, start_frame):
    patches = []

    interval_seconds = interval / 1000
    fps_multiplier = fps / metadata['fps']

    # Starts at middle of first color patch
    start = start_frame + (interval_seconds / 2) * fps

    try:
        file = (
            ffmpeg
            .input(path)
            .filter('select', 'gte(n,{}}'.format(start))
            .filter('fps', fps=(1/interval_seconds) / fps_multiplier)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="quiet", vframes=quantity)
            .run_async(pipe_stdout=True)
        )

        with tqdm(total=quantity) as pbar:
            pbar.set_description(f"Processing {Path(path).name}")

            while True:
                in_bytes = file.stdout.read(
                    metadata['height'] * metadata['width'] * 3)

                if not in_bytes:
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    [metadata['height'], metadata['width'], 3])

                sample = sample_colour(
                    frame, metadata['width'], metadata['height'])

                sample = scale_range_1(sample)

                patches.append(sample)

                pbar.update(1)

    except ffmpeg.Error as e:
        raise e

    except Exception as e:
        raise e

    return np.array(patches)


def load_patches(path):
    with open(path, 'r', newline='') as file:
        data = np.asarray(list(csv.reader(file)))

    return data.astype('float64')


def save_patches(patches, path):
    with open(path, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(patches)


def sample_colour(frame, width, height, sample_percent=0.05):
    sample_width = width * sample_percent
    sample_height = height * sample_percent

    values = (
        frame[
            int((height - sample_height) / 2):
            int((height + sample_height) / 2),
            int((width - sample_width) / 2):
            int((width + sample_width) / 2)
        ]
    )

    return np.array([np.mean(values[:, :, 0]), np.mean(
        values[:, :, 1]), np.mean(values[:, :, 2])])


def int_or_float(x):
    if int(float(x)) == float(x):
        return int(x)
    else:
        return float(x)
