import cv2
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent / "Videos"
FRAMES_DIR = Path(__file__).parent.parent / "Dataset" / "Frames"


def extract_frames(video_name: str, interval: float = 1.0, output_dir: Path = None) -> None:
    """
    Extracts frames from a video at a fixed time interval and saves them as PNGs.

    :param video_name: Name of the video file (e.g. "Level 1 Run 1.mp4"). Should be
        located in the Videos/ directory at the project root.
    :param interval: Time in seconds between extracted frames. Default is 1.0.
    :param output_dir: Directory to save frames. Defaults to Dataset/Frames/<video_stem>.
    """
    video_path = VIDEO_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(fps * interval)

    if output_dir is None:
        output_dir = FRAMES_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        filename = output_dir / f"frame_{frame_idx:06d}_{timestamp:.2f}s.png"
        cv2.imwrite(str(filename), frame)
        extracted += 1

        frame_idx += frame_step

    cap.release()


if __name__ == "__main__":
    extract_frames("Level 1 Run 1.mp4", interval=1.0)
