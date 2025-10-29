import subprocess
from pathlib import Path


def video_to_empty_yolo_dataset(video_path, output_dir, fps=1, val_split=0.1):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create folder structure
    train_img_dir = output_dir / "images/train"
    val_img_dir = output_dir / "images/val"
    train_lbl_dir = output_dir / "labels/train"
    val_lbl_dir = output_dir / "labels/val"
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract frames
    tmp_dir = output_dir / "tmp_frames"
    tmp_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            str(tmp_dir / "frame_%06d.jpg"),
        ],
        check=True,
    )

    # Get sorted list of frames
    frames = sorted(tmp_dir.glob("*.jpg"))
    n_val = int(len(frames) * val_split)

    # Move frames and create empty labels
    for i, frame in enumerate(frames):
        if i < n_val:
            img_dest = val_img_dir / frame.name
            lbl_dest = val_lbl_dir / f"{frame.stem}.txt"
        else:
            img_dest = train_img_dir / frame.name
            lbl_dest = train_lbl_dir / f"{frame.stem}.txt"

        frame.rename(img_dest)
        lbl_dest.touch()  # create empty label

    tmp_dir.rmdir()  # remove temporary folder if empty

    # Create data.yaml
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(f"""train: {train_img_dir} \nval: {val_img_dir} \nnc: 0 \nnames: []""")


if __name__ == "__main__":
    video_to_empty_yolo_dataset("899_IST_8_cut1.mp4", "dataset", fps=1)
