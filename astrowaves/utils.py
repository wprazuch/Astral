import tifffile
import imageio
import ffmpeg
import cv2
from pathlib import Path
import shutil


def generate_dummy_tiff_series(
    input_tiff_path: str, output_tiff_path: str, start_cord, end_cord
):
    """Generate smaller tiff file out of an input tiff file

    Parameters
    ----------
    input_tiff_path : str
        input tiff file path
    output_tiff_path : str
        output tiff file
    start_cord : tuple, optional
        start point of a subspace to take, (t, y, x)
    end_cord : tuple, optional
        end point of a subspace to take, (t, y, x)
    """
    file = tifffile.TiffFile(input_tiff_path)
    volume = file.asarray()
    volume = volume[
        start_cord[0] : end_cord[0],
        start_cord[1] : end_cord[1],
        start_cord[2] : end_cord[2],
    ]
    imageio.mimwrite(output_tiff_path, volume)


def normalize(img):
    img_new = img.copy()
    img_new = img_new - img_new.min()
    img_new = img_new / img_new.max()
    img_new = img_new * 255
    return img_new


def to_255(img):

    img_new = img.copy()

    img_new = img_new - img_new.min()
    img_new = img_new / img_new.max()
    img_new = img_new * 255
    img_new = img_new.astype("uint8")

    return img_new


def generate_video(img, output_path, output_filename):

    array_3d = img.copy()
    array_3d = to_255(array_3d)

    # create paths for video generation
    output_path = Path(output_path)

    tmp_path = output_path.joinpath("tmp")
    output_filepath = output_path.joinpath(output_filename)
    pattern_path = tmp_path.joinpath("img_%06d.jpg")

    tmp_path.mkdir(parents=True, exist_ok=True)

    # Write temp images
    for i in range(array_3d.shape[0]):
        cv2.imwrite(
            str(tmp_path.joinpath(f"img_{str(i).zfill(6)}.jpg")), array_3d[i, :, :]
        )

    # Create video
    try:
        ffmpeg.input(str(pattern_path), framerate=25).output(
            str(output_filepath)
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e

    shutil.rmtree(tmp_path)
