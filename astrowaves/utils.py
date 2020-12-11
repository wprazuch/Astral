import tifffile
import imageio


def generate_dummy_tiff_series(input_tiff_path: str, output_tiff_path: str, start_cord, end_cord):
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
    volume = volume[start_cord[0]:end_cord[0], start_cord[1]:end_cord[1], start_cord[2]:end_cord[2]]
    imageio.mimwrite(output_tiff_path, volume)
