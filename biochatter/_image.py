import base64
import io
import os
import subprocess
import tempfile

import pdf2image
from PIL import Image


def convert_and_resize_image(image: Image, max_size: int = 1024) -> Image:
    """Convert the image to RGB format.

    Do so if needed and resize it to have a maximum dimension of max_size.

    Args:
    ----
        image (PIL.Image): The input image.
        max_size (int): The maximum size for the image's width or height.

    Returns:
    -------
        PIL.Image: The converted and resized PIL image.

    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


def convert_to_png(image: Image) -> bytes:
    """Convert a PIL image to PNG format.

    Args:
    ----
        image (PIL.Image): The input image.

    Returns:
    -------
        bytes: The PNG image data.

    """
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def convert_to_pil_image(file_path: str, dpi: int = 300) -> Image:
    """Convert various image formats (PDF, EPS, TIFF, JPG, PNG) to a PIL image.

    Args:
    ----
        file_path (str): The path to the image file.
        dpi (int): Dots per inch for high-resolution EPS conversion.

    Returns:
    -------
        PIL.Image: The converted PIL image.

    """
    file_path = os.path.abspath(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        image = Image.open(file_path)
        return convert_and_resize_image(image)
    elif file_ext == ".pdf":
        pages = pdf2image.convert_from_path(file_path, dpi=dpi)
        if pages:
            return convert_and_resize_image(pages[0])
    elif file_ext == ".eps":
        output_path = file_path.replace(".eps", ".png")
        command = [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pngalpha",
            f"-r{dpi}",
            f"-sOutputFile={output_path}",
            file_path,
        ]
        subprocess.run(command, check=True)
        image = Image.open(output_path)
        return convert_and_resize_image(image)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def process_image(path: str, max_size: int) -> str:
    """Process an image, converting it to PNG and resizing if necessary, then
    encode to base64.

    Args:
    ----
        path (str): The path to the image file.
        max_size (int): The maximum size for the image's width or height.

    Returns:
    -------
        str: The base64 encoded image data.

    """
    image = convert_to_pil_image(path)
    png_image = convert_to_png(image)
    return base64.b64encode(png_image).decode("utf-8")


def encode_image(image_path):
    """Encode an image file to a base64 string, converting formats if necessary.

    Args:
    ----
        image_path (str): The path to the image file.

    Returns:
    -------
        str: The base64 encoded image data.

    """
    supported_formats = (".webp", ".jpg", ".jpeg", ".gif", ".png")
    file_ext = os.path.splitext(image_path)[1].lower()

    if file_ext in supported_formats:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return process_image(image_path, max_size=1024)


def encode_image_from_url(url: str) -> str:
    """Download an image from a URL, convert to base64, and return the base64
    string.

    Args:
    ----
        url (str): The URL of the image.

    Returns:
    -------
        str: The base64 encoded image data.

    """
    from urllib.request import urlopen

    # Get the file extension from the content type
    with urlopen(url) as response:
        content_type = response.info().get_content_type()
        extension = content_type.split("/")[-1]
        extension = "jpg" if extension == "jpeg" else extension  # normalize extension

    with (
        urlopen(url) as response,
        tempfile.NamedTemporaryFile(
            suffix=f".{extension}",
            delete=False,
        ) as tmp_file,
    ):
        tmp_file.write(response.read())
        tmp_file_path = tmp_file.name

    base64_string = encode_image(tmp_file_path)
    os.remove(tmp_file_path)

    return base64_string
