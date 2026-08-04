import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


def pdf_pages_to_images(
    pdf_path: str | Path,
    destination_folder: str | Path,
    image_format: str = "PNG",
    dpi: int = 300,
    prefix: str = "page",
) -> list[str]:
    """Extract all pages from a PDF file and save them as separate image files.

    Args:
        pdf_path: Path to the input PDF file
        destination_folder: Path to the folder where images will be saved
        image_format: Image format to save (PNG, JPEG, etc.). Defaults to PNG
        dpi: Resolution for the output images. Defaults to 150
        prefix: Prefix for the output image filenames. Defaults to "page"

    Returns:
        List of paths to the created image files

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the PDF file is invalid or corrupted
        OSError: If there's an error creating the destination folder or saving images

    """
    pdf_path = Path(pdf_path)
    destination_folder = Path(destination_folder)

    # Validate input PDF file
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    # Create destination folder if it doesn't exist
    try:
        destination_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating destination folder {destination_folder}: {e}")
        raise

    created_files = []

    try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)

        logger.info(f"Processing PDF with {len(pdf_document)} pages")

        # Extract each page as an image
        for page_num in range(len(pdf_document)):
            try:
                # Get the page
                page = pdf_document[page_num]

                # Create a transformation matrix for the desired DPI
                zoom = dpi / 72  # Convert DPI to zoom factor
                mat = fitz.Matrix(zoom, zoom)

                # Render page to an image (using correct PyMuPDF API)
                pix = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]

                # Convert pixmap to bytes and then to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Generate filename with zero-padded page numbers
                total_pages = len(pdf_document)
                padding = len(str(total_pages))
                filename = f"{prefix}_{page_num + 1:0{padding}d}.{image_format.lower()}"
                output_path = destination_folder / filename

                # Save the image
                img.save(output_path, format=image_format.upper())
                created_files.append(str(output_path))

                logger.debug(f"Saved page {page_num + 1} as {output_path}")

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                continue

        pdf_document.close()

    except Exception as e:
        logger.error(f"Error opening PDF file {pdf_path}: {e}")
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}") from e

    logger.info(f"Successfully extracted {len(created_files)} pages from {pdf_path}")
    return created_files
