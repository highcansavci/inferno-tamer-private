import rasterio
from PIL import Image, ImageDraw
import numpy as np
import os


class GeoTIFFProcessor:
    def __init__(self, input_file, output_dir, grid_size):
        """
        Initialize the GeoTIFF processor.
        :param input_file: Path to the input GeoTIFF file.
        :param output_dir: Directory to save the generated PNG with grid lines.
        :param grid_size: Tuple (rows, cols) for the grid division.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.grid_size = grid_size

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def convert_to_png(self):
        """
        Convert the GeoTIFF file to a PNG image.
        :return: Path to the saved PNG file.
        """
        with rasterio.open(self.input_file) as dataset:
            # Read the first 3 bands (for RGB)
            bands = [dataset.read(i + 1) for i in range(min(3, dataset.count))]
            rgb_array = np.stack(bands, axis=-1)

            # Save as PNG
            png_path = os.path.join(self.output_dir, "output.png")
            Image.fromarray(rgb_array).save(png_path)
            print(f"GeoTIFF converted to PNG: {png_path}")
            return png_path

    def add_grid_lines(self, png_path):
        """
        Draw grid lines on the PNG image and save it.
        :param png_path: Path to the PNG image.
        """
        image = Image.open(png_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        rows, cols = self.grid_size

        # Calculate grid cell dimensions
        cell_width = width // cols
        cell_height = height // rows

        print(f"Adding grid lines for a {rows}x{cols} grid...")

        # Draw vertical lines
        for col in range(1, cols):
            x = col * cell_width
            draw.line([(x, 0), (x, height)], fill="black", width=2)

        # Draw horizontal lines
        for row in range(1, rows):
            y = row * cell_height
            draw.line([(0, y), (width, y)], fill="black", width=2)

        # Save the image with grid lines
        grid_image_path = os.path.join(self.output_dir, "output_with_grid.png")
        image.save(grid_image_path)
        print(f"Grid lines added and saved: {grid_image_path}")


if __name__ == "__main__":
    # Inputs
    INPUT_TIFF = "../client/output.tif"  # Replace with your GeoTIFF file path
    OUTPUT_DIR = "../assets/images"      # Directory to store outputs
    GRID_NUMBER = (8, 8)                    # Divide into 4x4 grid

    # Process the GeoTIFF
    processor = GeoTIFFProcessor(INPUT_TIFF, OUTPUT_DIR, GRID_NUMBER)
    png_file = processor.convert_to_png()
    processor.add_grid_lines(png_file)
