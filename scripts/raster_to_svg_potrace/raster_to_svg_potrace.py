#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile
import uuid

from PIL import Image


def raster_to_svg_potrace(input_image, output_svg):
    """
    Converts a raster image to SVG format using Potrace.

    Parameters:
    - input_image (str): Path to the input raster image (e.g., PNG, JPEG).
    - output_svg (str): Path to the output SVG file.
    """
    # Check if input image exists
    if not os.path.isfile(input_image):
        print(f"Error: The file '{input_image}' does not exist.")
        sys.exit(1)
    
    # Convert the image to black and white (PBM format)
    try:
        img = Image.open(input_image).convert('1')  # Convert to black and white
    except Exception as e:
        print(f"Error: Unable to open or convert the image '{input_image}'. {e}")
        sys.exit(1)
    
    # Create a temporary PBM file
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as temp_pbm:
        pbm_path = temp_pbm.name
        try:
            img.save(pbm_path)
            print(f"Temporary PBM file created at '{pbm_path}'")
        except Exception as e:
            print(f"Error: Unable to save temporary PBM file. {e}")
            sys.exit(1)
    
    try:
        # Call Potrace to convert PBM to SVG
        subprocess.run(['potrace', pbm_path, '--svg', '--output', output_svg], check=True)
        print(f"SVG saved to '{output_svg}'")
    except subprocess.CalledProcessError as e:
        print(f"Error: Potrace failed to convert '{input_image}' to SVG. {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary PBM file
        if os.path.exists(pbm_path):
            try:
                os.remove(pbm_path)
                print(f"Temporary PBM file '{pbm_path}' deleted.")
            except Exception as e:
                print(f"Warning: Unable to delete temporary PBM file '{pbm_path}'. {e}")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description='Convert a raster image (e.g., PNG, JPEG) to SVG format using Potrace.'
    )
    parser.add_argument(
        'input_image',
        type=str,
        help='Path to the input raster image (e.g., pig.jpg)'
    )
    parser.add_argument(
        'output_svg',
        type=str,
        nargs='?',
        default=None,
        help='Path to the output SVG file (e.g., pig.svg). If not provided, the output filename will be the input filename with a .svg extension.'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Determine the output SVG path
    if args.output_svg:
        output_svg = args.output_svg
    else:
        base, _ = os.path.splitext(args.input_image)
        output_svg = f"{base}.svg"
    
    # Perform the conversion
    raster_to_svg_potrace(args.input_image, output_svg)

if __name__ == "__main__":
    main()


