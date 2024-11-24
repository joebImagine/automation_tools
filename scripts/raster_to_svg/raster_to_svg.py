#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from PIL import Image


def preprocess_image(image_path, handle_transparency=False):
    """
    Preprocess the image by converting transparent regions to white.

    Parameters:
    - image_path (str): Path to the input image.
    - handle_transparency (bool): Flag to handle transparency.

    Returns:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    """
    if handle_transparency:
        print("Handling transparency...")
        # Load image with alpha channel
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Image at path '{image_path}' not found.")
            sys.exit(1)

        if img.shape[2] == 4:
            # Separate alpha channel
            alpha = img[:, :, 3]
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Set transparent pixels to white
            img_gray[alpha < 255] = 255
        else:
            # Image does not have alpha channel
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Load in grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Error: Image at path '{image_path}' not found.")
            sys.exit(1)
    print(f"Preprocessed grayscale image shape: {img_gray.shape}")
    inverted_img_gray = cv2.bitwise_not(img_gray)
    return img_gray, inverted_img_gray


def create_contours(img_gray, retrieval_mode=cv2.RETR_CCOMP):
    """
    Find contours in the preprocessed grayscale image.

    Parameters:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    - retrieval_mode (int): OpenCV contour retrieval mode.

    Returns:
    - contours (list of np.ndarray): List of contour points.
    - hierarchy (np.ndarray): Contour hierarchy.
    """
    # Apply binary thresholding (no inversion)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Optional: Remove noise and smooth the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(thresh, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found in the image.")
        sys.exit(1)

    # Check if hierarchy is returned
    if hierarchy is None:
        print("Error: No hierarchy information found.")
        sys.exit(1)

    hierarchy = hierarchy[0]  # Simplify the hierarchy array

    return contours, hierarchy


def create_svg(contours, hierarchy, svg_path, image_size, scale=1.0):
    """
    Create an SVG file with each mask as a separate path.

    Parameters:
    - contours (list of np.ndarray): List of contour points.
    - hierarchy (np.ndarray): Contour hierarchy.
    - svg_path (str): Path to save the SVG file.
    - image_size (tuple): (width, height) of the image in pixels.
    - scale (float): Scaling factor for the SVG dimensions.
    """
    width, height = image_size
    dwg = svgwrite.Drawing(svg_path, size=(f"{width * scale}px", f"{height * scale}px"))
    dwg.add(dwg.rect(insert=(0, 0), size=(f"{width * scale}px", f"{height * scale}px"), fill='white'))

    mask_count = 0

    for idx, contour in enumerate(contours):
        # Only consider external contours
        if hierarchy[idx][3] == -1:
            mask_count += 1
            # Extract outer contour points
            outer = contour.squeeze().tolist()
            if not isinstance(outer[0], list):
                outer = [outer]

            # Start building path data
            path_data = ""

            # Outer contour
            if len(outer) > 0:
                path_data += f"M {outer[0][0] * scale},{outer[0][1] * scale} "
                for point in outer[1:]:
                    path_data += f"L {point[0] * scale},{point[1] * scale} "
                path_data += "Z "  # Close path

            # Handle holes (child contours)
            child_idx = hierarchy[idx][2]  # First child
            while child_idx != -1:
                hole = contours[child_idx].squeeze().tolist()
                if not isinstance(hole[0], list):
                    hole = [hole]

                if len(hole) > 0:
                    path_data += f"M {hole[0][0] * scale},{hole[0][1] * scale} "
                    for point in hole[1:]:
                        path_data += f"L {point[0] * scale},{point[1] * scale} "
                    path_data += "Z "  # Close path
                child_idx = hierarchy[child_idx][0]  # Move to next sibling

            # Add the path to SVG with 'evenodd' fill rule to handle holes
            dwg.add(dwg.path(d=path_data, fill='black', stroke='none', fill_rule='evenodd', id=f"mask_{mask_count}"))

    # Save SVG file
    dwg.save()
    print(f"SVG saved to: {os.path.abspath(svg_path)}")
    print(f"Total masks added to SVG: {mask_count}")


def display_masks(masks, max_display=5):
    """
    Display a subset of masks using matplotlib.

    Parameters:
    - masks (list of np.ndarray): List of mask images.
    - max_display (int): Maximum number of masks to display.
    """
    if not masks:
        print("No masks to display.")
        return

    num_masks_to_display = min(max_display, len(masks))  # Display up to max_display masks

    plt.figure(figsize=(15, 3))
    for i in range(num_masks_to_display):
        plt.subplot(1, num_masks_to_display, i+1)
        plt.imshow(masks[i], cmap='gray')
        plt.title(f'Mask {i+1}')
        plt.axis('off')
    plt.show()


def generate_mask_images(contours, hierarchy, img_shape):
    """
    Generate mask images for optional display purposes.

    Parameters:
    - contours (list of np.ndarray): List of contour points.
    - hierarchy (np.ndarray): Contour hierarchy.
    - img_shape (tuple): Shape of the original grayscale image.

    Returns:
    - masks (list of np.ndarray): List of mask images.
    """
    masks = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] == -1:
            mask = np.ones(img_shape, dtype=np.uint8) * 255  # Initialize mask as white
            # Draw the outer contour filled with black
            cv2.drawContours(mask, [contour], -1, color=0, thickness=-1)

            # Draw child contours (holes) filled with white
            child_idx = hierarchy[idx][2]  # First child
            while child_idx != -1:
                cv2.drawContours(mask, [contours[child_idx]], -1, color=255, thickness=-1)
                child_idx = hierarchy[child_idx][0]

            masks.append(mask)
    return masks


def main():
    """
    Main function to parse command-line arguments and execute mask creation.
    """
    parser = argparse.ArgumentParser(
        description="Create masks for each enclosed (including nested) white and/or transparent region in a black and white coloring book image and save them into a single SVG file."
    )
    parser.add_argument(
        'image_path',
        type=str,
        help="Path to the input black and white coloring book image."
    )
    parser.add_argument(
        '-s', '--svg_path',
        type=str,
        default='masks.svg',
        help="Path to save the output SVG file. Default is 'masks.svg'."
    )
    parser.add_argument(
        '-d', '--display',
        action='store_true',
        help="Display some of the created masks."
    )
    parser.add_argument(
        '-m', '--max_display',
        type=int,
        default=5,
        help="Maximum number of masks to display if --display is set. Default is 5."
    )
    parser.add_argument(
        '-r', '--retrieval_mode',
        type=str,
        default='CCOMP',
        choices=['EXTERNAL', 'LIST', 'CCOMP', 'TREE'],
        help="Contour retrieval mode. Options: EXTERNAL, LIST, CCOMP, TREE. Default is CCOMP."
    )
    parser.add_argument(
        '--handle_transparency',
        action='store_true',
        help="Handle transparency in the image by treating transparent regions as white."
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help="Scaling factor for the SVG dimensions. Default is 1.0 (no scaling)."
    )

    args = parser.parse_args()

    # Preprocess the image
    img_gray, inverted_img_gray = preprocess_image(args.image_path, handle_transparency=args.handle_transparency)

    # Get image dimensions
    height, width = img_gray.shape[:2]

    # Map string retrieval mode to OpenCV constants
    retrieval_modes = {
        'EXTERNAL': cv2.RETR_EXTERNAL,
        'LIST': cv2.RETR_LIST,
        'CCOMP': cv2.RETR_CCOMP,
        'TREE': cv2.RETR_TREE
    }

    retrieval_mode = retrieval_modes.get(args.retrieval_mode.upper(), cv2.RETR_CCOMP)

    # Find contours
    contours, hierarchy = create_contours(
        img_gray=img_gray,
        retrieval_mode=retrieval_mode
    )

    # Create SVG from contours
    create_svg(
        contours=contours,
        hierarchy=hierarchy,
        svg_path=args.svg_path,
        image_size=(width, height),
        scale=args.scale
    )

    # Optionally display masks
    if args.display:
        # Generate mask images for display purposes
        masks = generate_mask_images(contours, hierarchy, img_gray.shape)
        display_masks(masks, max_display=args.max_display)


if __name__ == "__main__":
    main()
