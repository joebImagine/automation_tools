#!/usr/bin/env python3

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from PIL import Image

# Print OpenCV version for debugging
print(f"OpenCV version: {cv2.__version__}")

def preprocess_image(image_path, handle_transparency=False):
    """
    Preprocess the image by converting transparent regions to white.

    Parameters:
    - image_path (str): Path to the input image.
    - handle_transparency (bool): Flag to handle transparency.

    Returns:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    """
    print("Starting image preprocessing...")
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
            print("Transparency handled: Transparent regions set to white.")
        else:
            # Image does not have alpha channel
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("Image does not have an alpha channel. Converted to grayscale.")
    else:
        # Load in grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Error: Image at path '{image_path}' not found.")
            sys.exit(1)
        print("Image loaded in grayscale.")

    print(f"Preprocessed grayscale image shape: {img_gray.shape}")
    return img_gray


def find_contours(img_gray, retrieval_mode=cv2.RETR_CCOMP):
    """
    Find contours in the preprocessed grayscale image.

    Parameters:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    - retrieval_mode (int): OpenCV contour retrieval mode.

    Returns:
    - contours (list of np.ndarray): List of contour points.
    - hierarchy (np.ndarray): Contour hierarchy.
    """
    print("Starting contour detection...")
    # Apply binary thresholding (no inversion)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Optional: Remove noise and smooth the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    print("Applied binary thresholding and morphological closing.")

    # Find contours with hierarchy
    contours_hierarchy = cv2.findContours(thresh, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    # Handle different OpenCV versions
    if len(contours_hierarchy) == 2:
        contours, hierarchy = contours_hierarchy
    elif len(contours_hierarchy) == 3:
        _, contours, hierarchy = contours_hierarchy
    else:
        print("Error: cv2.findContours returned an unexpected number of values.")
        sys.exit(1)

    if not contours:
        print("Error: No contours found in the image.")
        sys.exit(1)

    # Check if hierarchy is returned
    if hierarchy is None:
        print("Error: No hierarchy information found.")
        sys.exit(1)

    hierarchy = hierarchy[0]  # Simplify the hierarchy array
    print(f"Detected {len(contours)} contours.")
    print("Hierarchy for current image:")
    print(hierarchy)

    return contours, hierarchy


def merge_contours_hierarchies(contours1, hierarchy1, contours2, hierarchy2):
    """
    Merge two sets of contours and their hierarchies.

    Parameters:
    - contours1 (list of np.ndarray): First set of contours.
    - hierarchy1 (np.ndarray): First set of hierarchies.
    - contours2 (list of np.ndarray): Second set of contours.
    - hierarchy2 (np.ndarray): Second set of hierarchies.

    Returns:
    - merged_contours (list of np.ndarray): Merged list of contours.
    - merged_hierarchy (np.ndarray): Merged hierarchy.
    """
    print("Merging contours and hierarchies...")
    # Ensure contours1 and contours2 are lists
    if isinstance(contours1, tuple):
        contours1 = list(contours1)
    if isinstance(contours2, tuple):
        contours2 = list(contours2)

    merged_contours = contours1.copy()
    merged_hierarchy = hierarchy1.copy()

    offset = len(contours1)

    # Adjust hierarchy2 indices
    hierarchy2_adjusted = hierarchy2.copy()
    for i in range(len(hierarchy2_adjusted)):
        for j in range(4):
            if hierarchy2_adjusted[i][j] != -1:
                hierarchy2_adjusted[i][j] += offset

    # Combine hierarchies
    merged_hierarchy = np.vstack((merged_hierarchy, hierarchy2_adjusted))

    # Combine contours
    merged_contours += contours2

    print(f"Total contours after merging: {len(merged_contours)}")
    return merged_contours, merged_hierarchy


def find_and_merge_contours(img_gray, retrieval_mode=cv2.RETR_CCOMP):
    """
    Find and merge contours from both the original and inverted grayscale images.

    Parameters:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    - retrieval_mode (int): OpenCV contour retrieval mode.

    Returns:
    - merged_contours (list of np.ndarray): Merged list of contours.
    - merged_hierarchy (np.ndarray): Merged hierarchy.
    - merged_sources (list of str): List indicating the source of each contour ('original' or 'inverted').
    """
    print("Finding contours for the original image...")
    # Find contours for the original image
    contours1, hierarchy1 = find_contours(img_gray, retrieval_mode)
    sources1 = ['original'] * len(contours1)
    print(f"Contours from original image: {len(contours1)}")

    # Invert the grayscale image
    print("Inverting the grayscale image...")
    inverted_img_gray = cv2.bitwise_not(img_gray)

    print("Finding contours for the inverted image...")
    # Find contours for the inverted image
    contours2, hierarchy2 = find_contours(inverted_img_gray, retrieval_mode)
    sources2 = ['inverted'] * len(contours2)
    print(f"Contours from inverted image: {len(contours2)}")

    # Merge the two sets of contours and hierarchies
    merged_contours, merged_hierarchy = merge_contours_hierarchies(
        contours1, hierarchy1, contours2, hierarchy2
    )

    # Merge the sources
    merged_sources = sources1 + sources2
    print(f"Merged sources count: {len(merged_sources)}")

    return merged_contours, merged_hierarchy, merged_sources


def create_svg(contours, hierarchy, sources, svg_path, image_size, scale=1.0):
    """
    Create an SVG file with each mask as a separate path, differentiating between original and inverted contours.

    Parameters:
    - contours (list of np.ndarray): List of contour points.
    - hierarchy (np.ndarray): Contour hierarchy.
    - sources (list of str): List indicating the source of each contour ('original' or 'inverted').
    - svg_path (str): Path to save the SVG file.
    - image_size (tuple): (width, height) of the image in pixels.
    - scale (float): Scaling factor for the SVG dimensions.
    """
    print(f"Creating SVG file at '{svg_path}' with scale factor {scale}...")
    width, height = image_size
    dwg = svgwrite.Drawing(svg_path, size=(f"{width * scale}px", f"{height * scale}px"))
    # dwg.add(dwg.rect(insert=(0, 0), size=(f"{width * scale}px", f"{height * scale}px"), style="fill:#fff;", id="background"))
    print("Added white background to SVG.")

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

            # Determine fill color based on the source
            source = sources[idx]

            dwg.add(dwg.path(
                d=path_data,
                stroke='none',
                fill_rule='evenodd',
                **({"style": f"fill:#fff;"} if source == 'original' else {}),
                id=f"mask_{mask_count}",
            ))

            if mask_count % 10 == 0:
                print(f"Added {mask_count} masks to SVG...")

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
    print("Preparing to display masks...")
    if not masks:
        print("No masks to display.")
        return

    num_masks_to_display = min(max_display, len(masks))  # Display up to max_display masks
    print(f"Displaying {num_masks_to_display} masks...")

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
    print("Generating mask images for display...")
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
    print(f"Generated {len(masks)} mask images.")
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

    # Verify image path
    if not os.path.isfile(args.image_path):
        print(f"Error: The file '{args.image_path}' does not exist.")
        sys.exit(1)

    print("Starting raster to SVG conversion...")
    # Preprocess the image
    img_gray = preprocess_image(args.image_path, handle_transparency=args.handle_transparency)

    # Get image dimensions
    height, width = img_gray.shape[:2]
    print(f"Image dimensions: Width={width}, Height={height}")

    # Map string retrieval mode to OpenCV constants
    retrieval_modes = {
        'EXTERNAL': cv2.RETR_EXTERNAL,
        'LIST': cv2.RETR_LIST,
        'CCOMP': cv2.RETR_CCOMP,
        'TREE': cv2.RETR_TREE
    }

    retrieval_mode = retrieval_modes.get(args.retrieval_mode.upper(), cv2.RETR_CCOMP)
    print(f"Using contour retrieval mode: {args.retrieval_mode}")

    # Find and merge contours from original and inverted images
    merged_contours, merged_hierarchy, merged_sources = find_and_merge_contours(
        img_gray=img_gray,
        retrieval_mode=retrieval_mode
    )

    print("Merged Hierarchy:")
    print(merged_hierarchy)
    print("Merged Sources:")
    print(merged_sources)

    # Create SVG from merged contours and sources
    create_svg(
        contours=merged_contours,
        hierarchy=merged_hierarchy,
        sources=merged_sources,
        svg_path=args.svg_path,
        image_size=(width, height),
        scale=args.scale
    )

    # Optionally display masks
    if args.display:
        # Generate mask images for display purposes
        masks = generate_mask_images(merged_contours, merged_hierarchy, img_gray.shape)
        display_masks(masks, max_display=args.max_display)

    print("Raster to SVG conversion completed successfully.")


if __name__ == "__main__":
    main()
