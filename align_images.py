#!/usr/bin/env python3
"""
Image alignment script to find optimal scale and translation
for before/after image pairs using normalized cross-correlation.
"""

from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import zoom, shift
import sys

def load_and_preprocess(image_path, target_size=(600, 400)):
    """Load image and convert to grayscale numpy array"""
    img = Image.open(image_path)
    # Resize to manageable size for processing
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # Convert to grayscale
    img_gray = img.convert('L')
    return np.array(img_gray)

def normalized_cross_correlation(img1, img2):
    """Calculate normalized cross-correlation between two images"""
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    
    # Calculate correlation
    correlation = np.sum(img1_norm * img2_norm) / img1_norm.size
    return correlation

def find_best_alignment(before_img, after_img, scale_range=(0.95, 1.15), 
                        offset_range=(-30, 30), scale_step=0.01, offset_step=2):
    """
    Find the best scale and offset to align after_img with before_img
    """
    best_correlation = -1
    best_params = {'scale': 1.0, 'offset_x': 0, 'offset_y': 0}
    
    # Original image dimensions
    h, w = before_img.shape
    
    # Test different scales
    scales = np.arange(scale_range[0], scale_range[1], scale_step)
    offsets = range(offset_range[0], offset_range[1], offset_step)
    
    total_iterations = len(scales) * len(offsets) * len(offsets)
    iteration = 0
    
    print(f"Testing {len(scales)} scales x {len(offsets)}x{len(offsets)} offsets = {total_iterations} combinations...")
    
    for scale in scales:
        # Scale the after image
        scaled_shape = (int(h * scale), int(w * scale))
        
        # Skip if scaled image is too small or too large
        if scaled_shape[0] < h * 0.8 or scaled_shape[0] > h * 1.3:
            continue
        if scaled_shape[1] < w * 0.8 or scaled_shape[1] > w * 1.3:
            continue
            
        scaled_img = zoom(after_img, scale, order=1)
        
        # Crop or pad to match original size
        if scaled_img.shape[0] > h:
            # Crop
            start_y = (scaled_img.shape[0] - h) // 2
            scaled_img = scaled_img[start_y:start_y + h, :]
        elif scaled_img.shape[0] < h:
            # Pad
            pad_y = (h - scaled_img.shape[0]) // 2
            scaled_img = np.pad(scaled_img, ((pad_y, h - scaled_img.shape[0] - pad_y), (0, 0)), mode='edge')
            
        if scaled_img.shape[1] > w:
            # Crop
            start_x = (scaled_img.shape[1] - w) // 2
            scaled_img = scaled_img[:, start_x:start_x + w]
        elif scaled_img.shape[1] < w:
            # Pad
            pad_x = (w - scaled_img.shape[1]) // 2
            scaled_img = np.pad(scaled_img, ((0, 0), (pad_x, w - scaled_img.shape[1] - pad_x)), mode='edge')
        
        # Test different offsets
        for offset_x in offsets:
            for offset_y in offsets:
                iteration += 1
                if iteration % 100 == 0:
                    print(f"Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")
                
                # Apply offset
                shifted_img = shift(scaled_img, (offset_y, offset_x), mode='constant', cval=0)
                
                # Calculate correlation
                try:
                    correlation = normalized_cross_correlation(before_img, shifted_img)
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_params = {
                            'scale': scale,
                            'offset_x': offset_x,
                            'offset_y': offset_y,
                            'correlation': correlation
                        }
                except:
                    continue
    
    return best_params

def main():
    print("=== Image Alignment Analysis ===\n")
    
    # Analyze lawn images
    print("Processing LAWN images...")
    lawn_before = load_and_preprocess('lawn-transformation-before.jpg')
    lawn_after = load_and_preprocess('lawn-transformation-after.jpg')
    
    print(f"Lawn before shape: {lawn_before.shape}")
    print(f"Lawn after shape: {lawn_after.shape}")
    
    lawn_params = find_best_alignment(lawn_before, lawn_after)
    
    print(f"\nBest alignment for LAWN images:")
    print(f"  Scale: {lawn_params['scale']:.3f}")
    print(f"  Offset X: {lawn_params['offset_x']}px")
    print(f"  Offset Y: {lawn_params['offset_y']}px")
    print(f"  Correlation: {lawn_params.get('correlation', 0):.4f}")
    
    # Analyze garden images
    print("\n" + "="*40)
    print("Processing GARDEN images...")
    garden_before = load_and_preprocess('garden-transformation-after.jpg')  # Note: swapped
    garden_after = load_and_preprocess('garden-transformation-before.jpg')   # Note: swapped
    
    print(f"Garden before shape: {garden_before.shape}")
    print(f"Garden after shape: {garden_after.shape}")
    
    garden_params = find_best_alignment(garden_before, garden_after)
    
    print(f"\nBest alignment for GARDEN images:")
    print(f"  Scale: {garden_params['scale']:.3f}")
    print(f"  Offset X: {garden_params['offset_x']}px")
    print(f"  Offset Y: {garden_params['offset_y']}px")
    print(f"  Correlation: {garden_params.get('correlation', 0):.4f}")
    
    # Output CSS transforms
    print("\n" + "="*40)
    print("CSS TRANSFORMS TO APPLY:\n")
    
    print(f"Lawn after image:")
    print(f'  style="transform: scale({lawn_params["scale"]:.3f}) translateX({lawn_params["offset_x"]}px) translateY({lawn_params["offset_y"]}px);"')
    
    print(f"\nGarden after image:")
    print(f'  style="transform: scale({garden_params["scale"]:.3f}) translateX({garden_params["offset_x"]}px) translateY({garden_params["offset_y"]}px);"')

if __name__ == "__main__":
    main()