#!/usr/bin/env python3
"""
Faster image alignment script using coarse-to-fine search
"""

from PIL import Image
import numpy as np
from scipy.ndimage import zoom, shift

def load_and_preprocess(image_path, target_size=(400, 300)):
    """Load image and convert to grayscale numpy array"""
    img = Image.open(image_path)
    # Resize to smaller size for faster processing
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # Convert to grayscale
    img_gray = img.convert('L')
    return np.array(img_gray)

def calculate_correlation(img1, img2):
    """Fast correlation calculation"""
    # Use center crop for correlation to avoid edge effects
    h, w = img1.shape
    crop_h, crop_w = h // 2, w // 2
    start_h, start_w = h // 4, w // 4
    
    img1_crop = img1[start_h:start_h+crop_h, start_w:start_w+crop_w]
    img2_crop = img2[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Simple correlation
    img1_norm = (img1_crop - np.mean(img1_crop)) / (np.std(img1_crop) + 1e-6)
    img2_norm = (img2_crop - np.mean(img2_crop)) / (np.std(img2_crop) + 1e-6)
    
    return np.mean(img1_norm * img2_norm)

def quick_align(before_img, after_img):
    """Quick alignment using coarse search"""
    best_correlation = -1
    best_params = {'scale': 1.0, 'offset_x': 0, 'offset_y': 0}
    
    h, w = before_img.shape
    
    # Coarse search - test fewer combinations
    scales = [0.95, 1.0, 1.05, 1.1, 1.15]
    offsets = [-20, -10, 0, 10, 20]
    
    print(f"Testing {len(scales) * len(offsets) * len(offsets)} combinations...")
    
    for scale in scales:
        # Scale the after image
        scaled_img = zoom(after_img, scale, order=1)
        
        # Crop or pad to match original size
        if scaled_img.shape[0] > h:
            start_y = (scaled_img.shape[0] - h) // 2
            scaled_img = scaled_img[start_y:start_y + h, :]
        else:
            pad_y = (h - scaled_img.shape[0]) // 2
            pad_bottom = h - scaled_img.shape[0] - pad_y
            scaled_img = np.pad(scaled_img, ((pad_y, pad_bottom), (0, 0)), mode='edge')
            
        if scaled_img.shape[1] > w:
            start_x = (scaled_img.shape[1] - w) // 2
            scaled_img = scaled_img[:, start_x:start_x + w]
        else:
            pad_x = (w - scaled_img.shape[1]) // 2
            pad_right = w - scaled_img.shape[1] - pad_x
            scaled_img = np.pad(scaled_img, ((0, 0), (pad_x, pad_right)), mode='edge')
        
        for offset_x in offsets:
            for offset_y in offsets:
                # Apply offset
                shifted_img = shift(scaled_img, (offset_y, offset_x), mode='nearest')
                
                # Calculate correlation
                correlation = calculate_correlation(before_img, shifted_img)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_params = {
                        'scale': scale,
                        'offset_x': offset_x,
                        'offset_y': offset_y,
                        'correlation': correlation
                    }
                    print(f"  Better match: scale={scale:.2f}, x={offset_x}, y={offset_y}, corr={correlation:.3f}")
    
    return best_params

def main():
    print("=== Fast Image Alignment Analysis ===\n")
    
    # Analyze lawn images
    print("Processing LAWN images...")
    lawn_before = load_and_preprocess('lawn-transformation-before.jpg')
    lawn_after = load_and_preprocess('lawn-transformation-after.jpg')
    
    lawn_params = quick_align(lawn_before, lawn_after)
    
    print(f"\nBest alignment for LAWN images:")
    print(f"  Scale: {lawn_params['scale']:.3f}")
    print(f"  Offset X: {lawn_params['offset_x']}px")
    print(f"  Offset Y: {lawn_params['offset_y']}px")
    print(f"  Correlation: {lawn_params.get('correlation', 0):.4f}")
    
    # Analyze garden images (note they're swapped in HTML)
    print("\n" + "="*40)
    print("Processing GARDEN images...")
    garden_before = load_and_preprocess('garden-transformation-after.jpg')  # This is actually the "before"
    garden_after = load_and_preprocess('garden-transformation-before.jpg')   # This is actually the "after"
    
    garden_params = quick_align(garden_before, garden_after)
    
    print(f"\nBest alignment for GARDEN images:")
    print(f"  Scale: {garden_params['scale']:.3f}")
    print(f"  Offset X: {garden_params['offset_x']}px")
    print(f"  Offset Y: {garden_params['offset_y']}px")
    print(f"  Correlation: {garden_params.get('correlation', 0):.4f}")
    
    # Output CSS transforms
    print("\n" + "="*40)
    print("CSS TRANSFORMS TO APPLY:\n")
    
    print(f"Lawn after image:")
    print(f'  transform: scale({lawn_params["scale"]:.2f}) translateX({lawn_params["offset_x"]}px) translateY({lawn_params["offset_y"]}px);')
    
    print(f"\nGarden after image:")
    print(f'  transform: scale({garden_params["scale"]:.2f}) translateX({garden_params["offset_x"]}px) translateY({garden_params["offset_y"]}px);')

if __name__ == "__main__":
    main()