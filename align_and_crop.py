#!/usr/bin/env python3
"""
Create aligned and cropped versions of before/after images
"""

from PIL import Image
import numpy as np

def align_lawn_images():
    """Align lawn before and after images"""
    print("Processing lawn images...")
    
    # Load images
    before = Image.open('lawn-transformation-before.jpg')
    after = Image.open('lawn-transformation-after.jpg')
    
    print(f"Before size: {before.size}")
    print(f"After size: {after.size}")
    
    # The after image needs to be scaled up and shifted to align the tree
    # Based on visual inspection, scale up by 12% and shift
    after_scaled = after.resize((int(after.width * 1.12), int(after.height * 1.12)), Image.Resampling.LANCZOS)
    
    # Crop to same size as before image, with offset to align the tree
    # The tree is slightly left and up in the after image
    left_offset = 50  # Shift left
    top_offset = 30   # Shift up
    
    after_cropped = after_scaled.crop((
        left_offset,
        top_offset,
        left_offset + before.width,
        top_offset + before.height
    ))
    
    # Save aligned versions
    before.save('lawn-before-aligned.jpg', quality=90)
    after_cropped.save('lawn-after-aligned.jpg', quality=90)
    
    print(f"Saved aligned lawn images: {before.size}")
    

def align_garden_images():
    """Align garden before and after images"""
    print("\nProcessing garden images...")
    
    # Note: These are swapped in the display
    before = Image.open('garden-transformation-after.jpg')  # This is actually "before"
    after = Image.open('garden-transformation-before.jpg')  # This is actually "after"
    
    print(f"Before size: {before.size}")
    print(f"After size: {after.size}")
    
    # Scale the after image slightly
    after_scaled = after.resize((int(after.width * 1.05), int(after.height * 1.05)), Image.Resampling.LANCZOS)
    
    # Crop with slight offset to align house corner
    left_offset = 20
    top_offset = 0
    
    after_cropped = after_scaled.crop((
        left_offset,
        top_offset,
        left_offset + before.width,
        top_offset + before.height
    ))
    
    # Save aligned versions
    before.save('garden-before-aligned.jpg', quality=90)
    after_cropped.save('garden-after-aligned.jpg', quality=90)
    
    print(f"Saved aligned garden images: {before.size}")

if __name__ == "__main__":
    align_lawn_images()
    align_garden_images()
    print("\nDone! Created aligned image files.")