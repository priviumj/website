#!/usr/bin/env python3
"""
Use SIFT/ORB feature matching to find the optimal transformation between images
"""

import cv2
import numpy as np
from PIL import Image

def find_homography_transform(img1_path, img2_path, output_prefix):
    """
    Find the transformation needed to align img2 to img1 using feature matching
    """
    print(f"\nProcessing {output_prefix} images...")
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    print(f"Image 1 size: {img1.shape[:2]}")
    print(f"Image 2 size: {img2.shape[:2]}")
    
    # Try SIFT first (best but patented), fall back to ORB if needed
    try:
        # SIFT detector
        detector = cv2.SIFT_create()
        print("Using SIFT feature detector...")
    except:
        # ORB detector as fallback
        detector = cv2.ORB_create(nfeatures=1000)
        print("Using ORB feature detector...")
    
    # Find keypoints and descriptors
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    print(f"Found {len(kp1)} keypoints in image 1")
    print(f"Found {len(kp2)} keypoints in image 2")
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Not enough features found!")
        return None
    
    # Match features
    if isinstance(detector, cv2.SIFT):
        # For SIFT, use FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # For ORB, use BFMatcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Find matches
    try:
        matches = matcher.knnMatch(desc1, desc2, k=2)
    except:
        print("Matching failed, trying brute force...")
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:20]
    else:
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches")
    
    if len(good_matches) < 4:
        print("Not enough good matches found!")
        return None
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        print("Could not find homography!")
        return None
    
    # Apply transformation to align img2 to img1
    h, w = img1.shape[:2]
    aligned_img2 = cv2.warpPerspective(img2, M, (w, h))
    
    # Save aligned image
    output_path = f'{output_prefix}-after-aligned.jpg'
    cv2.imwrite(output_path, aligned_img2)
    
    # Also save the before image for consistency
    before_output = f'{output_prefix}-before-aligned.jpg'
    cv2.imwrite(before_output, img1)
    
    # Extract transformation parameters for CSS (approximation)
    # This is a simplification - homography is more complex than CSS transforms
    scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
    translate_x = M[0, 2]
    translate_y = M[1, 2]
    
    print(f"\nTransformation matrix:")
    print(M)
    print(f"\nApproximate CSS transform:")
    print(f"  Scale: {(scale_x + scale_y)/2:.3f}")
    print(f"  Translate X: {translate_x:.1f}px")
    print(f"  Translate Y: {translate_y:.1f}px")
    print(f"\nSaved aligned images: {before_output}, {output_path}")
    
    return M

def main():
    print("=== Feature-Based Image Alignment ===")
    
    # Process lawn images
    M_lawn = find_homography_transform(
        'lawn-transformation-before.jpg',
        'lawn-transformation-after.jpg',
        'lawn'
    )
    
    # Process garden images (swapped as noted)
    M_garden = find_homography_transform(
        'garden-transformation-after.jpg',  # This is actually before
        'garden-transformation-before.jpg', # This is actually after
        'garden'
    )
    
    print("\n=== Alignment Complete ===")
    print("Created aligned image files that can be used directly in the slider")

if __name__ == "__main__":
    main()