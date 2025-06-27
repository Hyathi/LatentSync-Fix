#!/usr/bin/env python3
"""
Test script for the improved video smoothing functionality.

This script demonstrates the different smoothing methods and allows for
easy comparison of results.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from latentsync.utils.image_processor import VideoProcessor
from latentsync.utils.util import write_video


def test_smoothing_methods(input_video: str, output_dir: str = "smoothing_test_outputs", 
                          resolution: int = 256, device: str = "cuda"):
    """
    Test all smoothing methods on the input video and save results for comparison.
    
    Args:
        input_video: Path to input video file
        output_dir: Directory to save output videos
        resolution: Video resolution for processing
        device: Device to use for processing ('cuda' or 'cpu')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video processor
    print(f"Initializing VideoProcessor with resolution={resolution}, device={device}")
    video_processor = VideoProcessor(resolution=resolution, device=device)
    
    # Test 1: Original method (backward compatibility)
    print("\n1. Testing original smoothing method...")
    start_time = time.time()
    try:
        video_frames_original = video_processor.affine_transform_video_smooth(
            input_video, enhanced_smoothing=False
        )
        original_time = time.time() - start_time
        output_path = os.path.join(output_dir, "original_smoothing.mp4")
        write_video(output_path, video_frames_original, fps=25)
        print(f"   ✓ Original method completed in {original_time:.2f}s")
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Original method failed: {e}")
        original_time = None
    
    # Test 2: Enhanced smoothing method
    print("\n2. Testing enhanced smoothing method...")
    start_time = time.time()
    try:
        video_frames_enhanced = video_processor.affine_transform_video_smooth(
            input_video, 
            enhanced_smoothing=True,
            landmark_smoothing_params={
                'primary_window': 15,
                'primary_poly': 2,
                'secondary_window': 7,
                'secondary_poly': 1,
                'gaussian_sigma': 1.0
            },
            matrix_smoothing_params={
                'window_length': 9,
                'poly_order': 2,
                'gaussian_sigma': 0.8
            }
        )
        enhanced_time = time.time() - start_time
        output_path = os.path.join(output_dir, "enhanced_smoothing.mp4")
        write_video(output_path, video_frames_enhanced, fps=25)
        print(f"   ✓ Enhanced method completed in {enhanced_time:.2f}s")
        print(f"   ✓ Saved to: {output_path}")
        
        if original_time:
            overhead = ((enhanced_time - original_time) / original_time) * 100
            print(f"   ℹ Processing overhead: {overhead:.1f}%")
    except Exception as e:
        print(f"   ✗ Enhanced method failed: {e}")
        enhanced_time = None
    
    # Test 3: Ultra-smooth method
    print("\n3. Testing ultra-smooth method...")
    start_time = time.time()
    try:
        video_frames_ultra = video_processor.affine_transform_video_ultra_smooth(
            input_video,
            temporal_alpha=0.3
        )
        ultra_time = time.time() - start_time
        output_path = os.path.join(output_dir, "ultra_smooth.mp4")
        write_video(output_path, video_frames_ultra, fps=25)
        print(f"   ✓ Ultra-smooth method completed in {ultra_time:.2f}s")
        print(f"   ✓ Saved to: {output_path}")
        
        if original_time:
            overhead = ((ultra_time - original_time) / original_time) * 100
            print(f"   ℹ Processing overhead: {overhead:.1f}%")
    except Exception as e:
        print(f"   ✗ Ultra-smooth method failed: {e}")
        ultra_time = None
    
    # Test 4: Custom parameters for high-motion content
    print("\n4. Testing high-motion optimized settings...")
    start_time = time.time()
    try:
        video_frames_motion = video_processor.affine_transform_video_ultra_smooth(
            input_video,
            landmark_params={
                'primary_window': 11,
                'primary_poly': 2,
                'secondary_window': 5,
                'secondary_poly': 1,
                'gaussian_sigma': 0.8
            },
            matrix_params={
                'window_length': 7,
                'poly_order': 2,
                'gaussian_sigma': 0.6
            },
            temporal_alpha=0.2
        )
        motion_time = time.time() - start_time
        output_path = os.path.join(output_dir, "high_motion_optimized.mp4")
        write_video(output_path, video_frames_motion, fps=25)
        print(f"   ✓ High-motion method completed in {motion_time:.2f}s")
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ High-motion method failed: {e}")
    
    print(f"\n🎉 Testing complete! Check the '{output_dir}' directory for results.")
    print("\nRecommended comparison order:")
    print("1. original_smoothing.mp4 (baseline)")
    print("2. enhanced_smoothing.mp4 (improved)")
    print("3. ultra_smooth.mp4 (maximum smoothing)")
    print("4. high_motion_optimized.mp4 (motion-specific)")


def main():
    parser = argparse.ArgumentParser(description="Test video smoothing improvements")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output-dir", default="smoothing_test_outputs", 
                       help="Output directory for test results")
    parser.add_argument("--resolution", type=int, default=256, 
                       help="Processing resolution")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device to use for processing")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_video):
        print(f"Error: Input video '{args.input_video}' not found!")
        sys.exit(1)
    
    print("🚀 Starting video smoothing improvement tests...")
    print(f"Input video: {args.input_video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Device: {args.device}")
    
    test_smoothing_methods(
        input_video=args.input_video,
        output_dir=args.output_dir,
        resolution=args.resolution,
        device=args.device
    )


if __name__ == "__main__":
    main()
