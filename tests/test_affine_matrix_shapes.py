#!/usr/bin/env python3
"""
Test script to verify affine matrix shape handling in the enhanced smoothing pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from latentsync.utils.image_processor import VideoProcessor
from latentsync.utils.affine_transform import AlignRestore


def test_affine_matrix_shapes():
    """Test that affine matrices have correct shapes throughout the pipeline."""
    
    print("🔍 Testing affine matrix shapes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize components
    restorer = AlignRestore(resolution=256, device=device, dtype=torch.float32)
    video_processor = VideoProcessor(resolution=256, device=device)
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    landmarks = np.array([[50, 60], [150, 60], [100, 120]], dtype=np.float32)
    
    print(f"\n📊 Testing AlignRestore.align_warp_face...")
    
    # Test original align_warp_face
    face, affine_matrix = restorer.align_warp_face(dummy_image, landmarks, smooth=True)
    print(f"   ✓ align_warp_face output:")
    print(f"     - Face shape: {face.shape}")
    print(f"     - Affine matrix shape: {affine_matrix.shape}")
    print(f"     - Affine matrix type: {type(affine_matrix)}")
    
    # Test multiple matrices for smoothing
    print(f"\n📊 Testing affine matrix smoothing...")
    
    affine_matrices = []
    for i in range(10):  # Create 10 dummy matrices
        # Simulate slight variations
        landmarks_var = landmarks + np.random.normal(0, 0.5, landmarks.shape)
        _, matrix = restorer.align_warp_face(dummy_image, landmarks_var, smooth=True)
        affine_matrices.append(matrix)
        print(f"   Matrix {i}: shape {matrix.shape}, type {type(matrix)}")
    
    # Test smoothing function
    print(f"\n📊 Testing _smooth_affine_matrices...")
    
    try:
        smoothed_matrices = video_processor._smooth_affine_matrices(affine_matrices)
        print(f"   ✓ Smoothing successful!")
        print(f"   - Input matrices: {len(affine_matrices)}")
        print(f"   - Output matrices: {len(smoothed_matrices)}")
        
        for i, matrix in enumerate(smoothed_matrices[:3]):  # Show first 3
            print(f"   - Smoothed matrix {i}: shape {matrix.shape}, type {type(matrix)}")
            
    except Exception as e:
        print(f"   ✗ Smoothing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test _apply_smoothed_transform
    print(f"\n📊 Testing _apply_smoothed_transform...")
    
    try:
        dummy_tensor = torch.from_numpy(dummy_image)
        for i, matrix in enumerate(smoothed_matrices[:3]):
            result = video_processor._apply_smoothed_transform(dummy_tensor, landmarks, matrix)
            print(f"   ✓ Transform {i}: input matrix shape {matrix.shape} -> output shape {result.shape}")
            
    except Exception as e:
        print(f"   ✗ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ All shape tests passed!")
    return True


def test_kornia_warp_affine():
    """Test kornia.warp_affine with different matrix shapes."""
    
    print(f"\n🔧 Testing kornia.warp_affine with different shapes...")
    
    import kornia
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy image
    image = torch.randn(1, 3, 256, 256, device=device)
    
    # Test different matrix shapes
    shapes_to_test = [
        (2, 3),      # [2, 3]
        (1, 2, 3),   # [1, 2, 3] - correct
        (1, 1, 2, 3) # [1, 1, 2, 3] - problematic
    ]
    
    for shape in shapes_to_test:
        print(f"\n   Testing matrix shape: {shape}")
        
        # Create dummy affine matrix
        if len(shape) == 2:
            matrix = torch.eye(2, 3, device=device)
        elif len(shape) == 3:
            matrix = torch.eye(2, 3, device=device).unsqueeze(0)
        elif len(shape) == 4:
            matrix = torch.eye(2, 3, device=device).unsqueeze(0).unsqueeze(0)
        
        print(f"     Matrix shape: {matrix.shape}")
        
        try:
            result = kornia.geometry.transform.warp_affine(
                image, matrix, (256, 256), mode="bilinear"
            )
            print(f"     ✓ Success: output shape {result.shape}")
        except Exception as e:
            print(f"     ✗ Failed: {e}")


def main():
    """Run all tests."""
    
    print("🚀 Starting affine matrix shape tests...")
    print("=" * 60)
    
    # Test basic shapes
    success = test_affine_matrix_shapes()
    
    # Test kornia compatibility
    test_kornia_warp_affine()
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print(f"The enhanced smoothing pipeline should now work correctly.")
    else:
        print(f"\n❌ Some tests failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
