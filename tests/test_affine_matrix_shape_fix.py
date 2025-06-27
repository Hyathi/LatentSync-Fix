#!/usr/bin/env python3
"""
Test script to verify the affine matrix shape fix in restore_img method.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from latentsync.utils.affine_transform import AlignRestore

def test_affine_matrix_shapes():
    """Test that restore_img handles different affine matrix shapes correctly."""
    print("🧪 Testing affine matrix shape handling in restore_img...")
    print("=" * 60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create AlignRestore instance
        restorer = AlignRestore(align_points=3, resolution=256, device=device)
        
        # Create dummy input data
        input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = torch.randn(3, 256, 256, device=device)  # Face tensor
        
        # Test different affine matrix shapes
        test_cases = [
            {
                "name": "Shape [2, 3] - No batch dimension",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32),
                "expected_shape": (1, 2, 3)
            },
            {
                "name": "Shape [1, 2, 3] - Correct batch dimension",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32).unsqueeze(0),
                "expected_shape": (1, 2, 3)
            },
            {
                "name": "Shape [1, 1, 2, 3] - Extra dimension",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                "expected_shape": (1, 2, 3)
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing {test_case['name']}")
            print(f"   Input shape: {test_case['matrix'].shape}")
            
            try:
                # This should work without throwing the shape error
                result = restorer.restore_img(input_img, face, test_case['matrix'])
                
                print(f"   ✅ Success! Output shape: {result.shape}")
                print(f"   Expected matrix shape after processing: {test_case['expected_shape']}")
                
                # Verify the result is a valid image
                assert isinstance(result, np.ndarray), "Result should be numpy array"
                assert result.shape == (256, 256, 3), f"Result should be (256, 256, 3), got {result.shape}"
                assert result.dtype == np.uint8, f"Result should be uint8, got {result.dtype}"
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False
        
        print(f"\n✅ All affine matrix shape tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_affine_matrix():
    """Test that numpy affine matrices are handled correctly."""
    print(f"\n🧪 Testing numpy affine matrix handling...")
    print("=" * 60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        restorer = AlignRestore(align_points=3, resolution=256, device=device)
        
        # Create dummy input data
        input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = torch.randn(3, 256, 256, device=device)
        
        # Test numpy affine matrix
        numpy_matrix = np.eye(2, 3, dtype=np.float32)
        print(f"Testing numpy matrix shape: {numpy_matrix.shape}")
        
        result = restorer.restore_img(input_img, face, numpy_matrix)
        
        print(f"✅ Numpy matrix test passed! Output shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Numpy matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Affine Matrix Shape Fix")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_affine_matrix_shapes()
    test2_passed = test_numpy_affine_matrix()
    
    print("\n" + "=" * 80)
    print("📋 Test Results Summary:")
    print(f"   Affine Matrix Shapes: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Numpy Matrix Handling: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The affine matrix shape fix is working correctly!")
        print("\n📝 The fix handles:")
        print("   - [2, 3] matrices (adds batch dimension)")
        print("   - [1, 2, 3] matrices (already correct)")
        print("   - [1, 1, 2, 3] matrices (removes extra dimension)")
        print("   - numpy arrays (converts to tensor with batch dimension)")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)
