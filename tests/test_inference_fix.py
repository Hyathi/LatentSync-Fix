#!/usr/bin/env python3
"""
Test script to verify that the inference pipeline works with the affine matrix fix.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.util import read_video

def test_inference_pipeline():
    """Test that the restore_img method works with different affine matrix shapes."""
    print("🧪 Testing restore_img method with different affine matrix shapes...")
    print("=" * 70)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Create AlignRestore (this is what's used in the pipeline)
        restorer = AlignRestore(align_points=3, resolution=256, device=device)
        
        # Create test data that simulates the inference scenario
        print("Creating test data that simulates the inference scenario...")

        # Create synthetic input image and face
        input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = torch.randn(3, 256, 256, device=device)

        # Test different affine matrix shapes that occur in the pipeline
        test_cases = [
            {
                "name": "Enhanced smoothing matrix [2, 3]",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32),
                "description": "This is the shape returned by enhanced smoothing"
            },
            {
                "name": "Original pipeline matrix [1, 2, 3]",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32).unsqueeze(0),
                "description": "This is the shape from original pipeline"
            },
            {
                "name": "Numpy matrix [2, 3]",
                "matrix": np.eye(2, 3, dtype=np.float32),
                "description": "This tests numpy input handling"
            }
        ]

        print(f"\n🔧 Testing restore_img with different matrix shapes...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"   {test_case['description']}")
            print(f"   Matrix shape: {test_case['matrix'].shape}")
            print(f"   Matrix type: {type(test_case['matrix'])}")

            try:
                # This is the exact call that was failing in the inference script
                result = restorer.restore_img(input_img, face, test_case['matrix'])

                print(f"   ✅ Success! Output shape: {result.shape}")
                print(f"   Output dtype: {result.dtype}")

                # Verify the result
                assert isinstance(result, np.ndarray), "Result should be numpy array"
                assert result.shape == (256, 256, 3), f"Result should be (256, 256, 3), got {result.shape}"
                assert result.dtype == np.uint8, f"Result should be uint8, got {result.dtype}"

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False
        
        print(f"\n✅ All restore_img tests passed! The fix is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error conditions."""
    print(f"\n🧪 Testing edge cases and error conditions...")
    print("=" * 70)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        restorer = AlignRestore(align_points=3, resolution=256, device=device)

        # Create test data
        input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = torch.randn(3, 256, 256, device=device)

        # Test edge cases
        test_cases = [
            {
                "name": "4D matrix [1, 1, 2, 3]",
                "matrix": torch.eye(2, 3, device=device).unsqueeze(0).unsqueeze(0),
                "should_work": True
            },
            {
                "name": "Wrong shape [3, 3]",
                "matrix": torch.eye(3, 3, device=device),
                "should_work": False
            },
            {
                "name": "Wrong shape [2, 2]",
                "matrix": torch.eye(2, 2, device=device),
                "should_work": False
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing {test_case['name']}")
            print(f"   Matrix shape: {test_case['matrix'].shape}")
            print(f"   Expected to work: {test_case['should_work']}")

            try:
                result = restorer.restore_img(input_img, face, test_case['matrix'])

                if test_case['should_work']:
                    print(f"   ✅ Success as expected! Output shape: {result.shape}")
                else:
                    print(f"   ⚠️ Unexpected success - this should have failed")
                    return False

            except Exception as e:
                if not test_case['should_work']:
                    print(f"   ✅ Failed as expected: {type(e).__name__}")
                else:
                    print(f"   ❌ Unexpected failure: {e}")
                    return False

        print(f"\n✅ All edge case tests passed!")
        return True

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Inference Pipeline with Affine Matrix Fix")
    print("=" * 80)

    # Run tests
    test1_passed = test_inference_pipeline()
    test2_passed = test_edge_cases()

    print("\n" + "=" * 80)
    print("📋 Test Results Summary:")
    print(f"   Restore Image Method: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Edge Cases: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The affine matrix fix is working correctly!")
        print("\n📝 The fix resolves:")
        print("   - ValueError: Input matrix must be a Bx2x3 tensor")
        print("   - Handles [2, 3], [1, 2, 3], and [1, 1, 2, 3] matrix shapes")
        print("   - Works with numpy arrays and torch tensors")
        print("   - Properly validates matrix shapes and provides clear error messages")
        print("\n🚀 The inference script should now work without the tensor shape error!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)
