#!/usr/bin/env python3
"""
Test script to verify device compatibility in the restore_img method.
This tests the fix for the device mismatch error.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from latentsync.utils.affine_transform import AlignRestore

def test_device_compatibility():
    """Test that all tensors are on the same device in restore_img."""
    print("🧪 Testing device compatibility in restore_img...")
    print("=" * 70)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create AlignRestore
        restorer = AlignRestore(align_points=3, resolution=256, device=device)
        
        # Create test data
        input_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = torch.randn(3, 256, 256, device=device)
        
        # Test different matrix types and devices
        test_cases = [
            {
                "name": "CPU tensor [2, 3]",
                "matrix": torch.eye(2, 3, dtype=torch.float32),  # CPU tensor
                "description": "Matrix starts on CPU, should be moved to device"
            },
            {
                "name": "CUDA tensor [2, 3]" if device == 'cuda' else "Device tensor [2, 3]",
                "matrix": torch.eye(2, 3, device=device, dtype=torch.float32),  # Device tensor
                "description": "Matrix already on correct device"
            },
            {
                "name": "Numpy array [2, 3]",
                "matrix": np.eye(2, 3, dtype=np.float32),  # Numpy array
                "description": "Numpy array should be converted and moved to device"
            },
            {
                "name": "CPU tensor [1, 2, 3]",
                "matrix": torch.eye(2, 3, dtype=torch.float32).unsqueeze(0),  # CPU tensor with batch
                "description": "Batched matrix on CPU, should be moved to device"
            }
        ]
        
        print(f"\n🔧 Testing device compatibility with different matrix types...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"   {test_case['description']}")
            print(f"   Matrix shape: {test_case['matrix'].shape}")
            print(f"   Matrix type: {type(test_case['matrix'])}")
            
            if isinstance(test_case['matrix'], torch.Tensor):
                print(f"   Matrix device: {test_case['matrix'].device}")
            
            try:
                # This should work without device mismatch errors
                result = restorer.restore_img(input_img, face, test_case['matrix'])
                
                print(f"   ✅ Success! Output shape: {result.shape}")
                print(f"   Output dtype: {result.dtype}")
                
                # Verify the result
                assert isinstance(result, np.ndarray), "Result should be numpy array"
                assert result.shape == (256, 256, 3), f"Result should be (256, 256, 3), got {result.shape}"
                assert result.dtype == np.uint8, f"Result should be uint8, got {result.dtype}"
                
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"   ❌ Device mismatch error: {e}")
                    return False
                else:
                    print(f"   ❌ Other RuntimeError: {e}")
                    return False
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                return False
        
        print(f"\n✅ All device compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_device_scenario():
    """Test the specific scenario that was causing the device error."""
    print(f"\n🧪 Testing mixed device scenario...")
    print("=" * 70)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        restorer = AlignRestore(align_points=3, resolution=256, device=device)
        
        # Simulate the exact scenario from the error:
        # - input_img: numpy array (CPU)
        # - face: tensor on device
        # - affine_matrix: tensor potentially on different device
        
        input_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)  # Different size
        face = torch.randn(3, 256, 256, device=device)
        
        # Test matrix that might be on CPU (common in pipeline)
        affine_matrix = torch.eye(2, 3, dtype=torch.float32)  # CPU tensor
        
        print(f"Input image shape: {input_img.shape} (numpy array)")
        print(f"Face shape: {face.shape} (device: {face.device})")
        print(f"Affine matrix shape: {affine_matrix.shape} (device: {affine_matrix.device})")
        
        print(f"\n🔧 Testing restore_img with mixed devices...")
        
        try:
            result = restorer.restore_img(input_img, face, affine_matrix)
            
            print(f"✅ Success! No device mismatch error!")
            print(f"   Output shape: {result.shape}")
            print(f"   Output dtype: {result.dtype}")
            
            # Verify output matches input image dimensions
            assert result.shape == input_img.shape, f"Output shape should match input: {result.shape} vs {input_img.shape}"
            
            return True
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"❌ Device mismatch error still occurs: {e}")
                return False
            else:
                print(f"❌ Other RuntimeError: {e}")
                return False
        
    except Exception as e:
        print(f"❌ Mixed device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Device Compatibility Fix")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_device_compatibility()
    test2_passed = test_mixed_device_scenario()
    
    print("\n" + "=" * 80)
    print("📋 Test Results Summary:")
    print(f"   Device Compatibility: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Mixed Device Scenario: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All device compatibility tests passed!")
        print("\n📝 The fix resolves:")
        print("   - RuntimeError: Expected all tensors to be on the same device")
        print("   - Properly moves affine matrices to the correct device")
        print("   - Handles numpy arrays, CPU tensors, and device tensors")
        print("   - Works with different input image sizes")
        print("\n🚀 The inference script should now work without device mismatch errors!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        sys.exit(1)
