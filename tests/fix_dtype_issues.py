#!/usr/bin/env python3
"""
Automatic fix script for dtype-related issues in LatentSync.

This script automatically patches common dtype compatibility problems
between PyTorch float16 and NumPy operations.
"""

import os
import sys
import re
import shutil
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"   📁 Created backup: {backup_path}")
    return backup_path


def fix_align_restore_dtype(file_path):
    """Fix dtype issues in AlignRestore class."""
    
    print(f"🔧 Fixing AlignRestore dtype in: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ⚠ File not found: {file_path}")
        return False
    
    # Backup original file
    backup_file(file_path)
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix default dtype from float16 to float32
    original_content = content
    content = re.sub(
        r'dtype=torch\.float16',
        'dtype=torch.float32',
        content
    )
    
    # Also fix any hardcoded float16 references
    content = re.sub(
        r'torch\.float16',
        'torch.float32',
        content
    )
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"   ✅ Fixed dtype issues in {file_path}")
        return True
    else:
        print(f"   ℹ No dtype issues found in {file_path}")
        return False


def fix_image_processor_dtype(file_path):
    """Fix dtype issues in ImageProcessor class."""
    
    print(f"🔧 Fixing ImageProcessor dtype in: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ⚠ File not found: {file_path}")
        return False
    
    # Backup original file
    backup_file(file_path)
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix AlignRestore initialization to explicitly use float32
    content = re.sub(
        r'self\.restorer = AlignRestore\(resolution=resolution, device=device\)',
        'self.restorer = AlignRestore(resolution=resolution, device=device, dtype=torch.float32)',
        content
    )
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"   ✅ Fixed ImageProcessor dtype initialization")
        return True
    else:
        print(f"   ℹ ImageProcessor already has correct dtype initialization")
        return False


def add_dtype_safety_checks(file_path):
    """Add safety checks for dtype operations."""
    
    print(f"🛡️ Adding dtype safety checks to: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ⚠ File not found: {file_path}")
        return False
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if safety checks are already present
    if 'astype(np.float32)' in content:
        print(f"   ℹ Safety checks already present in {file_path}")
        return False
    
    # This is a more complex fix that would require detailed analysis
    # For now, just report that manual review is needed
    print(f"   ⚠ Manual review recommended for additional safety checks")
    return False


def fix_preprocessing_scripts():
    """Fix dtype issues in preprocessing scripts."""
    
    print("🔧 Fixing preprocessing scripts...")
    
    preprocess_dir = "preprocess"
    if not os.path.exists(preprocess_dir):
        print(f"   ⚠ Preprocess directory not found: {preprocess_dir}")
        return
    
    # Look for Python files that might have dtype issues
    for file_path in Path(preprocess_dir).glob("*.py"):
        print(f"   📄 Checking: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if file uses VideoProcessor or AlignRestore
        if 'VideoProcessor' in content or 'AlignRestore' in content:
            print(f"   ℹ Found video processing code in {file_path}")
            # The main fixes should already be applied to the core classes
            # Additional script-specific fixes can be added here if needed


def verify_fixes():
    """Verify that the fixes have been applied correctly."""
    
    print("\n🔍 Verifying fixes...")
    
    # Check AlignRestore
    align_restore_path = "latentsync/utils/affine_transform.py"
    if os.path.exists(align_restore_path):
        with open(align_restore_path, 'r') as f:
            content = f.read()
        
        if 'dtype=torch.float32' in content:
            print("   ✅ AlignRestore uses float32")
        else:
            print("   ❌ AlignRestore still has dtype issues")
    
    # Check ImageProcessor
    image_processor_path = "latentsync/utils/image_processor.py"
    if os.path.exists(image_processor_path):
        with open(image_processor_path, 'r') as f:
            content = f.read()
        
        if 'dtype=torch.float32' in content:
            print("   ✅ ImageProcessor explicitly uses float32")
        else:
            print("   ❌ ImageProcessor may have dtype issues")


def main():
    """Run all dtype fixes."""
    
    print("🚀 Starting automatic dtype issue fixes...")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    fixes_applied = []
    
    # Fix core classes
    if fix_align_restore_dtype("latentsync/utils/affine_transform.py"):
        fixes_applied.append("AlignRestore dtype")
    
    if fix_image_processor_dtype("latentsync/utils/image_processor.py"):
        fixes_applied.append("ImageProcessor dtype")
    
    # Fix preprocessing scripts
    fix_preprocessing_scripts()
    
    # Verify fixes
    verify_fixes()
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 30)
    
    if fixes_applied:
        print("✅ Fixes applied:")
        for fix in fixes_applied:
            print(f"   - {fix}")
    else:
        print("ℹ No fixes needed - code already up to date")
    
    print("\n💡 Additional recommendations:")
    print("1. Test your video processing pipeline with a sample video")
    print("2. Monitor for any remaining dtype-related errors")
    print("3. Consider using the enhanced smoothing methods for better stability")
    print("4. Run debug_dtype_issues.py for detailed diagnostics")
    
    print("\n🔄 To restore original files (if needed):")
    print("   - Backup files are saved with .backup extension")
    print("   - Copy .backup files back to original names to restore")


if __name__ == "__main__":
    main()
