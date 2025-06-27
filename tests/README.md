# Test Scripts for LatentSync Video Processing

This directory contains test and utility scripts for validating and debugging the LatentSync video processing pipeline.

## 🧪 Test Scripts

### `debug_dtype_issues.py`
**Purpose**: Comprehensive dtype compatibility testing and diagnostics.

**Usage**:
```bash
cd /path/to/LatentSync-Fix
python tests/debug_dtype_issues.py
```

**What it does**:
- Tests PyTorch/NumPy dtype compatibility
- Validates AlignRestore with different dtypes
- Tests VideoProcessor configurations
- Provides diagnostic recommendations

### `test_affine_matrix_shapes.py`
**Purpose**: Validates tensor shape handling in the enhanced smoothing pipeline.

**Usage**:
```bash
cd /path/to/LatentSync-Fix
python tests/test_affine_matrix_shapes.py
```

**What it does**:
- Tests affine matrix shape consistency
- Validates smoothing function operations
- Tests kornia.warp_affine compatibility
- Ensures proper tensor transformations

### `test_smoothing_improvements.py`
**Purpose**: Comprehensive testing and comparison of all smoothing methods.

**Usage**:
```bash
cd /path/to/LatentSync-Fix
python tests/test_smoothing_improvements.py path/to/video.mp4 [options]
```

**Options**:
- `--device cuda/cpu`: Specify device (default: auto-detect)
- `--resolution 256`: Set processing resolution
- `--output-dir results`: Output directory for test videos

**What it does**:
- Processes video with all smoothing methods
- Generates comparison videos
- Measures processing times
- Creates quality comparison reports

## 🔧 Utility Scripts

### `fix_dtype_issues.py`
**Purpose**: Automatic detection and fixing of dtype-related issues.

**Usage**:
```bash
cd /path/to/LatentSync-Fix
python tests/fix_dtype_issues.py
```

**What it does**:
- Scans codebase for dtype issues
- Automatically applies fixes
- Creates backup files
- Verifies applied changes

## 📋 Running All Tests

To run a comprehensive test suite:

```bash
cd /path/to/LatentSync-Fix

# 1. Check for and fix any dtype issues
python tests/fix_dtype_issues.py

# 2. Run dtype compatibility tests
python tests/debug_dtype_issues.py

# 3. Validate tensor shape handling
python tests/test_affine_matrix_shapes.py

# 4. Test smoothing improvements (requires a video file)
python tests/test_smoothing_improvements.py data/sample_video.mp4
```

## 🎯 Expected Results

All tests should pass with output similar to:

```
✅ All dtype compatibility tests passed
✅ All shape tests passed
✅ Enhanced smoothing successful! Output shape: (N, 256, 256, 3)
✅ Ultra smoothing successful! Output shape: (N, 256, 256, 3)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **CUDA Errors**: Tests will automatically fall back to CPU if CUDA is unavailable
3. **Missing Video Files**: Use any MP4 file for testing smoothing improvements
4. **Permission Errors**: Ensure write permissions for output directories

### Getting Help

If tests fail:
1. Check the error messages for specific issues
2. Ensure all dependencies are installed
3. Verify you're using the correct Python environment
4. Check that the project structure is intact

## 📁 Test Output

Test scripts may create temporary files and directories:
- `temp_test/`: Temporary processing files
- `results/`: Output videos and comparison reports
- `*.backup`: Backup files created by fix scripts

These can be safely deleted after testing.

## 🔄 Continuous Testing

For development, you can set up automated testing:

```bash
# Create a simple test runner
cat > run_tests.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Running LatentSync tests..."
python tests/debug_dtype_issues.py && \
python tests/test_affine_matrix_shapes.py && \
echo "All tests passed!"
EOF

chmod +x run_tests.sh
./run_tests.sh
```

This ensures your changes don't break the video processing pipeline.
