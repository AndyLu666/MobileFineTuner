#!/bin/bash

# Operators Framework Build Script
# Provides unified compilation for the operators deep learning framework

echo "Building Operators Framework..."
echo "================================="

# Set compiler and flags
CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -I."

# Core source files (required for all builds)
CORE_SOURCES=(
    "core/tensor.cpp"
    "core/ops.cpp"
    "core/backward_functions.cpp"
    "core/utils.cpp"
)

# Check if all core files exist
echo "Checking core files..."
for file in "${CORE_SOURCES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing core file: $file"
        exit 1
    fi
    echo "Found: $file"
done

# Function to build a test program
build_test() {
    local test_name=$1
    local test_file=$2
    
    echo ""
    echo "Building $test_name..."
    
    if [[ ! -f "$test_file" ]]; then
        echo "Test file $test_file not found, skipping..."
        return 1
    fi
    
    $CXX $CXXFLAGS "$test_file" "${CORE_SOURCES[@]}" -o "${test_name}" 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
        echo "$test_name built successfully"
        return 0
    else
        echo "$test_name build failed"
        return 1
    fi
}

# Build available tests
echo ""
echo "Building test programs..."

build_test "test_enhanced" "test_enhanced.cpp"
build_test "test_fusion" "test_fusion.cpp"

echo ""
echo "Build Summary:"
echo "================"

# Count successful builds
successful_builds=0
total_builds=0

for test in "test_enhanced" "test_fusion"; do
    total_builds=$((total_builds + 1))
    if [[ -f "$test" ]]; then
        echo "$test - Ready"
        successful_builds=$((successful_builds + 1))
    else
        echo "$test - Failed"
    fi
done

echo ""
echo "Results: $successful_builds/$total_builds tests built successfully"

if [[ $successful_builds -eq $total_builds ]]; then
    echo "All builds completed successfully!"
    echo ""
    echo "Usage examples:"
    echo "  ./test_enhanced  # Test enhanced operators"
    echo "  ./test_fusion    # Test fusion operations"
    exit 0
else
    echo "Some builds failed. Check error messages above."
    exit 1
fi
