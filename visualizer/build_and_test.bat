@echo off
echo ============================================
echo WASM Module Build and Test
echo ============================================
echo.

cd C:\Users\Olivier\Documents\GitHub\CM3P\visualizer\wasm

echo Step 1: Running unit tests...
cargo test --release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Tests failed!
    pause
    exit /b 1
)
echo.
echo ✓ All tests passed!
echo.

echo Step 2: Building WASM module...
wasm-pack build --target web --release --out-dir pkg
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo.
echo ✓ WASM module built successfully!
echo.

echo Step 3: Listing output files...
dir pkg
echo.

echo ============================================
echo Build Complete!
echo ============================================
echo.
echo WASM module is ready at: wasm\pkg\
echo.
echo To test with your 244K dataset:
echo 1. Start a server: python -m http.server 8000
echo 2. Open: http://localhost:8000/test_integration.html
echo 3. Load: saved_logs\embeddings\beatmap_embeddings_rich.parquet
echo.
pause

