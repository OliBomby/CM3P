@echo off
REM Build script for WASM module on Windows

echo Building embeddings WASM module...
echo.

REM Check if Rust is installed
where rustc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust is not installed!
    echo Please install Rust from https://rustup.rs/
    echo After installation, restart your terminal and run this script again.
    pause
    exit /b 1
)

REM Check if wasm-pack is installed
where wasm-pack >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo wasm-pack not found. Installing...
    cargo install wasm-pack
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install wasm-pack
        pause
        exit /b 1
    )
)

echo.
echo Running Rust tests...
cd wasm
cargo test --release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Tests failed!
    pause
    exit /b 1
)

echo.
echo Building WASM module (release mode)...
wasm-pack build --target web --release --out-dir pkg
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo WASM module is ready at: wasm\pkg\
echo ========================================
echo.
echo To test the performance improvements:
echo 1. Start a local server: python -m http.server 8000
echo 2. Open http://localhost:8000/benchmark.html
echo.
pause

