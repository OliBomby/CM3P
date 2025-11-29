# Quick Start: Building and Testing WASM Module

## Build Commands

```bash
# Navigate to visualizer directory
cd C:\Users\Olivier\Documents\GitHub\CM3P\visualizer

# Run the build script
build_wasm.bat
```

This will:
1. Check for Rust installation
2. Install wasm-pack if needed
3. Run unit tests
4. Build the optimized WASM module
5. Output to `wasm/pkg/`

## Test with Real Data

```bash
# Start local server
python -m http.server 8000

# In browser, open:
# http://localhost:8000/test_integration.html
```

Load the file: `C:\Users\Olivier\Documents\GitHub\CM3P\saved_logs\embeddings\beatmap_embeddings_rich.parquet`

## Expected Results (244K rows, 128 dims)

| Operation | JS Time | WASM Time | Speedup |
|-----------|---------|-----------|---------|
| PCA | ~2000ms | ~400-500ms | 4-5x |
| K-Means | ~1500ms | ~250-350ms | 4-6x |
| Normalize | ~500ms | ~50-100ms | 5-10x |
| Neighbors | ~300ms | ~60-80ms | 4-5x |

## Integration

The new `worker_wasm.js` automatically uses WASM when available and falls back to JavaScript if not. Simply replace your current worker:

```bash
copy worker_wasm.js worker.js
```

## Verify It's Working

Check browser console for:
```
WASM module loaded successfully
PCA (WASM): 523.45ms
K-Means (WASM): 312.78ms
```

If you see `(JS)` instead of `(WASM)`, check SETUP.md for troubleshooting.

