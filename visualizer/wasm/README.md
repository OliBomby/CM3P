# Embeddings WASM Module

High-performance Rust WASM module for embedding calculations, optimized for large-scale vector operations.

## Features

- **PCA (Principal Component Analysis)**: Fast dimensionality reduction using power iteration
- **K-Means Clustering**: Efficient clustering algorithm with configurable k
- **Vector Normalization**: Batch normalization for cosine similarity calculations
- **Nearest Neighbors Search**: Fast similarity search using cosine distance

## Performance

Expected speedups over JavaScript implementation:
- PCA: 3-5x faster
- K-Means: 4-6x faster  
- Normalization: 5-10x faster
- Neighbor Search: 3-5x faster

## Building

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install wasm-pack:
   ```bash
   cargo install wasm-pack
   ```

### Build Commands

```bash
# Development build
cd wasm
wasm-pack build --target web --out-dir pkg

# Production build (optimized)
wasm-pack build --target web --release --out-dir pkg
```

This will generate the compiled WASM module in `wasm/pkg/`.

## Testing

Run the unit tests:
```bash
cd wasm
cargo test
```

Run browser-based benchmarks:
```bash
# Start a local server in the visualizer directory
python -m http.server 8000

# Open http://localhost:8000/benchmark.html
```

## Usage

### In JavaScript/Worker

```javascript
import init, { 
    pca_from_js, 
    kmeans_from_js, 
    normalize_from_js, 
    neighbors_from_js 
} from './wasm/pkg/embeddings_wasm.js';

// Initialize WASM module
await init();

// Prepare data (flat Float32Array)
const embeddings = new Float32Array([...]);
const n_samples = 1000;
const n_features = 128;

// PCA
const coords = pca_from_js(embeddings, n_samples, n_features);
// Returns Float32Array of length n_samples * 2 (x, y pairs)

// K-Means
const clusters = kmeans_from_js(embeddings, n_samples, n_features, k);
// Returns Int8Array of length n_samples (cluster labels)

// Normalize
const normalized = normalize_from_js(embeddings, n_samples, n_features);
// Returns Float32Array of same shape as input

// Find Neighbors
const neighbors = neighbors_from_js(normalized, n_samples, n_features, query_idx, k);
// Returns object: { indices: Uint32Array, distances: Float32Array }
```

## Data Format

All functions expect embeddings in **flat/row-major format**:
```
[emb0_feat0, emb0_feat1, ..., emb0_featN, 
 emb1_feat0, emb1_feat1, ..., emb1_featN,
 ...]
```

## API Reference

### `pca_from_js(embeddings, n_samples, n_features)`
Projects high-dimensional embeddings to 2D using PCA.

**Parameters:**
- `embeddings`: Float32Array - Flat array of embeddings
- `n_samples`: number - Number of samples
- `n_features`: number - Dimensionality of each embedding

**Returns:** Float32Array - 2D coordinates (x, y, x, y, ...)

### `kmeans_from_js(embeddings, n_samples, n_features, k)`
Performs k-means clustering.

**Parameters:**
- `embeddings`: Float32Array - Flat array of embeddings
- `n_samples`: number - Number of samples
- `n_features`: number - Dimensionality of each embedding
- `k`: number - Number of clusters

**Returns:** Int8Array - Cluster labels (0 to k-1)

### `normalize_from_js(embeddings, n_samples, n_features)`
Normalizes vectors to unit length for cosine similarity.

**Parameters:**
- `embeddings`: Float32Array - Flat array of embeddings
- `n_samples`: number - Number of samples
- `n_features`: number - Dimensionality of each embedding

**Returns:** Float32Array - Normalized embeddings (same shape as input)

### `neighbors_from_js(normalized_embeddings, n_samples, n_features, query_idx, n_neighbors)`
Finds k-nearest neighbors using cosine distance.

**Parameters:**
- `normalized_embeddings`: Float32Array - Normalized flat array
- `n_samples`: number - Number of samples
- `n_features`: number - Dimensionality
- `query_idx`: number - Index of query sample
- `n_neighbors`: number - Number of neighbors to find

**Returns:** Object with `indices` and `distances` arrays

## Benchmarking

Use the included `benchmark.html` to compare performance:

1. Build the WASM module (see Building section)
2. Start a local HTTP server in the visualizer directory
3. Open `benchmark.html` in your browser
4. Select dataset size and run benchmarks

The benchmark will compare JavaScript vs WASM implementations on identical data.

## Integration with Existing Worker

The new `worker.js` automatically detects and uses WASM when available, with fallback to JavaScript implementations. No changes needed to the main application code.

## License

MIT

