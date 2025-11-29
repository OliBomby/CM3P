import init, { 
    pca_from_js, 
    kmeans_from_js, 
    normalize_from_js 
} from './wasm/pkg/embeddings_wasm.js';

console.log('Loading WASM module...');
await init();
console.log('WASM module loaded!');

// Generate test data similar to your 244K embeddings
const n_samples = 10000;
const n_features = 128;

console.log(`\nGenerating test data: ${n_samples} samples x ${n_features} features`);
const embeddings = new Float32Array(n_samples * n_features);
for (let i = 0; i < n_samples * n_features; i++) {
    embeddings[i] = Math.random() * 10 - 5;
}

console.log('\n=== Running Benchmarks ===\n');

// Benchmark PCA
console.log('Testing PCA...');
const t0_pca = performance.now();
const pca_result = pca_from_js(embeddings, n_samples, n_features);
const t1_pca = performance.now();
console.log(`✓ PCA completed in ${(t1_pca - t0_pca).toFixed(2)}ms`);
console.log(`  Output: ${pca_result.length} coordinates (${n_samples * 2} expected)`);

// Benchmark K-means
console.log('\nTesting K-Means (k=10)...');
const t0_kmeans = performance.now();
const kmeans_result = kmeans_from_js(embeddings, n_samples, n_features, 10);
const t1_kmeans = performance.now();
console.log(`✓ K-Means completed in ${(t1_kmeans - t0_kmeans).toFixed(2)}ms`);
console.log(`  Output: ${kmeans_result.length} labels (${n_samples} expected)`);

// Benchmark Normalization
console.log('\nTesting Normalization...');
const t0_norm = performance.now();
const normalized = normalize_from_js(embeddings, n_samples, n_features);
const t1_norm = performance.now();
console.log(`✓ Normalization completed in ${(t1_norm - t0_norm).toFixed(2)}ms`);
console.log(`  Output: ${normalized.length} values (${n_samples * n_features} expected)`);

// Verify normalization (check first vector is unit length)
let mag_check = 0;
for (let j = 0; j < n_features; j++) {
    mag_check += normalized[j] * normalized[j];
}
console.log(`  First vector magnitude: ${Math.sqrt(mag_check).toFixed(6)} (should be ~1.0)`);

console.log('\n=== Benchmark Complete ===');
console.log('\nEstimated performance for 244K samples:');
console.log(`  PCA: ~${((t1_pca - t0_pca) * 244 / 10).toFixed(0)}ms`);
console.log(`  K-Means: ~${((t1_kmeans - t0_kmeans) * 244 / 10).toFixed(0)}ms`);
console.log(`  Normalize: ~${((t1_norm - t0_norm) * 244 / 10).toFixed(0)}ms`);

