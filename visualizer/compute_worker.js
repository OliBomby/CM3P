// Multi-threaded computation worker
// This worker receives a chunk of data and performs computations

self.onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'INIT_WASM') {
        // Load WASM module in this worker thread
        try {
            const wasmModule = await import('./wasm/pkg/embeddings_wasm.js');
            await wasmModule.default();
            self.wasmModule = wasmModule;
            self.postMessage({ type: 'READY' });
        } catch (err) {
            self.postMessage({ type: 'ERROR', error: err.message });
        }
    }

    if (type === 'PCA_CHUNK') {
        // Compute PCA scores for a chunk of data
        const { embeddings_flat, n_samples, n_features, mean, eigenvector, chunk_start } = data;
        const scores = new Float32Array(n_samples);

        for (let i = 0; i < n_samples; i++) {
            const offset = i * n_features;
            let score = 0.0;
            for (let j = 0; j < n_features; j++) {
                score += (embeddings_flat[offset + j] - mean[j]) * eigenvector[j];
            }
            scores[i] = score;
        }

        self.postMessage({ type: 'PCA_CHUNK_RESULT', scores, chunk_start }, [scores.buffer]);
    }

    if (type === 'KMEANS_ASSIGN_CHUNK') {
        // Assign points to clusters for a chunk
        const { embeddings_flat, n_samples, n_features, centroids, k, chunk_start } = data;
        const labels = new Int8Array(n_samples);

        for (let i = 0; i < n_samples; i++) {
            const sample_offset = i * n_features;
            let min_dist = Infinity;
            let best_cluster = 0;

            for (let c = 0; c < k; c++) {
                const centroid_offset = c * n_features;
                let dist = 0.0;

                // Optimized distance calculation
                for (let j = 0; j < n_features; j++) {
                    const diff = embeddings_flat[sample_offset + j] - centroids[centroid_offset + j];
                    dist += diff * diff;
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            labels[i] = best_cluster;
        }

        self.postMessage({ type: 'KMEANS_ASSIGN_RESULT', labels, chunk_start }, [labels.buffer]);
    }

    if (type === 'KMEANS_UPDATE_CHUNK') {
        // Compute partial sums for centroid update
        const { embeddings_flat, n_samples, n_features, labels, k, chunk_start } = data;
        const sums = new Float32Array(k * n_features);
        const counts = new Int32Array(k);

        for (let i = 0; i < n_samples; i++) {
            const cluster = labels[i];
            counts[cluster]++;
            const sample_offset = i * n_features;
            const sum_offset = cluster * n_features;

            for (let j = 0; j < n_features; j++) {
                sums[sum_offset + j] += embeddings_flat[sample_offset + j];
            }
        }

        self.postMessage({
            type: 'KMEANS_UPDATE_RESULT',
            sums,
            counts,
            chunk_start
        }, [sums.buffer, counts.buffer]);
    }

    if (type === 'NORMALIZE_CHUNK') {
        // Normalize a chunk of vectors
        const { embeddings_flat, n_samples, n_features, chunk_start } = data;
        const normalized = new Float32Array(n_samples * n_features);

        for (let i = 0; i < n_samples; i++) {
            const offset = i * n_features;
            let sum_sq = 0.0;

            for (let j = 0; j < n_features; j++) {
                const val = embeddings_flat[offset + j];
                sum_sq += val * val;
            }

            if (sum_sq > 0.0) {
                const inv_mag = 1.0 / Math.sqrt(sum_sq);
                for (let j = 0; j < n_features; j++) {
                    normalized[offset + j] = embeddings_flat[offset + j] * inv_mag;
                }
            }
        }

        self.postMessage({
            type: 'NORMALIZE_RESULT',
            normalized,
            chunk_start
        }, [normalized.buffer]);
    }
};

