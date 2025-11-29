// Parallel computation manager using Web Workers
class WorkerPool {
    constructor(numWorkers = navigator.hardwareConcurrency || 4) {
        this.numWorkers = Math.min(numWorkers, 8); // Cap at 8 workers
        this.workers = [];
        this.readyWorkers = 0;
        this.initPromise = this.initialize();
    }

    async initialize() {
        console.log(`Initializing ${this.numWorkers} worker threads...`);
        const promises = [];

        for (let i = 0; i < this.numWorkers; i++) {
            const worker = new Worker('./compute_worker.js', { type: 'module' });
            this.workers.push(worker);

            promises.push(new Promise((resolve) => {
                worker.onmessage = (e) => {
                    if (e.data.type === 'READY') {
                        this.readyWorkers++;
                        resolve();
                    }
                };
                worker.postMessage({ type: 'INIT_WASM' });
            }));
        }

        await Promise.all(promises);
        console.log(`✓ ${this.numWorkers} workers ready`);
    }

    async ready() {
        await this.initPromise;
    }

    // Parallel K-means clustering
    async kmeans(embeddings_flat, n_samples, n_features, k, maxIter = 5) {
        await this.ready();

        // Initialize centroids randomly
        let centroids = new Float32Array(k * n_features);
        for (let i = 0; i < k; i++) {
            const idx = Math.floor(Math.random() * n_samples);
            const src_offset = idx * n_features;
            const dst_offset = i * n_features;
            for (let j = 0; j < n_features; j++) {
                centroids[dst_offset + j] = embeddings_flat[src_offset + j];
            }
        }

        let labels = new Int8Array(n_samples);
        const chunk_size = Math.ceil(n_samples / this.numWorkers);

        // Lloyd's algorithm with parallel execution
        for (let iter = 0; iter < maxIter; iter++) {
            // PARALLEL ASSIGNMENT STEP
            const assignPromises = [];
            for (let w = 0; w < this.numWorkers; w++) {
                const start = w * chunk_size;
                const end = Math.min((w + 1) * chunk_size, n_samples);
                if (start >= end) continue;

                const chunk_samples = end - start;
                const chunk_data = embeddings_flat.slice(
                    start * n_features,
                    end * n_features
                );

                assignPromises.push(new Promise((resolve) => {
                    const worker = this.workers[w];
                    const handler = (e) => {
                        if (e.data.type === 'KMEANS_ASSIGN_RESULT') {
                            worker.removeEventListener('message', handler);
                            resolve({
                                labels: e.data.labels,
                                chunk_start: start
                            });
                        }
                    };
                    worker.addEventListener('message', handler);
                    worker.postMessage({
                        type: 'KMEANS_ASSIGN_CHUNK',
                        data: {
                            embeddings_flat: chunk_data,
                            n_samples: chunk_samples,
                            n_features,
                            centroids,
                            k,
                            chunk_start: start
                        }
                    }, [chunk_data.buffer]);
                }));
            }

            // Wait for all assignment results
            const assignResults = await Promise.all(assignPromises);

            // Merge labels
            for (const result of assignResults) {
                labels.set(result.labels, result.chunk_start);
            }

            // PARALLEL UPDATE STEP
            const updatePromises = [];
            for (let w = 0; w < this.numWorkers; w++) {
                const start = w * chunk_size;
                const end = Math.min((w + 1) * chunk_size, n_samples);
                if (start >= end) continue;

                const chunk_samples = end - start;
                const chunk_data = embeddings_flat.slice(
                    start * n_features,
                    end * n_features
                );
                const chunk_labels = labels.slice(start, end);

                updatePromises.push(new Promise((resolve) => {
                    const worker = this.workers[w];
                    const handler = (e) => {
                        if (e.data.type === 'KMEANS_UPDATE_RESULT') {
                            worker.removeEventListener('message', handler);
                            resolve({
                                sums: e.data.sums,
                                counts: e.data.counts
                            });
                        }
                    };
                    worker.addEventListener('message', handler);
                    worker.postMessage({
                        type: 'KMEANS_UPDATE_CHUNK',
                        data: {
                            embeddings_flat: chunk_data,
                            n_samples: chunk_samples,
                            n_features,
                            labels: chunk_labels,
                            k,
                            chunk_start: start
                        }
                    }, [chunk_data.buffer, chunk_labels.buffer]);
                }));
            }

            // Wait for all update results
            const updateResults = await Promise.all(updatePromises);

            // Merge sums and counts
            const total_sums = new Float32Array(k * n_features);
            const total_counts = new Int32Array(k);

            for (const result of updateResults) {
                for (let i = 0; i < k * n_features; i++) {
                    total_sums[i] += result.sums[i];
                }
                for (let i = 0; i < k; i++) {
                    total_counts[i] += result.counts[i];
                }
            }

            // Update centroids
            for (let c = 0; c < k; c++) {
                if (total_counts[c] > 0) {
                    const inv_count = 1.0 / total_counts[c];
                    for (let j = 0; j < n_features; j++) {
                        centroids[c * n_features + j] = total_sums[c * n_features + j] * inv_count;
                    }
                }
            }
        }

        return labels;
    }

    // Parallel normalization
    async normalize(embeddings_flat, n_samples, n_features) {
        await this.ready();

        const normalized = new Float32Array(n_samples * n_features);
        const chunk_size = Math.ceil(n_samples / this.numWorkers);
        const promises = [];

        for (let w = 0; w < this.numWorkers; w++) {
            const start = w * chunk_size;
            const end = Math.min((w + 1) * chunk_size, n_samples);
            if (start >= end) continue;

            const chunk_samples = end - start;
            const chunk_data = embeddings_flat.slice(
                start * n_features,
                end * n_features
            );

            promises.push(new Promise((resolve) => {
                const worker = this.workers[w];
                const handler = (e) => {
                    if (e.data.type === 'NORMALIZE_RESULT') {
                        worker.removeEventListener('message', handler);
                        resolve({
                            normalized: e.data.normalized,
                            chunk_start: start
                        });
                    }
                };
                worker.addEventListener('message', handler);
                worker.postMessage({
                    type: 'NORMALIZE_CHUNK',
                    data: {
                        embeddings_flat: chunk_data,
                        n_samples: chunk_samples,
                        n_features,
                        chunk_start: start
                    }
                }, [chunk_data.buffer]);
            }));
        }

        const results = await Promise.all(promises);

        // Merge normalized results
        for (const result of results) {
            normalized.set(result.normalized, result.chunk_start * n_features);
        }

        return normalized;
    }

    terminate() {
        for (const worker of this.workers) {
            worker.terminate();
        }
        this.workers = [];
    }
}

export { WorkerPool };

