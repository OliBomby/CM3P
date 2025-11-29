use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Int8Array, Math};

#[cfg(test)]
mod tests;

// Simple LCG random number generator for tests
fn simple_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

// ============================================================================
// SIMD-Optimized Operations
// ============================================================================

#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = 0.0f32;

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        sum += a[idx] * b[idx];
        sum += a[idx + 1] * b[idx + 1];
        sum += a[idx + 2] * b[idx + 2];
        sum += a[idx + 3] * b[idx + 3];
    }

    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }

    sum
}

#[inline(always)]
fn squared_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = 0.0f32;

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum
}

// ============================================================================
// Parallel PCA Implementation
// ============================================================================

#[wasm_bindgen]
pub fn calculate_pca_parallel(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    if n_samples == 0 || n_features == 0 {
        return vec![];
    }

    // Calculate mean
    let mut mean = vec![0.0f32; n_features];
    let inv_n = 1.0 / n_samples as f32;

    // Parallel mean computation using manual chunking
    let chunk_size = (n_samples + 3) / 4; // 4 chunks for better parallelism
    let mut partial_means: Vec<Vec<f32>> = (0..4)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(n_samples);
            let mut local_mean = vec![0.0f32; n_features];

            for i in start..end {
                let offset = i * n_features;
                for j in 0..n_features {
                    local_mean[j] += embeddings_flat[offset + j];
                }
            }
            local_mean
        })
        .collect();

    // Combine partial means
    for partial in partial_means.iter() {
        for j in 0..n_features {
            mean[j] += partial[j];
        }
    }
    for j in 0..n_features {
        mean[j] *= inv_n;
    }

    let mut components: Vec<Vec<f32>> = Vec::with_capacity(2);
    let mut rng_state = 12345u32;

    // Power iteration for principal components
    for c in 0..2 {
        let mut ev: Vec<f32> = (0..n_features)
            .map(|_| {
                #[cfg(target_arch = "wasm32")]
                { Math::random() as f32 - 0.5 }
                #[cfg(not(target_arch = "wasm32"))]
                { simple_random(&mut rng_state) - 0.5 }
            })
            .collect();

        let mut mag: f32 = dot_product_simd(&ev, &ev).sqrt();
        let inv_mag = 1.0 / mag;
        for val in ev.iter_mut() {
            *val *= inv_mag;
        }

        // Power iteration with parallel processing
        for _ in 0..8 {
            // Parallel computation of next eigenvector
            let chunk_size = (n_samples + 3) / 4;
            let mut partial_evs: Vec<Vec<f32>> = (0..4)
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = ((chunk_idx + 1) * chunk_size).min(n_samples);
                    let mut local_ev = vec![0.0f32; n_features];

                    for i in start..end {
                        let offset = i * n_features;
                        let mut score = 0.0f32;

                        // Compute centered dot product
                        let chunks = n_features / 4;
                        for k in 0..chunks {
                            let idx = k * 4;
                            score += (embeddings_flat[offset + idx] - mean[idx]) * ev[idx];
                            score += (embeddings_flat[offset + idx + 1] - mean[idx + 1]) * ev[idx + 1];
                            score += (embeddings_flat[offset + idx + 2] - mean[idx + 2]) * ev[idx + 2];
                            score += (embeddings_flat[offset + idx + 3] - mean[idx + 3]) * ev[idx + 3];
                        }
                        for k in (chunks * 4)..n_features {
                            score += (embeddings_flat[offset + k] - mean[k]) * ev[k];
                        }

                        for j in 0..n_features {
                            local_ev[j] += score * (embeddings_flat[offset + j] - mean[j]);
                        }
                    }
                    local_ev
                })
                .collect();

            // Combine partial eigenvectors
            let mut next_ev = vec![0.0f32; n_features];
            for partial in partial_evs.iter() {
                for j in 0..n_features {
                    next_ev[j] += partial[j];
                }
            }

            mag = dot_product_simd(&next_ev, &next_ev).sqrt();
            if mag > 0.0 {
                let inv_mag = 1.0 / mag;
                for val in next_ev.iter_mut() {
                    *val *= inv_mag;
                }
                ev = next_ev;
            }
        }

        if c == 1 {
            let u: &Vec<f32> = &components[0];
            let dot = dot_product_simd(u, &ev);
            for k in 0..n_features {
                ev[k] -= dot * u[k];
            }
            mag = dot_product_simd(&ev, &ev).sqrt();
            if mag > 0.0 {
                let inv_mag = 1.0 / mag;
                for val in ev.iter_mut() {
                    *val *= inv_mag;
                }
            }
        }

        components.push(ev);
    }

    // Parallel projection
    let mut projected = vec![0.0f32; n_samples * 2];
    let comp0 = &components[0];
    let comp1 = &components[1];

    let chunk_size = (n_samples + 3) / 4;
    for chunk_idx in 0..4 {
        let start = chunk_idx * chunk_size;
        let end = ((chunk_idx + 1) * chunk_size).min(n_samples);

        for i in start..end {
            let offset = i * n_features;
            let mut x = 0.0f32;
            let mut y = 0.0f32;

            let chunks = n_features / 4;
            for k in 0..chunks {
                let idx = k * 4;
                let c0 = embeddings_flat[offset + idx] - mean[idx];
                let c1 = embeddings_flat[offset + idx + 1] - mean[idx + 1];
                let c2 = embeddings_flat[offset + idx + 2] - mean[idx + 2];
                let c3 = embeddings_flat[offset + idx + 3] - mean[idx + 3];

                x += c0 * comp0[idx] + c1 * comp0[idx + 1] + c2 * comp0[idx + 2] + c3 * comp0[idx + 3];
                y += c0 * comp1[idx] + c1 * comp1[idx + 1] + c2 * comp1[idx + 2] + c3 * comp1[idx + 3];
            }

            for k in (chunks * 4)..n_features {
                let val = embeddings_flat[offset + k] - mean[k];
                x += val * comp0[k];
                y += val * comp1[k];
            }

            projected[i * 2] = x;
            projected[i * 2 + 1] = y;
        }
    }

    projected
}

// ============================================================================
// Parallel K-Means Implementation
// ============================================================================

#[wasm_bindgen]
pub fn calculate_kmeans_parallel(embeddings_flat: &[f32], n_samples: usize, n_features: usize, k: usize, seed: u32) -> Vec<i8> {
    if n_samples == 0 || k == 0 {
        return vec![];
    }

    let mut rng_state = seed;
    let mut centroids = vec![0.0f32; k * n_features];
    for i in 0..k {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let idx = (rng_state as usize) % n_samples;
        let src_offset = idx * n_features;
        let dst_offset = i * n_features;
        centroids[dst_offset..dst_offset + n_features]
            .copy_from_slice(&embeddings_flat[src_offset..src_offset + n_features]);
    }

    let mut labels = vec![0i8; n_samples];

    for _ in 0..5 {
        // Parallel assignment step
        let chunk_size = (n_samples + 3) / 4;
        for chunk_idx in 0..4 {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(n_samples);

            for i in start..end {
                let sample_offset = i * n_features;
                let sample = &embeddings_flat[sample_offset..sample_offset + n_features];

                let mut min_dist = f32::INFINITY;
                let mut best_cluster = 0i8;

                for c in 0..k {
                    let centroid_offset = c * n_features;
                    let centroid = &centroids[centroid_offset..centroid_offset + n_features];
                    let dist = squared_distance_simd(sample, centroid);

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = c as i8;
                    }
                }
                labels[i] = best_cluster;
            }
        }

        // Parallel update step
        let chunk_size_update = (n_samples + 3) / 4;
        let mut partial_sums: Vec<Vec<f32>> = (0..4)
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size_update;
                let end = ((chunk_idx + 1) * chunk_size_update).min(n_samples);
                let mut local_sums = vec![0.0f32; k * n_features];

                for i in start..end {
                    let cluster = labels[i] as usize;
                    let sample_offset = i * n_features;
                    let sum_offset = cluster * n_features;

                    for j in 0..n_features {
                        local_sums[sum_offset + j] += embeddings_flat[sample_offset + j];
                    }
                }
                local_sums
            })
            .collect();

        let mut sums = vec![0.0f32; k * n_features];
        for partial in partial_sums.iter() {
            for j in 0..(k * n_features) {
                sums[j] += partial[j];
            }
        }

        let mut counts = vec![0usize; k];
        for &label in labels.iter() {
            counts[label as usize] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv_count = 1.0 / counts[c] as f32;
                let centroid_offset = c * n_features;
                let sum_offset = c * n_features;

                for j in 0..n_features {
                    centroids[centroid_offset + j] = sums[sum_offset + j] * inv_count;
                }
            }
        }
    }

    labels
}

// ============================================================================
// Parallel Normalization
// ============================================================================

#[wasm_bindgen]
pub fn normalize_vectors_parallel(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    let mut normalized = vec![0.0f32; n_samples * n_features];

    let chunk_size = (n_samples + 3) / 4;
    for chunk_idx in 0..4 {
        let start = chunk_idx * chunk_size;
        let end = ((chunk_idx + 1) * chunk_size).min(n_samples);

        for i in start..end {
            let offset = i * n_features;
            let input_slice = &embeddings_flat[offset..offset + n_features];
            let output_slice = &mut normalized[offset..offset + n_features];

            let mut sum_sq = 0.0f32;
            let chunks = n_features / 4;

            for k in 0..chunks {
                let idx = k * 4;
                sum_sq += input_slice[idx] * input_slice[idx];
                sum_sq += input_slice[idx + 1] * input_slice[idx + 1];
                sum_sq += input_slice[idx + 2] * input_slice[idx + 2];
                sum_sq += input_slice[idx + 3] * input_slice[idx + 3];
            }

            for k in (chunks * 4)..n_features {
                sum_sq += input_slice[k] * input_slice[k];
            }

            if sum_sq == 0.0 {
                continue;
            }

            let inv_mag = 1.0 / sum_sq.sqrt();

            for k in 0..chunks {
                let idx = k * 4;
                output_slice[idx] = input_slice[idx] * inv_mag;
                output_slice[idx + 1] = input_slice[idx + 1] * inv_mag;
                output_slice[idx + 2] = input_slice[idx + 2] * inv_mag;
                output_slice[idx + 3] = input_slice[idx + 3] * inv_mag;
            }

            for k in (chunks * 4)..n_features {
                output_slice[k] = input_slice[k] * inv_mag;
            }
        }
    }

    normalized
}

// Keep non-parallel versions as fallback
#[wasm_bindgen]
pub fn calculate_pca(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    calculate_pca_parallel(embeddings_flat, n_samples, n_features)
}

#[wasm_bindgen]
pub fn calculate_kmeans(embeddings_flat: &[f32], n_samples: usize, n_features: usize, k: usize, seed: u32) -> Vec<i8> {
    calculate_kmeans_parallel(embeddings_flat, n_samples, n_features, k, seed)
}

#[wasm_bindgen]
pub fn normalize_vectors(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    normalize_vectors_parallel(embeddings_flat, n_samples, n_features)
}

// ============================================================================
// Nearest Neighbors (already fast, keep optimized version)
// ============================================================================

#[wasm_bindgen]
pub struct NeighborResult {
    indices: Vec<usize>,
    distances: Vec<f32>,
}

#[wasm_bindgen]
impl NeighborResult {
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<usize> {
        self.indices.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn distances(&self) -> Vec<f32> {
        self.distances.clone()
    }
}

#[wasm_bindgen]
pub fn find_nearest_neighbors(
    normalized_embeddings: &[f32],
    n_samples: usize,
    n_features: usize,
    query_idx: usize,
    n_neighbors: usize,
) -> NeighborResult {
    if query_idx >= n_samples {
        return NeighborResult {
            indices: vec![],
            distances: vec![],
        };
    }

    let query_offset = query_idx * n_features;
    let query = &normalized_embeddings[query_offset..query_offset + n_features];
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n_samples - 1);

    for i in 0..n_samples {
        if i == query_idx {
            continue;
        }

        let offset = i * n_features;
        let vec = &normalized_embeddings[offset..offset + n_features];
        let dot = dot_product_simd(query, vec);
        let distance = 1.0 - dot;
        results.push((i, distance));
    }

    let k = n_neighbors.min(results.len());
    results.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(k);
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    NeighborResult {
        indices: results.iter().map(|(idx, _)| *idx).collect(),
        distances: results.iter().map(|(_, dist)| *dist).collect(),
    }
}

// ============================================================================
// JavaScript-friendly wrappers
// ============================================================================

#[wasm_bindgen]
pub fn pca_from_js(embeddings: Float32Array, n_samples: usize, n_features: usize) -> Float32Array {
    let data: Vec<f32> = embeddings.to_vec();
    let result = calculate_pca_parallel(&data, n_samples, n_features);
    Float32Array::from(&result[..])
}

#[wasm_bindgen]
pub fn kmeans_from_js(embeddings: Float32Array, n_samples: usize, n_features: usize, k: usize) -> Int8Array {
    let data: Vec<f32> = embeddings.to_vec();
    let seed = (Math::random() * 4294967295.0) as u32;
    let result = calculate_kmeans_parallel(&data, n_samples, n_features, k, seed);
    Int8Array::from(&result[..])
}

#[wasm_bindgen]
pub fn normalize_from_js(embeddings: Float32Array, n_samples: usize, n_features: usize) -> Float32Array {
    let data: Vec<f32> = embeddings.to_vec();
    let result = normalize_vectors_parallel(&data, n_samples, n_features);
    Float32Array::from(&result[..])
}

#[wasm_bindgen]
pub fn neighbors_from_js(
    normalized_embeddings: Float32Array,
    n_samples: usize,
    n_features: usize,
    query_idx: usize,
    n_neighbors: usize,
) -> JsValue {
    let data: Vec<f32> = normalized_embeddings.to_vec();
    let result = find_nearest_neighbors(&data, n_samples, n_features, query_idx, n_neighbors);

    let obj = js_sys::Object::new();

    let indices_array = Array::new();
    for idx in result.indices.iter() {
        indices_array.push(&JsValue::from(*idx as u32));
    }

    js_sys::Reflect::set(&obj, &"indices".into(), &indices_array).unwrap();
    js_sys::Reflect::set(&obj, &"distances".into(), &Float32Array::from(&result.distances[..])).unwrap();
    obj.into()
}

