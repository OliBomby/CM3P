use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Int8Array, Math};

#[cfg(test)]
mod tests;

fn simple_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

// ============================================================================
// Ultra-Optimized SIMD Operations with Better Unrolling
// ============================================================================

#[inline(always)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    // Process 8 elements at a time with 2-way unrolling
    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        sum0 += a[idx] * b[idx] + a[idx + 1] * b[idx + 1];
        sum1 += a[idx + 2] * b[idx + 2] + a[idx + 3] * b[idx + 3];
        sum2 += a[idx + 4] * b[idx + 4] + a[idx + 5] * b[idx + 5];
        sum3 += a[idx + 6] * b[idx + 6] + a[idx + 7] * b[idx + 7];
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }

    sum
}

#[inline(always)]
fn squared_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        let d4 = a[idx + 4] - b[idx + 4];
        let d5 = a[idx + 5] - b[idx + 5];
        let d6 = a[idx + 6] - b[idx + 6];
        let d7 = a[idx + 7] - b[idx + 7];

        sum0 += d0 * d0 + d1 * d1;
        sum1 += d2 * d2 + d3 * d3;
        sum2 += d4 * d4 + d5 * d5;
        sum3 += d6 * d6 + d7 * d7;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (chunks * 8)..len {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum
}

// ============================================================================
// Hyper-Optimized PCA with Cache-Friendly Memory Access
// ============================================================================

#[wasm_bindgen]
pub fn calculate_pca(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    if n_samples == 0 || n_features == 0 {
        return vec![];
    }

    // Fast mean calculation
    let mut mean = vec![0.0f32; n_features];
    let inv_n = 1.0 / n_samples as f32;

    for i in 0..n_samples {
        let offset = i * n_features;
        let chunks = n_features / 8;
        for k in 0..chunks {
            let idx = k * 8;
            mean[idx] += embeddings_flat[offset + idx];
            mean[idx + 1] += embeddings_flat[offset + idx + 1];
            mean[idx + 2] += embeddings_flat[offset + idx + 2];
            mean[idx + 3] += embeddings_flat[offset + idx + 3];
            mean[idx + 4] += embeddings_flat[offset + idx + 4];
            mean[idx + 5] += embeddings_flat[offset + idx + 5];
            mean[idx + 6] += embeddings_flat[offset + idx + 6];
            mean[idx + 7] += embeddings_flat[offset + idx + 7];
        }
        for k in (chunks * 8)..n_features {
            mean[k] += embeddings_flat[offset + k];
        }
    }

    for j in 0..n_features {
        mean[j] *= inv_n;
    }

    let mut components: Vec<Vec<f32>> = Vec::with_capacity(2);
    let mut rng_state = 12345u32;

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

        // Optimized power iteration with fewer allocations
        for _ in 0..8 {
            let mut next_ev = vec![0.0f32; n_features];

            // Cache-friendly matrix-vector product
            for i in 0..n_samples {
                let offset = i * n_features;
                let mut score = 0.0f32;

                let chunks = n_features / 8;
                for k in 0..chunks {
                    let idx = k * 8;
                    let c0 = embeddings_flat[offset + idx] - mean[idx];
                    let c1 = embeddings_flat[offset + idx + 1] - mean[idx + 1];
                    let c2 = embeddings_flat[offset + idx + 2] - mean[idx + 2];
                    let c3 = embeddings_flat[offset + idx + 3] - mean[idx + 3];
                    let c4 = embeddings_flat[offset + idx + 4] - mean[idx + 4];
                    let c5 = embeddings_flat[offset + idx + 5] - mean[idx + 5];
                    let c6 = embeddings_flat[offset + idx + 6] - mean[idx + 6];
                    let c7 = embeddings_flat[offset + idx + 7] - mean[idx + 7];

                    score += c0 * ev[idx] + c1 * ev[idx + 1] + c2 * ev[idx + 2] + c3 * ev[idx + 3];
                    score += c4 * ev[idx + 4] + c5 * ev[idx + 5] + c6 * ev[idx + 6] + c7 * ev[idx + 7];
                }
                for k in (chunks * 8)..n_features {
                    score += (embeddings_flat[offset + k] - mean[k]) * ev[k];
                }

                // Accumulate in one pass
                for j in 0..n_features {
                    next_ev[j] += score * (embeddings_flat[offset + j] - mean[j]);
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

    // Ultra-fast projection
    let mut projected = vec![0.0f32; n_samples * 2];
    let comp0 = &components[0];
    let comp1 = &components[1];

    for i in 0..n_samples {
        let offset = i * n_features;
        let mut x = 0.0f32;
        let mut y = 0.0f32;

        let chunks = n_features / 8;
        for k in 0..chunks {
            let idx = k * 8;
            let c0 = embeddings_flat[offset + idx] - mean[idx];
            let c1 = embeddings_flat[offset + idx + 1] - mean[idx + 1];
            let c2 = embeddings_flat[offset + idx + 2] - mean[idx + 2];
            let c3 = embeddings_flat[offset + idx + 3] - mean[idx + 3];
            let c4 = embeddings_flat[offset + idx + 4] - mean[idx + 4];
            let c5 = embeddings_flat[offset + idx + 5] - mean[idx + 5];
            let c6 = embeddings_flat[offset + idx + 6] - mean[idx + 6];
            let c7 = embeddings_flat[offset + idx + 7] - mean[idx + 7];

            x += c0 * comp0[idx] + c1 * comp0[idx + 1] + c2 * comp0[idx + 2] + c3 * comp0[idx + 3];
            x += c4 * comp0[idx + 4] + c5 * comp0[idx + 5] + c6 * comp0[idx + 6] + c7 * comp0[idx + 7];

            y += c0 * comp1[idx] + c1 * comp1[idx + 1] + c2 * comp1[idx + 2] + c3 * comp1[idx + 3];
            y += c4 * comp1[idx + 4] + c5 * comp1[idx + 5] + c6 * comp1[idx + 6] + c7 * comp1[idx + 7];
        }

        for k in (chunks * 8)..n_features {
            let val = embeddings_flat[offset + k] - mean[k];
            x += val * comp0[k];
            y += val * comp1[k];
        }

        projected[i * 2] = x;
        projected[i * 2 + 1] = y;
    }

    projected
}

// ============================================================================
// Ultra-Optimized K-Means with Better Initialization
// ============================================================================

#[wasm_bindgen]
pub fn calculate_kmeans(embeddings_flat: &[f32], n_samples: usize, n_features: usize, k: usize, seed: u32) -> Vec<i8> {
    if n_samples == 0 || k == 0 {
        return vec![];
    }

    // K-means++ style initialization for better convergence
    let mut rng_state = seed;
    let mut centroids = vec![0.0f32; k * n_features];

    // First centroid randomly
    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
    let first_idx = (rng_state as usize) % n_samples;
    let src = first_idx * n_features;
    centroids[0..n_features].copy_from_slice(&embeddings_flat[src..src + n_features]);

    // Rest use distance-based selection
    let mut distances = vec![f32::INFINITY; n_samples];
    for i in 1..k {
        // Update distances to nearest centroid
        for j in 0..n_samples {
            let sample_offset = j * n_features;
            let sample = &embeddings_flat[sample_offset..sample_offset + n_features];
            let centroid_offset = (i - 1) * n_features;
            let centroid = &centroids[centroid_offset..centroid_offset + n_features];
            let dist = squared_distance_simd(sample, centroid);
            if dist < distances[j] {
                distances[j] = dist;
            }
        }

        // Select point with max distance
        let mut max_idx = 0;
        let mut max_dist = 0.0f32;
        for (j, &d) in distances.iter().enumerate() {
            if d > max_dist {
                max_dist = d;
                max_idx = j;
            }
        }

        let src = max_idx * n_features;
        let dst = i * n_features;
        centroids[dst..dst + n_features].copy_from_slice(&embeddings_flat[src..src + n_features]);
    }

    let mut labels = vec![0i8; n_samples];

    // Lloyd's algorithm with early stopping
    for iter in 0..10 {
        let mut changed = 0;

        // Assignment step with optimized distance
        for i in 0..n_samples {
            let sample_offset = i * n_features;
            let sample = &embeddings_flat[sample_offset..sample_offset + n_features];

            let mut min_dist = f32::INFINITY;
            let mut best_cluster = labels[i];

            for c in 0..k {
                let centroid_offset = c * n_features;
                let centroid = &centroids[centroid_offset..centroid_offset + n_features];
                let dist = squared_distance_simd(sample, centroid);

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c as i8;
                }
            }

            if labels[i] != best_cluster {
                changed += 1;
                labels[i] = best_cluster;
            }
        }

        // Early stopping if converged
        if iter > 0 && changed == 0 {
            break;
        }

        // Update step with vectorized operations
        let mut sums = vec![0.0f32; k * n_features];
        let mut counts = vec![0usize; k];

        for i in 0..n_samples {
            let cluster = labels[i] as usize;
            counts[cluster] += 1;
            let sample_offset = i * n_features;
            let sum_offset = cluster * n_features;

            let chunks = n_features / 8;
            for j in 0..chunks {
                let idx = j * 8;
                sums[sum_offset + idx] += embeddings_flat[sample_offset + idx];
                sums[sum_offset + idx + 1] += embeddings_flat[sample_offset + idx + 1];
                sums[sum_offset + idx + 2] += embeddings_flat[sample_offset + idx + 2];
                sums[sum_offset + idx + 3] += embeddings_flat[sample_offset + idx + 3];
                sums[sum_offset + idx + 4] += embeddings_flat[sample_offset + idx + 4];
                sums[sum_offset + idx + 5] += embeddings_flat[sample_offset + idx + 5];
                sums[sum_offset + idx + 6] += embeddings_flat[sample_offset + idx + 6];
                sums[sum_offset + idx + 7] += embeddings_flat[sample_offset + idx + 7];
            }
            for j in (chunks * 8)..n_features {
                sums[sum_offset + j] += embeddings_flat[sample_offset + j];
            }
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
// Ultra-Fast Normalization
// ============================================================================

#[wasm_bindgen]
pub fn normalize_vectors(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    let mut normalized = vec![0.0f32; n_samples * n_features];

    for i in 0..n_samples {
        let offset = i * n_features;
        let input_slice = &embeddings_flat[offset..offset + n_features];
        let output_slice = &mut normalized[offset..offset + n_features];

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks = n_features / 8;
        for k in 0..chunks {
            let idx = k * 8;
            sum0 += input_slice[idx] * input_slice[idx] + input_slice[idx + 1] * input_slice[idx + 1];
            sum1 += input_slice[idx + 2] * input_slice[idx + 2] + input_slice[idx + 3] * input_slice[idx + 3];
            sum2 += input_slice[idx + 4] * input_slice[idx + 4] + input_slice[idx + 5] * input_slice[idx + 5];
            sum3 += input_slice[idx + 6] * input_slice[idx + 6] + input_slice[idx + 7] * input_slice[idx + 7];
        }

        let mut sum_sq = sum0 + sum1 + sum2 + sum3;
        for k in (chunks * 8)..n_features {
            sum_sq += input_slice[k] * input_slice[k];
        }

        if sum_sq == 0.0 {
            continue;
        }

        let inv_mag = 1.0 / sum_sq.sqrt();

        for k in 0..chunks {
            let idx = k * 8;
            output_slice[idx] = input_slice[idx] * inv_mag;
            output_slice[idx + 1] = input_slice[idx + 1] * inv_mag;
            output_slice[idx + 2] = input_slice[idx + 2] * inv_mag;
            output_slice[idx + 3] = input_slice[idx + 3] * inv_mag;
            output_slice[idx + 4] = input_slice[idx + 4] * inv_mag;
            output_slice[idx + 5] = input_slice[idx + 5] * inv_mag;
            output_slice[idx + 6] = input_slice[idx + 6] * inv_mag;
            output_slice[idx + 7] = input_slice[idx + 7] * inv_mag;
        }

        for k in (chunks * 8)..n_features {
            output_slice[k] = input_slice[k] * inv_mag;
        }
    }

    normalized
}

// ============================================================================
// Nearest Neighbors
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
    let result = calculate_pca(&data, n_samples, n_features);
    Float32Array::from(&result[..])
}

#[wasm_bindgen]
pub fn kmeans_from_js(embeddings: Float32Array, n_samples: usize, n_features: usize, k: usize) -> Int8Array {
    let data: Vec<f32> = embeddings.to_vec();
    let seed = (Math::random() * 4294967295.0) as u32;
    let result = calculate_kmeans(&data, n_samples, n_features, k, seed);
    Int8Array::from(&result[..])
}

#[wasm_bindgen]
pub fn normalize_from_js(embeddings: Float32Array, n_samples: usize, n_features: usize) -> Float32Array {
    let data: Vec<f32> = embeddings.to_vec();
    let result = normalize_vectors(&data, n_samples, n_features);
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

