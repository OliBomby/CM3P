use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Int8Array, Math};

#[cfg(test)]
mod tests;

// Simple LCG random number generator for tests (works in native Rust)
fn simple_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

// ============================================================================
// PCA Implementation
// ============================================================================

#[wasm_bindgen]
pub fn calculate_pca(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    if n_samples == 0 || n_features == 0 {
        return vec![];
    }

    // Calculate mean
    let mut mean = vec![0.0f32; n_features];
    for i in 0..n_samples {
        let offset = i * n_features;
        for j in 0..n_features {
            mean[j] += embeddings_flat[offset + j];
        }
    }
    for j in 0..n_features {
        mean[j] /= n_samples as f32;
    }

    // Helper closure for centered data
    let get_centered = |i: usize, j: usize| -> f32 {
        embeddings_flat[i * n_features + j] - mean[j]
    };

    let mut components = Vec::with_capacity(2);
    let mut rng_state = 12345u32; // Seed for deterministic behavior in tests

    // Calculate first two principal components using power iteration
    for c in 0..2 {
        // Initialize eigenvector with random values
        let mut ev: Vec<f32> = (0..n_features)
            .map(|_| {
                #[cfg(target_arch = "wasm32")]
                {
                    Math::random() as f32 - 0.5
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    simple_random(&mut rng_state) - 0.5
                }
            })
            .collect();

        // Normalize
        let mut mag: f32 = ev.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in ev.iter_mut() {
            *val /= mag;
        }

        // Power iteration
        for _ in 0..8 {
            // Project data onto eigenvector
            let mut scores = vec![0.0f32; n_samples];
            for i in 0..n_samples {
                let mut s = 0.0f32;
                for j in 0..n_features {
                    s += get_centered(i, j) * ev[j];
                }
                scores[i] = s;
            }

            // Calculate next eigenvector estimate
            let mut next_ev = vec![0.0f32; n_features];
            for j in 0..n_features {
                let mut s = 0.0f32;
                for i in 0..n_samples {
                    s += scores[i] * get_centered(i, j);
                }
                next_ev[j] = s;
            }

            // Normalize
            mag = next_ev.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > 0.0 {
                for val in next_ev.iter_mut() {
                    *val /= mag;
                }
                ev = next_ev;
            }
        }

        // Orthogonalize second component
        if c == 1 {
            let u: &Vec<f32> = &components[0];
            let mut dot = 0.0f32;
            for k in 0..n_features {
                dot += u[k] * ev[k];
            }
            for k in 0..n_features {
                ev[k] -= dot * u[k];
            }
            mag = ev.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > 0.0 {
                for val in ev.iter_mut() {
                    *val /= mag;
                }
            }
        }

        components.push(ev);
    }

    // Project data onto principal components
    let mut projected = vec![0.0f32; n_samples * 2];
    for i in 0..n_samples {
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        for j in 0..n_features {
            let val = get_centered(i, j);
            x += val * components[0][j];
            y += val * components[1][j];
        }
        projected[i * 2] = x;
        projected[i * 2 + 1] = y;
    }

    projected
}

// ============================================================================
// K-Means Implementation
// ============================================================================

#[wasm_bindgen]
pub fn calculate_kmeans(embeddings_flat: &[f32], n_samples: usize, n_features: usize, k: usize, seed: u32) -> Vec<i8> {
    if n_samples == 0 || k == 0 {
        return vec![];
    }

    // Initialize centroids by randomly selecting k samples
    let mut rng_state = seed;
    let mut centroids = vec![0.0f32; k * n_features];
    for i in 0..k {
        // Simple LCG for deterministic random selection
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let idx = (rng_state as usize) % n_samples;
        for j in 0..n_features {
            centroids[i * n_features + j] = embeddings_flat[idx * n_features + j];
        }
    }

    let mut labels = vec![0i8; n_samples];

    // Run k-means iterations
    for _ in 0..5 {
        // Assignment step
        for i in 0..n_samples {
            let mut min_dist = f32::INFINITY;
            let mut best_cluster = 0i8;

            for c in 0..k {
                let mut dist = 0.0f32;
                for j in 0..n_features {
                    let diff = embeddings_flat[i * n_features + j] - centroids[c * n_features + j];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c as i8;
                }
            }
            labels[i] = best_cluster;
        }

        // Update step
        let mut sums = vec![0.0f32; k * n_features];
        let mut counts = vec![0usize; k];

        for i in 0..n_samples {
            let cluster = labels[i] as usize;
            counts[cluster] += 1;
            for j in 0..n_features {
                sums[cluster * n_features + j] += embeddings_flat[i * n_features + j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..n_features {
                    centroids[c * n_features + j] = sums[c * n_features + j] / counts[c] as f32;
                }
            }
        }
    }

    labels
}

// ============================================================================
// Vector Normalization
// ============================================================================

#[wasm_bindgen]
pub fn normalize_vectors(embeddings_flat: &[f32], n_samples: usize, n_features: usize) -> Vec<f32> {
    let mut normalized = vec![0.0f32; n_samples * n_features];

    for i in 0..n_samples {
        let offset = i * n_features;

        // Calculate magnitude
        let mut sum_sq = 0.0f32;
        for j in 0..n_features {
            let val = embeddings_flat[offset + j];
            sum_sq += val * val;
        }

        if sum_sq == 0.0 {
            // Zero vector stays zero
            continue;
        }

        let inv_mag = 1.0 / sum_sq.sqrt();
        for j in 0..n_features {
            normalized[offset + j] = embeddings_flat[offset + j] * inv_mag;
        }
    }

    normalized
}

// ============================================================================
// Nearest Neighbors Search
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
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(n_samples - 1);

    // Calculate cosine distances (using normalized vectors, so just 1 - dot product)
    for i in 0..n_samples {
        if i == query_idx {
            continue;
        }

        let offset = i * n_features;
        let mut dot = 0.0f32;
        for j in 0..n_features {
            dot += normalized_embeddings[query_offset + j] * normalized_embeddings[offset + j];
        }

        let distance = 1.0 - dot;
        results.push((i, distance));
    }

    // Partial sort to get top k neighbors
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

    // Convert indices to JS array
    let indices_array = Array::new();
    for idx in result.indices.iter() {
        indices_array.push(&JsValue::from(*idx as u32));
    }

    js_sys::Reflect::set(&obj, &"indices".into(), &indices_array).unwrap();
    js_sys::Reflect::set(&obj, &"distances".into(), &Float32Array::from(&result.distances[..])).unwrap();
    obj.into()
}
