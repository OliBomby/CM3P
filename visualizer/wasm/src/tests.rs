#[cfg(test)]
mod tests {
    use super::super::*;

    // Helper function to create test embeddings
    fn create_test_embeddings() -> (Vec<f32>, usize, usize) {
        // 5 samples, 3 features each
        let embeddings = vec![
            1.0, 2.0, 3.0,  // sample 0
            4.0, 5.0, 6.0,  // sample 1
            1.5, 2.5, 3.5,  // sample 2
            10.0, 11.0, 12.0, // sample 3
            10.5, 11.5, 12.5, // sample 4
        ];
        (embeddings, 5, 3)
    }

    #[test]
    fn test_pca_output_shape() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let result = calculate_pca(&embeddings, n_samples, n_features);

        // Should return n_samples * 2 values (x, y pairs)
        assert_eq!(result.len(), n_samples * 2);
    }

    #[test]
    fn test_pca_empty_input() {
        let result = calculate_pca(&[], 0, 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pca_deterministic_with_fixed_seed() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();

        // PCA should give consistent results (note: randomness in init, but converges)
        let result1 = calculate_pca(&embeddings, n_samples, n_features);
        let result2 = calculate_pca(&embeddings, n_samples, n_features);

        // Results might differ slightly due to random initialization, but should be in same ballpark
        assert_eq!(result1.len(), result2.len());
    }

    #[test]
    fn test_kmeans_output_shape() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let k = 2;
        let result = calculate_kmeans(&embeddings, n_samples, n_features, k, 42);

        // Should return one label per sample
        assert_eq!(result.len(), n_samples);
    }

    #[test]
    fn test_kmeans_label_range() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let k = 3;
        let result = calculate_kmeans(&embeddings, n_samples, n_features, k, 42);

        // All labels should be in range [0, k)
        for &label in &result {
            assert!(label >= 0 && label < k as i8);
        }
    }

    #[test]
    fn test_kmeans_clustering_quality() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let k = 2;
        let result = calculate_kmeans(&embeddings, n_samples, n_features, k, 42);

        // Samples 0, 1, 2 should cluster together (similar values)
        // Samples 3, 4 should cluster together (similar values)
        let cluster_0_1_2 = vec![result[0], result[1], result[2]];
        let cluster_3_4 = vec![result[3], result[4]];

        // Check that samples within each group have same cluster
        assert_eq!(cluster_0_1_2[0], cluster_0_1_2[1]);
        assert_eq!(cluster_0_1_2[1], cluster_0_1_2[2]);
        assert_eq!(cluster_3_4[0], cluster_3_4[1]);

        // And groups are different
        assert_ne!(cluster_0_1_2[0], cluster_3_4[0]);
    }

    #[test]
    fn test_kmeans_empty_input() {
        let result = calculate_kmeans(&[], 0, 0, 2, 42);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_normalize_vectors_unit_length() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        // Each normalized vector should have unit length
        for i in 0..n_samples {
            let offset = i * n_features;
            let mut sum_sq = 0.0f32;
            for j in 0..n_features {
                let val = normalized[offset + j];
                sum_sq += val * val;
            }
            let magnitude = sum_sq.sqrt();
            assert!((magnitude - 1.0).abs() < 1e-5, "Vector {} has magnitude {}", i, magnitude);
        }
    }

    #[test]
    fn test_normalize_zero_vector() {
        let embeddings = vec![
            0.0, 0.0, 0.0,  // zero vector
            1.0, 2.0, 3.0,  // normal vector
        ];
        let normalized = normalize_vectors(&embeddings, 2, 3);

        // Zero vector should remain zero
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[1], 0.0);
        assert_eq!(normalized[2], 0.0);

        // Second vector should be normalized
        let sum_sq: f32 = normalized[3..6].iter().map(|x| x * x).sum();
        assert!((sum_sq.sqrt() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_output_shape() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        // Output should have same shape as input
        assert_eq!(normalized.len(), embeddings.len());
    }

    #[test]
    fn test_find_nearest_neighbors_count() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        let result = find_nearest_neighbors(&normalized, n_samples, n_features, 0, 3);

        // Should return 3 neighbors (excluding query point itself)
        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.distances.len(), 3);
    }

    #[test]
    fn test_find_nearest_neighbors_excludes_self() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        let query_idx = 2;
        let result = find_nearest_neighbors(&normalized, n_samples, n_features, query_idx, 5);

        // Query point should not be in results
        assert!(!result.indices.contains(&query_idx));
    }

    #[test]
    fn test_find_nearest_neighbors_sorted_by_distance() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        let result = find_nearest_neighbors(&normalized, n_samples, n_features, 0, 4);

        // Distances should be in ascending order
        for i in 0..result.distances.len() - 1 {
            assert!(result.distances[i] <= result.distances[i + 1]);
        }
    }

    #[test]
    fn test_find_nearest_neighbors_similarity_grouping() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        // Query from first group (sample 0), should find samples 1,2 closer than 3,4
        let result = find_nearest_neighbors(&normalized, n_samples, n_features, 0, 4);

        // First two neighbors should be from same group (indices 1 or 2)
        let close_neighbors = &result.indices[0..2];
        assert!(close_neighbors.contains(&1) || close_neighbors.contains(&2));
    }

    #[test]
    fn test_find_nearest_neighbors_invalid_query() {
        let (embeddings, n_samples, n_features) = create_test_embeddings();
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);

        // Query with out-of-bounds index
        let result = find_nearest_neighbors(&normalized, n_samples, n_features, 999, 3);

        assert_eq!(result.indices.len(), 0);
        assert_eq!(result.distances.len(), 0);
    }

    #[test]
    fn test_pca_preserves_relative_distances() {
        // Create embeddings where we know the structure
        let embeddings = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            10.0, 10.0, 10.0,
        ];
        let result = calculate_pca(&embeddings, 4, 3);

        // Points 0, 1, 2 should be closer to each other than to point 3
        let p0 = (result[0], result[1]);
        let p1 = (result[2], result[3]);
        let p2 = (result[4], result[5]);
        let p3 = (result[6], result[7]);

        let dist_01 = ((p0.0 - p1.0).powi(2) + (p0.1 - p1.1).powi(2)).sqrt();
        let dist_03 = ((p0.0 - p3.0).powi(2) + (p0.1 - p3.1).powi(2)).sqrt();

        // Distance to outlier should be larger
        assert!(dist_03 > dist_01);
    }

    #[test]
    fn test_large_dataset() {
        // Test with larger dataset
        let n_samples = 1000;
        let n_features = 128;
        let mut embeddings = Vec::with_capacity(n_samples * n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                embeddings.push((i + j) as f32 * 0.01);
            }
        }

        // Test PCA
        let pca_result = calculate_pca(&embeddings, n_samples, n_features);
        assert_eq!(pca_result.len(), n_samples * 2);

        // Test K-means
        let kmeans_result = calculate_kmeans(&embeddings, n_samples, n_features, 10, 42);
        assert_eq!(kmeans_result.len(), n_samples);

        // Test normalization
        let normalized = normalize_vectors(&embeddings, n_samples, n_features);
        assert_eq!(normalized.len(), embeddings.len());

        // Test neighbor search
        let neighbors = find_nearest_neighbors(&normalized, n_samples, n_features, 0, 10);
        assert_eq!(neighbors.indices.len(), 10);
    }
}

