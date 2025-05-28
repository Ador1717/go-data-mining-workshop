package main

import (
    "fmt"
    "github.com/sjwhitworth/golearn/base"
    "github.com/sjwhitworth/golearn/cluster"
)

func main() {
    // Load dataset
    rawData, err := base.ParseCSVToInstances("datasets/zone_clustering.csv", true)
    if err != nil {
        panic(err)
    }

    // Note: Normalize data here if needed (GoLearn doesn't provide built-in scaler, so normalization must be manual or skipped)

    // Initialize KMeans with 3 clusters
    kmeans := cluster.NewKMeans(3)

    // Fit the model to the dataset
    clusters, err := kmeans.Fit(rawData)
    if err != nil {
        panic(err)
    }

    // Print cluster assignment for each instance
    fmt.Println("Cluster assignments:")
    for i, clusterID := range clusters {
        fmt.Printf("Zone %d -> Cluster %d\n", i, clusterID)
    }

    // Student tasks:
    // - Change number of clusters (e.g. 2, 4, 5)
    // - Engineer new features by modifying CSV before loading
}
