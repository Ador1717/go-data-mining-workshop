package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/clustering"
	"github.com/sjwhitworth/golearn/metrics/pairwise"
)

// loadCSV loads a CSV file and returns the parsed data
func loadCSV(filename string) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %v", err)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	return records[1:], nil // Skip header
}

// convertToFloat64 converts string data to float64
func convertToFloat64(data [][]string) ([][]float64, error) {
	result := make([][]float64, 0, len(data))
	
	for _, row := range data {
		if len(strings.TrimSpace(strings.Join(row, ""))) == 0 {
			continue
		}
		
		floatRow := make([]float64, 0, len(row)-1) // Skip first column (zone_id)
		for i := 1; i < len(row); i++ { // Start from index 1 to skip zone_id
			val := row[i]
			if strings.TrimSpace(val) == "" {
				continue
			}
			
			f, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
			if err != nil {
				return nil, fmt.Errorf("error converting '%s' to float64: %v", val, err)
			}
			floatRow = append(floatRow, f)
		}
		
		if len(floatRow) > 0 {
			result = append(result, floatRow)
		}
	}
	
	return result, nil
}

// createDataGrid creates a GoLearn FixedDataGrid from float64 data
func createDataGrid(data [][]float64) base.FixedDataGrid {
	cols := len(data[0])
	
	// Create attributes
	attrs := make([]base.Attribute, cols)
	featureNames := []string{"Avg_Speed", "Air_Quality", "Noise_Level", "Public_Transport"}
	for i := 0; i < cols; i++ {
		attrs[i] = base.NewFloatAttribute(featureNames[i])
	}
	
	// Create instances
	instances := base.NewDenseInstances()
	
	// Add attributes
	for _, attr := range attrs {
		instances.AddAttribute(attr)
	}
	
	// Extend to accommodate data
	instances.Extend(len(data))
	
	// Add data
	for rowIdx, row := range data {
		for colIdx, value := range row {
			attrSpec, err := instances.GetAttribute(attrs[colIdx])
			if err != nil {
				log.Printf("Error getting attribute spec: %v", err)
				continue
			}
			instances.Set(attrSpec, rowIdx, base.PackFloatToBytes(value))
		}
	}
	
	return instances
}

// performDBSCAN performs DBSCAN clustering using GoLearn
func performDBSCAN(data [][]float64) (clustering.ClusterMap, error) {
	fmt.Println("\n🔍 Performing DBSCAN Clustering...")
	
	// Create data grid
	dataGrid := createDataGrid(data)
	
	// Set up DBSCAN parameters
	params := clustering.DBSCANParameters{
		ClusterParameters: clustering.ClusterParameters{
			Attributes: dataGrid.AllAttributes(),
			Metric:     pairwise.NewEuclidean(),
		},
		Eps:      15.0, // Maximum distance for neighborhood
		MinCount: 5,    // Minimum points to form a cluster
	}
	
	// Perform DBSCAN clustering
	clusterMap, err := clustering.DBSCAN(dataGrid, params)
	if err != nil {
		return nil, fmt.Errorf("DBSCAN failed: %v", err)
	}
	
	return clusterMap, nil
}

// performExpectationMaximization performs EM clustering using GoLearn
func performExpectationMaximization(data [][]float64, nComponents int) (clustering.ClusterMap, error) {
	fmt.Println("\n🔍 Performing Expectation Maximization Clustering...")
	
	// Create data grid
	dataGrid := createDataGrid(data)
	
	// Create EM clusterer
	em, err := clustering.NewExpectationMaximization(nComponents)
	if err != nil {
		return nil, fmt.Errorf("failed to create EM clusterer: %v", err)
	}
	
	// Fit the model
	err = em.Fit(dataGrid)
	if err != nil {
		return nil, fmt.Errorf("EM fit failed: %v", err)
	}
	
	// Predict clusters
	clusterMap, err := em.Predict(dataGrid)
	if err != nil {
		return nil, fmt.Errorf("EM predict failed: %v", err)
	}
	
	return clusterMap, nil
}

// convertClusterMapToAssignments converts GoLearn ClusterMap to simple assignments array
func convertClusterMapToAssignments(clusterMap clustering.ClusterMap, dataSize int) []int {
	assignments := make([]int, dataSize)
	
	// Initialize all points as noise (-1)
	for i := range assignments {
		assignments[i] = -1
	}
	
	// Assign cluster IDs
	for clusterID, pointIndices := range clusterMap {
		for _, pointIndex := range pointIndices {
			if pointIndex < dataSize {
				assignments[pointIndex] = clusterID
			}
		}
	}
	
	return assignments
}

// calculateClusterCentroids calculates centroids for each cluster
func calculateClusterCentroids(data [][]float64, assignments []int) map[int][]float64 {
	centroids := make(map[int][]float64)
	counts := make(map[int]int)
	
	// Initialize centroids
	for _, clusterID := range assignments {
		if clusterID >= 0 {
			if _, exists := centroids[clusterID]; !exists {
				centroids[clusterID] = make([]float64, len(data[0]))
				counts[clusterID] = 0
			}
		}
	}
	
	// Sum up points in each cluster
	for i, clusterID := range assignments {
		if clusterID >= 0 {
			for j, val := range data[i] {
				centroids[clusterID][j] += val
			}
			counts[clusterID]++
		}
	}
	
	// Calculate averages
	for clusterID := range centroids {
		if counts[clusterID] > 0 {
			for j := range centroids[clusterID] {
				centroids[clusterID][j] /= float64(counts[clusterID])
			}
		}
	}
	
	return centroids
}

// performClustering performs clustering analysis using GoLearn
func performClustering() {
	fmt.Println("=== CLUSTERING: Urban Zones Analysis (GoLearn) ===")
	
	// Load data
	csvData, err := loadCSV("../../datasets/zones_clustering.csv")
	if err != nil {
		log.Fatalf("Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records\n", len(csvData))

	// Convert to float64
	data, err := convertToFloat64(csvData)
	if err != nil {
		log.Fatalf("Error converting data: %v", err)
	}

	fmt.Printf("Processed %d valid records with %d features\n", len(data), len(data[0]))

	// Perform DBSCAN clustering
	dbscanClusters, err := performDBSCAN(data)
	if err != nil {
		log.Printf("DBSCAN failed: %v", err)
	} else {
		fmt.Printf("DBSCAN found %d clusters\n", len(dbscanClusters))
		
		// Convert to assignments and calculate centroids
		dbscanAssignments := convertClusterMapToAssignments(dbscanClusters, len(data))
		dbscanCentroids := calculateClusterCentroids(data, dbscanAssignments)
		
		// Print cluster statistics
		printClusterStatistics("DBSCAN", dbscanAssignments, dbscanCentroids, data)
	}

	// Perform Expectation Maximization clustering
	emClusters, err := performExpectationMaximization(data, 3)
	if err != nil {
		log.Printf("Expectation Maximization failed: %v", err)
	} else {
		fmt.Printf("Expectation Maximization found %d clusters\n", len(emClusters))
		
		// Convert to assignments and calculate centroids
		emAssignments := convertClusterMapToAssignments(emClusters, len(data))
		emCentroids := calculateClusterCentroids(data, emAssignments)
		
		// Print cluster statistics
		printClusterStatistics("Expectation Maximization", emAssignments, emCentroids, data)
	}
}

// printClusterStatistics prints detailed statistics for each cluster
func printClusterStatistics(algorithmName string, assignments []int, centroids map[int][]float64, data [][]float64) {
	fmt.Printf("\n📊 %s Cluster Analysis:\n", algorithmName)
	featureNames := []string{"Avg Speed (km/h)", "Air Quality Index", "Noise Level (dB)", "Public Transport Use"}
	
	// Count points in each cluster
	clusterCounts := make(map[int]int)
	for _, clusterID := range assignments {
		if clusterID >= 0 {
			clusterCounts[clusterID]++
		}
	}
	
	// Count noise points (cluster ID = -1)
	noiseCount := 0
	for _, clusterID := range assignments {
		if clusterID == -1 {
			noiseCount++
		}
	}
	
	if noiseCount > 0 {
		fmt.Printf("   🔸 Noise points: %d\n", noiseCount)
	}
	
	for clusterID, centroid := range centroids {
		count := clusterCounts[clusterID]
		fmt.Printf("\n🏙️  Cluster %d (%d zones):\n", clusterID, count)
		fmt.Printf("   Centroid: [%.2f, %.2f, %.2f, %.2f]\n", 
			centroid[0], centroid[1], centroid[2], centroid[3])
		
		// Calculate detailed statistics for this cluster
		var clusterData [][]float64
		for i, assignedCluster := range assignments {
			if assignedCluster == clusterID {
				clusterData = append(clusterData, data[i])
			}
		}
		
		if len(clusterData) > 0 {
			for f := 0; f < len(featureNames); f++ {
				var values []float64
				for _, point := range clusterData {
					values = append(values, point[f])
				}
				
				sum := 0.0
				min, max := values[0], values[0]
				for _, v := range values {
					sum += v
					if v < min {
						min = v
					}
					if v > max {
						max = v
					}
				}
				mean := sum / float64(len(values))
				
				fmt.Printf("   %s: Mean=%.2f, Range=[%.2f, %.2f]\n", 
					featureNames[f], mean, min, max)
			}
		}
	}
	
	fmt.Printf("\n📈 %s Quality Metrics:\n", algorithmName)
	fmt.Printf("   Total data points: %d\n", len(data))
	fmt.Printf("   Number of clusters: %d\n", len(centroids))
	fmt.Printf("   Noise points: %d\n", noiseCount)
	if len(centroids) > 0 {
		fmt.Printf("   Average points per cluster: %.1f\n", float64(len(data)-noiseCount)/float64(len(centroids)))
	}
}

func main() {
	fmt.Println("🏙️  Urban Zones Clustering Exercise")
	fmt.Println("===================================")
	fmt.Println("Using GoLearn Clustering Algorithms")
	fmt.Println()

	performClustering()

	fmt.Println("\n✅ Clustering Exercise Complete!")
	fmt.Println("💡 Algorithms: DBSCAN & Expectation Maximization")
	fmt.Println("🔧 Implementation: GoLearn Machine Learning Library")
} 