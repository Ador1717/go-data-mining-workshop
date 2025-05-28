package clustering

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// Point represents a data point with multiple dimensions
type Point []float64

// Cluster represents a cluster with a centroid and assigned points
type Cluster struct {
	Centroid Point
	Points   []Point
}

// Distance calculates Euclidean distance between two points
func distance(p1, p2 Point) float64 {
	if len(p1) != len(p2) {
		return math.Inf(1)
	}
	
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// FindClosestCluster returns the index of the closest cluster centroid
func findClosestCluster(point Point, centroids []Point) int {
	minDist := math.Inf(1)
	closestIdx := 0
	
	for i, centroid := range centroids {
		dist := distance(point, centroid)
		if dist < minDist {
			minDist = dist
			closestIdx = i
		}
	}
	return closestIdx
}

// CalculateCentroid calculates the centroid of a set of points
func calculateCentroid(points []Point) Point {
	if len(points) == 0 {
		return nil
	}
	
	dimensions := len(points[0])
	centroid := make(Point, dimensions)
	
	for _, point := range points {
		for i := range point {
			centroid[i] += point[i]
		}
	}
	
	for i := range centroid {
		centroid[i] /= float64(len(points))
	}
	
	return centroid
}

// KMeans performs K-means clustering
func kmeans(data []Point, k int, maxIterations int) []Cluster {
	if len(data) == 0 || k <= 0 {
		return nil
	}
	
	rand.Seed(time.Now().UnixNano())
	
	// Initialize centroids randomly
	centroids := make([]Point, k)
	for i := 0; i < k; i++ {
		// Pick a random point as initial centroid
		randomIdx := rand.Intn(len(data))
		centroids[i] = make(Point, len(data[randomIdx]))
		copy(centroids[i], data[randomIdx])
	}
	
	for iteration := 0; iteration < maxIterations; iteration++ {
		// Create clusters
		clusters := make([]Cluster, k)
		for i := range clusters {
			clusters[i].Centroid = make(Point, len(centroids[i]))
			copy(clusters[i].Centroid, centroids[i])
			clusters[i].Points = []Point{}
		}
		
		// Assign points to closest clusters
		for _, point := range data {
			closestIdx := findClosestCluster(point, centroids)
			clusters[closestIdx].Points = append(clusters[closestIdx].Points, point)
		}
		
		// Update centroids
		converged := true
		for i := range clusters {
			if len(clusters[i].Points) > 0 {
				newCentroid := calculateCentroid(clusters[i].Points)
				if distance(centroids[i], newCentroid) > 1e-6 {
					converged = false
				}
				centroids[i] = newCentroid
			}
		}
		
		if converged {
			fmt.Printf("Converged after %d iterations\n", iteration+1)
			return clusters
		}
	}
	
	// Final cluster assignment
	clusters := make([]Cluster, k)
	for i := range clusters {
		clusters[i].Centroid = centroids[i]
		clusters[i].Points = []Point{}
	}
	
	for _, point := range data {
		closestIdx := findClosestCluster(point, centroids)
		clusters[closestIdx].Points = append(clusters[closestIdx].Points, point)
	}
	
	return clusters
}

// LoadData loads CSV data for clustering
func loadData(filename string) ([]Point, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("not enough data")
	}

	// Skip header row and filter out empty lines
	var validRecords [][]string
	for i, row := range records {
		if i == 0 {
			continue // Skip header
		}
		// Skip empty lines or lines with empty values
		if len(row) == 0 || strings.Join(row, "") == "" {
			continue
		}
		// Check if all values are non-empty
		allValid := true
		for _, val := range row {
			if strings.TrimSpace(val) == "" {
				allValid = false
				break
			}
		}
		if allValid {
			validRecords = append(validRecords, row)
		}
	}

	points := make([]Point, len(validRecords))

	for i, row := range validRecords {
		// Skip first column (area names) and only use numeric features
		point := make(Point, len(row)-1)
		for j := 1; j < len(row); j++ { // Start from index 1 to skip first column
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing value at row %d, col %d: %v", i+1, j, err)
			}
			point[j-1] = val // Adjust index since we're skipping first column
		}
		points[i] = point
	}

	return points, nil
}

// Run executes the clustering exercise
func Run() {
	fmt.Println("Running Clustering Exercise...")

	// Load dataset
	data, err := loadData("datasets/zones_clustering.csv")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded %d data points with %d dimensions\n", len(data), len(data[0]))

	// Perform K-means clustering with k=3
	k := 3
	maxIterations := 100
	clusters := kmeans(data, k, maxIterations)

	// Display results
	fmt.Printf("\nK-means clustering results (k=%d):\n", k)
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d: %d points, Centroid: ", i+1, len(cluster.Points))
		for j, coord := range cluster.Centroid {
			if j > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.2f", coord)
		}
		fmt.Println()
	}

	// Show some example points from each cluster
	fmt.Println("\nExample points from each cluster:")
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d examples:\n", i+1)
		for j := 0; j < 3 && j < len(cluster.Points); j++ {
			fmt.Print("  [")
			for k, coord := range cluster.Points[j] {
				if k > 0 {
					fmt.Print(", ")
				}
				fmt.Printf("%.2f", coord)
			}
			fmt.Println("]")
		}
	}

	// Student tasks:
	// - Try different values of k (2, 4, 5)
	// - Experiment with different initialization methods
	// - Add visualization of clusters (if you implement plotting)
} 