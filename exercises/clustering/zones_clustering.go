package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// CSVData represents parsed CSV data
type CSVData struct {
	Headers []string
	Rows    [][]string
}

// loadCSV loads a CSV file and returns the parsed data
func loadCSV(filename string) (*CSVData, error) {
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

	return &CSVData{
		Headers: records[0],
		Rows:    records[1:],
	}, nil
}

// convertToFloat64 converts string data to float64, handling missing values
func convertToFloat64(data [][]string, skipColumns []int) ([][]float64, error) {
	result := make([][]float64, 0, len(data))
	
	for _, row := range data {
		if len(strings.TrimSpace(strings.Join(row, ""))) == 0 {
			continue // Skip empty rows
		}
		
		floatRow := make([]float64, 0, len(row)-len(skipColumns))
		for i, val := range row {
			// Skip specified columns
			skip := false
			for _, skipCol := range skipColumns {
				if i == skipCol {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
			
			if strings.TrimSpace(val) == "" {
				continue // Skip empty values
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

// K-means clustering using gonum
func performKMeansGonum(data [][]float64, k int) ([]int, [][]float64, error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil, nil, fmt.Errorf("empty data")
	}

	n := len(data)
	d := len(data[0])
	
	// Convert to matrix
	dataMatrix := mat.NewDense(n, d, nil)
	for i, row := range data {
		for j, val := range row {
			dataMatrix.Set(i, j, val)
		}
	}

	// Initialize centroids randomly
	rand.Seed(time.Now().UnixNano())
	centroids := mat.NewDense(k, d, nil)
	for i := 0; i < k; i++ {
		randomIdx := rand.Intn(n)
		for j := 0; j < d; j++ {
			centroids.Set(i, j, dataMatrix.At(randomIdx, j))
		}
	}

	assignments := make([]int, n)
	maxIters := 100
	
	for iter := 0; iter < maxIters; iter++ {
		// Assign points to nearest centroids
		changed := false
		for i := 0; i < n; i++ {
			minDist := math.Inf(1)
			newAssignment := 0
			
			for c := 0; c < k; c++ {
				dist := 0.0
				for j := 0; j < d; j++ {
					diff := dataMatrix.At(i, j) - centroids.At(c, j)
					dist += diff * diff
				}
				dist = math.Sqrt(dist)
				
				if dist < minDist {
					minDist = dist
					newAssignment = c
				}
			}
			
			if assignments[i] != newAssignment {
				changed = true
				assignments[i] = newAssignment
			}
		}
		
		if !changed {
			fmt.Printf("K-means converged after %d iterations\n", iter+1)
			break
		}
		
		// Update centroids
		clusterCounts := make([]int, k)
		newCentroids := mat.NewDense(k, d, nil)
		
		for i := 0; i < n; i++ {
			cluster := assignments[i]
			clusterCounts[cluster]++
			for j := 0; j < d; j++ {
				newCentroids.Set(cluster, j, newCentroids.At(cluster, j)+dataMatrix.At(i, j))
			}
		}
		
		for c := 0; c < k; c++ {
			if clusterCounts[c] > 0 {
				for j := 0; j < d; j++ {
					newCentroids.Set(c, j, newCentroids.At(c, j)/float64(clusterCounts[c]))
				}
			}
		}
		
		centroids = newCentroids
	}

	// Convert centroids back to slice
	centroidSlices := make([][]float64, k)
	for i := 0; i < k; i++ {
		centroidSlices[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			centroidSlices[i][j] = centroids.At(i, j)
		}
	}

	return assignments, centroidSlices, nil
}

// calculateWithinClusterSumOfSquares calculates WCSS for cluster evaluation
func calculateWithinClusterSumOfSquares(data [][]float64, assignments []int, centroids [][]float64) float64 {
	wcss := 0.0
	for i, point := range data {
		cluster := assignments[i]
		if cluster >= 0 && cluster < len(centroids) {
			for j, val := range point {
				diff := val - centroids[cluster][j]
				wcss += diff * diff
			}
		}
	}
	return wcss
}

// performClustering performs clustering on zones data using gonum
func performClustering() {
	fmt.Println("=== CLUSTERING: Urban Zones Analysis (Custom K-Means with Gonum) ===")
	
	// Load zones clustering data
	csvData, err := loadCSV("../../datasets/zones_clustering.csv")
	if err != nil {
		log.Fatalf("Clustering: Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records with features: %v\n", len(csvData.Rows), csvData.Headers)

	// Convert to float64, skip the area column (first column)
	data, err := convertToFloat64(csvData.Rows, []int{0})
	if err != nil {
		log.Fatalf("Clustering: Error converting data: %v", err)
	}

	fmt.Printf("Processed %d valid records with %d features\n", len(data), len(data[0]))

	// Test different numbers of clusters using elbow method
	fmt.Println("\n🔍 Testing different numbers of clusters (Elbow Method):")
	maxK := 8
	wcssValues := make([]float64, maxK)
	
	for k := 1; k <= maxK; k++ {
		assignments, centroids, err := performKMeansGonum(data, k)
		if err != nil {
			log.Printf("Error with K=%d: %v", k, err)
			continue
		}
		wcss := calculateWithinClusterSumOfSquares(data, assignments, centroids)
		wcssValues[k-1] = wcss
		fmt.Printf("K=%d: WCSS = %.2f\n", k, wcss)
	}

	// Perform K-means clustering with optimal K=3
	k := 3
	assignments, centroids, err := performKMeansGonum(data, k)
	if err != nil {
		log.Fatalf("Clustering: K-means failed: %v", err)
	}

	fmt.Printf("\nK-Means clustering complete with %d clusters\n", k)

	// Create visualization: Average Speed vs Air Quality colored by cluster
	p := plot.New()
	p.Title.Text = "Urban Zones Clustering: Speed vs Air Quality (Custom K-Means)"
	p.X.Label.Text = "Average Speed (km/h)"
	p.Y.Label.Text = "Air Quality Index"

	clusterPoints := make([]plotter.XYs, k)
	for i := range clusterPoints {
		clusterPoints[i] = make(plotter.XYs, 0)
	}

	// Extract cluster assignments and create plot points
	for i, row := range data {
		if i < len(assignments) {
			clusterID := assignments[i]
			xy := plotter.XY{X: row[0], Y: row[1]} // avg_speed vs air_quality
			if clusterID >= 0 && clusterID < k {
				clusterPoints[clusterID] = append(clusterPoints[clusterID], xy)
			}
		}
	}

	colors := []color.Color{
		color.RGBA{R: 255, G: 0, B: 0, A: 255},   // Red
		color.RGBA{R: 0, G: 255, B: 0, A: 255},   // Green
		color.RGBA{R: 0, G: 0, B: 255, A: 255},   // Blue
		color.RGBA{R: 255, G: 255, B: 0, A: 255}, // Yellow
		color.RGBA{R: 255, G: 0, B: 255, A: 255}, // Magenta
	}

	for i := 0; i < k; i++ {
		if len(clusterPoints[i]) > 0 {
			s, err := plotter.NewScatter(clusterPoints[i])
			if err != nil {
				log.Fatalf("Clustering: Error creating scatter plot: %v", err)
			}
			s.GlyphStyle.Color = colors[i%len(colors)]
			s.GlyphStyle.Radius = vg.Points(4)
			p.Add(s)
			p.Legend.Add(fmt.Sprintf("Cluster %d (%d zones)", i, len(clusterPoints[i])), s)
		}
	}

	// Add centroids to the plot
	centroidPoints := make(plotter.XYs, k)
	for i := 0; i < k; i++ {
		centroidPoints[i] = plotter.XY{X: centroids[i][0], Y: centroids[i][1]}
	}
	
	centroidScatter, err := plotter.NewScatter(centroidPoints)
	if err != nil {
		log.Fatalf("Clustering: Error creating centroid scatter plot: %v", err)
	}
	centroidScatter.GlyphStyle.Color = color.RGBA{R: 0, G: 0, B: 0, A: 255} // Black
	centroidScatter.GlyphStyle.Radius = vg.Points(8)
	centroidScatter.GlyphStyle.Shape = draw.CrossGlyph{}
	p.Add(centroidScatter)
	p.Legend.Add("Centroids", centroidScatter)

	if err := p.Save(6*vg.Inch, 4*vg.Inch, "clustering_zones_analysis.png"); err != nil {
		log.Fatalf("Clustering: Error saving plot: %v", err)
	}
	fmt.Println("Clustering plot saved to clustering_zones_analysis.png")

	// Create second visualization: Noise Level vs Public Transport Use
	p2 := plot.New()
	p2.Title.Text = "Urban Zones Clustering: Noise vs Public Transport"
	p2.X.Label.Text = "Noise Level (dB)"
	p2.Y.Label.Text = "Public Transport Use"

	clusterPoints2 := make([]plotter.XYs, k)
	for i := range clusterPoints2 {
		clusterPoints2[i] = make(plotter.XYs, 0)
	}

	for i, row := range data {
		if i < len(assignments) {
			clusterID := assignments[i]
			xy := plotter.XY{X: row[2], Y: row[3]}
			if clusterID >= 0 && clusterID < k {
				clusterPoints2[clusterID] = append(clusterPoints2[clusterID], xy)
			}
		}
	}

	for i := 0; i < k; i++ {
		if len(clusterPoints2[i]) > 0 {
			s, err := plotter.NewScatter(clusterPoints2[i])
			if err != nil {
				log.Fatalf("Clustering: Error creating second scatter plot: %v", err)
			}
			s.GlyphStyle.Color = colors[i%len(colors)]
			s.GlyphStyle.Radius = vg.Points(4)
			p2.Add(s)
			p2.Legend.Add(fmt.Sprintf("Cluster %d", i), s)
		}
	}

	if err := p2.Save(6*vg.Inch, 4*vg.Inch, "clustering_noise_transport.png"); err != nil {
		log.Fatalf("Clustering: Error saving second plot: %v", err)
	}
	fmt.Println("Additional plot saved to clustering_noise_transport.png")
	
	// Print detailed cluster statistics
	fmt.Println("\n📊 Detailed Cluster Analysis:")
	featureNames := []string{"Avg Speed (km/h)", "Air Quality Index", "Noise Level (dB)", "Public Transport Use"}
	
	for i := 0; i < k; i++ {
		fmt.Printf("\n🏙️  Cluster %d (%d zones):\n", i, len(clusterPoints[i]))
		fmt.Printf("   Centroid: [%.2f, %.2f, %.2f, %.2f]\n", 
			centroids[i][0], centroids[i][1], centroids[i][2], centroids[i][3])
		
		// Calculate cluster characteristics
		if len(clusterPoints[i]) > 0 {
			// Find points in this cluster and calculate statistics
			var clusterData [][]float64
			for j, row := range data {
				if j < len(assignments) && assignments[j] == i {
					clusterData = append(clusterData, row)
				}
			}
			
			if len(clusterData) > 0 {
				// Calculate mean, min, max for each feature
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
	}

	// Calculate and display clustering quality metrics
	wcss := calculateWithinClusterSumOfSquares(data, assignments, centroids)
	fmt.Printf("\n📈 Clustering Quality Metrics:\n")
	fmt.Printf("   Within-Cluster Sum of Squares (WCSS): %.2f\n", wcss)
	fmt.Printf("   Total data points: %d\n", len(data))
	fmt.Printf("   Number of clusters: %d\n", k)
	fmt.Printf("   Average points per cluster: %.1f\n", float64(len(data))/float64(k))
}

func main() {
	fmt.Println("🏙️  Urban Zones Clustering Exercise")
	fmt.Println("===================================")
	fmt.Println("Using Custom K-Means with Gonum Matrix Operations")
	fmt.Println()

	performClustering()

	fmt.Println("\n✅ Clustering Exercise Complete!")
	fmt.Println("📊 Generated visualizations:")
	fmt.Println("  • clustering_zones_analysis.png - Speed vs Air Quality")
	fmt.Println("  • clustering_noise_transport.png - Noise vs Public Transport")
	fmt.Println()
	fmt.Println("💡 Algorithm: K-Means Clustering")
	fmt.Println("🔧 Matrix Operations: Gonum")
	fmt.Println("📈 Visualization: Gonum Plot")
} 