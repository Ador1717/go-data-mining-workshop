package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

// FixedDataGrid represents a GoLearn-style data structure
type FixedDataGrid struct {
	data   [][]float64
	labels []string
	attrs  []string
}

// Size returns the number of rows and columns
func (f *FixedDataGrid) Size() (int, int) {
	if len(f.data) == 0 {
		return 0, 0
	}
	return len(f.data), len(f.data[0])
}

// KnnClassifier represents a GoLearn-style KNN classifier
type KnnClassifier struct {
	distance string
	weighting string
	k int
	trainData *FixedDataGrid
}

// NewKnnClassifier creates a new GoLearn-style KNN classifier
func NewKnnClassifier(distance, weighting string, k int) *KnnClassifier {
	return &KnnClassifier{
		distance: distance,
		weighting: weighting,
		k: k,
	}
}

// Fit trains the classifier with training data (GoLearn-style API)
func (knn *KnnClassifier) Fit(trainData *FixedDataGrid) error {
	knn.trainData = trainData
	return nil
}

// euclideanDistance calculates the Euclidean distance between two points
func (knn *KnnClassifier) euclideanDistance(point1, point2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(point1); i++ {
		diff := point1[i] - point2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Predict makes predictions on test data (GoLearn-style API)
func (knn *KnnClassifier) Predict(testData *FixedDataGrid) (*FixedDataGrid, error) {
	predictions := make([]string, len(testData.data))
	
	for i, testPoint := range testData.data {
		// Calculate distances to all training points
		type neighbor struct {
			distance float64
			label    string
		}
		
		neighbors := make([]neighbor, len(knn.trainData.data))
		for j, trainPoint := range knn.trainData.data {
			dist := knn.euclideanDistance(testPoint, trainPoint)
			neighbors[j] = neighbor{distance: dist, label: knn.trainData.labels[j]}
		}
		
		// Sort by distance
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].distance < neighbors[b].distance
		})
		
		// Count votes from K nearest neighbors
		votes := make(map[string]int)
		for j := 0; j < knn.k && j < len(neighbors); j++ {
			votes[neighbors[j].label]++
		}
		
		// Find the most voted class
		maxVotes := 0
		var prediction string
		for label, count := range votes {
			if count > maxVotes {
				maxVotes = count
				prediction = label
			}
		}
		
		predictions[i] = prediction
	}
	
	// Return predictions in GoLearn-style format
	return &FixedDataGrid{
		data:   testData.data,
		labels: predictions,
		attrs:  testData.attrs,
	}, nil
}

// createInstances creates GoLearn-style instances from CSV data
func createInstances(data *CSVData) (*FixedDataGrid, error) {
	features := make([][]float64, len(data.Rows))
	labels := make([]string, len(data.Rows))
	
	for i, row := range data.Rows {
		if len(row) < 5 {
			continue
		}
		
		// Parse features (first 4 columns)
		feature := make([]float64, 4)
		for j := 0; j < 4; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing feature at row %d, col %d: %v", i, j, err)
			}
			feature[j] = val
		}
		
		features[i] = feature
		labels[i] = strings.TrimSpace(row[4])
	}
	
	return &FixedDataGrid{
		data:   features,
		labels: labels,
		attrs:  data.Headers[:4],
	}, nil
}

// InstancesTrainTestSplit splits data into training and test sets (GoLearn-style API)
func InstancesTrainTestSplit(instances *FixedDataGrid, trainRatio float64) (*FixedDataGrid, *FixedDataGrid) {
	trainSize := int(float64(len(instances.data)) * trainRatio)
	
	trainData := &FixedDataGrid{
		data:   instances.data[:trainSize],
		labels: instances.labels[:trainSize],
		attrs:  instances.attrs,
	}
	
	testData := &FixedDataGrid{
		data:   instances.data[trainSize:],
		labels: instances.labels[trainSize:],
		attrs:  instances.attrs,
	}
	
	return trainData, testData
}

// GetConfusionMatrix calculates accuracy (simplified version of GoLearn's evaluation)
func GetConfusionMatrix(testData, predictions *FixedDataGrid) (map[string]interface{}, error) {
	correct := 0
	total := len(testData.labels)
	
	for i := 0; i < total; i++ {
		if testData.labels[i] == predictions.labels[i] {
			correct++
		}
	}
	
	accuracy := float64(correct) / float64(total)
	
	return map[string]interface{}{
		"accuracy": accuracy,
		"correct":  correct,
		"total":    total,
	}, nil
}

// GetSummary returns a summary string (GoLearn-style API)
func GetSummary(confusionMatrix map[string]interface{}) string {
	accuracy := confusionMatrix["accuracy"].(float64)
	correct := confusionMatrix["correct"].(int)
	total := confusionMatrix["total"].(int)
	
	return fmt.Sprintf("Overall accuracy: %.4f (%d/%d correct predictions)", 
		accuracy, correct, total)
}

// performGoLearnKNN performs classification using GoLearn-style KNN classifier
func performGoLearnKNN(instances *FixedDataGrid, k int) error {
	fmt.Printf("\n🔍 Performing KNN Classification (k=%d) using GoLearn-style API...\n", k)

	// Split data into training and test sets (80/20)
	trainData, testData := InstancesTrainTestSplit(instances, 0.8)

	fmt.Printf("Training samples: %d\n", func() int { rows, _ := trainData.Size(); return rows }())
	fmt.Printf("Test samples: %d\n", func() int { rows, _ := testData.Size(); return rows }())

	// Create KNN classifier
	classifier := NewKnnClassifier("euclidean", "linear", k)

	// Train the classifier
	err := classifier.Fit(trainData)
	if err != nil {
		return fmt.Errorf("failed to fit KNN classifier: %v", err)
	}

	// Make predictions
	predictions, err := classifier.Predict(testData)
	if err != nil {
		return fmt.Errorf("failed to make predictions: %v", err)
	}

	// Calculate accuracy using GoLearn-style evaluation
	confusionMatrix, err := GetConfusionMatrix(testData, predictions)
	if err != nil {
		return fmt.Errorf("failed to get confusion matrix: %v", err)
	}

	fmt.Printf("\n📊 GoLearn-style KNN Results:\n")
	fmt.Println(GetSummary(confusionMatrix))

	// Show some example predictions
	fmt.Println("\nSample Predictions:")
	for i := 0; i < 5 && i < len(testData.labels); i++ {
		fmt.Printf("Actual: %-15s | Predicted: %-15s | %s\n", 
			testData.labels[i], predictions.labels[i], 
			func() string { if testData.labels[i] == predictions.labels[i] { return "✓" } else { return "✗" } }())
	}

	return nil
}

// createSleepPatternChart creates a single chart showing sleep patterns
func createSleepPatternChart(data *CSVData) {
	fmt.Println("\n📈 Creating Sleep Pattern Visualization...")

	p := plot.New()
	p.Title.Text = "Sleep Hours vs Caffeine Intake by Sleep Type"
	p.X.Label.Text = "Hours of Sleep"
	p.Y.Label.Text = "Caffeine Intake (mg)"

	// Separate data by sleep type
	var morningPoints, nightPoints plotter.XYs

	for _, row := range data.Rows {
		if len(row) < 5 {
			continue
		}

		caffeine, err1 := strconv.ParseFloat(strings.TrimSpace(row[0]), 64)
		sleepHours, err2 := strconv.ParseFloat(strings.TrimSpace(row[1]), 64)
		sleepType := strings.TrimSpace(row[4])

		if err1 != nil || err2 != nil {
			continue
		}

		point := plotter.XY{X: sleepHours, Y: caffeine}

		if sleepType == "Morning Person" {
			morningPoints = append(morningPoints, point)
		} else if sleepType == "Night Owl" {
			nightPoints = append(nightPoints, point)
		}
	}

	// Create scatter plots
	if len(morningPoints) > 0 {
		morningScatter, err := plotter.NewScatter(morningPoints)
		if err == nil {
			morningScatter.GlyphStyle.Color = plotter.DefaultGlyphStyle.Color
			morningScatter.GlyphStyle.Radius = vg.Points(3)
			p.Add(morningScatter)
			p.Legend.Add("Morning Person", morningScatter)
		}
	}

	if len(nightPoints) > 0 {
		nightScatter, err := plotter.NewScatter(nightPoints)
		if err == nil {
			nightScatter.GlyphStyle.Color = plotter.DefaultLineStyle.Color
			nightScatter.GlyphStyle.Radius = vg.Points(3)
			p.Add(nightScatter)
			p.Legend.Add("Night Owl", nightScatter)
		}
	}

	// Save plot
	filename := "sleep_pattern_analysis.png"
	if err := p.Save(8*vg.Inch, 6*vg.Inch, filename); err != nil {
		log.Printf("Error saving plot: %v", err)
	} else {
		fmt.Printf("Chart saved as %s\n", filename)
	}
}

// performClassification performs the complete classification analysis
func performClassification() {
	fmt.Println("=== CLASSIFICATION: Sleep Pattern Analysis (GoLearn-style KNN) ===")

	// Load data
	data, err := loadCSV("../../datasets/sleep_classification.csv")
	if err != nil {
		log.Fatalf("Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records with %d features\n", len(data.Rows), len(data.Headers)-1)
	fmt.Printf("Features: %s\n", strings.Join(data.Headers[:len(data.Headers)-1], ", "))
	fmt.Printf("Target: %s\n", data.Headers[len(data.Headers)-1])

	// Create GoLearn-style instances
	instances, err := createInstances(data)
	if err != nil {
		log.Fatalf("Error creating instances: %v", err)
	}

	// Perform GoLearn-style KNN classification
	err = performGoLearnKNN(instances, 3)
	if err != nil {
		log.Printf("GoLearn-style KNN failed: %v", err)
	}

	// Create visualization
	createSleepPatternChart(data)
}

func main() {
	fmt.Println("🧠 Sleep Pattern Classification Exercise")
	fmt.Println("========================================")
	fmt.Println("Using GoLearn-style KNN Classifier")
	fmt.Println()

	performClassification()

	fmt.Println("\n✅ Classification Exercise Complete!")
	fmt.Println("💡 Algorithm: K-Nearest Neighbors (GoLearn-style API)")
	fmt.Println("🔧 Implementation: GoLearn-compatible Interface")
	fmt.Println("📈 Visualization: Single Sleep Pattern Chart")
} 