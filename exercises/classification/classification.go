package classification

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Data represents training data for classification
type Data struct {
	Features [][]float64
	Labels   []string
}

// KNNClassifier represents a k-nearest neighbors classifier
type KNNClassifier struct {
	K        int
	TrainData *Data
}

// DistanceLabel represents a distance and its corresponding label
type DistanceLabel struct {
	Distance float64
	Label    string
}

// LoadData loads CSV data for classification
func loadData(filename string) (*Data, error) {
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

	features := make([][]float64, len(validRecords))
	labels := make([]string, len(validRecords))

	for i, row := range validRecords {
		// Assume last column is label, rest are features
		featureRow := make([]float64, len(row)-1)
		for j := 0; j < len(row)-1; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing feature at row %d, col %d: %v", i+1, j, err)
			}
			featureRow[j] = val
		}
		features[i] = featureRow
		labels[i] = strings.TrimSpace(row[len(row)-1]) // Last column is the label
	}

	return &Data{Features: features, Labels: labels}, nil
}

// TrainTestSplit splits classification data into training and testing sets
func trainTestSplit(data *Data, trainRatio float64) (*Data, *Data) {
	rand.Seed(time.Now().UnixNano())
	
	n := len(data.Features)
	trainSize := int(float64(n) * trainRatio)
	
	// Create indices and shuffle them
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	
	trainFeatures := make([][]float64, trainSize)
	trainLabels := make([]string, trainSize)
	testFeatures := make([][]float64, n-trainSize)
	testLabels := make([]string, n-trainSize)
	
	for i := 0; i < trainSize; i++ {
		idx := indices[i]
		trainFeatures[i] = data.Features[idx]
		trainLabels[i] = data.Labels[idx]
	}
	
	for i := trainSize; i < n; i++ {
		idx := indices[i]
		testFeatures[i-trainSize] = data.Features[idx]
		testLabels[i-trainSize] = data.Labels[idx]
	}
	
	return &Data{Features: trainFeatures, Labels: trainLabels},
		   &Data{Features: testFeatures, Labels: testLabels}
}

// EuclideanDistance calculates the Euclidean distance between two feature vectors
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// NewKNNClassifier creates a new KNN classifier
func NewKNNClassifier(k int) *KNNClassifier {
	return &KNNClassifier{K: k}
}

// Fit trains the KNN classifier (just stores the training data)
func (knn *KNNClassifier) Fit(data *Data) {
	knn.TrainData = data
}

// Predict predicts the label for a single sample
func (knn *KNNClassifier) Predict(features []float64) string {
	if knn.TrainData == nil {
		return ""
	}
	
	// Calculate distances to all training samples
	distances := make([]DistanceLabel, len(knn.TrainData.Features))
	for i, trainFeatures := range knn.TrainData.Features {
		dist := euclideanDistance(features, trainFeatures)
		distances[i] = DistanceLabel{Distance: dist, Label: knn.TrainData.Labels[i]}
	}
	
	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})
	
	// Count votes from k nearest neighbors
	votes := make(map[string]int)
	for i := 0; i < knn.K && i < len(distances); i++ {
		votes[distances[i].Label]++
	}
	
	// Find the label with the most votes
	maxVotes := 0
	predictedLabel := ""
	for label, count := range votes {
		if count > maxVotes {
			maxVotes = count
			predictedLabel = label
		}
	}
	
	return predictedLabel
}

// PredictBatch predicts labels for multiple samples
func (knn *KNNClassifier) PredictBatch(data *Data) []string {
	predictions := make([]string, len(data.Features))
	for i, features := range data.Features {
		predictions[i] = knn.Predict(features)
	}
	return predictions
}

// CalculateAccuracy calculates the accuracy of predictions
func calculateAccuracy(actual, predicted []string) float64 {
	if len(actual) != len(predicted) {
		return 0.0
	}
	
	correct := 0
	for i := range actual {
		if actual[i] == predicted[i] {
			correct++
		}
	}
	
	return float64(correct) / float64(len(actual))
}

// PrintConfusionMatrix prints a simple confusion matrix
func printConfusionMatrix(actual, predicted []string) {
	// Get unique labels
	labelSet := make(map[string]bool)
	for _, label := range actual {
		labelSet[label] = true
	}
	for _, label := range predicted {
		labelSet[label] = true
	}
	
	labels := make([]string, 0, len(labelSet))
	for label := range labelSet {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	
	// Create confusion matrix
	matrix := make(map[string]map[string]int)
	for _, label := range labels {
		matrix[label] = make(map[string]int)
	}
	
	for i := range actual {
		matrix[actual[i]][predicted[i]]++
	}
	
	// Print matrix
	fmt.Println("\nConfusion Matrix:")
	fmt.Print("Actual\\Predicted\t")
	for _, label := range labels {
		fmt.Printf("%s\t", label)
	}
	fmt.Println()
	
	for _, actualLabel := range labels {
		fmt.Printf("%s\t\t", actualLabel)
		for _, predictedLabel := range labels {
			fmt.Printf("%d\t", matrix[actualLabel][predictedLabel])
		}
		fmt.Println()
	}
}

// Run executes the classification exercise
func Run() {
	fmt.Println("Running Classification Exercise...")

	// Load dataset
	data, err := loadData("datasets/sleep_classification.csv")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded %d samples with %d features\n", len(data.Features), len(data.Features[0]))

	// Split dataset (70% train, 30% test)
	trainData, testData := trainTestSplit(data, 0.7)
	fmt.Printf("Training samples: %d, Test samples: %d\n", len(trainData.Features), len(testData.Features))

	// Initialize KNN classifier with k=3
	knn := NewKNNClassifier(3)

	// Train model (KNN just stores the training data)
	knn.Fit(trainData)

	// Predict on test set
	predictions := knn.PredictBatch(testData)

	// Evaluate accuracy
	accuracy := calculateAccuracy(testData.Labels, predictions)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)

	// Print confusion matrix
	printConfusionMatrix(testData.Labels, predictions)

	// Print some example predictions
	fmt.Println("\nExample predictions (first 10 samples):")
	for i := 0; i < 10 && i < len(testData.Labels); i++ {
		fmt.Printf("Predicted: %s, Actual: %s\n", predictions[i], testData.Labels[i])
	}

	// Student tasks:
	// - Adjust k value (e.g. 1, 5, 7)
	// - Try different train/test split ratios
	// - Add feature normalization
} 