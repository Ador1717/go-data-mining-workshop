package classification

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

// loadCSV loads a CSV file and returns features and labels
func loadCSV(filename string) ([][]float64, []string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var features [][]float64
	var labels []string

	for i, row := range records {
		if i == 0 {
			continue // skip header
		}
		if len(row) < 5 {
			continue
		}
		// First 4 are features
		feat := make([]float64, 4)
		for j := 0; j < 4; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, nil, err
			}
			feat[j] = val
		}
		features = append(features, feat)
		labels = append(labels, strings.TrimSpace(row[4]))
	}
	return features, labels, nil
}

// euclideanDistance computes Euclidean distance between two points
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// knnPredict predicts the labels of test samples based on k nearest neighbors
func knnPredict(trainX [][]float64, trainY []string, testX [][]float64, k int) []string {
	predictions := make([]string, len(testX))

	for i, testPoint := range testX {
		type neighbor struct {
			dist  float64
			label string
		}
		var neighbors []neighbor
		for j, trainPoint := range trainX {
			d := euclideanDistance(testPoint, trainPoint)
			neighbors = append(neighbors, neighbor{d, trainY[j]})
		}
		// Sort by distance
		sort.Slice(neighbors, func(i, j int) bool {
			return neighbors[i].dist < neighbors[j].dist
		})
		// Count votes
		votes := make(map[string]int)
		for j := 0; j < k && j < len(neighbors); j++ {
			votes[neighbors[j].label]++
		}
		// Select label with most votes
		maxVotes := 0
		chosen := ""
		for label, count := range votes {
			if count > maxVotes {
				maxVotes = count
				chosen = label
			}
		}
		predictions[i] = chosen
	}
	return predictions
}

// computeAccuracy compares true labels with predictions
func computeAccuracy(trueLabels, predicted []string) float64 {
	correct := 0
	for i := range trueLabels {
		if trueLabels[i] == predicted[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(trueLabels))
}

// trainTestSplit splits data into train/test sets
func trainTestSplit(X [][]float64, Y []string, ratio float64) ([][]float64, []string, [][]float64, []string) {
	n := len(X)
	// Create a local rand.Rand with a fixed seed for reproducibility
	r := rand.New(rand.NewSource(42))

	perm := r.Perm(n)
	trainSize := int(float64(n) * ratio)

	var trainX, testX [][]float64
	var trainY, testY []string

	for i, idx := range perm {
		if i < trainSize {
			trainX = append(trainX, X[idx])
			trainY = append(trainY, Y[idx])
		} else {
			testX = append(testX, X[idx])
			testY = append(testY, Y[idx])
		}
	}
	return trainX, trainY, testX, testY
}

// printConfusionMatrix prints a confusion matrix of actual vs predicted labels
func printConfusionMatrix(actual []string, predicted []string) {
	if len(actual) != len(predicted) {
		fmt.Println("Length mismatch in confusion matrix data")
		return
	}

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
	// Sort labels alphabetically
	sort.Strings(labels)

	// Initialize matrix
	matrix := make(map[string]map[string]int)
	for _, actualLabel := range labels {
		matrix[actualLabel] = make(map[string]int)
		for _, predictedLabel := range labels {
			matrix[actualLabel][predictedLabel] = 0
		}
	}

	// Populate matrix
	for i := range actual {
		matrix[actual[i]][predicted[i]]++
	}

	// Print matrix
	fmt.Println("\n Confusion Matrix (Actual vs Predicted):")
	fmt.Printf("%-15s", "Actual \\ Pred")
	for _, label := range labels {
		fmt.Printf("%-15s", label)
	}
	fmt.Println()
	for _, actualLabel := range labels {
		fmt.Printf("%-15s", actualLabel)
		for _, predictedLabel := range labels {
			fmt.Printf("%-15d", matrix[actualLabel][predictedLabel])
		}
		fmt.Println()
	}
}

// reads 4 float64 features from user input
func readUserInput() ([]float64, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\n👤 Enter if you drink energy drinks (0 or 1), hours of sleep, screen time before bed (hrs) and wake up hour separated by space (e.g. 0 6 2.4 9):")

	line, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	line = strings.TrimSpace(line)
	parts := strings.Fields(line)
	if len(parts) != 4 {
		return nil, fmt.Errorf("expected 4 values, got %d", len(parts))
	}

	var input []float64
	for _, p := range parts {
		val, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float value: %v", err)
		}
		input = append(input, val)
	}
	return input, nil
}

func Run() {
	fmt.Println("Simple KNN Classifier")

	X, Y, err := loadCSV("datasets/sleep_classification.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}

	k:=3
	 // <- change this value for different k and find the best one

	trainX, trainY, testX, testY := trainTestSplit(X, Y, 0.8)
	predicted := knnPredict(trainX, trainY, testX, k)
	accuracy := computeAccuracy(testY, predicted)
	fmt.Printf("\nk= %d\n", k)
	fmt.Printf("\nAccuracy: %.2f%%\n", accuracy*100)

	fmt.Println("\nSample Predictions:")
	for i := 0; i < 10 && i < len(testY); i++ {
		fmt.Printf("Actual: %-15s Predicted: %-15s\n", testY[i], predicted[i])
	}

	printConfusionMatrix(testY, predicted)

	//prediction based on input
	userInput, err := readUserInput()
	if err != nil {
		log.Fatal("Invalid input:", err)
	}

	//// knnPredict expects a slice of test points, so wrap userInput in a slice
	userPred := knnPredict(trainX, trainY, [][]float64{userInput}, k)
	fmt.Printf("\nPrediction for your input: %s\n", userPred[0])
}
