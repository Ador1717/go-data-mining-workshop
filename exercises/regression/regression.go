package regression

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

// LinearRegression represents a simple linear regression model
type LinearRegression struct {
	Weights []float64
	Bias    float64
}

// TrainData represents training data
type TrainData struct {
	Features [][]float64
	Targets  []float64
}

// LoadData loads CSV data for regression
func loadData(filename string) (*TrainData, error) {
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
	targets := make([]float64, len(validRecords))

	for i, row := range validRecords {
		// Assume last column is target, rest are features
		featureRow := make([]float64, len(row)-1)
		for j := 0; j < len(row)-1; j++ {
			val, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil {
				return nil, fmt.Errorf("error parsing feature at row %d, col %d: %v", i+1, j, err)
			}
			featureRow[j] = val
		}
		features[i] = featureRow

		// Parse target (last column)
		target, err := strconv.ParseFloat(strings.TrimSpace(row[len(row)-1]), 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing target at row %d: %v", i+1, err)
		}
		targets[i] = target
	}

	return &TrainData{Features: features, Targets: targets}, nil
}

// TrainTestSplit splits data into training and testing sets
func trainTestSplit(data *TrainData, trainRatio float64) (*TrainData, *TrainData) {
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
	trainTargets := make([]float64, trainSize)
	testFeatures := make([][]float64, n-trainSize)
	testTargets := make([]float64, n-trainSize)
	
	for i := 0; i < trainSize; i++ {
		idx := indices[i]
		trainFeatures[i] = data.Features[idx]
		trainTargets[i] = data.Targets[idx]
	}
	
	for i := trainSize; i < n; i++ {
		idx := indices[i]
		testFeatures[i-trainSize] = data.Features[idx]
		testTargets[i-trainSize] = data.Targets[idx]
	}
	
	return &TrainData{Features: trainFeatures, Targets: trainTargets},
		   &TrainData{Features: testFeatures, Targets: testTargets}
}

// Fit trains the linear regression model using gradient descent
func (lr *LinearRegression) Fit(data *TrainData, learningRate float64, epochs int) {
	if len(data.Features) == 0 {
		return
	}
	
	numFeatures := len(data.Features[0])
	lr.Weights = make([]float64, numFeatures)
	lr.Bias = 0.0
	
	// Initialize weights randomly
	rand.Seed(time.Now().UnixNano())
	for i := range lr.Weights {
		lr.Weights[i] = rand.Float64()*0.01 - 0.005 // Small random values
	}
	
	n := float64(len(data.Features))
	
	for epoch := 0; epoch < epochs; epoch++ {
		// Calculate gradients
		weightGrads := make([]float64, numFeatures)
		biasGrad := 0.0
		
		for i, features := range data.Features {
			prediction := lr.predict(features)
			error := prediction - data.Targets[i]
			
			// Update gradients
			for j, feature := range features {
				weightGrads[j] += error * feature
			}
			biasGrad += error
		}
		
		// Update weights and bias
		for j := range lr.Weights {
			lr.Weights[j] -= learningRate * weightGrads[j] / n
		}
		lr.Bias -= learningRate * biasGrad / n
	}
}

// predict makes a prediction for a single sample
func (lr *LinearRegression) predict(features []float64) float64 {
	prediction := lr.Bias
	for i, weight := range lr.Weights {
		prediction += weight * features[i]
	}
	return prediction
}

// Predict makes predictions for multiple samples
func (lr *LinearRegression) Predict(data *TrainData) []float64 {
	predictions := make([]float64, len(data.Features))
	for i, features := range data.Features {
		predictions[i] = lr.predict(features)
	}
	return predictions
}

// CalculateMSE calculates Mean Squared Error
func calculateMSE(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		return math.NaN()
	}
	
	sum := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		sum += diff * diff
	}
	return sum / float64(len(actual))
}

// normalizeFeatures normalizes features to have zero mean and unit variance
func normalizeFeatures(data *TrainData) (*TrainData, []float64, []float64) {
	if len(data.Features) == 0 {
		return data, nil, nil
	}
	
	numFeatures := len(data.Features[0])
	means := make([]float64, numFeatures)
	stds := make([]float64, numFeatures)
	
	// Calculate means
	for _, features := range data.Features {
		for j, feature := range features {
			means[j] += feature
		}
	}
	for j := range means {
		means[j] /= float64(len(data.Features))
	}
	
	// Calculate standard deviations
	for _, features := range data.Features {
		for j, feature := range features {
			diff := feature - means[j]
			stds[j] += diff * diff
		}
	}
	for j := range stds {
		stds[j] = math.Sqrt(stds[j] / float64(len(data.Features)))
		if stds[j] == 0 {
			stds[j] = 1 // Prevent division by zero
		}
	}
	
	// Normalize features
	normalizedFeatures := make([][]float64, len(data.Features))
	for i, features := range data.Features {
		normalizedFeatures[i] = make([]float64, numFeatures)
		for j, feature := range features {
			normalizedFeatures[i][j] = (feature - means[j]) / stds[j]
		}
	}
	
	return &TrainData{Features: normalizedFeatures, Targets: data.Targets}, means, stds
}

// normalizeTestFeatures normalizes test features using training means and stds
func normalizeTestFeatures(data *TrainData, means, stds []float64) *TrainData {
	if len(data.Features) == 0 || len(means) == 0 || len(stds) == 0 {
		return data
	}
	
	normalizedFeatures := make([][]float64, len(data.Features))
	for i, features := range data.Features {
		normalizedFeatures[i] = make([]float64, len(features))
		for j, feature := range features {
			normalizedFeatures[i][j] = (feature - means[j]) / stds[j]
		}
	}
	
	return &TrainData{Features: normalizedFeatures, Targets: data.Targets}
}

// Run executes the regression exercise
func Run() {
	fmt.Println("Running Regression Exercise...")

	// Load dataset
	data, err := loadData("datasets/housing_prices.csv")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded %d samples with %d features\n", len(data.Features), len(data.Features[0]))

	// Split dataset into train/test (80/20)
	trainData, testData := trainTestSplit(data, 0.8)
	fmt.Printf("Training samples: %d, Test samples: %d\n", len(trainData.Features), len(testData.Features))

	// Normalize features
	normalizedTrainData, means, stds := normalizeFeatures(trainData)
	normalizedTestData := normalizeTestFeatures(testData, means, stds)

	// Initialize and train Linear Regression model with smaller learning rate
	lr := &LinearRegression{}
	lr.Fit(normalizedTrainData, 0.001, 1000) // Much smaller learning rate: 0.001

	// Predict on test data
	predictions := lr.Predict(normalizedTestData)

	// Evaluate MSE (Mean Squared Error)
	mse := calculateMSE(testData.Targets, predictions)
	fmt.Printf("Mean Squared Error: %.2f\n", mse)

	// Print predicted vs actual price for first 10 test samples
	fmt.Println("Predicted vs Actual (first 10 samples):")
	for i := 0; i < 10 && i < len(testData.Targets); i++ {
		fmt.Printf("Predicted: %.2f, Actual: %.2f\n", predictions[i], testData.Targets[i])
	}

	// Student tasks:
	// - Adjust learning rate and epochs
	// - Try different normalization methods
	// - Try different train/test split ratios
} 