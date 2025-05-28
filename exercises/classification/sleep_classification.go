package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
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

// ConfusionMatrix represents a confusion matrix
type ConfusionMatrix struct {
	TruePositive  int
	TrueNegative  int
	FalsePositive int
	FalseNegative int
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

// Simple KNN classifier implementation that returns predictions
func performKNNClassificationWithPredictions(trainX, testX [][]float64, trainY, testY []float64, k int) ([]float64, float64) {
	predictions := make([]float64, len(testX))
	correct := 0
	
	for i, testPoint := range testX {
		// Calculate distances to all training points
		distances := make([]struct {
			distance float64
			label    float64
		}, len(trainX))
		
		for j, trainPoint := range trainX {
			dist := 0.0
			for d := 0; d < len(testPoint); d++ {
				diff := testPoint[d] - trainPoint[d]
				dist += diff * diff
			}
			distances[j] = struct {
				distance float64
				label    float64
			}{
				distance: math.Sqrt(dist),
				label:    trainY[j],
			}
		}
		
		// Sort by distance
		sort.Slice(distances, func(a, b int) bool {
			return distances[a].distance < distances[b].distance
		})
		
		// Get k nearest neighbors and find most common label
		labelCounts := make(map[float64]int)
		for j := 0; j < k && j < len(distances); j++ {
			labelCounts[distances[j].label]++
		}
		
		// Find most common label
		maxCount := 0
		var prediction float64
		for label, count := range labelCounts {
			if count > maxCount {
				maxCount = count
				prediction = label
			}
		}
		
		predictions[i] = prediction
		if prediction == testY[i] {
			correct++
		}
	}
	
	return predictions, float64(correct) / float64(len(testX))
}

// Simple KNN classifier implementation (original for K testing)
func performKNNClassification(trainX, testX [][]float64, trainY, testY []float64, k int) float64 {
	_, accuracy := performKNNClassificationWithPredictions(trainX, testX, trainY, testY, k)
	return accuracy
}

// calculateConfusionMatrix calculates confusion matrix for binary classification
func calculateConfusionMatrix(actual, predicted []float64) ConfusionMatrix {
	var cm ConfusionMatrix
	
	for i := 0; i < len(actual); i++ {
		if actual[i] == 1 && predicted[i] == 1 {
			cm.TruePositive++
		} else if actual[i] == 0 && predicted[i] == 0 {
			cm.TrueNegative++
		} else if actual[i] == 0 && predicted[i] == 1 {
			cm.FalsePositive++
		} else if actual[i] == 1 && predicted[i] == 0 {
			cm.FalseNegative++
		}
	}
	
	return cm
}

// printConfusionMatrix prints confusion matrix with metrics
func printConfusionMatrix(cm ConfusionMatrix) {
	fmt.Println("\n📊 Confusion Matrix:")
	fmt.Println("                    Predicted")
	fmt.Println("                Morning  Night")
	fmt.Printf("Actual Morning    %4d    %4d\n", cm.TrueNegative, cm.FalsePositive)
	fmt.Printf("       Night      %4d    %4d\n", cm.FalseNegative, cm.TruePositive)
	
	// Calculate metrics with safety checks
	total := cm.TruePositive + cm.TrueNegative + cm.FalsePositive + cm.FalseNegative
	fmt.Printf("\nTotal test samples: %d\n", total)
	
	if total == 0 {
		fmt.Println("❌ Error: No test samples found!")
		return
	}
	
	accuracy := float64(cm.TruePositive+cm.TrueNegative) / float64(total)
	
	var precision, recall, f1Score float64
	if cm.TruePositive+cm.FalsePositive > 0 {
		precision = float64(cm.TruePositive) / float64(cm.TruePositive+cm.FalsePositive)
	}
	if cm.TruePositive+cm.FalseNegative > 0 {
		recall = float64(cm.TruePositive) / float64(cm.TruePositive+cm.FalseNegative)
	}
	if precision+recall > 0 {
		f1Score = 2 * (precision * recall) / (precision + recall)
	}
	
	fmt.Println("\n📈 Classification Metrics:")
	fmt.Printf("Accuracy:  %.3f (%.1f%%)\n", accuracy, accuracy*100)
	fmt.Printf("Precision: %.3f (%.1f%%)\n", precision, precision*100)
	fmt.Printf("Recall:    %.3f (%.1f%%)\n", recall, recall*100)
	fmt.Printf("F1-Score:  %.3f\n", f1Score)
	
	fmt.Println("\n📋 Metric Explanations:")
	fmt.Println("• Accuracy: Overall correctness of predictions")
	fmt.Println("• Precision: Of predicted Night Owls, how many were actually Night Owls")
	fmt.Println("• Recall: Of actual Night Owls, how many were correctly identified")
	fmt.Println("• F1-Score: Harmonic mean of precision and recall")
	
	// Force output flush and add separator
	fmt.Println()
	fmt.Println("=" + strings.Repeat("=", 50))
}

// createSleepCoffeeCharts creates descriptive charts for sleep and coffee patterns
func createSleepCoffeeCharts(X [][]float64, y []float64) {
	// Chart 1: Sleep Hours Distribution by Sleep Type
	p1 := plot.New()
	p1.Title.Text = "Sleep Hours Distribution by Sleep Type"
	p1.X.Label.Text = "Sleep Hours"
	p1.Y.Label.Text = "Frequency"
	
	// Separate data by sleep type
	var morningHours, nightHours []float64
	for i, features := range X {
		sleepHours := features[1] // hours_sleep is index 1
		if y[i] == 0 { // Morning Person
			morningHours = append(morningHours, sleepHours)
		} else { // Night Owl
			nightHours = append(nightHours, sleepHours)
		}
	}
	
	// Create histogram for morning people
	morningHist, err := plotter.NewHist(plotter.Values(morningHours), 10)
	if err == nil {
		morningHist.FillColor = color.RGBA{R: 0, G: 150, B: 255, A: 150} // Blue
		p1.Add(morningHist)
		p1.Legend.Add("Morning People", morningHist)
	}
	
	// Create histogram for night owls
	nightHist, err := plotter.NewHist(plotter.Values(nightHours), 10)
	if err == nil {
		nightHist.FillColor = color.RGBA{R: 255, G: 100, B: 0, A: 150} // Orange
		p1.Add(nightHist)
		p1.Legend.Add("Night Owls", nightHist)
	}
	
	if err := p1.Save(8*vg.Inch, 6*vg.Inch, "sleep_hours_distribution.png"); err != nil {
		log.Printf("Warning: Could not save sleep hours chart: %v", err)
	} else {
		fmt.Println("Sleep hours distribution chart saved to sleep_hours_distribution.png")
	}
	
	// Chart 2: Coffee Intake Distribution by Sleep Type
	p2 := plot.New()
	p2.Title.Text = "Coffee Intake Distribution by Sleep Type"
	p2.X.Label.Text = "Caffeine (mg)"
	p2.Y.Label.Text = "Frequency"
	
	// Separate caffeine data by sleep type
	var morningCaffeine, nightCaffeine []float64
	for i, features := range X {
		caffeine := features[0] // caffeine_mg is index 0
		if y[i] == 0 { // Morning Person
			morningCaffeine = append(morningCaffeine, caffeine)
		} else { // Night Owl
			nightCaffeine = append(nightCaffeine, caffeine)
		}
	}
	
	// Create histogram for morning people caffeine
	morningCafHist, err := plotter.NewHist(plotter.Values(morningCaffeine), 12)
	if err == nil {
		morningCafHist.FillColor = color.RGBA{R: 0, G: 150, B: 255, A: 150} // Blue
		p2.Add(morningCafHist)
		p2.Legend.Add("Morning People", morningCafHist)
	}
	
	// Create histogram for night owls caffeine
	nightCafHist, err := plotter.NewHist(plotter.Values(nightCaffeine), 12)
	if err == nil {
		nightCafHist.FillColor = color.RGBA{R: 255, G: 100, B: 0, A: 150} // Orange
		p2.Add(nightCafHist)
		p2.Legend.Add("Night Owls", nightCafHist)
	}
	
	if err := p2.Save(8*vg.Inch, 6*vg.Inch, "coffee_intake_distribution.png"); err != nil {
		log.Printf("Warning: Could not save coffee chart: %v", err)
	} else {
		fmt.Println("Coffee intake distribution chart saved to coffee_intake_distribution.png")
	}
	
	// Chart 3: Sleep vs Coffee Correlation
	p3 := plot.New()
	p3.Title.Text = "Sleep Hours vs Coffee Intake by Sleep Type"
	p3.X.Label.Text = "Sleep Hours"
	p3.Y.Label.Text = "Caffeine (mg)"
	
	morningPersonPts := make(plotter.XYs, 0)
	nightOwlPts := make(plotter.XYs, 0)
	
	for i, features := range X {
		xy := plotter.XY{X: features[1], Y: features[0]} // sleep hours vs caffeine
		if y[i] == 0 { // Morning Person
			morningPersonPts = append(morningPersonPts, xy)
		} else { // Night Owl
			nightOwlPts = append(nightOwlPts, xy)
		}
	}
	
	s0, err := plotter.NewScatter(morningPersonPts)
	if err == nil {
		s0.GlyphStyle.Color = color.RGBA{R: 0, G: 150, B: 255, A: 255} // Blue
		s0.GlyphStyle.Radius = vg.Points(4)
		p3.Add(s0)
		p3.Legend.Add("Morning People", s0)
	}
	
	s1, err := plotter.NewScatter(nightOwlPts)
	if err == nil {
		s1.GlyphStyle.Color = color.RGBA{R: 255, G: 100, B: 0, A: 255} // Orange
		s1.GlyphStyle.Radius = vg.Points(4)
		p3.Add(s1)
		p3.Legend.Add("Night Owls", s1)
	}
	
	if err := p3.Save(8*vg.Inch, 6*vg.Inch, "sleep_vs_coffee_correlation.png"); err != nil {
		log.Printf("Warning: Could not save correlation chart: %v", err)
	} else {
		fmt.Println("Sleep vs coffee correlation chart saved to sleep_vs_coffee_correlation.png")
	}
	
	// Print descriptive statistics
	fmt.Println("\n📊 Sleep and Coffee Pattern Analysis:")
	
	// Calculate averages
	avgMorningSleep := calculateMean(morningHours)
	avgNightSleep := calculateMean(nightHours)
	avgMorningCaffeine := calculateMean(morningCaffeine)
	avgNightCaffeine := calculateMean(nightCaffeine)
	
	fmt.Printf("☀️  Morning People (n=%d):\n", len(morningHours))
	fmt.Printf("   Average Sleep: %.1f hours\n", avgMorningSleep)
	fmt.Printf("   Average Caffeine: %.0f mg\n", avgMorningCaffeine)
	
	fmt.Printf("🌙 Night Owls (n=%d):\n", len(nightHours))
	fmt.Printf("   Average Sleep: %.1f hours\n", avgNightSleep)
	fmt.Printf("   Average Caffeine: %.0f mg\n", avgNightCaffeine)
	
	fmt.Printf("\n🔍 Key Insights:\n")
	if avgMorningSleep > avgNightSleep {
		fmt.Printf("• Morning people sleep %.1f hours more on average\n", avgMorningSleep-avgNightSleep)
	} else {
		fmt.Printf("• Night owls sleep %.1f hours more on average\n", avgNightSleep-avgMorningSleep)
	}
	
	if avgMorningCaffeine > avgNightCaffeine {
		fmt.Printf("• Morning people consume %.0f mg more caffeine on average\n", avgMorningCaffeine-avgNightCaffeine)
	} else {
		fmt.Printf("• Night owls consume %.0f mg more caffeine on average\n", avgNightCaffeine-avgMorningCaffeine)
	}
}

// calculateMean calculates the mean of a slice of float64 values
func calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// performClassification performs classification on sleep data using KNN
func performClassification() {
	fmt.Println("=== CLASSIFICATION: Sleep Pattern Analysis (Custom KNN) ===")
	
	// Load sleep classification data
	csvData, err := loadCSV("../../datasets/sleep_classification.csv")
	if err != nil {
		log.Fatalf("Classification: Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records with features: %v\n", len(csvData.Rows), csvData.Headers)

	// Convert data for KNN
	var X [][]float64
	var y []float64
	
	for _, row := range csvData.Rows {
		if len(row) < 5 {
			continue
		}
		
		// Convert numeric features
		features := make([]float64, 4)
		var parseErr error
		for i := 0; i < 4; i++ {
			features[i], parseErr = strconv.ParseFloat(strings.TrimSpace(row[i]), 64)
			if parseErr != nil {
				break
			}
		}
		if parseErr != nil {
			continue
		}
		
		// Convert sleep_type to numeric: Morning Person = 0, Night Owl = 1
		sleepType := strings.TrimSpace(row[4])
		var label float64
		if sleepType == "Morning Person" {
			label = 0
		} else if sleepType == "Night Owl" {
			label = 1
		} else {
			continue
		}
		
		X = append(X, features)
		y = append(y, label)
	}

	// Split data into train/test (80/20)
	trainSize := int(0.8 * float64(len(X)))
	trainX := X[:trainSize]
	trainY := y[:trainSize]
	testX := X[trainSize:]
	testY := y[trainSize:]

	// Perform KNN classification with different K values
	fmt.Println("\nTesting different K values:")
	for k := 1; k <= 7; k += 2 {
		accuracy := performKNNClassification(trainX, testX, trainY, testY, k)
		fmt.Printf("K=%d: Accuracy = %.2f%%\n", k, accuracy*100)
	}

	// Use K=3 for final analysis with confusion matrix
	predictions, accuracy := performKNNClassificationWithPredictions(trainX, testX, trainY, testY, 3)
	fmt.Printf("\nFinal KNN (K=3) Accuracy: %.2f%%\n", accuracy*100)

	// Calculate and display confusion matrix
	cm := calculateConfusionMatrix(testY, predictions)
	printConfusionMatrix(cm)
	
	// Force output flush before plotting
	fmt.Println("🎨 Generating visualizations...")

	// Print prediction examples
	fmt.Println("\n🔍 Sample Predictions:")
	fmt.Println("Test Sample | Actual | Predicted | Correct?")
	fmt.Println("------------|--------|-----------|----------")
	for i := 0; i < min(10, len(testY)); i++ {
		actualLabel := "Morning"
		if testY[i] == 1 {
			actualLabel = "Night"
		}
		predictedLabel := "Morning"
		if predictions[i] == 1 {
			predictedLabel = "Night"
		}
		correct := "✅"
		if testY[i] != predictions[i] {
			correct = "❌"
		}
		fmt.Printf("    %2d      | %7s | %9s | %8s\n", i+1, actualLabel, predictedLabel, correct)
	}

	// Create sleep and coffee descriptive charts
	createSleepCoffeeCharts(X, y)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	fmt.Println("🧠 Sleep Pattern Classification Exercise")
	fmt.Println("========================================")
	fmt.Println("Using Custom KNN Implementation")
	fmt.Println()

	performClassification()

	fmt.Println("\n✅ Classification Exercise Complete!")
	fmt.Println("📊 Generated visualizations:")
	fmt.Println("  • sleep_hours_distribution.png - Sleep Hours Distribution by Type")
	fmt.Println("  • coffee_intake_distribution.png - Coffee Intake Distribution by Type")
	fmt.Println("  • sleep_vs_coffee_correlation.png - Sleep vs Coffee Correlation")
	fmt.Println()
	fmt.Println("📈 Analysis includes:")
	fmt.Println("  • Confusion Matrix with Precision, Recall, F1-Score")
	fmt.Println("  • Sleep and coffee pattern analysis")
	fmt.Println("  • Descriptive statistics and insights")
	fmt.Println()
	fmt.Println("💡 Algorithm: K-Nearest Neighbors (KNN)")
	fmt.Println("📈 Visualization: Gonum Plot")
} 