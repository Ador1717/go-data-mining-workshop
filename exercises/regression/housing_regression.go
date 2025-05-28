package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
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

// Simple Linear Regression implementation
func performLinearRegression(X, y []float64) (slope, intercept, r2 float64) {
	n := float64(len(X))
	sumX, sumY, sumXY, sumXX := 0.0, 0.0, 0.0, 0.0
	
	for i := range X {
		sumX += X[i]
		sumY += y[i]
		sumXY += X[i] * y[i]
		sumXX += X[i] * X[i]
	}
	
	slope = (n*sumXY - sumX*sumY) / (n*sumXX - sumX*sumX)
	intercept = (sumY - slope*sumX) / n
	
	// Calculate R-squared manually
	meanY := sumY / n
	ssTotal := 0.0
	ssRes := 0.0
	
	for i := range X {
		predicted := slope*X[i] + intercept
		ssRes += (y[i] - predicted) * (y[i] - predicted)
		ssTotal += (y[i] - meanY) * (y[i] - meanY)
	}
	
	r2 = 1.0 - (ssRes / ssTotal)
	return slope, intercept, r2
}

// Multiple Linear Regression implementation
func performMultipleLinearRegression(X [][]float64, y []float64) (coefficients []float64, r2 float64) {
	if len(X) == 0 || len(X[0]) == 0 {
		return nil, 0
	}

	n := len(X)
	numFeatures := len(X[0]) + 1 // +1 for intercept

	// Create design matrix with intercept column
	designMatrix := make([][]float64, n)
	for i := range designMatrix {
		designMatrix[i] = make([]float64, numFeatures)
		designMatrix[i][0] = 1.0 // intercept term
		for j := 0; j < len(X[i]); j++ {
			designMatrix[i][j+1] = X[i][j]
		}
	}

	// Solve normal equations: (X^T * X) * β = X^T * y
	// First compute X^T * X
	XTX := make([][]float64, numFeatures)
	for i := range XTX {
		XTX[i] = make([]float64, numFeatures)
		for j := range XTX[i] {
			for k := 0; k < n; k++ {
				XTX[i][j] += designMatrix[k][i] * designMatrix[k][j]
			}
		}
	}

	// Compute X^T * y
	XTy := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		for k := 0; k < n; k++ {
			XTy[i] += designMatrix[k][i] * y[k]
		}
	}

	// Solve using Gaussian elimination (simplified for small matrices)
	coefficients = gaussianElimination(XTX, XTy)

	// Calculate R-squared
	meanY := 0.0
	for _, val := range y {
		meanY += val
	}
	meanY /= float64(len(y))

	ssTotal := 0.0
	ssRes := 0.0
	for i := 0; i < n; i++ {
		predicted := coefficients[0] // intercept
		for j := 0; j < len(X[i]); j++ {
			predicted += coefficients[j+1] * X[i][j]
		}
		ssRes += (y[i] - predicted) * (y[i] - predicted)
		ssTotal += (y[i] - meanY) * (y[i] - meanY)
	}

	r2 = 1.0 - (ssRes / ssTotal)
	return coefficients, r2
}

// Simple Gaussian elimination for solving linear systems
func gaussianElimination(A [][]float64, b []float64) []float64 {
	n := len(A)
	
	// Create augmented matrix
	augmented := make([][]float64, n)
	for i := range augmented {
		augmented[i] = make([]float64, n+1)
		copy(augmented[i][:n], A[i])
		augmented[i][n] = b[i]
	}

	// Forward elimination
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(augmented[k][i]) > math.Abs(augmented[maxRow][i]) {
				maxRow = k
			}
		}
		
		// Swap rows
		augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]

		// Make all rows below this one 0 in current column
		for k := i + 1; k < n; k++ {
			if augmented[i][i] != 0 {
				factor := augmented[k][i] / augmented[i][i]
				for j := i; j <= n; j++ {
					augmented[k][j] -= factor * augmented[i][j]
				}
			}
		}
	}

	// Back substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = augmented[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= augmented[i][j] * x[j]
		}
		if augmented[i][i] != 0 {
			x[i] /= augmented[i][i]
		}
	}

	return x
}

// calculateMeanAbsoluteError calculates MAE for model evaluation
func calculateMeanAbsoluteError(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		return 0
	}
	
	sum := 0.0
	for i := range actual {
		sum += math.Abs(actual[i] - predicted[i])
	}
	return sum / float64(len(actual))
}

// calculateRootMeanSquaredError calculates RMSE for model evaluation
func calculateRootMeanSquaredError(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		return 0
	}
	
	sum := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(actual)))
}

// performRegression performs regression on housing price data
func performRegression() {
	fmt.Println("=== REGRESSION: Housing Price Prediction (Custom Linear Regression) ===")
	
	// Load housing price data
	csvData, err := loadCSV("../../datasets/housing_prices.csv")
	if err != nil {
		log.Fatalf("Regression: Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records with features: %v\n", len(csvData.Rows), csvData.Headers)

	// Convert to float64
	data, err := convertToFloat64(csvData.Rows, []int{})
	if err != nil {
		log.Fatalf("Regression: Error converting data: %v", err)
	}

	fmt.Printf("Processed %d valid records\n", len(data))

	// === SIMPLE LINEAR REGRESSION: Size vs Price ===
	fmt.Println("\n🏠 Simple Linear Regression: Size vs Price")
	
	// Use size_sqm as feature and price_eur as target
	var sizeX, priceY []float64
	for _, row := range data {
		if len(row) >= 6 {
			sizeX = append(sizeX, row[0]) // size_sqm
			priceY = append(priceY, row[5]) // price_eur
		}
	}

	// Perform simple linear regression
	slope, intercept, r2Simple := performLinearRegression(sizeX, priceY)
	fmt.Printf("Simple Linear Regression Results:\n")
	fmt.Printf("  R² = %.4f\n", r2Simple)
	fmt.Printf("  Equation: price = %.2f * size + %.2f\n", slope, intercept)

	// === MULTIPLE LINEAR REGRESSION: All features vs Price ===
	fmt.Println("\n🏠 Multiple Linear Regression: All Features vs Price")
	
	// Use all features except price as predictors
	var multiX [][]float64
	var multiY []float64
	for _, row := range data {
		if len(row) >= 6 {
			features := row[:5] // size_sqm, num_rooms, floor, year_built, distance_to_center_km
			multiX = append(multiX, features)
			multiY = append(multiY, row[5]) // price_eur
		}
	}

	// Perform multiple linear regression
	coefficients, r2Multiple := performMultipleLinearRegression(multiX, multiY)
	fmt.Printf("Multiple Linear Regression Results:\n")
	fmt.Printf("  R² = %.4f\n", r2Multiple)
	fmt.Printf("  Equation: price = %.2f + %.2f*size + %.2f*rooms + %.2f*floor + %.2f*year + %.2f*distance\n",
		coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5])

	// Calculate predictions and error metrics for multiple regression
	var actualPrices, predictedPrices []float64
	for i, features := range multiX {
		predicted := coefficients[0] // intercept
		for j, feature := range features {
			predicted += coefficients[j+1] * feature
		}
		actualPrices = append(actualPrices, multiY[i])
		predictedPrices = append(predictedPrices, predicted)
	}

	mae := calculateMeanAbsoluteError(actualPrices, predictedPrices)
	rmse := calculateRootMeanSquaredError(actualPrices, predictedPrices)
	fmt.Printf("  Mean Absolute Error (MAE): €%.2f\n", mae)
	fmt.Printf("  Root Mean Squared Error (RMSE): €%.2f\n", rmse)

	// === VISUALIZATION 1: Simple Linear Regression ===
	p1 := plot.New()
	p1.Title.Text = "Housing Price Prediction: Size vs Price (Simple Linear Regression)"
	p1.X.Label.Text = "Size (sqm)"
	p1.Y.Label.Text = "Price (EUR)"

	// Scatter plot of original data
	points := make(plotter.XYs, len(sizeX))
	for i := range sizeX {
		points[i] = plotter.XY{X: sizeX[i], Y: priceY[i]}
	}

	s, err := plotter.NewScatter(points)
	if err != nil {
		log.Fatalf("Regression: Error creating scatter plot: %v", err)
	}
	s.GlyphStyle.Color = color.RGBA{B: 255, A: 255}
	s.GlyphStyle.Radius = vg.Points(2)
	p1.Add(s)

	// Create regression line
	minX, maxX := sizeX[0], sizeX[0]
	for _, val := range sizeX {
		if val < minX {
			minX = val
		}
		if val > maxX {
			maxX = val
		}
	}

	linePoints := make(plotter.XYs, 100)
	for i := 0; i < 100; i++ {
		xVal := minX + (maxX-minX)*float64(i)/99.0
		yVal := slope*xVal + intercept
		linePoints[i] = plotter.XY{X: xVal, Y: yVal}
	}

	l, err := plotter.NewLine(linePoints)
	if err != nil {
		log.Fatalf("Regression: Error creating line plot: %v", err)
	}
	l.LineStyle.Width = vg.Points(2)
	l.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p1.Add(l)

	p1.Legend.Add("Housing Data", s)
	p1.Legend.Add("Linear Regression", l)

	if err := p1.Save(6*vg.Inch, 4*vg.Inch, "regression_housing_simple.png"); err != nil {
		log.Fatalf("Regression: Error saving plot: %v", err)
	}
	fmt.Println("Simple regression plot saved to regression_housing_simple.png")

	// === VISUALIZATION 2: Multiple Regression - Actual vs Predicted ===
	p2 := plot.New()
	p2.Title.Text = "Multiple Linear Regression: Actual vs Predicted Prices"
	p2.X.Label.Text = "Actual Price (EUR)"
	p2.Y.Label.Text = "Predicted Price (EUR)"

	// Scatter plot of actual vs predicted
	actualVsPredicted := make(plotter.XYs, len(actualPrices))
	for i := range actualPrices {
		actualVsPredicted[i] = plotter.XY{X: actualPrices[i], Y: predictedPrices[i]}
	}

	s2, err := plotter.NewScatter(actualVsPredicted)
	if err != nil {
		log.Fatalf("Regression: Error creating second scatter plot: %v", err)
	}
	s2.GlyphStyle.Color = color.RGBA{G: 150, B: 255, A: 255}
	s2.GlyphStyle.Radius = vg.Points(3)
	p2.Add(s2)

	// Add perfect prediction line (y = x)
	minPrice := math.Min(minValue(actualPrices), minValue(predictedPrices))
	maxPrice := math.Max(maxValue(actualPrices), maxValue(predictedPrices))
	
	perfectLine := make(plotter.XYs, 2)
	perfectLine[0] = plotter.XY{X: minPrice, Y: minPrice}
	perfectLine[1] = plotter.XY{X: maxPrice, Y: maxPrice}

	l2, err := plotter.NewLine(perfectLine)
	if err != nil {
		log.Fatalf("Regression: Error creating perfect line: %v", err)
	}
	l2.LineStyle.Width = vg.Points(2)
	l2.LineStyle.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	l2.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	p2.Add(l2)

	p2.Legend.Add("Predictions", s2)
	p2.Legend.Add("Perfect Prediction", l2)

	if err := p2.Save(6*vg.Inch, 4*vg.Inch, "regression_actual_vs_predicted.png"); err != nil {
		log.Fatalf("Regression: Error saving second plot: %v", err)
	}
	fmt.Println("Actual vs predicted plot saved to regression_actual_vs_predicted.png")

	// === FEATURE ANALYSIS ===
	fmt.Println("\n📊 Feature Analysis:")
	featureNames := []string{"Size (sqm)", "Num Rooms", "Floor", "Year Built", "Distance to Center (km)"}
	
	for i, name := range featureNames {
		var values []float64
		for _, row := range multiX {
			values = append(values, row[i])
		}
		
		mean := 0.0
		for _, v := range values {
			mean += v
		}
		mean /= float64(len(values))
		
		min, max := values[0], values[0]
		for _, v := range values {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		
		fmt.Printf("  %s: Mean=%.2f, Range=[%.2f, %.2f], Coefficient=%.2f\n", 
			name, mean, min, max, coefficients[i+1])
	}

	// === PRICE ANALYSIS ===
	fmt.Println("\n💰 Price Analysis:")
	meanPrice := 0.0
	for _, price := range priceY {
		meanPrice += price
	}
	meanPrice /= float64(len(priceY))
	
	minPrice = priceY[0]
	maxPrice = priceY[0]
	for _, price := range priceY {
		if price < minPrice {
			minPrice = price
		}
		if price > maxPrice {
			maxPrice = price
		}
	}
	
	fmt.Printf("  Mean Price: €%.2f\n", meanPrice)
	fmt.Printf("  Price Range: €%.2f - €%.2f\n", minPrice, maxPrice)
	fmt.Printf("  Total Properties Analyzed: %d\n", len(priceY))
}

// Helper functions
func minValue(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	min := slice[0]
	for _, v := range slice {
		if v < min {
			min = v
		}
	}
	return min
}

func maxValue(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice {
		if v > max {
			max = v
		}
	}
	return max
}

func main() {
	fmt.Println("🏠 Housing Price Regression Exercise")
	fmt.Println("===================================")
	fmt.Println("Using Custom Linear Regression Implementation")
	fmt.Println()

	performRegression()

	fmt.Println("\n✅ Regression Exercise Complete!")
	fmt.Println("📊 Generated visualizations:")
	fmt.Println("  • regression_housing_simple.png - Simple Linear Regression (Size vs Price)")
	fmt.Println("  • regression_actual_vs_predicted.png - Multiple Regression (Actual vs Predicted)")
	fmt.Println()
	fmt.Println("💡 Algorithms: Simple & Multiple Linear Regression")
	fmt.Println("🔧 Implementation: Custom least squares with Gaussian elimination")
	fmt.Println("📈 Visualization: Gonum Plot")
} 