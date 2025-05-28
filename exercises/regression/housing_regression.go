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

	"gonum.org/v1/gonum/mat" // Required for Gonum regression
	"gonum.org/v1/gonum/stat/regression"
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

// performLinearRegressionGonum uses Gonum for Simple Linear Regression and its R²
func performLinearRegressionGonum(X, y []float64) (slope, intercept, r2 float64) {
	if len(X) != len(y) {
		log.Println("Error: X and y must have the same length for regression.")
		return 0, 0, 0
	}
	if len(X) == 0 {
		log.Println("Error: Input data is empty for simple linear regression.")
		return 0, 0, 0
	}

	// For simple linear regression, X is an n-by-1 matrix.
	xMatrix := mat.NewDense(len(X), 1, X)

	// Use regression.Fit to get the Result struct, which includes R2.
	// The last argument 'true' indicates that an intercept should be fitted.
	result, err := regression.Fit(xMatrix, y, nil, true)
	if err != nil {
		log.Printf("Error fitting simple linear regression model with Gonum: %v\n", err)
		return 0, 0, 0
	}

	// Coefficients: result.Coefficients[0] is the intercept, result.Coefficients[1] is the slope.
	if len(result.Coefficients) < 2 {
		log.Println("Error: Not enough coefficients returned by Gonum regression for simple linear model.")
		return 0, 0, 0
	}
	intercept = result.Coefficients[0]
	slope = result.Coefficients[1]
	r2 = result.R2

	return slope, intercept, r2
}

// performMultipleLinearRegressionGonum uses Gonum for Multiple Linear Regression and its R²
func performMultipleLinearRegressionGonum(X [][]float64, y []float64) (coefficients []float64, r2 float64) {
	if len(X) == 0 || len(X[0]) == 0 {
		log.Println("Error: Input X data is empty for multiple regression.")
		return nil, 0
	}
	if len(X) != len(y) {
		log.Println("Error: Number of samples in X and y must match for multiple regression.")
		return nil, 0
	}

	rows := len(X)
	cols := len(X[0])

	// Convert X (slice of slices) to a Gonum Dense matrix.
	xData := make([]float64, rows*cols)
	for i, row := range X {
		for j, val := range row {
			xData[i*cols+j] = val
		}
	}
	xMatrix := mat.NewDense(rows, cols, xData)

	// Use regression.Fit to get the Result struct.
	// The last argument 'true' indicates that an intercept should be fitted.
	result, err := regression.Fit(xMatrix, y, nil, true)
	if err != nil {
		log.Printf("Error fitting multiple linear regression model with Gonum: %v\n", err)
		return nil, 0
	}

	coefficients = result.Coefficients
	r2 = result.R2

	return coefficients, r2
}

// calculateMeanAbsoluteError calculates MAE for model evaluation
func calculateMeanAbsoluteError(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) || len(actual) == 0 {
		log.Println("Warning: MAE calculation input length mismatch or empty.")
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
	if len(actual) != len(predicted) || len(actual) == 0 {
		log.Println("Warning: RMSE calculation input length mismatch or empty.")
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
	fmt.Println("=== REGRESSION: Housing Price Prediction (Gonum Linear Regression & R²) ===")

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
	if len(data) == 0 {
		log.Fatalf("No data to process after conversion.")
		return
	}

	// === SIMPLE LINEAR REGRESSION: Size vs Price ===
	fmt.Println("\n🏠 Simple Linear Regression: Size vs Price (using Gonum)")

	var sizeX, priceYSimple []float64
	for _, row := range data {
		if len(row) >= 6 {
			sizeX = append(sizeX, row[0])
			priceYSimple = append(priceYSimple, row[5])
		}
	}
	if len(sizeX) == 0 { // Check if data was actually populated
		log.Println("No data available for simple linear regression after filtering.")
		// Decide if you want to return or continue to multiple regression
	}

	slope, intercept, r2Simple := performLinearRegressionGonum(sizeX, priceYSimple)
	fmt.Printf("Simple Linear Regression Results (Gonum):\n")
	fmt.Printf("  R² (from Gonum) = %.4f\n", r2Simple)
	fmt.Printf("  Equation: price = %.2f * size + %.2f\n", slope, intercept)

	// === MULTIPLE LINEAR REGRESSION: All features vs Price ===
	fmt.Println("\n🏠 Multiple Linear Regression: All Features vs Price (using Gonum)")

	var multiX [][]float64
	var multiY []float64
	for _, row := range data {
		if len(row) >= 6 {
			features := row[:5]
			multiX = append(multiX, features)
			multiY = append(multiY, row[5])
		}
	}
	if len(multiX) == 0 { // Check if data was actually populated
		log.Println("No data available for multiple linear regression after filtering.")
		// Decide if you want to return or continue
	}

	coefficients, r2Multiple := performMultipleLinearRegressionGonum(multiX, multiY)
	if coefficients == nil {
		log.Println("Multiple linear regression failed to produce coefficients. Skipping further analysis for multiple regression.")
	} else {
		fmt.Printf("Multiple Linear Regression Results (Gonum):\n")
		fmt.Printf("  R² (from Gonum) = %.4f\n", r2Multiple)
		if len(coefficients) >= 6 { // Intercept + 5 features
			fmt.Printf("  Equation: price = %.2f + %.2f*size + %.2f*rooms + %.2f*floor + %.2f*year + %.2f*distance\n",
				coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4], coefficients[5])
		} else {
			fmt.Println("  Could not display full equation due to unexpected number of coefficients.")
			fmt.Printf("  Coefficients: %v\n", coefficients)
		}

		// Calculate predictions and error metrics for multiple regression
		var actualPrices, predictedPrices []float64
		for i, featuresRow := range multiX {
			if len(coefficients) > len(featuresRow) { // Check: num coeffs must be num features + 1 (for intercept)
				predicted := coefficients[0] // intercept
				for j, featureVal := range featuresRow {
					predicted += coefficients[j+1] * featureVal
				}
				actualPrices = append(actualPrices, multiY[i])
				predictedPrices = append(predictedPrices, predicted)
			} else {
				log.Printf("Skipping prediction for multiple regression row %d due to coefficient/feature mismatch.\n", i)
			}
		}

		if len(actualPrices) > 0 {
			mae := calculateMeanAbsoluteError(actualPrices, predictedPrices)
			rmse := calculateRootMeanSquaredError(actualPrices, predictedPrices)
			fmt.Printf("  Mean Absolute Error (MAE): €%.2f\n", mae)
			fmt.Printf("  Root Mean Squared Error (RMSE): €%.2f\n", rmse)
		} else {
			fmt.Println("  Could not calculate MAE/RMSE for multiple regression as no predictions were made or data was insufficient.")
		}
	}

	// === VISUALIZATION 1: Simple Linear Regression ===
	if len(sizeX) > 0 && len(priceYSimple) > 0 { // Ensure data exists for plotting
		p1 := plot.New()
		p1.Title.Text = "Housing Price Prediction: Size vs Price (Simple Linear Regression with Gonum)"
		p1.X.Label.Text = "Size (sqm)"
		p1.Y.Label.Text = "Price (EUR)"

		points := make(plotter.XYs, len(sizeX))
		for i := range sizeX {
			points[i] = plotter.XY{X: sizeX[i], Y: priceYSimple[i]}
		}

		s, err := plotter.NewScatter(points)
		if err != nil {
			log.Printf("Regression: Error creating scatter plot: %v\n", err)
		} else {
			s.GlyphStyle.Color = color.RGBA{B: 255, A: 255}
			s.GlyphStyle.Radius = vg.Points(2)
			p1.Add(s)
			p1.Legend.Add("Housing Data", s)
		}

		minXVal, maxXVal := minValue(sizeX), maxValue(sizeX) // Use helper for clarity if data is guaranteed non-empty
		if len(sizeX) > 0 {                                  // Ensure minXVal/maxXVal are valid before use
			linePoints := make(plotter.XYs, 2)
			linePoints[0] = plotter.XY{X: minXVal, Y: slope*minXVal + intercept}
			linePoints[1] = plotter.XY{X: maxXVal, Y: slope*maxXVal + intercept}

			l, err := plotter.NewLine(linePoints)
			if err != nil {
				log.Printf("Regression: Error creating line plot: %v\n", err)
			} else {
				l.LineStyle.Width = vg.Points(2)
				l.LineStyle.Color = color.RGBA{R: 255, A: 255}
				p1.Add(l)
				p1.Legend.Add("Linear Regression (Gonum)", l)
			}
		}

		if err := p1.Save(6*vg.Inch, 4*vg.Inch, "regression_housing_simple_gonum.png"); err != nil {
			log.Printf("Regression: Error saving plot: %v\n", err)
		} else {
			fmt.Println("Simple regression plot saved to regression_housing_simple_gonum.png")
		}
	}

	// === VISUALIZATION 2: Multiple Regression - Actual vs Predicted ===
	// This section depends on 'actualPrices' and 'predictedPrices' from the multiple regression block
	if coefficients != nil && len(actualPrices) > 0 && len(predictedPrices) > 0 {
		p2 := plot.New()
		p2.Title.Text = "Multiple Linear Regression: Actual vs Predicted Prices (Gonum)"
		p2.X.Label.Text = "Actual Price (EUR)"
		p2.Y.Label.Text = "Predicted Price (EUR)"

		actualVsPredicted := make(plotter.XYs, len(actualPrices))
		for i := range actualPrices {
			actualVsPredicted[i] = plotter.XY{X: actualPrices[i], Y: predictedPrices[i]}
		}

		s2, err := plotter.NewScatter(actualVsPredicted)
		if err != nil {
			log.Printf("Regression: Error creating second scatter plot: %v\n", err)
		} else {
			s2.GlyphStyle.Color = color.RGBA{G: 150, B: 255, A: 255}
			s2.GlyphStyle.Radius = vg.Points(3)
			p2.Add(s2)
			p2.Legend.Add("Predictions (Gonum)", s2)
		}

		minPrice := math.Min(minValue(actualPrices), minValue(predictedPrices))
		maxPrice := math.Max(maxValue(actualPrices), maxValue(predictedPrices))

		perfectLine := make(plotter.XYs, 2)
		perfectLine[0] = plotter.XY{X: minPrice, Y: minPrice}
		perfectLine[1] = plotter.XY{X: maxPrice, Y: maxPrice}

		l2, err := plotter.NewLine(perfectLine)
		if err != nil {
			log.Printf("Regression: Error creating perfect line: %v\n", err)
		} else {
			l2.LineStyle.Width = vg.Points(2)
			l2.LineStyle.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
			l2.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
			p2.Add(l2)
			p2.Legend.Add("Perfect Prediction", l2)
		}

		if err := p2.Save(6*vg.Inch, 4*vg.Inch, "regression_actual_vs_predicted_gonum.png"); err != nil {
			log.Printf("Regression: Error saving second plot: %v\n", err)
		} else {
			fmt.Println("Actual vs predicted plot saved to regression_actual_vs_predicted_gonum.png")
		}
	}

	// === FEATURE ANALYSIS ===
	fmt.Println("\n📊 Feature Analysis (Coefficients from Gonum):")
	featureNames := []string{"Size (sqm)", "Num Rooms", "Floor", "Year Built", "Distance to Center (km)"}

	if coefficients != nil && len(coefficients) > len(featureNames) { // intercept + features
		fmt.Printf("  Intercept: %.2f\n", coefficients[0])
		for i, name := range featureNames {
			var values []float64
			for _, row := range multiX { // multiX should be populated if coefficients are valid
				if i < len(row) {
					values = append(values, row[i])
				}
			}

			if len(values) > 0 {
				mean := 0.0
				for _, v := range values {
					mean += v
				}
				mean /= float64(len(values))
				minV, maxV := minValue(values), maxValue(values)
				fmt.Printf("  %s: Mean=%.2f, Range=[%.2f, %.2f], Coefficient=%.2f\n",
					name, mean, minV, maxV, coefficients[i+1])
			} else {
				// This case should ideally not happen if multiX was properly populated for regression
				fmt.Printf("  %s: No data for stats, Coefficient=%.2f\n", name, coefficients[i+1])
			}
		}
	} else {
		fmt.Println("  Could not perform detailed feature analysis: insufficient or nil coefficients.")
		if coefficients != nil {
			fmt.Printf("  Raw Coefficients: %v\n", coefficients)
		}
	}

	// === PRICE ANALYSIS ===
	fmt.Println("\n💰 Price Analysis:")
	if len(priceYSimple) > 0 { // Using priceYSimple as a proxy for available price data
		meanPrice := 0.0
		for _, price := range priceYSimple {
			meanPrice += price
		}
		meanPrice /= float64(len(priceYSimple))
		minP, maxP := minValue(priceYSimple), maxValue(priceYSimple)

		fmt.Printf("  Mean Price: €%.2f\n", meanPrice)
		fmt.Printf("  Price Range: €%.2f - €%.2f\n", minP, maxP)
		fmt.Printf("  Total Properties Analyzed (used for simple regression): %d\n", len(priceYSimple))
	} else {
		fmt.Println("  No price data available for analysis (based on simple regression data).")
	}
}

// Helper functions
func minValue(slice []float64) float64 {
	if len(slice) == 0 {
		log.Println("Warning: minValue called on empty slice.")
		return math.NaN() // Return NaN for empty slice to indicate undefined
	}
	min := slice[0]
	for _, v := range slice[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxValue(slice []float64) float64 {
	if len(slice) == 0 {
		log.Println("Warning: maxValue called on empty slice.")
		return math.NaN() // Return NaN for empty slice
	}
	max := slice[0]
	for _, v := range slice[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func main() {
	fmt.Println("🏠 Housing Price Regression Exercise")
	fmt.Println("===================================")
	fmt.Println("Using Gonum Library for Linear Regression and R²")
	fmt.Println()

	performRegression()

	fmt.Println("\n✅ Regression Exercise Complete!")
	fmt.Println("📊 Generated visualizations:")
	fmt.Println("  • regression_housing_simple_gonum.png - Simple Linear Regression (Size vs Price)")
	fmt.Println("  • regression_actual_vs_predicted_gonum.png - Multiple Regression (Actual vs Predicted)")
	fmt.Println()
	fmt.Println("💡 Algorithm: Linear Regression from Gonum Library")
	fmt.Println("🔧 Implementation: gonum.org/v1/gonum/stat/regression.Fit")
	fmt.Println("📈 Visualization: Gonum Plot")
}
