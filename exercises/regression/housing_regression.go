package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// XYData implements plotter.XYer interface for gonum plotting
type XYData struct {
	X []float64
	Y []float64
}

func (d XYData) Len() int {
	return len(d.X)
}

func (d XYData) XY(i int) (x, y float64) {
	return d.X[i], d.Y[i]
}

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
			continue // Skip empty rows
		}
		
		floatRow := make([]float64, 0, len(row))
		for _, val := range row {
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

// performRegression performs linear regression using gonum/stat
func performRegression() {
	fmt.Println("=== REGRESSION: Housing Price Prediction (gonum/stat) ===")
	
	// Load housing price data
	csvData, err := loadCSV("../../datasets/housing_prices.csv")
	if err != nil {
		log.Fatalf("Error loading CSV: %v", err)
	}

	fmt.Printf("Loaded %d records\n", len(csvData))

	// Convert to float64
	data, err := convertToFloat64(csvData)
	if err != nil {
		log.Fatalf("Error converting data: %v", err)
	}

	fmt.Printf("Processed %d valid records\n", len(data))

	// Extract size and price data
	var sizeX, priceY []float64
	for _, row := range data {
		if len(row) >= 6 {
			sizeX = append(sizeX, row[0]) // size_sqm
			priceY = append(priceY, row[5]) // price_eur
		}
	}

	// Perform linear regression using gonum/stat
	intercept, slope := stat.LinearRegression(sizeX, priceY, nil, false)
	
	// Calculate R-squared
	meanY := stat.Mean(priceY, nil)
	var ssTotal, ssRes float64
	
	for i := range sizeX {
		predicted := slope*sizeX[i] + intercept
		ssRes += (priceY[i] - predicted) * (priceY[i] - predicted)
		ssTotal += (priceY[i] - meanY) * (priceY[i] - meanY)
	}
	
	r2 := 1.0 - (ssRes / ssTotal)

	// Display results
	fmt.Printf("\n🏠 Linear Regression Results:\n")
	fmt.Printf("  R² = %.4f\n", r2)
	fmt.Printf("  Equation: price = %.2f * size + %.2f\n", slope, intercept)
	fmt.Printf("  Formula: %.3f*x + %.3f\n", slope, intercept)

	// Create visualization
	createPlot(sizeX, priceY, slope, intercept, r2)
}

// createPlot creates a plot with PNG output
func createPlot(X, y []float64, slope, intercept, r2 float64) {
	fmt.Println("\n📊 Creating Visualization...")
	
	// Create XYData for plotting
	data := XYData{X: X, Y: y}
	
	// Create plot
	p := plot.New()
	p.Title.Text = "Housing Price Prediction: Size vs Price"
	p.X.Label.Text = "Size (sqm)"
	p.Y.Label.Text = "Price (EUR)"
	
	// Set default styles
	plotter.DefaultLineStyle.Width = vg.Points(2)
	plotter.DefaultGlyphStyle.Radius = vg.Points(3)
	
	// Create scatter plot
	scatter, err := plotter.NewScatter(data)
	if err != nil {
		log.Printf("Error creating scatter plot: %v", err)
		return
	}
	scatter.GlyphStyle.Color = color.RGBA{R: 0, G: 100, B: 255, A: 200}
	
	// Create regression line function
	line := plotter.NewFunction(func(x float64) float64 { 
		return slope*x + intercept 
	})
	line.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	line.Width = vg.Points(2)
	
	// Add to plot
	p.Add(scatter, line)
	p.Legend.Add("Housing Data", scatter)
	p.Legend.Add(fmt.Sprintf("Linear Regression (R²=%.3f)", r2), line)
	
	// Save as PNG
	if err := p.Save(8*vg.Inch, 6*vg.Inch, "regression_simple.png"); err != nil {
		log.Printf("Error saving PNG: %v", err)
	} else {
		fmt.Printf("Plot saved as regression_simple.png\n")
	}
}

func main() {
	fmt.Println("🏠 Simple Housing Price Regression")
	fmt.Println("==================================")
	fmt.Println("Using gonum/stat.LinearRegression")
	fmt.Println()

	performRegression()

	fmt.Println("\n✅ Regression Exercise Complete!")
	fmt.Println("📊 Generated visualizations:")
	fmt.Println("  • regression_simple.png - Linear Regression Plot")
	fmt.Println()
	fmt.Println("💡 Algorithm: Linear Regression")
	fmt.Println("🔧 Implementation: gonum/stat.LinearRegression")
	fmt.Println("📈 Visualization: Gonum Plot")
} 