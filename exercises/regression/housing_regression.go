package regression

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type CSVColumn int

const (
	SizeSQM            CSVColumn = iota // 0
	NumRooms                            // 1
	Floor                               // 2
	YearBuilt                           // 3
	DistanceToCenterKM                  // 4
	PriceEUR                            // 5
)

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

func loadCSV(filename string) ([]string, [][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening file %s: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("error reading CSV: %v", err)
	}

	if len(records) == 0 {
		return nil, nil, fmt.Errorf("empty CSV file")
	}

	header := records[0]
	data := records[1:]
	return header, data, nil
}

func convertToFloat64(data [][]string) ([][]float64, error) {
	result := make([][]float64, 0, len(data))

	for rIdx, row := range data {
		if len(strings.TrimSpace(strings.Join(row, ""))) == 0 {
			continue
		}

		floatRow := make([]float64, 0, len(row))
		for cIdx, val := range row {
			trimmedVal := strings.TrimSpace(val)
			if trimmedVal == "" {
				floatRow = append(floatRow, 0.0)
				continue
			}

			f, err := strconv.ParseFloat(trimmedVal, 64)
			if err != nil {
				return nil, fmt.Errorf("error converting '%s' (row %d, col %d) to float64: %v", val, rIdx+1, cIdx+1, err)
			}
			floatRow = append(floatRow, f)
		}

		if len(floatRow) > 0 {
			result = append(result, floatRow)
		}
	}
	return result, nil
}

func performRegression(csvFilePath string, featureCol CSVColumn, targetCol CSVColumn) {
	featureIndex := int(featureCol)
	targetIndex := int(targetCol)

	fmt.Printf("\n=== REGRESSION ANALYSIS ===\n")

	header, csvData, err := loadCSV(csvFilePath)
	if err != nil {
		log.Printf("Error loading CSV: %v", err)
		return
	}

	if featureIndex >= len(header) || targetIndex >= len(header) {
		log.Printf("Error: Column index out of bounds")
		return
	}

	featureName := header[featureIndex]
	targetName := header[targetIndex]
	fmt.Printf("Feature: %s vs Target: %s\n", featureName, targetName)

	numericData, err := convertToFloat64(csvData)
	if err != nil {
		log.Printf("Error converting data: %v", err)
		return
	}

	var featureX, targetY []float64
	for _, row := range numericData {
		if len(row) > featureIndex && len(row) > targetIndex {
			featureX = append(featureX, row[featureIndex])
			targetY = append(targetY, row[targetIndex])
		}
	}

	if len(featureX) < 2 {
		log.Printf("Insufficient data points for regression")
		return
	}

	intercept, slope := stat.LinearRegression(featureX, targetY, nil, false)
	r2 := stat.RSquared(featureX, targetY, nil, intercept, slope)

	fmt.Printf("\nResults:\n")
	fmt.Printf("  R² = %.4f\n", r2)
	fmt.Printf("  Equation: %s = %.3f * %s + %.3f\n", targetName, slope, featureName, intercept)

	outputDir := "exercises/regression"
	os.MkdirAll(outputDir, os.ModePerm)

	safeFeatureName := strings.ToLower(strings.ReplaceAll(featureName, " ", "_"))
	safeTargetName := strings.ToLower(strings.ReplaceAll(targetName, " ", "_"))
	plotFileName := fmt.Sprintf("%s_vs_%s_regression.png", safeFeatureName, safeTargetName)
	fullPlotPath := filepath.Join(outputDir, plotFileName)

	createPlot(featureX, targetY, slope, intercept, r2, featureName, targetName, fullPlotPath)

	// Uncomment the lines below to make a prediction using the regression model
	// testValue := 75.0 // Example: input value for feature (e.g., size in sqm)
	// predicted := predict(testValue, slope, intercept)
	// fmt.Printf("\nPrediction:\n")
	// fmt.Printf("  Given %s = %.2f, predicted %s = %.2f\n", featureName, testValue, targetName, predicted)
}

func createPlot(X, Y []float64, slope, intercept, r2 float64, featureName string, targetName string, outputPath string) {
	fmt.Printf("Creating plot: %s\n", outputPath)

	pts := make(plotter.XYs, len(X))
	for i := range X {
		pts[i].X = X[i]
		pts[i].Y = Y[i]
	}

	p := plot.New()
	p.Title.Text = fmt.Sprintf("%s vs %s (R² = %.3f)", featureName, targetName, r2)
	p.X.Label.Text = featureName
	p.Y.Label.Text = targetName

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Printf("Error creating scatter plot: %v", err)
		return
	}
	scatter.GlyphStyle.Color = color.RGBA{B: 128, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(2.5)

	lineFunc := plotter.NewFunction(func(x float64) float64 {
		return slope*x + intercept
	})
	lineFunc.Color = color.RGBA{R: 255, A: 255}
	lineFunc.Width = vg.Points(1.5)

	p.Add(scatter, lineFunc)

	if err := p.Save(6*vg.Inch, 4.5*vg.Inch, outputPath); err != nil {
		log.Printf("Error saving plot: %v", err)
	} else {
		fmt.Printf("Plot saved: %s\n", outputPath)
	}
}

func predict(featureValue, slope, intercept float64) float64 {
	return slope*featureValue + intercept
}

func Run() {
	fmt.Println("Housing Data Regression Analysis")
	fmt.Println("================================")

	csvFilePath := "datasets/housing_prices.csv"

	// Dynamic example: Number of rooms vs Floor level
	var featureCol CSVColumn = SizeSQM
	var targetCol CSVColumn = PriceEUR

	performRegression(csvFilePath, featureCol, targetCol)

	fmt.Println("\nRegression analysis complete!")
	fmt.Println("Check 'exercises/regression/' for the generated plot.")
}
