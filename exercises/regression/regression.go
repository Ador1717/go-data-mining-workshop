package main

import (
    "fmt"
    "github.com/sjwhitworth/golearn/base"
    "github.com/sjwhitworth/golearn/evaluation"
    "github.com/sjwhitworth/golearn/linear_models"
    "github.com/sjwhitworth/golearn/filters"
    "math/rand"
)

func main() {
    // Load dataset
    rawData, err := base.ParseCSVToInstances("data/smartcity_housing_prices.csv", true)
    if err != nil {
        panic(err)
    }

    // Select which features to use: Adjust here by filtering attributes
    // Example: keep all features except "floor" or "year_built"
    // Uncomment this block to remove 'floor'
    /*
    filter := filters.NewAttributeFilter()
    filter.AddAttribute("floor")
    filter.Train(rawData)
    rawData = filter.Filter(rawData)
    */

    // Shuffle and split dataset into train/test (80/20)
    trainData, testData := base.InstancesTrainTestSplit(rawData, 0.8)

    // Initialize Linear Regression model
    lr := linear_models.NewLinearRegression()

    // Train model
    lr.Fit(trainData)

    // Predict on test data
    predictions, err := lr.Predict(testData)
    if err != nil {
        panic(err)
    }

    // Evaluate MSE (Mean Squared Error)
    mse := evaluation.GetRegressionMeanSquaredError(testData, predictions)
    fmt.Printf("Mean Squared Error: %.2f\n", mse)

    // Optional: Print predicted vs actual price for first 10 test samples
    fmt.Println("Predicted vs Actual (first 10 samples):")
    for i := 0; i < 10 && i < testData.Rows; i++ {
        predicted := predictions.RowString(i)
        actual := testData.Get(testData.GetClassAttribute(), i)
        fmt.Printf("Predicted: %s, Actual: %s\n", predicted, actual)
    }
}
