package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
)

func RunClassification() {
	fmt.Println("Running Classification Exercise...")

	// Load dataset
	rawData, err := base.ParseCSVToInstances("datasets/sleep_classification.csv", true)
	if err != nil {
		panic(err)
	}

	// Split dataset (70% train, 30% test)
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.7)

	// Initialize Random Forest classifier with 100 trees
	rf := ensemble.NewRandomForest(100, 3)

	// Train model
	rf.Fit(trainData)

	// Predict on test set
	predictions, err := rf.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Evaluate accuracy
	confMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}
	fmt.Println(evaluation.GetSummary(confMat))

	// Student task:
	// - Adjust number of trees (e.g. 50, 200)
	// - Remove or add features by filtering attributes before training
}
