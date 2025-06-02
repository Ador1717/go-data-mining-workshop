package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Ador1717/go-data-mining-workshop/exercises/classification"
	"github.com/Ador1717/go-data-mining-workshop/exercises/clustering"
	"github.com/Ador1717/go-data-mining-workshop/exercises/regression"
)

func main() {
	fmt.Println("Go Data Mining Workshop")
	fmt.Println("==========================")
	fmt.Println("Machine Learning Exercises with Custom Implementations")
	fmt.Println()

	// Check if datasets exist first
	checkDatasets()

	for {
		showMenu()
		choice := getUserChoice()

		if choice == 0 {
			fmt.Println("👋 Thanks for using the Go Data Mining Workshop! Goodbye!")
			break
		}

		runExercise(choice)

		fmt.Println()
		fmt.Print("Press Enter to return to menu...")
		bufio.NewReader(os.Stdin).ReadBytes('\n')
		fmt.Println()
	}
}

func showMenu() {
	fmt.Println("📚 Available Exercises:")
	fmt.Println()
	fmt.Println("1. REGRESSION - Housing Price Prediction")
	fmt.Println("   🔧 Algorithm: Linear Regression")
	fmt.Println("   📊 Dataset: housing_prices.csv")
	fmt.Println("   🎯 Goal: Predict housing prices based on features")
	fmt.Println()
	fmt.Println("2. CLASSIFICATION - Sleep Pattern Analysis")
	fmt.Println("   🔧 Algorithm: K-Nearest Neighbors (KNN)")
	fmt.Println("   📊 Dataset: sleep_classification.csv")
	fmt.Println("   🎯 Goal: Classify sleep patterns (Morning Person vs Night Owl)")
	fmt.Println()
	fmt.Println("3. CLUSTERING - Lifestyle Analysis")
	fmt.Println("   🔧 Algorithm: K-Means Clustering")
	fmt.Println("   📊 Dataset: clustering.csv")
	fmt.Println("   🎯 Goal: Group by lifestyle characteristics")
	fmt.Println()
	fmt.Println("0. 🚪 Exit")
	fmt.Println()
}

func getUserChoice() int {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("Enter your choice (0-3): ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("❌ Error reading input. Please try again.")
			continue
		}

		input = strings.TrimSpace(input)
		choice, err := strconv.Atoi(input)
		if err != nil || choice < 0 || choice > 3 {
			fmt.Println("❌ Invalid choice. Please enter a number between 0 and 3.")
			continue
		}

		return choice
	}
}

func runExercise(choice int) {
	fmt.Println()
	fmt.Println("Starting exercise...")
	fmt.Println()

	switch choice {
	case 1:
		regression.Run()
	case 2:
		classification.Run()
	case 3:
		clustering.Run()
	default:
		fmt.Println("❌ Invalid exercise choice")
		return
	}
}

func checkDatasets() {
	datasets := []string{
		"datasets/sleep_classification.csv",
		"datasets/clustering.csv",
		"datasets/housing_prices.csv",
	}

	fmt.Println("🔍 Dataset Status Check:")
	allExist := true
	for _, dataset := range datasets {
		if _, err := os.Stat(dataset); err == nil {
			fmt.Printf("   ✅ %s - Found\n", dataset)
		} else {
			fmt.Printf("   ❌ %s - Missing\n", dataset)
			allExist = false
		}
	}

	if !allExist {
		fmt.Println()
		fmt.Println("⚠️  Warning: Some datasets are missing. Please ensure all datasets are in the 'datasets/' directory.")
	}

	fmt.Println()
	fmt.Println("🎓 Learning Objectives:")
	fmt.Println("   • Understand different ML algorithm types")
	fmt.Println("   • Implement algorithms from scratch")
	fmt.Println("   • Work with real-world data")
	fmt.Println("   • Create meaningful visualizations")
	fmt.Println("   • Evaluate model performance")
	fmt.Println()
}
