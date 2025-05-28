package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func main() {
	fmt.Println("🚀 Go Data Mining Workshop")
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
	fmt.Println("1. 🧠 CLASSIFICATION - Sleep Pattern Analysis")
	fmt.Println("   🔧 Algorithm: K-Nearest Neighbors (KNN)")
	fmt.Println("   📊 Dataset: sleep_classification.csv")
	fmt.Println("   🎯 Goal: Classify sleep patterns (Morning Person vs Night Owl)")
	fmt.Println()
	fmt.Println("2. 🏙️  CLUSTERING - Urban Zones Analysis")
	fmt.Println("   🔧 Algorithm: K-Means Clustering with Gonum")
	fmt.Println("   📊 Dataset: zones_clustering.csv")
	fmt.Println("   🎯 Goal: Group urban zones by characteristics")
	fmt.Println()
	fmt.Println("3. 🏠 REGRESSION - Housing Price Prediction")
	fmt.Println("   🔧 Algorithm: Simple & Multiple Linear Regression")
	fmt.Println("   📊 Dataset: housing_prices.csv")
	fmt.Println("   🎯 Goal: Predict housing prices based on features")
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
	fmt.Println("🚀 Starting exercise...")
	fmt.Println()
	
	var cmd *exec.Cmd
	var exerciseName string
	
	switch choice {
	case 1:
		exerciseName = "Classification"
		cmd = exec.Command("go", "run", "sleep_classification.go")
		cmd.Dir = "exercises/classification"
	case 2:
		exerciseName = "Clustering"
		cmd = exec.Command("go", "run", "zones_clustering.go")
		cmd.Dir = "exercises/clustering"
	case 3:
		exerciseName = "Regression"
		cmd = exec.Command("go", "run", "housing_regression.go")
		cmd.Dir = "exercises/regression"
	default:
		fmt.Println("❌ Invalid exercise choice")
		return
	}
	
	fmt.Printf("🔄 Running %s exercise...\n", exerciseName)
	fmt.Println("=" + strings.Repeat("=", len(exerciseName)+18))
	
	// Set up command to show output in real-time
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	
	err := cmd.Run()
	if err != nil {
		fmt.Printf("❌ Error running %s exercise: %v\n", exerciseName, err)
		fmt.Println("💡 Make sure you're in the correct directory and all dependencies are installed.")
		return
	}
	
	fmt.Println()
	fmt.Printf("✅ %s exercise completed successfully!\n", exerciseName)
}

func checkDatasets() {
	datasets := []string{
		"datasets/sleep_classification.csv",
		"datasets/zones_clustering.csv",
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