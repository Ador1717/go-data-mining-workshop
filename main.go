package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"

    "go-datamining-workshop/exercises"
)

func main() {
    reader := bufio.NewReader(os.Stdin)

    fmt.Println("Go Data Mining Workshop")
    fmt.Println("Choose an exercise to run:")
    fmt.Println("1 - Regression (Housing Prices)")
    fmt.Println("2 - Classification (Sleep Type)")
    fmt.Println("3 - Clustering (City Zones)")
    fmt.Print("Enter choice: ")

    choice, _ := reader.ReadString('\n')
    choice = strings.TrimSpace(choice)

    switch choice {
    case "1":
        exercises.RunRegression()
    case "2":
        exercises.RunClassification()
    case "3":
        exercises.RunClustering()
    default:
        fmt.Println("Invalid choice")
    }
}
