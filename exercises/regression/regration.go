package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "os"
    "strconv"

    "gonum.org/v1/gonum/stat"
)

func main() {
    file, err := os.Open("../../datasets/housing_prices.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    reader.Read() // skip header

    var sizeSqm, priceEur []float64
    for {
        record, err := reader.Read()
        if err != nil {
            break
        }

        size, _ := strconv.ParseFloat(record[0], 64)  // size_sqm
        price, _ := strconv.ParseFloat(record[5], 64) // price_eur

        sizeSqm = append(sizeSqm, size)
        priceEur = append(priceEur, price)
    }

    alpha, beta := stat.LinearRegression(sizeSqm, priceEur, nil, false)
    fmt.Printf("Model: price = %.2f + %.2f * size_sqm\n", alpha, beta)

    testSize := 85.0
    predicted := alpha + beta*testSize
    fmt.Printf("Predicted price for %.1f m² = €%.2f\n", testSize, predicted)
}
