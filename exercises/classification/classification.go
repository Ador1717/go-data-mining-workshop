package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "os"
)

func main() {
    file, err := os.Open("../../datasets/sleep_classification.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    reader.Read() // skip header

    var good, poor int
    for {
        record, err := reader.Read()
        if err != nil {
            break
        }

        // Sleep type is in column 4
        sleepType := record[4]
        if sleepType == "Morning Person" {
            good++
        } else if sleepType == "Night Owl" {
            poor++
        }
    }

    fmt.Printf("Morning Person (Good Sleep): %d\n", good)
    fmt.Printf("Night Owl (Poor Sleep): %d\n", poor)
}
