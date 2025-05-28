package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "math"
    "math/rand"
    "os"
    "strconv"
    "time"
)

func euclidean(a, b []float64) float64 {
    sum := 0.0
    for i := range a {
        diff := a[i] - b[i]
        sum += diff * diff
    }
    return math.Sqrt(sum)
}

func assignToClusters(data [][]float64, centroids [][]float64) []int {
    assignments := make([]int, len(data))
    for i, point := range data {
        minDist := math.MaxFloat64
        minIndex := 0
        for j, centroid := range centroids {
            dist := euclidean(point, centroid)
            if dist < minDist {
                minDist = dist
                minIndex = j
            }
        }
        assignments[i] = minIndex
    }
    return assignments
}

func updateCentroids(data [][]float64, assignments []int, k int) [][]float64 {
    dim := len(data[0])
    centroids := make([][]float64, k)
    counts := make([]int, k)

    for i := range centroids {
        centroids[i] = make([]float64, dim)
    }

    for i, point := range data {
        cluster := assignments[i]
        for d := 0; d < dim; d++ {
            centroids[cluster][d] += point[d]
        }
        counts[cluster]++
    }

    for i := range centroids {
        for d := 0; d < dim; d++ {
            if counts[i] > 0 {
                centroids[i][d] /= float64(counts[i])
            }
        }
    }

    return centroids
}

func main() {
    file, err := os.Open("../../datasets/zones_clustering.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    reader.Read() // skip header

    var data [][]float64
    for {
        record, err := reader.Read()
        if err != nil {
            break
        }

        speed, _ := strconv.ParseFloat(record[1], 64)
        air, _ := strconv.ParseFloat(record[2], 64)
        noise, _ := strconv.ParseFloat(record[3], 64)

        data = append(data, []float64{speed, air, noise})
    }

    k := 3
    rand.Seed(time.Now().UnixNano())

    centroids := make([][]float64, k)
    for i := range centroids {
        centroids[i] = data[rand.Intn(len(data))]
    }

    var assignments []int
    for iter := 0; iter < 10; iter++ {
        assignments = assignToClusters(data, centroids)
        centroids = updateCentroids(data, assignments, k)
    }

    for i, cluster := range assignments {
        fmt.Printf("Zone %d → Cluster %d\n", i+1, cluster)
    }
}
