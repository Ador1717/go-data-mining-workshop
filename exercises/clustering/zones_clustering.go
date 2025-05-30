package clustering

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/mpraski/clusters"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func loadCSV(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data [][]float64
	for _, row := range records[1:] { // skip header
		if len(row) < 4 {
			continue // skip incomplete rows
		}

		var point []float64
		for i := range 4 { // read all 4 columns, indexes 0 to 3
			val := strings.TrimSpace(row[i])
			if val == "" {
				return nil, fmt.Errorf("empty value in row: %v", row)
			}
			f, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("parse error in row %v: %w", row, err)
			}
			point = append(point, f)
		}
		data = append(data, point)
	}
	return data, nil
}


func createClusterPlot(assignments []int, data [][]float64, xIdx, yIdx int, features []string, filename string) error {
	p := plot.New()
	p.Title.Text = fmt.Sprintf("Clustering: %s vs %s", features[xIdx], features[yIdx])
	p.X.Label.Text = features[xIdx]
	p.Y.Label.Text = features[yIdx]

	colors := []color.RGBA{
		{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
		{255, 165, 0, 255}, {128, 0, 128, 255}, {128, 128, 128, 255},
	}

	clusterPoints := make(map[int]plotter.XYs)
	for i, clusterID := range assignments {
		coords := data[i]
		clusterPoints[clusterID] = append(clusterPoints[clusterID], plotter.XY{
			X: coords[xIdx],
			Y: coords[yIdx],
		})
	}

	for clusterID, pts := range clusterPoints {
		scatter, err := plotter.NewScatter(pts)
		if err != nil {
			return err
		}
		scatter.GlyphStyle.Color = colors[clusterID%len(colors)]
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Radius = vg.Points(3)

		p.Add(scatter)
		p.Legend.Add(fmt.Sprintf("Cluster %d", clusterID), scatter)
	}

	return p.Save(8*vg.Inch, 6*vg.Inch, filename)
}

// sanitizeFilename replaces or removes characters that are unsafe in filenames
func sanitizeFilename(name string) string {
	replacer := strings.NewReplacer(" ", "_", "(", "", ")", "", "/", "_", "%", "percent")
	return replacer.Replace(strings.ToLower(name))
}

func Run() {
	fmt.Println("Clustering Analysis")

	data, err := loadCSV("datasets/clustering.csv")
	if err != nil {
		log.Fatal("Error loading CSV:", err)
	}

	// KMeans: 1000 iterations, 3 clusters, using Euclidean distance
	c, err := clusters.KMeans(1000, 3, clusters.EuclideanDistance) //adjust number of clusters here
	if err != nil {
		log.Fatal("Failed to create KMeans clusterer:", err)
	}

	err = c.Learn(data)
	if err != nil {
		log.Fatal("KMeans Learn failed:", err)
	}

	assignments := c.Guesses()

	// Print cluster frequencies
	freq := make(map[int]int)
	for _, clusterID := range assignments {
		freq[clusterID]++
	}
	fmt.Println("Cluster frequencies:")
	for clusterID, count := range freq {
		fmt.Printf("Cluster %d: %d points\n", clusterID, count)
	}

	// Use only 2 selected features for plotting
	features := []string{
		"Monthly entertainment spending (euros)",
		"Gaming hours per week",
		"Weekly exercise hours",
		"Hours of sleep per night",
	}

	//Change these to select different features for the plot
	xIdx := 0 // entertainment spending (euros)
	yIdx := 1 // gaming hours per week

	filename := fmt.Sprintf("cluster_%s_vs_%s.png",
		sanitizeFilename(features[xIdx]),
		sanitizeFilename(features[yIdx]))

	err = createClusterPlot(assignments, data, xIdx, yIdx, features, filename)
	if err != nil {
		log.Printf("Warning: Could not create plot %s: %v", filename, err)
	}
}
