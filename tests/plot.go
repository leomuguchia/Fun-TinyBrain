// File: tests/plot.go
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	file, err := os.Open("spike_records.csv")
	if err != nil {
		log.Fatalf("failed to open CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	if len(rows) < 2 {
		log.Fatalf("not enough data")
	}

	type Spike struct {
		Timestep int
		NeuronID int
	}

	var spikes []Spike

	// Parse membrane potentials and collect spike events
	for i := 1; i < len(rows); i++ {
		timestep, _ := strconv.Atoi(rows[i][0])
		for j := 1; j < len(rows[i]); j++ {
			val, _ := strconv.ParseFloat(rows[i][j], 64)
			if val == 0 {
				spikes = append(spikes, Spike{
					Timestep: timestep,
					NeuronID: j - 1,
				})
			}
		}
	}

	// Raster plot
	p := plot.New()
	p.Title.Text = "Raster Plot of Spikes"
	p.X.Label.Text = "Time Step"
	p.Y.Label.Text = "Neuron ID"

	pts := make(plotter.XYs, len(spikes))
	for i, s := range spikes {
		pts[i].X = float64(s.Timestep)
		pts[i].Y = float64(s.NeuronID)
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatalf("failed to create scatter: %v", err)
	}
	scatter.GlyphStyle.Radius = vg.Points(1.5)

	p.Add(scatter)

	if err := p.Save(10*vg.Inch, 6*vg.Inch, "raster_plot.png"); err != nil {
		log.Fatalf("failed to save plot: %v", err)
	}

	fmt.Println("✅ Raster plot saved as raster_plot.png")
}
