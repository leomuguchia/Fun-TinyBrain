package main

import (
	"encoding/json"
	"io/ioutil"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type Connection struct {
	Weight float64 `json:"weight"`
}
type Neuron struct {
	Connections []Connection `json:"connections"`
}
type Layer struct {
	Neurons []Neuron `json:"neurons"`
}
type Network struct {
	Layers []Layer `json:"layers"`
}

func main() {
	f, _ := os.Open("network_state.json")
	defer f.Close()
	data, _ := ioutil.ReadAll(f)
	var net Network
	json.Unmarshal(data, &net)

	// Example: plot weights of first neuron in first layer
	pts := make(plotter.XYs, len(net.Layers[0].Neurons[0].Connections))
	for i, c := range net.Layers[0].Neurons[0].Connections {
		pts[i].X = float64(i)
		pts[i].Y = c.Weight
	}

	p := plot.New()
	p.Title.Text = "Weights of first neuron"
	line, _ := plotter.NewLine(pts)
	p.Add(line)
	p.Save(4*vg.Inch, 4*vg.Inch, "weights.png")
}
