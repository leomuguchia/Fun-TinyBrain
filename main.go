package main

import (
	"encoding/csv"
	"fmt"
	neuron "network/metal"
	"os"
	"strconv"
)

func main() {
	const numLayers = 8
	const neuronsPerLayer = 8
	const timeSteps = 30

	// Build an 8×8 layered network
	layers := []*neuron.Layer{}
	for i := 0; i < numLayers; i++ {
		layer := []neuron.SpikingNeuron{}
		inputSize := neuronsPerLayer
		if i == 0 {
			inputSize = 4 // input layer has fixed size inputs
		}
		for j := 0; j < neuronsPerLayer; j++ {
			weights := make([]float64, inputSize)
			for k := range weights {
				weights[k] = 0.5
			}
			layer = append(layer, neuron.SpikingNeuron{
				Weights:   weights,
				Bias:      0.1,
				Threshold: 1.0,
				Decay:     0.95,
			})
		}
		layers = append(layers, neuron.NewLayer(layer))
	}
	net := neuron.NewNetwork(layers)

	// Use same input every timestep
	input := []float64{1.0, 0.8, 0.5, -0.4}

	// CSV output
	file, err := os.Create("spike_records.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write CSV headers
	headers := []string{"t"}
	for l := 0; l < numLayers; l++ {
		for n := 0; n < neuronsPerLayer; n++ {
			headers = append(headers, fmt.Sprintf("L%d_N%d", l, n))
		}
	}
	writer.Write(headers)

	// Run for multiple steps
	for t := 0; t < timeSteps; t++ {
		output := net.Forward(input)
		row := []string{strconv.Itoa(t)}
		for _, layer := range net.Layers() {
			for _, neuron := range layer.Neurons() {
				row = append(row, fmt.Sprintf("%.2f", neuron.MembranePotential))
			}
		}
		writer.Write(row)
		fmt.Printf("Step %d: Final Output: %v\n", t, output)
	}
}
