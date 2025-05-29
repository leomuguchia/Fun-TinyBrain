package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	neuron "network/metal"
	"os"
	"strconv"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	const numLayers = 8
	const neuronsPerLayer = 8
	const timeSteps = 30

	// Build an 8×8 layered network with randomized weights, biases, thresholds, decay
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
				weights[k] = 0.3 + 0.4*rand.Float64() // weights between 0.3 and 0.7
			}
			layer = append(layer, neuron.SpikingNeuron{
				Weights:   weights,
				Bias:      -0.1 + 0.3*rand.Float64(), // bias between -0.1 and 0.2
				Threshold: 1.2 + 0.5*rand.Float64(),  // threshold between 1.2 and 1.7
				Decay:     0.90 + 0.1*rand.Float64(), // decay between 0.9 and 1.0
			})
		}
		layers = append(layers, neuron.NewLayer(layer))
	}
	net := neuron.NewNetwork(layers)

	// CSV output setup
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

	// Run for multiple steps with time-varying input
	for t := 0; t < timeSteps; t++ {
		input := []float64{
			1.0 * (0.5 + 0.5*float64(t)/float64(timeSteps)), // ramping from 0.5 to 1.0
			0.8 * (1.0 - 0.5*float64(t)/float64(timeSteps)), // ramping down from 0.8 to 0.4
			0.5,                         // constant
			-0.4 * (float64(t%2)*2 - 1), // alternating sign every timestep
		}

		output := net.Forward(input)

		row := []string{strconv.Itoa(t)}
		for _, layer := range net.Layers() {
			for _, neuron := range layer.Neurons() {
				row = append(row, fmt.Sprintf("%.2f", neuron.MembranePotential))
			}
		}
		writer.Write(row)

		fmt.Printf("Step %d: Input: %v Final Output: %v\n", t, input, output)
	}
}
