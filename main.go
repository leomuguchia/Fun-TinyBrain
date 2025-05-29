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

	const numLayers = 88
	const neuronsPerLayer = 88
	const timeSteps = 300

	// Build an 8×8 layered network with randomized weights, biases, thresholds, decay
	layers := []*neuron.Layer{}
	for i := 0; i < numLayers; i++ {
		layer := []neuron.SpikingNeuron{}
		inputSize := neuronsPerLayer
		if i == 0 {
			inputSize = 4 // input layer has fixed size inputs
		}
		for j := 0; j < neuronsPerLayer; j++ {
			connections := make([]neuron.Connection, inputSize)
			for k := range connections {
				connections[k] = neuron.Connection{
					Weight: rand.Float64()*2 - 1,
				}
			}
			layer = append(layer, neuron.SpikingNeuron{
				Connections:      connections,
				Bias:             rand.NormFloat64() * 0.5,
				Threshold:        1.0 + rand.Float64(),
				Decay:            0.8 + 0.2*rand.Float64(),
				RefractoryPeriod: rand.Intn(4) + 1,
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

		output := net.Forward(input, t)

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
