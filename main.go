package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	neuron "tinybrain/metal"
)

// createNewNetwork builds a fresh network (your existing network setup code)
func createNewNetwork() *neuron.Network {
	const numLayers = 8
	const neuronsPerLayer = 8

	layers := []*neuron.Layer{}
	for i := 0; i < numLayers; i++ {
		layer := []neuron.SpikingNeuron{}
		inputSize := neuronsPerLayer
		if i == 0 {
			inputSize = 8 // input layer has 8 inputs
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
	return neuron.NewNetwork(layers)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	const timeSteps = 100
	const patternSwitchInterval = 50 // every 10 steps, switch pattern
	const numLayers = 8
	const neuronsPerLayer = 8

	// Try to load existing network state
	net := &neuron.Network{}
	err := net.Load("network_state.json")
	if err != nil {
		fmt.Println("No saved network, creating new one")
		net = createNewNetwork()
	} else {
		fmt.Println("Loaded network from network_state.json")
	}

	// Setup CSV file for spike recording
	file, err := os.Create("tests/spike_records_patterns.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// CSV Headers
	headers := []string{"t", "pattern"}
	for l := 0; l < numLayers; l++ {
		for n := 0; n < neuronsPerLayer; n++ {
			headers = append(headers, fmt.Sprintf("L%d_N%d", l, n))
		}
	}
	writer.Write(headers)

	// Define input patterns
	patternA := []float64{1, 1, 1, 1, 0, 0, 0, 0} // Left side lit
	patternB := []float64{1, 0, 1, 0, 1, 0, 1, 0} // Alternating

	for t := 0; t < timeSteps; t++ {
		var input []float64
		patternLabel := "A"
		if (t/patternSwitchInterval)%2 == 0 {
			input = patternA
		} else {
			input = patternB
			patternLabel = "B"
		}

		output := net.Forward(input, t)

		row := []string{strconv.Itoa(t), patternLabel}
		for _, layer := range net.Layers {
			for _, neuron := range layer.Neurons {
				row = append(row, fmt.Sprintf("%.2f", neuron.MembranePotential))
			}
		}
		writer.Write(row)

		fmt.Printf("Step %d [Pattern %s]: Output: %v\n", t, patternLabel, output)
		// Example: track weight of first neuron in first layer from first input
		if t%10 == 0 {
			trackedWeight := net.Layers[0].Neurons[0].Connections[0].Weight
			fmt.Printf("  ↳ Tracked Weight L0.N0.C0: %.4f\n", trackedWeight)
		}

	}

	// Save network state after training
	if err := net.Save("network_state.json"); err != nil {
		fmt.Println("Error saving network:", err)
	} else {
		fmt.Println("Network state saved to network_state.json")
	}

	fmt.Println("Done. Check tests/spike_records_patterns.csv for potential learning traces.")
}
