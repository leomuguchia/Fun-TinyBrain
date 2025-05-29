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
		// In createNewNetwork()
		for j := 0; j < neuronsPerLayer; j++ {
			connections := make([]neuron.Connection, inputSize)
			for k := range connections {
				connections[k] = neuron.Connection{
					Weight: 0.1 + rand.Float64()*0.4, // Random weights 0.1-0.5
				}
			}
			layer = append(layer, neuron.SpikingNeuron{
				Connections:       connections,
				Bias:              -0.3 + rand.Float64()*0.6, // Wider bias range
				Threshold:         1.0 + rand.Float64()*0.5,  // Slightly higher thresholds
				Decay:             0.7 + rand.Float64()*0.2,  // Faster decay
				RefractoryPeriod:  2 + rand.Intn(3),          // Longer refractory
				MinWeight:         -1.5,                      // Expanded weight range
				MaxWeight:         1.5,
				AdaptiveThreshold: 0, // Initialize to 0
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

	// Stronger, more distinct patterns
	patternA := []float64{1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1} // Left side active
	patternB := []float64{0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0} // Alternating strong/weak

	for t := 0; t < timeSteps; t++ {
		var input []float64
		patternLabel := "A"
		if (t/patternSwitchInterval)%2 == 0 {
			input = patternA
		} else {
			input = patternB
			patternLabel = "B"
		}

		learningRate := 0.05
		output := net.Forward(input, t, learningRate)

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

func clipWeights(n *neuron.SpikingNeuron, min, max float64) {
	for i := range n.Connections {
		if n.Connections[i].Weight < min {
			n.Connections[i].Weight = min
		}
		if n.Connections[i].Weight > max {
			n.Connections[i].Weight = max
		}
	}
}
