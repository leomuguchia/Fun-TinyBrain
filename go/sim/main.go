// main.go
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

func createNewNetwork() *neuron.Network {
	const numLayers = 3 // Simpler hierarchy
	const neuronsPerLayer = 32

	layers := []*neuron.Layer{}
	for i := 0; i < numLayers; i++ {
		neurons := []neuron.SpikingNeuron{}
		inputSize := neuronsPerLayer
		if i == 0 {
			inputSize = 8 // Input layer
		}

		for j := 0; j < neuronsPerLayer; j++ {
			connections := make([]neuron.Connection, inputSize)
			for k := range connections {
				connections[k] = neuron.Connection{
					Weight: 0.5 + rand.Float64()*0.5, // 0.5-1.0
				}
			}

			neurons = append(neurons, neuron.SpikingNeuron{
				Connections:      connections,
				Threshold:        1.0,
				Decay:            0.95,
				Bias:             0.1,
				RefractoryPeriod: 1,
				MinWeight:        -1.0,
				MaxWeight:        2.0,
			})
		}
		layers = append(layers, neuron.NewLayer(neurons))
	}
	return neuron.NewNetwork(layers)
}

// Generate spike train for pattern
func generateSpikeTrain(pattern []float64, timesteps int) [][]float64 {
	spikeTrain := make([][]float64, timesteps)
	for t := range spikeTrain {
		spikes := make([]float64, len(pattern))
		for i, rate := range pattern {
			if rand.Float64() < rate {
				spikes[i] = 1.0
			}
		}
		spikeTrain[t] = spikes
	}
	return spikeTrain
}

func main() {
	rand.Seed(time.Now().UnixNano())

	const timesteps = 500
	const patternDuration = 50
	const numPatterns = 2

	net := createNewNetwork()

	// Create CSV file
	file, err := os.Create("spike_records.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Headers
	headers := []string{"timestep", "pattern"}
	for l := range net.Layers {
		for n := range net.Layers[l].Neurons {
			headers = append(headers, fmt.Sprintf("L%d_N%d", l, n))
		}
	}
	writer.Write(headers)

	// Spike train patterns
	patterns := [][][]float64{
		generateSpikeTrain([]float64{0.8, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.8}, patternDuration), // Pattern A
		generateSpikeTrain([]float64{0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8, 0.1}, patternDuration), // Pattern B
	}

	for t := 0; t < timesteps; t++ {
		patternIdx := (t / patternDuration) % numPatterns
		pattern := patterns[patternIdx][t%patternDuration]

		// Forward pass
		output := net.Forward(pattern, t)

		// Update weights with STDP
		net.UpdateWeights(t, 0.01)

		// Record data
		record := []string{strconv.Itoa(t), strconv.Itoa(patternIdx)}
		for _, layer := range net.Layers {
			for _, neuron := range layer.Neurons {
				record = append(record, fmt.Sprintf("%.2f", neuron.MembranePotential))
			}
		}
		writer.Write(record)

		// Print status
		if t%50 == 0 {
			fmt.Printf("Timestep %d, Pattern %d\n", t, patternIdx)
			fmt.Printf("Output spikes: %v\n", output)
		}
	}

	fmt.Println("SNN simulation complete. Data saved to spike_records.csv")
}
