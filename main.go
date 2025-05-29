package main

import (
	"fmt"
	neuron "network/metal"
)

func main() {
	layer1 := neuron.NewLayer([]neuron.SpikingNeuron{
		{Weights: []float64{0.5, -0.6, 0.1}, Bias: 0.1, Threshold: 1.0, Decay: 0.9},
		{Weights: []float64{0.3, 0.2, -0.4}, Bias: 0.0, Threshold: 1.2, Decay: 0.85},
	})

	layer2 := neuron.NewLayer([]neuron.SpikingNeuron{
		{Weights: []float64{0.4, 0.6}, Bias: 0.2, Threshold: 0.8, Decay: 0.9},
	})

	network := neuron.NewNetwork([]*neuron.Layer{layer1, layer2})

	inputSequence := [][]float64{
		{0.1, 0.1, 0.1},
		{0.1, 0.1, 0.1},
		{1.0, 2.0, -1.0},
	}

	for t, inputs := range inputSequence {
		output := network.Forward(inputs)
		fmt.Printf("Time %d, Final Output: %v\n", t, output)
	}
}
