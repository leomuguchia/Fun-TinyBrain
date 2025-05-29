package main

import (
	"fmt"
	neuron "network/metal"
)

func main() {
	layer1 := neuron.NewLayer([]neuron.SpikingNeuron{
		{Weights: []float64{1.5, -1.0, 0.5}, Bias: 0.2, Threshold: 0.7, Decay: 0.9},
		{Weights: []float64{1.0, 0.8, -0.7}, Bias: 0.1, Threshold: 0.8, Decay: 0.85},
	})

	layer2 := neuron.NewLayer([]neuron.SpikingNeuron{
		{Weights: []float64{0.9, 1.2}, Bias: 0.15, Threshold: 0.6, Decay: 0.9},
	})

	network := neuron.NewNetwork([]*neuron.Layer{layer1, layer2})

	inputSequence := [][]float64{
		{0.5, 0.5, 0.5},
		{0.6, 0.7, 0.4},
		{1.0, 1.5, -0.5},
	}

	for t, inputs := range inputSequence {
		output := network.Forward(inputs)
		fmt.Printf("Time %d, Final Output: %v\n", t, output)
	}
}
