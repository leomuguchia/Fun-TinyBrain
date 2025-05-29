package neuron

// Spiking neural network
type Network struct {
	layers []*Layer
	time   int // Global time step
}

func NewNetwork(layers []*Layer) *Network {
	return &Network{layers: layers, time: 0}
}

func (n *Network) Forward(input []float64, currentTime int) []int {
	for _, layer := range n.layers {
		input = floatSlice(layer.Forward(input, currentTime))
	}
	return intSlice(input)
}

func floatSlice(inputs []int) []float64 {
	result := make([]float64, len(inputs))
	for i, v := range inputs {
		result[i] = float64(v)
	}
	return result
}

func intSlice(inputs []float64) []int {
	result := make([]int, len(inputs))
	for i, v := range inputs {
		if v >= 1 {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}
	return result
}
