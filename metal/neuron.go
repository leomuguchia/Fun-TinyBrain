package neuron

type SpikingNeuron struct {
	Weights           []float64
	Bias              float64
	MembranePotential float64
	Threshold         float64
	Decay             float64
}

func (n *SpikingNeuron) Forward(inputs []float64) int {
	if len(inputs) != len(n.Weights) {
		panic("inputs and weights length mismatch")
	}

	n.MembranePotential *= n.Decay

	sum := 0.0
	for i := 0; i < len(inputs); i++ {
		sum += inputs[i] * n.Weights[i]
	}
	sum += n.Bias

	n.MembranePotential += sum

	if n.MembranePotential < 0 {
		n.MembranePotential = 0
	}

	if n.MembranePotential >= n.Threshold {
		n.MembranePotential = 0
		return 1
	}

	return 0
}
