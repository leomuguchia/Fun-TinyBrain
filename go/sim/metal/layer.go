// layer.go
package neuron

type Layer struct {
	Neurons []SpikingNeuron `json:"neurons"`
}

func NewLayer(neurons []SpikingNeuron) *Layer {
	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(inputs []float64, currentTime int) []float64 {
	outputs := make([]float64, len(l.Neurons))

	// First process all neurons
	for i := range l.Neurons {
		l.Neurons[i].Forward(inputs, currentTime)
	}

	// Apply lateral inhibition
	maxPotential := 0.0
	for _, neuron := range l.Neurons {
		if neuron.MembranePotential > maxPotential {
			maxPotential = neuron.MembranePotential
		}
	}

	for i := range l.Neurons {
		if l.Neurons[i].Fired {
			// Winner-takes-all inhibition
			l.Neurons[i].Inhibition = maxPotential * 0.8
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
		}
	}

	return outputs
}

func (l *Layer) UpdateWeights(currentTime int, learningRate float64) {
	for i := range l.Neurons {
		l.Neurons[i].UpdateWeights(currentTime, learningRate)
	}
}
