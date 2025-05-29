// File: metal/accessors.go
package neuron

func (n *Network) Layers() []*Layer {
	return n.layers
}

func (l *Layer) Neurons() []*SpikingNeuron {
	neurons := make([]*SpikingNeuron, len(l.neurons))
	for i := range l.neurons {
		neurons[i] = &l.neurons[i]
	}
	return neurons
}
