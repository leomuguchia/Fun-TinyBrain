// File: metal/accessors.go
package neuron

func (n *Network) Layers() []*Layer {
	return n.layers
}

func (l *Layer) Neurons() []SpikingNeuron {
	return l.neurons
}
