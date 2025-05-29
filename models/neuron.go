package models

type SpikingNeuron struct {
	MembranePotential float64
	Threshold         float64
	Decay             float64
	Bias              float64
	Weights           []float64
	RefractoryPeriod  int
	RefractoryTimer   int
}

type Connection struct {
	Weight        float64
	LastPreSpike  int
	LastPostSpike int
}

// Neuron struct
type Neuron struct {
	weights []float64
	bias    float64
}
