package models

type SpikingNeuron struct {
	weights           []float64
	bias              float64
	membranePotential float64
	threshold         float64
}
