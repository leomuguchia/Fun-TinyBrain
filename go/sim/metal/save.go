package neuron

import (
	"encoding/json"
	"os"
)

// Save saves the network state to a file
func (n *Network) Save(filename string) error {
	return saveJSON(filename, n)
}

// Load loads the network state from a file
func (n *Network) Load(filename string) error {
	file, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	return json.Unmarshal(file, n)
}

// helper to save any object as JSON
func saveJSON(filename string, v any) error {
	bytes, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, bytes, 0644)
}
