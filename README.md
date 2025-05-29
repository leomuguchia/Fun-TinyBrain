# Fun Tiny Brain

*A tiny spiking neural network built in Go — where neurons either fire or don’t (no in-between).*

---

## What Makes It Different?

Unlike traditional neural networks (where neurons output a range of values and require activation functions), each neuron in Fun Tiny Brain:

- **Accumulates input signals over time**
- **Fires (spikes) only when a threshold is reached**
- **Resets after firing**
- **Gradually loses potential if the threshold isn’t met**

The output is always binary:  
**fire (1)** or **no fire (0)**.

---

## Why Does This Matter?

This simple model is closer to how real neurons work — switching on/off, not sliding smoothly.  
It eliminates the need for activation functions and keeps things straightforward.