# <img src="resources/PowerOwl.svg" width="200" alt="PowerOwl">
PowerOwl models multi-layered power grid architectures, covering the technical grid, 
individual facilities, organizational units, a corresponding network topology,
individual network assets, communication configurations and more.
Its primary use is to model and simulate the scenarios used by [Wattson](https://wattson.it)
or other co-simulation approaches.

## Setup
```bash
git clone https://github.com/fkie-cad/powerowl.git
python3 -m pip install -e powerowl
``` 

## Example usage
```python
from powerowl.power_owl import PowerOwl
from pathlib import Path

# Create PowerOwl instance
owl = PowerOwl()
# Load MLG from the examples folder
owl.load_from_file(Path("example-models/cigre_mv_mlg.json"))
# Show the model in the browser
owl.draw()
```

## Disclaimer
For now, the part of PowerOwl that allows to derive the whole grid architecture
based just on a power grid topology remains private. 
Thus, only models that are published along with Wattson are available.
