# Data Standard

## Canonical timeseries columns

Required:

- `time_s`
- `current_A`
- `voltage_V`
- `temperature_K`

Recommended additional columns:

- `cell_id`
- `experiment_id`
- `cycle_index`
- `step_index`
- `charge_Ah`
- `energy_Wh`
- `ambient_temperature_K`
- `soc_reference`
- `soh_reference`

## Sign convention

The research configuration and exported dataset must explicitly state current sign convention. The baseline PyBaMM runner records the applied current in its manifest.

## Raw/interim/processed

- `data/raw`: immutable instrument exports
- `data/interim`: parsing/alignment products
- `data/processed`: model-ready versioned datasets

Never overwrite raw files during preprocessing.
