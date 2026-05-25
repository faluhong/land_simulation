# Land change simulation workflow in Hispaniola

## Overview
A clean, modular workflow for simulating historical and future land-cover change
— with a focus on primary forest dynamics — across the island of Hispaniola (Haiti and the Dominican Republic).

The pre-required land cover dataset from 1996 to 2022 is available at: https://doi.org/10.6084/m9.figshare.28100408 
The topography information (DEM, slope and aspect) is available at: https://drive.google.com/drive/folders/1aZSa_7aFEeQwvHR9SfOvxNlLJV86umlA?usp=sharing

```text
land_simulation/
├── data/           # Auxiliary datasets for the land change simulation
├── pythoncode/     # Core Python scripts
├── results/        # Example simulation outputs
└── README.md       # Project documentation
```

The _**pythoncode**_ folder contains the main code of land change simulation.
- **1. _change_matrix_**: get the time series of change matrix from the observed land cover dataset
- **2. _change_matrix_prediction_extrapolation_**: predict the future change matrix by extrapolating the observed change matrix
- **3. _change_matrix_hindcast_pf_interpolation_**: get the historical change matrix based on the observed change matrix and the historical literature
- **4. _change_probability_map_**: generate the land change probability map to determine which pixel is more likely to change
- **5. _land_change_simulation_**: allocate the land change by combining the change quantity and change probability map


## Simulation Workflow Overview

### 1. Change quantity determination
(1) generate the change matrix based on the observed land cover time series  
(2) determine the historical land change quantity by compiling historical literature  
(3) predict the future land change quantity and uncertainty using bootstrapping

Related code: **_change_matrix_**,  **_change_matrix_prediction_extrapolation_**,  **_change_matrix_hindcast_pf_interpolation_**

### 2. Change probability map generation
(1) extract the land change related predictor variables     
(2) generate the land change probability map using the Random Forest model

Related code: **_change_probability_map_**

### 3. Land change allocation
allocate the land change by combining the change quantity and change probability map

Related code: **_land_change_simulation_**


## Contact
For questions or further information, please contact:
- Falu Hong (faluhong@uconn.edu)