# Workflow of land change simulation 

Major steps of the land change simulation workflow are as follows:

## 1. Change quantity determination
(1) generate the change matrix based on the observed land cover time series  
(2) determine the historical land change quantity by compiling the historical literature  
(3) predict the future land change quantity and uncertainty using bootstrapping

## 2. Change probability map generation
(1) extract the land change related predictor variables  
(2) generate the land change probability map using the Random Forest model

## 3. Land change simulation
(1) allocate the land change by combining the change quantity and change probability map
(2) determine the uncertainty of the land change simulation using bootstrapping strategy
