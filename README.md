## Optimized Snow Sensor Location
This repository contains software to optimally locate a limited number of locations in a basin that best captures the feature space variance of the basin. These tools were developed with the intent of locating snowpack sensing instrumentation sites. 

The funcs.py file contains functions to run a Guassian mixture model (GMM) over a physiographic feature space. The model combines multiple, multivariate distributions of the features to converge on locations which optimize the ability of the limited number of sensor sites to represent all points in the feature space. 
