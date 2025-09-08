# SunPlace

SunPlace is a tool to obtain suitable PV and battery sizes given the following input:

- 3D building mesh (triangular): The target building on whose roof the panels should be mounted.
- 3D surroundings mesh (triangular): A mesh of the surrounding occlusions which could cast shade on the roof (trees, other buildings,..).
- Load trace: Hourly household electricity data.
- Location: Lat/Lon of the building.
- Optional EV Usage data: Trace of EV departure and arrival times as well as charging policy.

SunPlace uses the input to:

1. Extract coherent roof segments from the building mesh and find possible panel locations.
2. Get the hourly production of each roof segment with its individual tilt/azimuth using PVLib which incorporated PVGIS weather data.
3. Carry out a shading analysis for each possible panel location.
4. Carry out a sizing simulation to obtain a suitable system size (PV and battery) to meet a specified self-sufficiency target provided either in EUE or LOLP metric.
5. Place the required number of panels in a suitable roof location to maximize overall production.

# How to use SunPlace

All of the configurations can be specified in sunplace.py which also needs to be run to start the simulation. Furthermore, changes to the sizing simulation can be implemented and compiled in /sizing_simulation.

We provide the data used for the case study in /data
