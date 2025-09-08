import datetime
import sys
import os

import numpy as np
import shutil
import solar_simulation
from global_land_mask import globe
import placement
import visualize
import subprocess

import datetime
import pytz
from timezonefinder import TimezoneFinder

# Solar Simulation Configuration
simulation_year = 2021
grid_size = 1 # in m^2 for each mesh cell
target_faces = 100 # Target number of roof faces to reduce to
angle_tolerance = 1.0 # Angle tolerance in degrees for merging faces
max_roof_tilt = 60.0 # Maximum roof tilt in degrees to consider a face as rooftop

# Panel Configuration
panel_size_kW = 0.4  # Each panel produces 400 W
panel_height=1.8 # in m
panel_width=1 # in m

# Location for Solothurn, Switzerland
lat, lon = 47.21, 7.54
altitude = 500 # in m

# Sizing Configuration
eue_target = 0.2 # 80% of the load should be covered by solar
conf = 0.75 # Confidence level for robustness of the sizing simulation
pv_cost = 1000 # per kW
battery_cost = 600 # per kWh

# EV Configuration
ev_path = './data/EV.csv'
op = 'safe_departure'


def get_timezone(lat, lon, year=2021):
    """Get timezone name and offset from GMT from coordinates"""
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lng=lon, lat=lat)
    tz = pytz.timezone(timezone_str)

    # Choose a date in DST (January 1 for north, July 1 for south)
    dt = datetime.datetime(year, 1, 1) if lat > 0 else datetime.datetime(year, 7, 1)

    # If this date is in DST, shift to the other half of the year
    if tz.dst(dt) != datetime.timedelta(0):
        dt = datetime.datetime(year, 7, 1) if lat > 0 else datetime.datetime(year, 1, 1)

    gmt_offset = tz.utcoffset(dt).total_seconds() / 3600
    gmt_offset = np.ceil(gmt_offset) # We round up half hour time zones as we consider hourly time steps
    return timezone_str, gmt_offset


def get_config(filename):
    if not globe.is_land(lat=lat, lon=lon):
        print("Please choose coordinates on land.")
        sys.exit()

    timezone_str, offset_hours = get_timezone(lat, lon)

    config = {
        'building_filename': f"./data/{filename}/building.obj",
        'surroundings_filename': f"./data/{filename}/surroundings3D.obj",
        'geo_centroid': (lat, lon),
        'altitude': altitude,

        'timezone': timezone_str, 
        'gmt_offset': offset_hours,
        'grid_size': grid_size, # in m^2 for each mesh cell
        'target_faces': target_faces,

        'angle_tolerance': angle_tolerance, # Angle tolerance in degrees for merging faces
        'max_roof_tilt': max_roof_tilt, # Maximum roof tilt in degrees to consider a face as rooftop

        'simulation_params': {
            'start': datetime.datetime(simulation_year, 1, 1, 0, 0),
            'end': datetime.datetime(simulation_year, 12, 31, 23, 0), # Simulate 365 days of the year
        }
    }
    return config


def simulate(output_dir, filename):
    output_dir = output_dir + filename
    config = get_config(filename)

    # Run a complete simulation
    scene = solar_simulation.initialize_scene(config)
    simulation_results = solar_simulation.run_simulation(scene, config)
    
    # Save results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    solar_simulation.save_hourly_data_to_txt(simulation_results, output_dir)
    comprehensive_results = solar_simulation.create_comprehensive_results(scene, simulation_results)
    solar_simulation.save_comprehensive_results(comprehensive_results, output_dir)

    # Get optimal sizing
    command = f"sizing_simulation/sim {pv_cost} {battery_cost} 1 1 30 1 {eue_target} {conf} 365 ./data/load_detached.txt {output_dir} 0.8 0.2 60 7.4 {op} {ev_path} 20"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)
    result = result.stdout.split("\t")
    if len(result) != 4:
        print("Error in simulator execution")
        return False
    battery, solar = result[0], result[1]
    if 'inf' in battery or float(battery) == -1.0:
        print("Simulator returned inf - no valid solution found for given SSR.")
        return True
    
    num_panels = int(np.ceil(float(solar) / panel_size_kW))  # Assume each panel produces 400 W
    print(battery, "kWh,", solar, "kW")
    print("Placing", num_panels, "panels on roof...")

    # Panel placement
    comprehensive_results = placement.read_results(output_dir)
    panel_placement = placement.panel_placement(
        comprehensive_results,
        num_panels=num_panels,
        quad_size=1,
        panel_height=panel_height,
        panel_width=panel_width
    )

    visualize.visualize_quads_and_panels(comprehensive_results, panel_placement, (scene.building_V, scene.united_faces), kwh_per_panel=panel_size_kW)
    return True

if __name__ == '__main__': 
    files = ['building_A', 'building_B', 'building_C']
    out_dir = './test/'

    for file in files:
        print("Simulating", file)
        simulate(out_dir, file)