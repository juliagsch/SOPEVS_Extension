import datetime
import sys
import json
import os
import pytz
import solar_optimizer

import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib.iotools import get_pvgis_hourly
from read_mesh import read_building, read_surroundings
from coplanarity_mesh import Scene
import shutil
from timezonefinder import TimezoneFinder
from global_land_mask import globe
import visualize
from raytracing import ray_triangle_intersection

enable_visualize = True


def solar_vector(azimuth, zenith):
    """Convert solar angles to 3D direction vector"""
    az_rad = np.radians(azimuth)
    zen_rad = np.radians(zenith)
    return np.array([
        np.sin(zen_rad) * np.sin(az_rad),
        np.sin(zen_rad) * np.cos(az_rad),
        np.cos(zen_rad)
    ])


def is_shaded(origin, solar_dir, occlusion_triangles):
    """Check shading using multiple rays across the triangle"""
    ray_origin = origin + solar_dir * 0.1  # Offset by 10 cm to avoid self-intersection

    # Check against centeroid
    for tri in occlusion_triangles:
        if ray_triangle_intersection(ray_origin, solar_dir, tri):
            return True  # Shaded if any ray hits

    return False  # Not shaded if all rays clear


def get_pv_production(config, tilt, azimuth, year=2022):
    ts_offset = int(config['gmt_offset'])
    timezone_str=config['timezone']
    lat, lon = config['geo_centroid']

    # Set name of the radiation database. 
    # "PVGIS-SARAH" for Europe, Africa and Asia or 
    # "PVGIS-NSRDB" for the Americas between 60°N and 20°S, 
    # "PVGIS-ERA5" and "PVGIS-COSMO" for Europe (including high-latitudes)
    if 'Australia' in timezone_str or 'America' in timezone_str:
        raddatabase='PVGIS-ERA5'
    else:
        raddatabase='PVGIS-SARAH3'
        
    poa, _ = get_pvgis_hourly(
        latitude=lat,
        longitude=lon,
        components=False,
        # We need to simulate 3 years as the output is given in GMT+0 but we need to support all timezones.
        # Therefore, we need values from the last day of the year before the start date and the first day of the year after the end year.
        # The function only simulates full years.
        start=year-1, 
        end=year+1,
        raddatabase=raddatabase,
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        usehorizon=True,
        pvcalculation=True,
        map_variables=True,
        peakpower=1
    )
    poa = poa['poa_global']
    pv_production = poa[8760-ts_offset : (8760*2)-ts_offset].values # Shift to local timezone by taking values from following or previous year
    pv_production = pv_production
    np.savetxt(f'{round(tilt)}_{round(azimuth)}.txt', pv_production/1000)

    return pv_production


def create_comprehensive_results(scene, simulation_results):
    comprehensive_results = {}

    for roof_segment_idx, per_panel_production in simulation_results.items():
        # find the roof_segment object with matching idx
        roof_segment = next((rs for rs in scene.roof_segments if rs.idx == roof_segment_idx), None)
        if roof_segment is None:
            print(f"Warning: Roof segment {roof_segment_idx} not found.")
            continue

        for panel_idx, panel_production in enumerate(per_panel_production):
            mesh_id = f"Segment{roof_segment_idx}_Panel{panel_idx}"
            try:
                panel_coords = np.array(roof_segment.panels[panel_idx])
                comprehensive_results[mesh_id] = {
                    'average_radiance': round(float(np.mean(panel_production)), 1),
                    'original_coordinates': panel_coords.tolist(),
                    'centroid': np.mean(panel_coords, axis=0).tolist(),
                }
            except (ValueError, IndexError) as e:
                print(f"Skipping invalid mesh ID {mesh_id}: {str(e)}")

    return comprehensive_results


def initialize_scene(config):
    """Initialize scene consisting of the building and its surroundings."""

    building_V, building_F, offset = read_building(config['building_filename'])
    surroundings_V, surroundings_F = read_surroundings(config['surroundings_filename'], offset)        

    scene = Scene(
        building_V=building_V,
        building_F=building_F,
        surroundings_V=surroundings_V,
        surroundings_F=surroundings_F,
        grid_size=config['grid_size'],
        target_faces=config['target_faces']
    )

    if enable_visualize:
        visualize.plot_building(scene.building_V, scene.building_F)
        visualize.plot_rooftops(scene.building_V, scene.roof_faces)
        visualize.plot_building_with_mesh_grid(scene.building_V, scene.building_F, scene.panels)
        visualize.plot_rooftops_with_mesh_points(scene.building_V, scene.roof_faces, scene.panels)
        visualize.plot_rooftops_with_mesh_grid(scene.building_V, scene.roof_faces, scene.panels, scene.surroundings_V, scene.surroundings_F)
    return scene


def simulate_roof_segment(scene, roof_segment, config):
    timezone_str=config['timezone']
    start_time=config['simulation_params']['start']
    end_time=config['simulation_params']['end']
    lat, lon = config['geo_centroid']

    # Generate time range
    start = pd.Timestamp(start_time).tz_localize(timezone_str)
    end = pd.Timestamp(end_time).tz_localize(timezone_str)
    times = pd.date_range(start=start, end=end, freq='h', tz=timezone_str)

    # Get the production without shading
    pv_production = get_pv_production(config, roof_segment.tilt, roof_segment.azimuth, year=start.year)

    site = Location(lat, lon, tz=timezone_str, altitude=config['altitude'])
    solar_pos = site.get_solarposition(times)

    production_per_panel = [pv_production.copy() for _ in range(len(roof_segment.panels))]

    # Shading simulation for every panel position on roof segment
    for t_idx, (ts, pos) in enumerate(solar_pos.iterrows()):
        print(t_idx)
        # Iteration can be skipped if there is no production in that hour
        if pv_production[t_idx] == 0:
            continue

        # Get solar vector
        solar_azimuth = pos['azimuth']
        solar_zenith = pos['apparent_zenith']
        solar_dir = solar_vector(solar_azimuth, solar_zenith)

        # Keep track of processed grid points to avoid duplicate computations
        shaded_mesh_pts = set()
        processed_pts = set()

        for panel_idx, panel in enumerate(roof_segment.panels):
            panel_shaded = False
            for square_edge in panel:
                # Check if square edge has already been processed
                xyz = tuple(square_edge)
                if xyz in processed_pts and xyz in shaded_mesh_pts:
                    panel_shaded = True
                    break
                if xyz in processed_pts:
                    continue

                # Carry out raytracing
                if is_shaded(square_edge, solar_dir, scene.surroundings_triangles):
                    panel_shaded = True
                    shaded_mesh_pts.add(xyz)
                processed_pts.add(xyz)
            
            if panel_shaded:
                production_per_panel[panel_idx][t_idx] = 0
            else:
                centroid = np.mean(np.array(panel))
                if is_shaded(centroid, solar_dir, scene.surroundings_triangles):
                    production_per_panel[panel_idx][t_idx] = 0

    return production_per_panel


# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def run_simulation(scene, config):
    """Execute full solar potential simulation"""
    results = {}

    for roof_segment in scene.roof_segments:
        print("Run simulation for roof segment ", roof_segment.idx)
        results[roof_segment.idx] = simulate_roof_segment(scene, roof_segment, config)

    return results


def save_hourly_data_to_txt(simulation_results, output_dir):
    """Save hourly production data to text files."""
    for roof_segment_idx, per_panel_production in simulation_results.items():
        for panel_idx, panel_production in enumerate(per_panel_production):
            filename = os.path.join(output_dir, f"Segment{roof_segment_idx}_Panel{panel_idx}_hourly.txt")
            with open(filename, 'w') as f:
                for hourly in panel_production:
                    # Convert from W to kW
                    avg_production = hourly / 1000.0
                    f.write(f"{avg_production:.16f}\n")


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    return obj


def save_comprehensive_results(comprehensive_results, output_dir):
    """Save comprehensive results to text file with proper serialization"""
    filename = os.path.join(output_dir, "comprehensive_results.txt")
    
    try:
        # Convert numpy types to JSON-serializable formats
        converted_results = convert_to_serializable(comprehensive_results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
            
        print(f"Successfully saved results to {filename}")
    except TypeError as e:
        print(f"Serialization error: {str(e)}")
    except IOError as e:
        print(f"File I/O error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def get_timezone(lat, lon, year=2022):
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


def main():
    if len(sys.argv) != 2:
        # print("Usage: python solar_optimizer.py <output_dir>")
        # sys.exit(1)
        output_dir = "./out_noshading" 
    else:
        output_dir = sys.argv[1] 

    filename = "2615150_1233155_2615160_1233165"
    filename = "2613504_1233184_2613514_1233192"
    lat, lon = 47.21, 7.54
    # lat, lon = 50.2088, -90.5323 #ontario
    # lat, lon = -33.2088, 150.5323 #sydney

    altitude = 500
    
    if not globe.is_land(lat=lat, lon=lon):
        print("Please choose coordinates on land.")
        sys.exit()

    timezone_str, offset_hours = get_timezone(lat, lon)

    CONVERSION_PARAMS = {
        'building_filename': f"./samples/{filename}/building.obj",
        'surroundings_filename': f"./samples/{filename}/surroundings3D.obj",
        'earth_radius': 6378137.0,
        'geo_centroid': (lat, lon),
        'altitude': altitude,

        'unit_scaling': (1.0, 1.0, 1.0),
        'timezone': timezone_str, 
        'gmt_offset': offset_hours,
        'grid_size': 1.0, # in m^2 for each mesh cell
        'target_faces': 500,

        'simulation_params': {
            'start': datetime.datetime(2022, 1, 1, 0, 0),
            'end': datetime.datetime(2022, 12, 31, 23, 0), # Simulate 365 days of the year
        }
    }

    print(CONVERSION_PARAMS)

    # Run a complete simulation
    scene = initialize_scene(CONVERSION_PARAMS)
    simulation_results = run_simulation(scene, CONVERSION_PARAMS)
    
    # Save results
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    save_hourly_data_to_txt(simulation_results, output_dir)
    comprehensive_results = create_comprehensive_results(scene, simulation_results)
    save_comprehensive_results(comprehensive_results, output_dir)
    
    comprehensive_results = solar_optimizer.read_results(output_dir)
    panel_placement = solar_optimizer.optimize_panel_placement(
        comprehensive_results,
        num_panels=4,
        quad_size=1,
        panel_length=1,
        panel_width=1
    )
    
    visualize.visualize_quads_and_panels(comprehensive_results, panel_placement, (scene.surroundings_V, scene.surroundings_F))


    return True

if __name__ == '__main__':
    main()