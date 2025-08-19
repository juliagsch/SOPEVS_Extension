import datetime
import time
import sys
import json
import os
import pprint
import pytz

import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
from pvlib.iotools import get_pvgis_hourly
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel
from cartesian_lonlat import convert_coordinate_system, visualize_3d_mesh, convert_coordinate_system_building, interleave_mesh, building_format
import matplotlib.pyplot as plt
import shutil
from timezonefinder import TimezoneFinder
from global_land_mask import globe

visualize = False

def compute_centroid(triangle):
    """Calculate centroid with robust type checking"""
    # Convert to numpy array if needed
    if not isinstance(triangle, np.ndarray):
        try:
            triangle = np.array(triangle, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Invalid triangle structure: {triangle}") from e

    # Verify triangle shape (3 vertices, 3 coordinates each)
    if triangle.shape != (3, 3):
        raise ValueError(f"Invalid triangle dimensions: {triangle.shape}. Should be (3,3)")

    return np.mean(triangle, axis=0)

def ray_triangle_intersection(ray_origin, ray_dir, triangle, epsilon=1e-10):
    """Möller–Trumbore algorithm with dynamic epsilon"""
    v0, v1, v2 = [np.array(p) for p in triangle]
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Auto-calculate epsilon if not provided
    if epsilon is None:
        avg_edge = (np.linalg.norm(edge1) + np.linalg.norm(edge2) + np.linalg.norm(v2 - v1)) / 3
        epsilon = max(1e-6, avg_edge * 0.01)

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return False

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return t > epsilon

def calculate_tilt_azimuth(triangle):
    """Calculate surface orientation from triangle geometry with type safety"""
    # Convert to numpy array if needed
    if not isinstance(triangle, np.ndarray):
        triangle = np.array(triangle, dtype=np.float64)

    # Ensure we have valid 3D coordinates
    if triangle.shape != (3, 3):
        raise ValueError(f"Invalid triangle shape {triangle.shape}. Expected 3 vertices with 3 coordinates each")

    # Calculate normal vector
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = np.cross(v1, v2)

    if normal[2] < 0:
        normal = -normal

    # Normalize and calculate angles
    normal /= np.linalg.norm(normal)
    tilt = np.degrees(np.arccos(normal[2]))
    azimuth = np.degrees(np.arctan2(normal[1], normal[0])) # Counter clock-wise, 0 degrees pointing East
    azimuth = (360-azimuth) % 360 # Convert to clock-wise
    azimuth = (azimuth + 90) % 360 # Convert from 0 degrees pointing East to 0 degrees pointing North
    print(azimuth)

    if np.isclose(tilt, 0.0):
        azimuth = 0.0 

    return tilt, azimuth

def solar_vector(azimuth, zenith):
    """Convert solar angles to 3D direction vector"""
    az_rad = np.radians(azimuth)
    zen_rad = np.radians(zenith)
    return np.array([
        np.sin(zen_rad) * np.sin(az_rad),
        np.sin(zen_rad) * np.cos(az_rad),
        np.cos(zen_rad)
    ])

def is_shaded(centroid, solar_dir, building_triangles, num_samples):
    """Check shading using multiple rays across the triangle"""
    ray_origin = centroid + solar_dir * 0.1  # Offset to avoid self-intersection

    # Check against centeroid
    for tri in building_triangles:
        if ray_triangle_intersection(ray_origin, solar_dir, tri):
            return True  # Shaded if any ray hits

    return False  # Not shaded if all rays clear

def extract_meshes(nested_list):
    """Flatten nested structure to extract individual meshes (each with 4 points)."""
    meshes = []
    for item in nested_list:
        if isinstance(item, list):
            if len(item) == 4 and all(len(sub) == 3 for sub in item):
                meshes.append(item)
            else:
                meshes.extend(extract_meshes(item))
    return meshes


def calculate_mesh_averages(results):
    """Calculate average radiance for each mesh."""
    return {mesh_id: np.mean(list(data.values())) for mesh_id, data in results.items()}

def create_mesh_coordinate_map(mesh_triangles):
    return {
        idx: {
            'vertices': geometry,  # First element of tuple
            'centroid': compute_centroid(geometry),  # Process geometry only
            'mesh_index': mesh_idx  # Second element of tuple
        }
        for idx, (geometry, mesh_idx) in enumerate(mesh_triangles)  # Unpack tuple here
    }


def simulate_period_with_shading(lat, lon, tilt, azimuth, mesh_triangles, building_triangles, current_idx, config):
    """Simulate solar flux for a mesh triangle considering shading from other meshes and buildings."""
    # Set up location and site
    timezone_str=config['timezone']
    start_time=config['simulation_params']['start']
    end_time=config['simulation_params']['end']
    # time_base=config['simulation_params']['resolution']
    num_samples=config['simulation_params']['shading_samples']
    ts_offset = int(config['gmt_offset'])
    
    # Generate time range
    start = pd.Timestamp(start_time).tz_localize(timezone_str)
    end = pd.Timestamp(end_time).tz_localize(timezone_str)
    times = pd.date_range(start=start, end=end, freq='h', tz=timezone_str)
    
    # if time_base == 'hourly':
    #     times = pd.date_range(start=start, end=end, freq='h', tz=timezone_str)
    # elif time_base == 'daily':
    #     start_date = start.floor('D')
    #     end_date = end.ceil('D') - pd.Timedelta(seconds=1)
    #     times = pd.date_range(start=start_date, end=end_date, freq='h', tz=timezone_str)
    # elif time_base == 'weekly':
    #     start_date = start.floor('D')
    #     end_date = end.ceil('D') - pd.Timedelta(seconds=1)
    #     times = pd.date_range(start=start_date, end=end_date, freq='h', tz=timezone_str)
    # else:
    #     raise ValueError("Invalid time_base")

    # times = times[(times >= start) & (times <= end)]
    # if times.empty:
    #     return {}

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
        # We need to simulate three years as the output is given in GMT+0 but we need to support all timezones.
        # Therefore, we need values from the last day of the year before the start date and the first day of the year after the end year.
        # The function only simulates full years.
        start=start.year-1, 
        end=start.year+1,
        raddatabase=raddatabase,
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        usehorizon=True,
        pvcalculation=True,
        map_variables=True,
        peakpower=1
    )
    poa = poa['poa_global']
    poa = poa[8760-ts_offset:17520-ts_offset] # Shift to local timezone by taking values from following or previous year

    site = Location(lat, lon, tz=timezone_str, altitude=config['altitude'])
    solar_pos = site.get_solarposition(times)

    # Precompute shading status for each time point
    shaded_mask = np.zeros(len(times), dtype=bool)
    triangle_geometry = np.array(mesh_triangles[current_idx])
    centroid = np.mean(triangle_geometry, axis=0)
    for i, (ts, pos) in enumerate(solar_pos.iterrows()):
        # Only run shading on Mondays (0) — skip others
        if ts.dayofweek != 0:
            shaded_mask[i] = shaded_mask[i-(24*ts.dayofweek)]
        else:
            solar_azimuth = pos['azimuth']
            solar_zenith = pos['apparent_zenith']
            solar_dir = solar_vector(solar_azimuth, solar_zenith)

            # Check if the current triangle is shaded at this time
            shaded_mask[i] = is_shaded(centroid, solar_dir, building_triangles, num_samples)



    # Apply shading mask (set shaded timesteps to zero)
    total_flux = poa.clip(lower=0)

    # with open("poa_unshaded.txt", 'w') as f:
    #     # Write header
    #     #f.write("Timestamp,Solar_Trace_kW_per_m2\n")

    #     for i in total_flux:
    #         f.write(f"{i/1000:.16f}\n")
    
    # total_flux[shaded_mask] = 0


    # with open("poa_shaded.txt", 'w') as f:
    #     # Write header
    #     #f.write("Timestamp,Solar_Trace_kW_per_m2\n")

    #     for i in total_flux:
    #         f.write(f"{i/1000:.16f}\n")
    


    # with open("mask.txt", 'w') as f:
    #     # Write header
    #     #f.write("Timestamp,Solar_Trace_kW_per_m2\n")

    #     for i in shaded_mask:
    #         if i:
    #             f.write(f"{0.5:.16f}\n")
    #         else:
    #             f.write(f"{0:.16f}\n")


    # Aggregate results based on time_base
    # if time_base == 'hourly':
    #     aggregated = total_flux
    # elif time_base == 'daily':
    #     aggregated = total_flux.resample('D').sum()
    # elif time_base == 'weekly':
    #     aggregated = total_flux.resample('W').sum()

    return {ts.to_pydatetime(): val for ts, val in total_flux.items()}


def create_comprehensive_results(averages, coordinate_map):
    results = {}
    for mesh_id, avg in averages.items():
        try:
            idx = int(mesh_id.split('_')[1]) - 1
            if idx not in coordinate_map:
                raise KeyError(f"No coordinate data for index {idx}")

            results[mesh_id] = {
                'average_radiance': avg,
                'original_coordinates': coordinate_map[idx]['vertices'],
                'centroid': coordinate_map[idx]['centroid'],
                'mesh_index': coordinate_map[idx]['mesh_index']
            }
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid mesh ID {mesh_id}: {str(e)}")

    return results

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def load_and_process_building(params):
    """Load and process building geometry data"""
    vertices, faces, offset = read_polyshape_3d.read_polyshape(params['input_file'])
    roof = RoofSolarPanel(
        V=vertices,
        F=faces,
        **params['panel_config']
    )
    if visualize:
        roof.display_building_and_rooftops()
        roof.plot_building_with_mesh_grid()
        roof.plot_rooftops_with_mesh_points()


    ground_centroid = roof.get_ground_centroid()[:2]

    # Convert building coordinates
    converted_building = convert_coordinate_system_building(
        params, roof.V,
        *ground_centroid,
        *params['geo_centroid'],
        *params['unit_scaling']
    )

    # Generate building triangles
    building_triangles = []
    for face in roof.triangular_F:
        building_triangles.append([converted_building[i] for i in face])

    original_building_triangles = []
    vertices, faces = read_polyshape_3d.read_surroundings(params['obstruction_file'], offset)
    if visualize:
        roof.plot_rooftops_with_mesh_grid(vertices, faces)
    original_building = building_format(vertices)

    for face in faces:
        original_building_triangles.append([original_building[i] for i in face])

    return roof, converted_building, building_triangles, original_building, original_building_triangles

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def process_solar_meshes(roof, params):
    ground_centroid = roof.get_ground_centroid()[:2]
    """Process and triangulate solar panel meshes"""
    converted_mesh = convert_coordinate_system(
        params, roof.mesh_objects,
        *ground_centroid,
        *params['geo_centroid'],
        *params['unit_scaling']
    )

    original_mesh = interleave_mesh(roof.mesh_objects)

    # Extract and triangulate meshes
    converted_mesh_triangles = []
    for mesh_idx, square in enumerate(extract_meshes(converted_mesh)):
        tri1 = [square[0], square[2], square[1]]
        tri2 = [square[0], square[3], square[2]]
        converted_mesh_triangles.append((tri1, mesh_idx))
        converted_mesh_triangles.append((tri2, mesh_idx))

    
    original_mesh_triangles = []
    for mesh_idx, square in enumerate(extract_meshes(original_mesh)):
        tri1 = [square[0], square[2], square[1]]
        tri2 = [square[0], square[3], square[2]]
        original_mesh_triangles.append((tri1, mesh_idx))
        original_mesh_triangles.append((tri2, mesh_idx))

    return converted_mesh, converted_mesh_triangles, original_mesh, original_mesh_triangles


# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def process_results(raw_results, coordinate_map):
    """Process and enrich simulation results"""
    averages = calculate_mesh_averages(raw_results)
    return create_comprehensive_results(averages, coordinate_map)

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def initialize_components(config):
    """Initialize and return building and solar mesh data"""
    # Process building geometry
    roof, converted_building, building_triangles, original_building, original_building_triangles = load_and_process_building(config)
    # Process solar panel meshes
    converted_mesh, mesh_triangles, original_mesh, original_mesh_triangles = process_solar_meshes(roof, config)
    # visualize_3d_mesh(converted_mesh, config)
    # print(converted_mesh)
    # print(converted_building)
    # print(building_triangles)
    # visualize_3d_mesh(converted_building, config)

    # Create coordinate map during initialization
    coordinate_map = create_mesh_coordinate_map(mesh_triangles)

    return (
        {
            # 'building_triangles': building_triangles,
            'roof': roof,
            # 'converted_building': converted_building,
            'original_building': original_building,
            'original_building_triangles': original_building_triangles,
        },
        {
            # 'mesh_triangles': mesh_triangles,
            # 'converted_mesh': converted_mesh,
            'coordinate_map': coordinate_map,
            'original_mesh': original_mesh,
            'original_mesh_triangles': original_mesh_triangles,
        }
    )

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def run_complete_simulation(building_data, solar_meshes, config):
    """Execute full solar potential simulation"""
    # Prepare simulation data
    mesh_data = prepare_mesh_data(solar_meshes['original_mesh_triangles'])
    lat, lon = config['geo_centroid']
    # mesh_data = prepare_mesh_data(building_data['roof'].mesh_objects)

    # Run simulation for each mesh element
    import time
    results = {}
    breaking = False
    for idx, orientation in enumerate(mesh_data['orientations']):
        print(orientation, " faces: ", len(building_data))
        start = time.time()
        results[f"Mesh_{idx + 1}"] = simulate_period_with_shading(
            lat=lat,
            lon=lon,
            tilt=orientation[0],
            azimuth=orientation[1],
            mesh_triangles=[tri for tri, _ in solar_meshes['original_mesh_triangles']],
            building_triangles=building_data['original_building_triangles'],
            current_idx=idx,
            config=config,
        )
        print("duration: ", time.time()-start)
        
        if breaking:
            break
        breaking = True

    return results

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def prepare_mesh_data(original_mesh_triangles):
    """Prepare mesh data for simulation"""
    return {
        'orientations': [calculate_tilt_azimuth(tri) for tri, _ in original_mesh_triangles],
        # 'centroids': [compute_centroid(tri) for tri, _ in mesh_triangles]
    }

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def final_results(raw_results, solar_meshes):
    """Process results"""
    # Calculate averages and create comprehensive dataset
    averages = calculate_mesh_averages(raw_results)

    # Use coordinate map from solar_meshes
    comprehensive_results = create_comprehensive_results(averages, solar_meshes['coordinate_map'])

    return comprehensive_results


# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def create_3d_visualization(results_data):
    #Generate 3D visualization of results with mesh labels.
    radiances = [mesh['average_radiance'] for mesh in results_data.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot meshes with labels
    for mesh_id, mesh in results_data.items():
        # Plot the polygon
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        polygon.set_facecolor(cmap(norm(mesh['average_radiance'])))
        ax.add_collection3d(polygon)

        # Add mesh number label at centroid
        mesh_number = int(mesh_id.split('_')[1])  # Extract number from "Mesh_X"
        centroid = mesh['centroid']
        ax.text(
            centroid[0], centroid[1], centroid[2],
            str(mesh_number),
            color='white',
            fontsize=9,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle="round",
                facecolor='black',
                alpha=0.7,
                edgecolor='none'
            ),
            zorder=4  # Ensure labels stay on top
        )

    # Configure axes and colorbar
    ax.set_xlabel('X Axis', fontsize=9)
    ax.set_ylabel('Y Axis', fontsize=9)
    ax.set_zlabel('Elevation', fontsize=9)
    ax.grid(True)

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Average Radiance (W/m²)', fontsize=10)

    # Set viewing angle
    ax.view_init(elev=45, azim=-45)
    plt.tight_layout()
    plt.show()


def save_hourly_data_to_txt(simulation_results, output_dir):
    """Save hourly irradiance data for mesh pairs to text files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Sort mesh IDs numerically
    mesh_ids = sorted(simulation_results.keys(),
                      key=lambda x: int(x.split('_')[1]))

    # Group meshes in pairs
    groups = []
    for i in range(0, len(mesh_ids), 2):
        pair = mesh_ids[i:i + 2]
        group_name = f"Group_{i // 2 + 1}_Meshes_{'-'.join([m.split('_')[1] for m in pair])}"
        groups.append((group_name, pair))

    # Process each group
    for group_name, mesh_pair in groups:
        filename = os.path.join(output_dir, f"{group_name}_hourly.txt")

        # Get common timestamps (assuming all meshes have same timestamps)
        timestamps = list(simulation_results[mesh_pair[0]].keys())

        with open(filename, 'w') as f:
            # Write header
            #f.write("Timestamp,Solar_Trace_kW_per_m2\n")

            for ts in timestamps:
                # Get values for all meshes in pair
                values = []
                for mesh_id in mesh_pair:
                    values.append(simulation_results[mesh_id].get(ts, 0))

                # Calculate average
                avg_production = (np.mean(values) /1000) if values else 0
                #meshes_list = ','.join([m.split('_')[1] for m in mesh_pair])
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

def save_comprehensive_results(results, output_dir):
    """Save comprehensive results to text file with proper serialization"""
    filename = os.path.join(output_dir, "comprehensive_results.txt")
    
    try:
        # Convert numpy types to JSON-serializable formats
        converted_results = convert_to_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
            
        print(f"Successfully saved results to {filename}")
    except TypeError as e:
        print(f"Serialization error: {str(e)}")
    except IOError as e:
        print(f"File I/O error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def get_timezone(lat, lon, year):
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
    
    filename = "2613471_1232999_2613482_1233008_"
    # lat, lon = 47.21, 7.54# ontario 50.2088, -90.5323
    lat, lon = 50.2088, -90.5323
    lat, lon = -33.2088, 150.5323

    altitude = 500
    year = 2010

    if year > 2022 or year < 2006:
        raise Exception("Choose a year between 2006 and 2022")

    if not globe.is_land(lat=lat, lon=lon):
        print("Please choose coordinates on land.")
        sys.exit()

    timezone_str, offset_hours = get_timezone(lat, lon, year)

    CONVERSION_PARAMS = {
        'input_file': f"./roofs/{filename}.obj",
        'obstruction_file': f"./surroundings/{filename}.obj",
        'earth_radius': 6378137.0,
        # 'geo_centroid': (52.1986125198786, 0.11358089726501427), #England
        #'geo_centroid': (-69.3872203183979, -67.70319583227294), #antarctic
        #'geo_centroid': (-42.2812963915156, 172.8486127892288), #New Zealand
        #'geo_centroid': (-33.36185896158091, 23.879985430205615), # South Africa
        'geo_centroid': (lat, lon), #Switzerland
        'altitude': altitude,

        # for variation of latitude
        # 'geo_centroid': (0, 23.879985430205615), # 
        #'geo_centroid': (20, 23.879985430205615), # 
        # 'geo_centroid': (-40, 23.879985430205615), # 
        #'geo_centroid': (60, 23.879985430205615), # 
        #'geo_centroid': (70, 23.879985430205615), # 
        #'geo_centroid': (80, 23.879985430205615), # 


        'unit_scaling': (1.0, 1.0, 1.0),
        'timezone': timezone_str, 
        'gmt_offset': offset_hours,
        'panel_config': {
            'panel_dx': 1.0,
            'panel_dy': 1.0,
            'max_panels': 10,
            'b_scale_x': 1,
            'b_scale_y': 1,
            'b_scale_z': 1,
            # 'exclude_face_indices': [],   # 2 for test 2, 9 for test
            'grid_size': 1.0                #m^2 for each mesh grid
        },
        'simulation_params': {
            'start': datetime.datetime(year, 1, 1, 0, 0),
            'end': datetime.datetime(year, 12, 30, 23, 0) if year % 4 == 0 else datetime.datetime(year, 12, 31, 23, 0), # Simulate 365 days of the year
            'resolution': 'hourly',
            'shading_samples': 1     # when simulating the ray-tracing algo, the number of samples drawn
        },
        # 'visualization': {
        #     'face_color': plt.cm.viridis(0.5),
        #     'edge_color': 'k',
        #     'alpha': 0.5,
        #     'labels': ('Longitude', 'Latitude', 'Elevation (m)')
        # }
    }

    print(CONVERSION_PARAMS)

    # Run a complete simulation
    building_data, solar_meshes = initialize_components(CONVERSION_PARAMS)
    simulation_results = run_complete_simulation(building_data, solar_meshes, CONVERSION_PARAMS)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Recreate the directory
    os.makedirs(output_dir)

    save_hourly_data_to_txt(simulation_results,output_dir)
    comprehensive_results = final_results(simulation_results, solar_meshes)
    save_comprehensive_results(comprehensive_results, output_dir)  # Comprehensive results to script dir

    return True

if __name__ == '__main__':
    main()