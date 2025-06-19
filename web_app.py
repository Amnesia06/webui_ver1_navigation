from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import sys
import io
import base64
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import numpy as np

# Import your path planner module
try:
    from path import (
        generate_fixed_path, 
        analyze_path_sequence_fixed,
        get_movement_analysis,
        LiveTelemetryLogger
    )
except ImportError as e:
    print(f"Error importing path: {e}")
    sys.exit(1)

app = Flask(__name__)

# Add this helper function at the top of your file
def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON, handling infinity and NaN values
    """
    if isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    else:
        return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_path', methods=['POST'])
def generate_path():
    try:
        data = request.json
        
        # Extract parameters
        farm_width_m = float(data['farm_width'])
        farm_breadth_m = float(data['farm_breadth'])
        rover_width_m = float(data['rover_width'])
        rover_length_m = float(data['rover_length'])
        exit_type = data['exit_type']
        waypoint_spacing_cm = float(data['waypoint_spacing'])
        gap_size = 1

        # Convert cm to meters
        waypoint_spacing_m = waypoint_spacing_cm / 100.0
        
        # Validate inputs
        if any(val <= 0 for val in [farm_width_m, farm_breadth_m, rover_width_m, rover_length_m]):
            return jsonify({'error': 'All dimensions must be positive numbers'}), 400
        
        if waypoint_spacing_cm < 1 or waypoint_spacing_cm > 10:
            return jsonify({'error': 'Waypoint spacing must be between 1-10cm'}), 400
        
        # Calculate grid
        num_lanes_x = int(farm_width_m / rover_width_m)
        num_lanes_y = int(farm_breadth_m / rover_length_m)
        
        if num_lanes_x < 3:
            return jsonify({'error': f'Farm width too small. Need at least {3 * rover_width_m}m for rover width {rover_width_m}m'}), 400
        if num_lanes_y < 1:
            return jsonify({'error': f'Farm breadth too small. Need at least {rover_length_m}m for rover length {rover_length_m}m'}), 400
        
        max_lane_idx_x = num_lanes_x - 1
        max_lane_idx_y = num_lanes_y - 1
        n_inner_x_sweeps = max(0, num_lanes_x - 2)
        
        # Determine exit point
        is_corner_exit = exit_type == 'corner'
        
        if is_corner_exit:
            corner_choice = data['corner_choice']
            corner_map = {
                'top_left': (0, max_lane_idx_y),
                'top_right': (max_lane_idx_x, max_lane_idx_y),
                'bottom_left': (0, 0),
                'bottom_right': (max_lane_idx_x, 0)
            }
            exit_point_lanes = corner_map[corner_choice]
        else:
            # Custom exit validation - FIXED
            exit_x = int(data['exit_x'])
            exit_y = int(data['exit_y'])
            
            # Check if coordinates are within valid range
            if exit_x < 0 or exit_x > max_lane_idx_x or exit_y < 0 or exit_y > max_lane_idx_y:
                return jsonify({'error': f'Exit coordinates out of range. X: 0-{max_lane_idx_x}, Y: 0-{max_lane_idx_y}'}), 400
            
            # Check if the exit point is on the boundary (edge or corner)
            is_on_left_edge = (exit_x == 0)
            is_on_right_edge = (exit_x == max_lane_idx_x)
            is_on_bottom_edge = (exit_y == 0)
            is_on_top_edge = (exit_y == max_lane_idx_y)
            
            is_on_boundary = (is_on_left_edge or is_on_right_edge or is_on_bottom_edge or is_on_top_edge)
            
            if not is_on_boundary:
                return jsonify({'error': f'Custom exit must be on farm boundary. Current position ({exit_x}, {exit_y}) is not on boundary. Valid boundary positions: X=0 or X={max_lane_idx_x}, Y=0 or Y={max_lane_idx_y}'}), 400
            
            exit_point_lanes = (exit_x, exit_y)
        
        # Generate path
        path_data = generate_fixed_path(
            n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, 
            exit_point_lanes, is_corner_exit, gap_size
        )
        
        path_lanes_list = path_data['points_lanes']
        sow_flags_list = path_data['sow_flags']
        
        if not path_lanes_list or len(path_lanes_list) < 2:
            return jsonify({'error': 'Path generation failed'}), 500
        
        # Generate subdivided waypoints
        detailed_waypoints = generate_detailed_waypoints(
            path_lanes_list, rover_width_m, rover_length_m, waypoint_spacing_m
        )
        
        # Generate waypoint analysis data
        waypoint_data = analyze_waypoints(
            path_lanes_list, detailed_waypoints, sow_flags_list, 
            rover_width_m, rover_length_m
        )
        
        # Generate visualization
        plot_data = generate_plot(
            path_lanes_list, sow_flags_list, exit_point_lanes,
            farm_width_m, farm_breadth_m, rover_width_m, rover_length_m,
            max_lane_idx_x, max_lane_idx_y, n_inner_x_sweeps,
            detailed_waypoints
        )
        
        # Calculate statistics
        total_segments = len(path_lanes_list) - 1
        sown_segments = total_segments
        unsown_segments = 0
        efficiency = 100.0

        # Calculate distances
        total_distance = 0
        sown_distance = 0

        for i in range(len(path_lanes_list) - 1):
            lx1, ly1 = path_lanes_list[i]
            lx2, ly2 = path_lanes_list[i + 1]
            distance = abs(lx2 - lx1) * rover_width_m + abs(ly2 - ly1) * rover_length_m
            total_distance += distance
            if i < len(sow_flags_list) and sow_flags_list[i]:
                sown_distance += distance

        efficiency = (sown_segments / total_segments * 100) if total_segments > 0 else 0
        path_efficiency = (sown_distance / total_distance * 100) if total_distance > 0 else 0

        # Calculate coverage
        total_coverage_area = num_lanes_x * num_lanes_y * rover_width_m * rover_length_m
        farm_area = farm_width_m * farm_breadth_m
        coverage_percentage = (total_coverage_area / farm_area * 100) if farm_area > 0 else 0
        
        # Use safe serialization for the return
        return jsonify({
            'success': True,
            'plot_data': plot_data,
            'waypoint_data': safe_json_serialize(waypoint_data),
            'statistics': safe_json_serialize({
                'grid_size': f"{num_lanes_x} × {num_lanes_y}",
                'total_waypoints': len(detailed_waypoints),
                'original_waypoints': len(path_lanes_list),
                'waypoint_spacing': f"{waypoint_spacing_cm}cm",
                'total_segments': total_segments,
                'sown_segments': sown_segments,
                'unsown_segments': unsown_segments,
                'total_distance': f"{total_distance:.1f}m",
                'sown_distance': f"{sown_distance:.1f}m",
                'efficiency': f"{efficiency:.1f}%",
                'coverage': f"{coverage_percentage:.1f}%",
                'inner_sweeps': n_inner_x_sweeps,
                'exit_position': f"Lane ({exit_point_lanes[0]}, {exit_point_lanes[1]})"
            })
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


def generate_detailed_waypoints(path_lanes_list, rover_width_m, rover_length_m, waypoint_spacing_m):
    """
    Generate detailed waypoints along the path with specified spacing
    """
    detailed_waypoints = []
    
    # Convert original lane coordinates to metric centers
    original_points = [
        ((ln_x + 0.5) * rover_width_m, (ln_y + 0.5) * rover_length_m) 
        for ln_x, ln_y in path_lanes_list
    ]
    
    # Always add the first point
    detailed_waypoints.append(original_points[0])
    
    # Generate intermediate points between each pair of original points
    for i in range(len(original_points) - 1):
        x1, y1 = original_points[i]
        x2, y2 = original_points[i + 1]
        
        # Calculate distance between points
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        # Calculate number of intermediate points needed
        if distance > waypoint_spacing_m:
            num_intermediate = int(distance / waypoint_spacing_m)
            
            # Generate intermediate points
            for j in range(1, num_intermediate + 1):
                ratio = j / (num_intermediate + 1)
                intermediate_x = x1 + (x2 - x1) * ratio
                intermediate_y = y1 + (y2 - y1) * ratio
                detailed_waypoints.append((intermediate_x, intermediate_y))
        
        # Add the end point
        detailed_waypoints.append((x2, y2))
    
    return detailed_waypoints

def analyze_waypoints(path_lanes_list, detailed_waypoints, sow_flags_list, rover_width_m, rover_length_m):
    """
    Analyze waypoints and provide detailed information about each one
    """
    waypoint_analysis = []
    
    # Convert original lane coordinates to metric centers
    original_points = [
        ((ln_x + 0.5) * rover_width_m, (ln_y + 0.5) * rover_length_m) 
        for ln_x, ln_y in path_lanes_list
    ]
    
    # Count actual distinct rows
    vrow_columns = set()
    hrow_y_positions = set()
    
    # Determine row types for original segments
    row_info = []
    vrow_counter = 1
    hrow_counter = 1
    
    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]
        lx2, ly2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        
        if lx1 == lx2 and ly1 != ly2:  # Vertical movement
            if is_sown:
                row_type = f"VRow{vrow_counter}"
                vrow_columns.add(lx1)
                vrow_counter += 1
            else:
                row_type = "Transition"
        elif ly1 == ly2 and lx1 != lx2:  # Horizontal movement
            if is_sown:
                row_type = f"HRow{hrow_counter}"
                hrow_y_positions.add(ly1)
                hrow_counter += 1
            else:
                row_type = "Transition"
        else:
            row_type = "Turn"
        
        row_info.append({
            'type': row_type,
            'is_sown': is_sown,
            'start_lane': (lx1, ly1),
            'end_lane': (lx2, ly2)
        })
    
    # Analyze each detailed waypoint
    cumulative_distance = 0.0
    
    for idx, (x, y) in enumerate(detailed_waypoints):
        # Find which original segment this waypoint belongs to
        segment_info = find_waypoint_segment(x, y, original_points, row_info)
        
        # Calculate distance from previous waypoint
        if idx == 0:
            distance_from_previous = 0.0
        else:
            prev_x, prev_y = detailed_waypoints[idx - 1]
            distance_from_previous = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
            
            # Handle potential infinity/NaN values
            if not isinstance(distance_from_previous, (int, float)) or distance_from_previous != distance_from_previous:
                distance_from_previous = 0.0
            elif distance_from_previous == float('inf'):
                distance_from_previous = 0.0
            
            cumulative_distance += distance_from_previous
        
        # NEW: Calculate movement command
        command = calculate_movement_command(idx, detailed_waypoints)
        
        # Ensure all numeric values are valid
        x_coord = float(x) if isinstance(x, (int, float)) and x == x else 0.0
        y_coord = float(y) if isinstance(y, (int, float)) and y == y else 0.0
        
        # Calculate lane coordinates safely
        lane_x = int(x_coord / rover_width_m) if rover_width_m > 0 else 0
        lane_y = int(y_coord / rover_length_m) if rover_length_m > 0 else 0
        
        waypoint_analysis.append({
            'id': idx + 1,
            'coordinates': {
                'x': round(x_coord, 3),
                'y': round(y_coord, 3)
            },
            'lane_coordinates': {
                'x': lane_x,
                'y': lane_y
            },
            'row_type': segment_info['type'],
            'is_sown': segment_info['is_sown'],
            'distance_from_previous': round(distance_from_previous, 3),
            'cumulative_distance': round(cumulative_distance, 3),
            'action': 'SOW' if segment_info['is_sown'] else 'MOVE',
            'command': command,  # NEW: Movement command
            'total_vrows': len(vrow_columns),
            'total_hrows': len(hrow_y_positions)
        })
    
    return waypoint_analysis

def calculate_movement_command(idx, waypoints):
    """
    Calculate the movement command for a waypoint based on direction changes
    """
    if idx == 0:
        return "START"
    
    if idx >= len(waypoints) - 1:
        return "STOP"
    
    # Get current, previous, and next waypoints
    prev_x, prev_y = waypoints[idx - 1]
    curr_x, curr_y = waypoints[idx]
    
    # For the last waypoint, just check previous direction
    if idx == len(waypoints) - 1:
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        if abs(dx) > abs(dy):
            return "MOVE EAST" if dx > 0 else "MOVE WEST"
        else:
            return "MOVE NORTH" if dy > 0 else "MOVE SOUTH"
    
    next_x, next_y = waypoints[idx + 1]
    
    # Calculate direction vectors
    prev_dx = curr_x - prev_x
    prev_dy = curr_y - prev_y
    next_dx = next_x - curr_x
    next_dy = next_y - curr_y
    
    # Normalize small movements to avoid noise
    threshold = 0.001
    if abs(prev_dx) < threshold: prev_dx = 0
    if abs(prev_dy) < threshold: prev_dy = 0
    if abs(next_dx) < threshold: next_dx = 0
    if abs(next_dy) < threshold: next_dy = 0
    
    # Determine previous and next directions
    prev_direction = get_direction(prev_dx, prev_dy)
    next_direction = get_direction(next_dx, next_dy)
    
    # If directions are the same, it's straight movement
    if prev_direction == next_direction:
        if next_direction == "NORTH":
            return "MOVE STRAIGHT NORTH"
        elif next_direction == "SOUTH":
            return "MOVE STRAIGHT SOUTH"
        elif next_direction == "EAST":
            return "MOVE STRAIGHT EAST"
        elif next_direction == "WEST":
            return "MOVE STRAIGHT WEST"
        else:
            return "MOVE STRAIGHT"
    
    # Determine turn direction
    turn_type = get_turn_type(prev_direction, next_direction)
    return f"TURN {turn_type} TO {next_direction}"

def get_direction(dx, dy):
    """
    Get cardinal direction from movement vector
    """
    if abs(dx) > abs(dy):
        return "EAST" if dx > 0 else "WEST"
    else:
        return "NORTH" if dy > 0 else "SOUTH"

def get_turn_type(from_dir, to_dir):
    """
    Determine if it's a left or right turn
    """
    # Define clockwise direction order
    directions = ["NORTH", "EAST", "SOUTH", "WEST"]
    
    try:
        from_idx = directions.index(from_dir)
        to_idx = directions.index(to_dir)
        
        # Calculate turn direction
        diff = (to_idx - from_idx) % 4
        
        if diff == 1:
            return "RIGHT"
        elif diff == 3:
            return "LEFT"
        elif diff == 2:
            return "AROUND"  # 180-degree turn
        else:
            return "STRAIGHT"
    except ValueError:
        return "UNKNOWN"


def find_waypoint_segment(x, y, original_points, row_info):
    """
    Find which segment a waypoint belongs to
    """
    min_distance = float('inf')
    best_segment = {'type': 'Unknown', 'is_sown': False}
    
    for i in range(len(original_points) - 1):
        x1, y1 = original_points[i]
        x2, y2 = original_points[i + 1]
        
        # Calculate distance from waypoint to line segment
        distance = point_to_line_distance(x, y, x1, y1, x2, y2)
                
        # Handle potential infinity values
        if not isinstance(distance, (int, float)) or distance != distance:
            distance = float('inf')
        
        if distance < min_distance and distance != float('inf'):
            min_distance = distance
            if i < len(row_info):
                best_segment = row_info[i]
    
    return best_segment

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Calculate distance from point to line segment
    """
    try:
        # Vector from point 1 to point 2
        dx = x2 - x1
        dy = y2 - y1
        
        # If the line segment is actually a point
        if dx == 0 and dy == 0:
            return ((px - x1)**2 + (py - y1)**2)**0.5
        
        # Calculate the parameter t
        denominator = dx * dx + dy * dy
        if denominator == 0:
            return ((px - x1)**2 + (py - y1)**2)**0.5
            
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / denominator))
        
        # Find the closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Return distance to closest point
        distance = ((px - closest_x)**2 + (py - closest_y)**2)**0.5
        
        # Ensure we return a valid number
        if not isinstance(distance, (int, float)) or distance != distance:
            return 0.0
        elif distance == float('inf'):
            return 0.0
        else:
            return distance
            
    except (ZeroDivisionError, ValueError, TypeError):
        # Return a safe default value if any calculation fails
        return 0.0

def generate_plot(path_lanes_list, sow_flags_list, exit_point_lanes,
                 farm_width_m, farm_breadth_m, rover_width_m, rover_length_m,
                 max_lane_idx_x, max_lane_idx_y, n_inner_x_sweeps,
                 detailed_waypoints=None):
    
    # Convert lane coordinates to metric centers (for path drawing)
    path_metric_centers = [
        ((ln_x + 0.5) * rover_width_m, (ln_y + 0.5) * rover_length_m) 
        for ln_x, ln_y in path_lanes_list
    ]
    
    # Use detailed waypoints if provided, otherwise use original
    waypoints_to_draw = detailed_waypoints if detailed_waypoints else path_metric_centers
    
    exit_metric_center = (
        (exit_point_lanes[0] + 0.5) * rover_width_m,
        (exit_point_lanes[1] + 0.5) * rover_length_m
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    plot_padding_m = max(rover_width_m, rover_length_m) * 0.5
    ax.set_xlim(-plot_padding_m, farm_width_m + plot_padding_m)
    ax.set_ylim(-plot_padding_m, farm_breadth_m + plot_padding_m)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f'Farm Path Plan: {farm_width_m}×{farm_breadth_m}m, Rover: {rover_width_m}×{rover_length_m}m', 
                fontsize=14, pad=20)
    
    # Draw farm boundary
    ax.add_patch(Rectangle((0, 0), farm_width_m, farm_breadth_m, 
                          fill=False, edgecolor='darkgray', lw=2, zorder=1))
    
    # Draw path segments
    single_brown_color = '#8B4513'

    # First, draw all path segments as brown (unsown trace)
    for i in range(len(path_metric_centers) - 1):
        x1, y1 = path_metric_centers[i]
        x2, y2 = path_metric_centers[i + 1]
        
        if x1 == x2:  # Vertical movement
            rect_x = x1 - rover_width_m / 2
            rect_y = min(y1, y2) - rover_length_m / 2
            rect_width = rover_width_m
            rect_height = abs(y2 - y1) + rover_length_m
        else:  # Horizontal movement
            rect_x = min(x1, x2) - rover_width_m / 2
            rect_y = y1 - rover_length_m / 2
            rect_width = abs(x2 - x1) + rover_width_m
            rect_height = rover_length_m
        
        # Draw brown background for all paths
        path_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                            color=single_brown_color, alpha=0.4, zorder=2)
        ax.add_patch(path_rect)

    # Then, draw green rectangles over sown segments
    for i in range(len(path_metric_centers) - 1):
        if i < len(sow_flags_list) and sow_flags_list[i]:
            x1, y1 = path_metric_centers[i]
            x2, y2 = path_metric_centers[i + 1]
            
            if x1 == x2:  # Vertical movement
                rect_x = x1 - rover_width_m / 2
                rect_y = min(y1, y2) - rover_length_m / 2
                rect_width = rover_width_m
                rect_height = abs(y2 - y1) + rover_length_m
            else:  # Horizontal movement
                rect_x = min(x1, x2) - rover_width_m / 2
                rect_y = y1 - rover_length_m / 2
                rect_width = abs(x2 - x1) + rover_width_m
                rect_height = rover_length_m
            
            # Draw green overlay for sown areas
            sown_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                                color='#006400', alpha=0.8, zorder=3)
            ax.add_patch(sown_rect)

    # Add detailed waypoints as small directional arrows
    arrow_size = min(rover_width_m, rover_length_m) * 0.05

    # Show directional arrows for waypoints (every few waypoints to avoid clutter)
    step_size = max(1, len(waypoints_to_draw) // 100)

    for i in range(0, len(waypoints_to_draw) - 1, step_size):
        x1, y1 = waypoints_to_draw[i]
        x2, y2 = waypoints_to_draw[i + 1] if i + 1 < len(waypoints_to_draw) else waypoints_to_draw[i]
        
        # Calculate direction
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize direction (avoid division by zero)
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            dx_norm = dx / length * arrow_size
            dy_norm = dy / length * arrow_size
            
            # Draw small red directional arrow
            ax.annotate('', xy=(x1 + dx_norm, y1 + dy_norm), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, 
                                    alpha=0.8, shrinkA=0, shrinkB=0),
                    zorder=6)
        else:
            # For stationary points, draw a small red dot
            ax.plot(x1, y1, 'ro', markersize=2, alpha=0.8, zorder=6)

    # Add waypoint arrows showing direction (keep using original points for cleaner arrows)
    for i in range(len(path_metric_centers) - 1):
        x1, y1 = path_metric_centers[i]
        x2, y2 = path_metric_centers[i + 1]
        
        # Calculate arrow direction
        dx = x2 - x1
        dy = y2 - y1
        
        # Draw arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
                zorder=5)

    # Add row labels
    add_row_labels(ax, path_lanes_list, sow_flags_list, rover_width_m, rover_length_m, 
                  plot_padding_m, max_lane_idx_y)
    
    # Add start marker (rectangular rover shape)
    start_center = path_metric_centers[0]
    rover_rect_width = rover_width_m
    rover_rect_height = rover_length_m
    rover_start_x = start_center[0] - rover_rect_width/2
    rover_start_y = start_center[1] - rover_rect_height/2

    ax.add_patch(Rectangle((rover_start_x, rover_start_y), rover_rect_width, rover_rect_height,
                        color='orange', fill=True, lw=1.5, zorder=4,
                        edgecolor='black'))
    
    # Add exit gate
    add_exit_gate(ax, exit_point_lanes, exit_metric_center, rover_width_m, rover_length_m,
                 farm_width_m, farm_breadth_m, max_lane_idx_x, max_lane_idx_y)
    
    # Add legend with detailed waypoint info
    legend_handles = [
        Line2D([0], [0], c=single_brown_color, lw=10, alpha=0.4, 
               label=f'Unsown Path'),
        Line2D([0], [0], c='#006400', lw=10, alpha=0.8, 
               label=f'Sown Area'),
        Rectangle((0, 0), 1, 1, fc='orange', ec='black',
               label=f'🤖 Rover ({start_center[0]:.1f}, {start_center[1]:.1f})m'),
        Line2D([0], [0], color='red', marker='>', ms=4, ls='None',
               label=f'Waypoint Arrows ({len(waypoints_to_draw)} total)'),
        Line2D([0], [0], color='red', lw=2, alpha=0.7,
            label='Path Direction →'),
        Rectangle((0, 0), 1, 1, fc='red', ec='darkred', 
               label=f'EXIT ({exit_metric_center[0]:.1f}, {exit_metric_center[1]:.1f})m')
    ]
    
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.28, 1.02), fontsize=8)
    plt.subplots_adjust(right=0.75)
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def add_row_labels(ax, path_lanes_list, sow_flags_list, rover_width_m, rover_length_m, 
                  plot_padding_m, max_lane_idx_y):
    # Add VRow labels
    vrow_columns = set()
    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]
        lx2, ly2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        
        if lx1 == lx2 and ly1 != ly2 and is_sown:
            vrow_columns.add(lx1)
    
    sorted_vrow_columns = sorted(vrow_columns)
    for i, column_x in enumerate(sorted_vrow_columns):
        tx_m = (column_x + 0.5) * rover_width_m
        ty_m = -plot_padding_m * 0.7
        
        label = f'VRow{i+1}'
        ax.text(tx_m, ty_m, label, fontsize=10, color='navy',
               ha='center', va='center', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', 
                        alpha=0.9, ec='navy', lw=2))
    
    # Add HRow labels
    horizontal_y_positions = set()
    for i in range(len(path_lanes_list) - 1):
        x1, y1 = path_lanes_list[i]
        x2, y2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        
        if y1 == y2 and x1 != x2 and is_sown:
            horizontal_y_positions.add(y1)
    
    sorted_horizontal_positions = sorted(horizontal_y_positions)
    
    for i, y_pos in enumerate(sorted_horizontal_positions):
        tx_m = -plot_padding_m * 0.7
        ty_m = (y_pos + 0.5) * rover_length_m
        
        label = f'HRow{i+1}'
        ax.text(tx_m, ty_m, label, fontsize=10, color='darkgreen',
               ha='center', va='center', weight='bold', rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', 
                        alpha=0.9, ec='darkgreen', lw=2))

def add_exit_gate(ax, exit_point_lanes, exit_metric_center, rover_width_m, rover_length_m,
                 farm_width_m, farm_breadth_m, max_lane_idx_x, max_lane_idx_y):
    gate_width = rover_width_m * 0.8
    gate_height = rover_length_m * 0.3
    
    exit_x, exit_y = exit_point_lanes
    
    if exit_x == 0:  # Left border
        gate_x = -gate_height/2
        gate_y = exit_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_x == max_lane_idx_x:  # Right border
        gate_x = farm_width_m - gate_height/2
        gate_y = exit_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_y == 0:  # Bottom border
        gate_x = exit_metric_center[0] - gate_width/2
        gate_y = -gate_height/2
        gate_w, gate_h = gate_width, gate_height
    else:  # Top border
        gate_x = exit_metric_center[0] - gate_width/2
        gate_y = farm_breadth_m - gate_height/2
        gate_w, gate_h = gate_width, gate_height
    
    exit_gate = Rectangle((gate_x, gate_y), gate_w, gate_h,
                         color='red', alpha=0.8, zorder=5,
                         edgecolor='darkred', linewidth=2)
    ax.add_patch(exit_gate)

@app.route('/download_csv')
def download_csv():
    csv_filename = "navigation_log.csv"
    if os.path.exists(csv_filename):
        return send_file(csv_filename, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


