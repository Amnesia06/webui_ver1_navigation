import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import csv
from datetime import datetime
import os
import math

# --- Helper Functions ---
SOWN_SEGMENTS_LOG = set()

def _commit_point_to_path(points_list_lanes, sow_flags_list, new_lane_point, sow_flag_requested, context=""):
    # new_lane_point is (lane_x, lane_y)
    if not points_list_lanes:
        points_list_lanes.append(new_lane_point)
        return
    if points_list_lanes[-1] == new_lane_point:
        return

    previous_lane_point = points_list_lanes[-1]
    current_segment = frozenset({previous_lane_point, new_lane_point})
    actual_sow_flag_for_this_segment = False

    if sow_flag_requested:
        if current_segment not in SOWN_SEGMENTS_LOG:
            actual_sow_flag_for_this_segment = True
            SOWN_SEGMENTS_LOG.add(current_segment)
    points_list_lanes.append(new_lane_point)
    sow_flags_list.append(actual_sow_flag_for_this_segment)

def _add_headland_segment_custom_exit(current_lane_x, current_lane_y, 
                                     target_lane_x, target_lane_y, 
                                     exit_point_lanes, 
                                     _points_list_lanes, _sow_flags_list, 
                                     segment_label="",
                                     is_designated_unsown_positioning_leg=False,
                                     gap_size=1):
    ex_lane_x, ey_lane_y = exit_point_lanes
    if (current_lane_x, current_lane_y) == exit_point_lanes:
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 1")
        return target_lane_x, target_lane_y

    on_segment_path = False
    if current_lane_x == target_lane_x == ex_lane_x and min(current_lane_y, target_lane_y) <= ey_lane_y <= max(current_lane_y, target_lane_y):
        on_segment_path = True
    elif current_lane_y == target_lane_y == ey_lane_y and min(current_lane_x, target_lane_x) <= ex_lane_x <= max(current_lane_x, target_lane_x):
        on_segment_path = True

    if on_segment_path:
        # Calculate stop point before exit to leave gap
        if current_lane_x == target_lane_x == ex_lane_x:  # Vertical movement
            if current_lane_y < ey_lane_y:  # Moving upward
                sow_stop_y = max(current_lane_y, ey_lane_y - gap_size)
            else:  # Moving downward
                sow_stop_y = min(current_lane_y, ey_lane_y + gap_size)
            
            # Sow until stop point
            if sow_stop_y != current_lane_y:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, sow_stop_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
                
        elif current_lane_y == target_lane_y == ey_lane_y:  # Horizontal movement
            if current_lane_x < ex_lane_x:  # Moving rightward
                sow_stop_x = max(current_lane_x, ex_lane_x - gap_size)
            else:  # Moving leftward
                sow_stop_x = min(current_lane_x, ex_lane_x + gap_size)
            
            # Sow until stop point
            if sow_stop_x != current_lane_x:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (sow_stop_x, ey_lane_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
    else: 
        sow_request = not is_designated_unsown_positioning_leg
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3")
            
    return target_lane_x, target_lane_y

def get_user_choice_corner_lanes(max_lx_idx, max_ly_idx):
    print(f"\nChoose exit corner (0-indexed lanes: X up to {max_lx_idx}, Y up to {max_ly_idx}):")
    print(f"1. Top-Left Lane (0, {max_ly_idx})")
    print(f"2. Top-Right Lane ({max_lx_idx}, {max_ly_idx})")
    print(f"3. Bottom-Left Lane (0, 0)")
    print(f"4. Bottom-Right Lane ({max_lx_idx}, 0)")
    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if choice == 1: return (0, max_ly_idx)
            if choice == 2: return (max_lx_idx, max_ly_idx)
            if choice == 3: return (0, 0)
            if choice == 4: return (max_lx_idx, 0)
        except ValueError: print("Invalid input.")

def get_user_defined_exit_lanes(max_lx_idx, max_ly_idx):
    print(f"\nDefine custom exit lane (X: 0-{max_lx_idx}, Y: 0-{max_ly_idx}):")
    print(f"Note: Corner positions are excluded (available in Fixed Corner Exit option)")
    print(f"1. Top boundary (Y lane = {max_ly_idx}) - excluding corners")
    print(f"2. Bottom boundary (Y lane = 0) - excluding corners") 
    print(f"3. Left boundary (X lane = 0) - excluding corners")
    print(f"4. Right boundary (X lane = {max_lx_idx}) - excluding corners")
    while True:
        try:
            b_choice = int(input("Choose boundary for exit (1-4): "))
            if 1 <= b_choice <= 4: break
        except ValueError: print("Invalid input.")
    
    ex_l, ey_l = -1, -1
    while True:
        try:
            if b_choice == 1:  # Top boundary - exclude corners (0,max_y) and (max_x,max_y)
                ey_l = max_ly_idx
                ex_l = int(input(f"Enter X-lane (1-{max_lx_idx-1}, excluding corners): "))
                assert 1 <= ex_l <= max_lx_idx-1
                break
            elif b_choice == 2:  # Bottom boundary - exclude corners (0,0) and (max_x,0)
                ey_l = 0
                ex_l = int(input(f"Enter X-lane (1-{max_lx_idx-1}, excluding corners): "))
                assert 1 <= ex_l <= max_lx_idx-1
                break
            elif b_choice == 3:  # Left boundary - exclude corners (0,0) and (0,max_y)
                ex_l = 0
                ey_l = int(input(f"Enter Y-lane (1-{max_ly_idx-1}, excluding corners): "))
                assert 1 <= ey_l <= max_ly_idx-1
                break
            elif b_choice == 4:  # Right boundary - exclude corners (max_x,0) and (max_x,max_y)
                ex_l = max_lx_idx
                ey_l = int(input(f"Enter Y-lane (1-{max_ly_idx-1}, excluding corners): "))
                assert 1 <= ey_l <= max_ly_idx-1
                break
        except (ValueError, AssertionError): 
            print("Invalid lane index. Must be non-corner boundary position.")
    return (ex_l, ey_l)


def _commit_partial_vertical_sweep(points_list, sow_flags_list, curr_x, start_y, end_y, gap_size=1):
    """
    Creates a partial vertical sweep with gaps at both ends.
    
    Args:
        points_list: List of lane points
        sow_flags_list: List of sowing flags
        curr_x: Current X lane position
        start_y: Starting Y lane position
        end_y: Ending Y lane position  
        gap_size: Size of gap to leave at each end (in lane units)
    """
    if start_y == end_y:
        return end_y
    
    # Determine direction and calculate total distance
    direction = 1 if end_y > start_y else -1
    total_distance = abs(end_y - start_y)
    
    # If total distance is too small for meaningful gaps, do full sown movement
    if total_distance <= 2 * gap_size:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), True, "PartialSweep_FullSown")
        return end_y
    
    # Calculate sowing start and end positions
    # Leave gap_size at the beginning and end
    sow_start_y = start_y + (gap_size * direction)
    sow_end_y = end_y - (gap_size * direction)
    
    # Phase 1: Unsown movement to sowing start position (creating initial gap)
    if sow_start_y != start_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, sow_start_y), False, "PartialSweep_Phase1_Gap")
    
    # Phase 2: Sown movement (main productive sweep)
    _commit_point_to_path(points_list, sow_flags_list, (curr_x, sow_end_y), True, "PartialSweep_Phase2_Sown")
    
    # Phase 3: Unsown movement to final position (creating final gap)
    if sow_end_y != end_y:
        _commit_point_to_path(points_list, sow_flags_list, (curr_x, end_y), False, "PartialSweep_Phase3_Gap")
    
    return end_y

# --- Path Generation (Operates in 0-indexed Lane Numbers) ---
# Updated helper function to properly handle gap_size parameter
def _add_headland_segment_custom_exit_with_gaps(current_lane_x, current_lane_y, 
                                     target_lane_x, target_lane_y, 
                                     exit_point_lanes, 
                                     _points_list_lanes, _sow_flags_list, 
                                     segment_label="",
                                     is_designated_unsown_positioning_leg=False,
                                     gap_size=1):
    ex_lane_x, ey_lane_y = exit_point_lanes
    if (current_lane_x, current_lane_y) == exit_point_lanes:
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 1")
        return target_lane_x, target_lane_y

    on_segment_path = False
    if current_lane_x == target_lane_x == ex_lane_x and min(current_lane_y, target_lane_y) <= ey_lane_y <= max(current_lane_y, target_lane_y):
        on_segment_path = True
    elif current_lane_y == target_lane_y == ey_lane_y and min(current_lane_x, target_lane_x) <= ex_lane_x <= max(current_lane_x, target_lane_x):
        on_segment_path = True

    if on_segment_path:
        # [Existing gap logic when exit is on the path - keep unchanged]
        # Calculate stop point before exit to leave gap
        if current_lane_x == target_lane_x == ex_lane_x:  # Vertical movement
            if current_lane_y < ey_lane_y:  # Moving upward
                sow_stop_y = max(current_lane_y, ey_lane_y - gap_size)
            else:  # Moving downward
                sow_stop_y = min(current_lane_y, ey_lane_y + gap_size)
            
            # Sow until stop point
            if sow_stop_y != current_lane_y:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, sow_stop_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
                
        elif current_lane_y == target_lane_y == ey_lane_y:  # Horizontal movement
            if current_lane_x < ex_lane_x:  # Moving rightward
                sow_stop_x = max(current_lane_x, ex_lane_x - gap_size)
            else:  # Moving leftward
                sow_stop_x = min(current_lane_x, ex_lane_x + gap_size)
            
            # Sow until stop point
            if sow_stop_x != current_lane_x:
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (sow_stop_x, ey_lane_y), True, f"AHLCE {segment_label} Case 2 SowToGap")
            
            # Unsown movement to exit
            _commit_point_to_path(_points_list_lanes, _sow_flags_list, (ex_lane_x, ey_lane_y), False, f"AHLCE {segment_label} Case 2 ToExit")
            
            # Continue unsown to target if needed
            if (ex_lane_x, ey_lane_y) != (target_lane_x, target_lane_y):
                _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), False, f"AHLCE {segment_label} Case 2 PastExit")
    else: 
        # Exit not on this segment - FIXED: Always sow perimeter segments regardless of gap size
        sow_request = not is_designated_unsown_positioning_leg
        _commit_point_to_path(_points_list_lanes, _sow_flags_list, (target_lane_x, target_lane_y), sow_request, f"AHLCE {segment_label} Case 3_Fixed")
            
    return target_lane_x, target_lane_y


def generate_fixed_path(n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, exit_point_lanes, is_corner_exit, gap_size=1):
    global SOWN_SEGMENTS_LOG
    SOWN_SEGMENTS_LOG.clear()

    _points_lanes = [] # Stores (lane_x, lane_y)
    _sow_flags = []
    ex_ln_x, ex_ln_y = exit_point_lanes # Exit point in lane indices
    
    def _commit(new_ln_pt, sow_req, ctx=""):
        _commit_point_to_path(_points_lanes, _sow_flags, new_ln_pt, sow_req, ctx)

    # [Previous initialization code remains the same...]
    # Determine start_lane_y for boustrophedon (0 or max_lane_idx_y)
    natural_start_ln_y = 0
    if ex_ln_y == max_lane_idx_y: natural_start_ln_y = max_lane_idx_y
    elif ex_ln_y == 0: natural_start_ln_y = 0
    else:
        natural_start_ln_y = max_lane_idx_y if abs(ex_ln_y - max_lane_idx_y) < abs(ex_ln_y - 0) else 0
    
    b_start_lane_y = natural_start_ln_y
    if n_inner_x_sweeps % 2 != 0: # If odd number of inner sweeps, flip start Y
        b_start_lane_y = 0 if natural_start_ln_y == max_lane_idx_y else max_lane_idx_y
    
    # Determine start_lane_x for boustrophedon (inner lanes are 1 to max_lane_idx_x - 1)
    start_sweep_lane_x = -1
    lanes_to_sweep_x = [] # List of inner lane X-indices (1 to max_lane_idx_x - 1)
    
    first_inner_lane_x = 1
    last_inner_lane_x = max_lane_idx_x - 1

    if n_inner_x_sweeps == 0:
        start_sweep_lane_x = 0 if ex_ln_x <= max_lane_idx_x / 2.0 else max_lane_idx_x

    if is_corner_exit:
        if n_inner_x_sweeps > 0:
            start_sweep_lane_x = last_inner_lane_x if ex_ln_x <= max_lane_idx_x / 2.0 else first_inner_lane_x
        
        initial_sweep_dir = -1 if start_sweep_lane_x == last_inner_lane_x else 1
        end_col_for_range = (first_inner_lane_x -1) if initial_sweep_dir == -1 else (last_inner_lane_x + 1)
        if n_inner_x_sweeps > 0:
             lanes_to_sweep_x = list(range(start_sweep_lane_x, end_col_for_range, initial_sweep_dir))
    else: # Custom Exit - Different starting point based on exit side
        if n_inner_x_sweeps == 0:
            start_sweep_lane_x = 0
        else:
            # Different starting point and direction based on exit side
            if ex_ln_x == 0:  # LEFT BOUNDARY EXIT - start from VRow4 (rightmost inner)
                start_sweep_lane_x = last_inner_lane_x  # Start from VRow4
                if n_inner_x_sweeps == 1:
                    lanes_to_sweep_x = [start_sweep_lane_x]
                else:
                    # REVERSE order: VRow4 → VRow3 → VRow2 → VRow1
                    lanes_to_sweep_x = list(range(last_inner_lane_x, last_inner_lane_x - n_inner_x_sweeps, -1))
            elif ex_ln_x == max_lane_idx_x:  # RIGHT BOUNDARY EXIT - start from VRow1 (leftmost inner)
                start_sweep_lane_x = first_inner_lane_x  # Start from VRow1
                if n_inner_x_sweeps == 1:
                    lanes_to_sweep_x = [start_sweep_lane_x]
                else:
                    # SEQUENTIAL order: VRow1 → VRow2 → VRow3 → VRow4
                    lanes_to_sweep_x = list(range(first_inner_lane_x, first_inner_lane_x + n_inner_x_sweeps))
            elif ex_ln_y == max_lane_idx_y or ex_ln_y == 0:  # TOP or BOTTOM BOUNDARY EXIT
                if ex_ln_x <= max_lane_idx_x / 2:  # Exit is on left half
                    # Start from VRow3 (rightmost/FARTHEST) and work leftward: VRow3 → VRow2 → VRow1
                    start_sweep_lane_x = last_inner_lane_x  # Start from farthest VRow
                    if n_inner_x_sweeps == 1:
                        lanes_to_sweep_x = [start_sweep_lane_x]
                    else:
                        lanes_to_sweep_x = list(range(last_inner_lane_x, last_inner_lane_x - n_inner_x_sweeps, -1))
                else:  # Exit is on right half  
                    # Start from VRow1 (leftmost/FARTHEST) and work rightward: VRow1 → VRow2 → VRow3
                    start_sweep_lane_x = first_inner_lane_x  # Start from farthest VRow
                    if n_inner_x_sweeps == 1:
                        lanes_to_sweep_x = [start_sweep_lane_x]
                    else:
                        lanes_to_sweep_x = list(range(first_inner_lane_x, first_inner_lane_x + n_inner_x_sweeps))


    initial_pos_lane_x = start_sweep_lane_x if n_inner_x_sweeps > 0 else (0 if ex_ln_x <= max_lane_idx_x / 2.0 else max_lane_idx_x)
    _points_lanes.append((initial_pos_lane_x, b_start_lane_y))
    curr_ln_x, curr_ln_y = initial_pos_lane_x, b_start_lane_y

    # Inner vertical sweeps
    for i, sweep_ln_x in enumerate(lanes_to_sweep_x):
        curr_ln_x = sweep_ln_x
        target_ln_y = 0 if curr_ln_y == max_lane_idx_y else max_lane_idx_y
        
        curr_ln_y = _commit_partial_vertical_sweep(_points_lanes, _sow_flags, curr_ln_x, curr_ln_y, target_ln_y, gap_size)

        if i < len(lanes_to_sweep_x) - 1:
            next_sweep_lane_x = lanes_to_sweep_x[i+1]
            _commit((next_sweep_lane_x, curr_ln_y), False, f"InnerSweep H-Turn{i+1}")
            curr_ln_x = next_sweep_lane_x

    # BRANCHING LOGIC
    # Replace the existing corner exit logic (around lines 280-320) with this:

    if is_corner_exit:
        # EFFICIENT CORNER EXIT LOGIC - Stop inner sweeps one row earlier and turn RIGHT
        
        # Determine efficient perimeter sequence based on exit corner and current position
        ex_ln_x, ex_ln_y = exit_point_lanes
        
        if ex_ln_x == 0 and ex_ln_y == max_lane_idx_y:  # TOP-LEFT corner exit (0, max_y)
            if curr_ln_y == 0:  # Currently at bottom after inner sweeps
                # Efficient sequence: HRow2(up) → VRow5(right) → HRow1(down) → VRow1(to exit)
                _commit((curr_ln_x, max_lane_idx_y), True, "HRow2_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((max_lane_idx_x, curr_ln_y), True, "VRow5_EfficientRight") 
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, 0), True, "HRow1_EfficientDown")
                curr_ln_y = 0
                _commit((0, curr_ln_y), True, "VRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, ex_ln_y), True, "VRow1_ToExit")
            else:  # Currently at top after inner sweeps
                # Efficient sequence: HRow2(right) → VRow5(down) → HRow1(left) → VRow1(to exit)
                _commit((max_lane_idx_x, curr_ln_y), True, "HRow2_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, 0), True, "VRow5_EfficientDown")
                curr_ln_y = 0
                _commit((0, curr_ln_y), True, "HRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, ex_ln_y), True, "VRow1_ToExit")
        
        elif ex_ln_x == max_lane_idx_x and ex_ln_y == max_lane_idx_y:  # TOP-RIGHT corner exit
            if curr_ln_y == 0:  # Currently at bottom after inner sweeps
                # Efficient sequence: HRow2(up) → VRow1(left) → HRow1(down) → VRow5(to exit)
                _commit((curr_ln_x, max_lane_idx_y), True, "HRow2_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((0, curr_ln_y), True, "VRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, 0), True, "HRow1_EfficientDown")
                curr_ln_y = 0
                _commit((max_lane_idx_x, curr_ln_y), True, "VRow5_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, ex_ln_y), True, "VRow5_ToExit")
            else:  # Currently at top after inner sweeps
                # Efficient sequence: HRow2(left) → VRow1(down) → HRow1(right) → VRow5(to exit)
                _commit((0, curr_ln_y), True, "HRow2_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, 0), True, "VRow1_EfficientDown")
                curr_ln_y = 0
                _commit((max_lane_idx_x, curr_ln_y), True, "HRow1_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, ex_ln_y), True, "VRow5_ToExit")
        
        elif ex_ln_x == 0 and ex_ln_y == 0:  # BOTTOM-LEFT corner exit
            if curr_ln_y == max_lane_idx_y:  # Currently at top after inner sweeps
                # Efficient sequence: HRow1(down) → VRow5(right) → HRow2(up) → VRow1(to exit)
                _commit((curr_ln_x, 0), True, "HRow1_EfficientDown")
                curr_ln_y = 0
                _commit((max_lane_idx_x, curr_ln_y), True, "VRow5_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, max_lane_idx_y), True, "HRow2_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((0, curr_ln_y), True, "VRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, ex_ln_y), True, "VRow1_ToExit")
            else:  # Currently at bottom after inner sweeps
                # Efficient sequence: HRow1(right) → VRow5(up) → HRow2(left) → VRow1(to exit)
                _commit((max_lane_idx_x, curr_ln_y), True, "HRow1_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, max_lane_idx_y), True, "VRow5_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((0, curr_ln_y), True, "HRow2_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, ex_ln_y), True, "VRow1_ToExit")
        
        else:  # BOTTOM-RIGHT corner exit (max_x, 0)
            if curr_ln_y == max_lane_idx_y:  # Currently at top after inner sweeps
                # Efficient sequence: HRow1(down) → VRow1(left) → HRow2(up) → VRow5(to exit)
                _commit((curr_ln_x, 0), True, "HRow1_EfficientDown")
                curr_ln_y = 0
                _commit((0, curr_ln_y), True, "VRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, max_lane_idx_y), True, "HRow2_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((max_lane_idx_x, curr_ln_y), True, "VRow5_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, ex_ln_y), True, "VRow5_ToExit")
            else:  # Currently at bottom after inner sweeps
                # Efficient sequence: HRow1(left) → VRow1(up) → HRow2(right) → VRow5(to exit)
                _commit((0, curr_ln_y), True, "HRow1_EfficientLeft")
                curr_ln_x = 0
                _commit((curr_ln_x, max_lane_idx_y), True, "VRow1_EfficientUp")
                curr_ln_y = max_lane_idx_y
                _commit((max_lane_idx_x, curr_ln_y), True, "HRow2_EfficientRight")
                curr_ln_x = max_lane_idx_x
                _commit((curr_ln_x, ex_ln_y), True, "VRow5_ToExit")
        
        # Final alignment to exact exit point
        if curr_ln_x != ex_ln_x or curr_ln_y != ex_ln_y:
            _commit((ex_ln_x, ex_ln_y), True, "FinalAlignmentToExit")




    
    else:
        # CORRECTED CUSTOM EXIT LOGIC: Stop exactly before exit, then retrace
        
        if ex_ln_x == max_lane_idx_x:  # RIGHT BOUNDARY EXIT (e.g., 5,2)
            print(f"DEBUG: VRow4 ended at ({curr_ln_x}, {curr_ln_y}), Exit at ({ex_ln_x}, {ex_ln_y})")
            
            # Step 1: Move unsown directly to exit X-coordinate
            if curr_ln_x != ex_ln_x:
                _commit((ex_ln_x, curr_ln_y), False, "CustomExit_MoveToExitColumn")
                curr_ln_x = ex_ln_x
                print(f"DEBUG: Moved to exit column: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 2: Calculate position just before exit (with gap) and STOP there
            if curr_ln_y != ex_ln_y:
                if ex_ln_y > curr_ln_y:  # Exit is above current position
                    stop_y = max(curr_ln_y, ex_ln_y - gap_size)
                else:  # Exit is below current position   
                    stop_y = min(curr_ln_y, ex_ln_y + gap_size)
                
                if stop_y != curr_ln_y:
                    _commit((curr_ln_x, stop_y), False, "CustomExit_StopBeforeExit")
                    curr_ln_y = stop_y
                    print(f"DEBUG: STOPPED just before exit at: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 3: Now retrace back and sow perimeter
            print(f"DEBUG: Starting retrace from: ({curr_ln_x}, {curr_ln_y})")
            
            # Determine which corner to go to first for retracing
            if curr_ln_y <= max_lane_idx_y / 2:
                # Go to bottom-right corner first
                retrace_corner_y = 0
            else:
                # Go to top-right corner first   
                retrace_corner_y = max_lane_idx_y
            
            # Retrace: Sow vertically to corner
            if curr_ln_y != retrace_corner_y:
                curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                    curr_ln_x, curr_ln_y, curr_ln_x, retrace_corner_y,
                    exit_point_lanes, _points_lanes, _sow_flags,
                    "CustomExit_RetraceToCorner", gap_size=gap_size)
                print(f"DEBUG: Retraced to corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal to left boundary
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, 0, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowHorizontalToLeft", gap_size=gap_size)
            print(f"DEBUG: Sowed to left boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical to opposite corner
            opposite_corner_y = max_lane_idx_y if retrace_corner_y == 0 else 0
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, opposite_corner_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalToOpposite", gap_size=gap_size)
            print(f"DEBUG: Sowed to opposite corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal back to right boundary
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, max_lane_idx_x, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowHorizontalToRight", gap_size=gap_size)
            print(f"DEBUG: Sowed back to right boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Final sowing approach to actual exit
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, ex_ln_x, ex_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_FinalSowToExit", gap_size=gap_size)
            print(f"DEBUG: Final approach to exit: ({curr_ln_x}, {curr_ln_y})")
              
        elif ex_ln_x == 0:  # LEFT BOUNDARY EXIT (opposite of right boundary)
            print(f"DEBUG: VRow4 ended at ({curr_ln_x}, {curr_ln_y}), Exit at ({ex_ln_x}, {ex_ln_y})")
            
            # Step 1: Move unsown directly to exit X-coordinate (left boundary = 0)
            if curr_ln_x != ex_ln_x:
                _commit((ex_ln_x, curr_ln_y), False, "CustomExit_MoveToExitColumn")
                curr_ln_x = ex_ln_x
                print(f"DEBUG: Moved to exit column: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 2: Calculate position just before exit (with gap) and STOP there
            if curr_ln_y != ex_ln_y:
                if ex_ln_y > curr_ln_y:  # Exit is above current position
                    stop_y = max(curr_ln_y, ex_ln_y - gap_size)
                else:  # Exit is below current position
                    stop_y = min(curr_ln_y, ex_ln_y + gap_size)
                
                if stop_y != curr_ln_y:
                    _commit((curr_ln_x, stop_y), False, "CustomExit_StopBeforeExit")
                    curr_ln_y = stop_y
                    print(f"DEBUG: STOPPED just before exit at: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 3: Now retrace back and sow perimeter (OPPOSITE direction of right boundary)
            print(f"DEBUG: Starting retrace from: ({curr_ln_x}, {curr_ln_y})")
            
            # Determine which corner to go to first for retracing (same logic as right)
            if curr_ln_y <= max_lane_idx_y / 2:
                retrace_corner_y = 0  # Go to bottom-left corner first
            else:
                retrace_corner_y = max_lane_idx_y  # Go to top-left corner first
            
            # Retrace: Sow vertically to corner (LEFT boundary vertical)
            if curr_ln_y != retrace_corner_y:
                curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                    curr_ln_x, curr_ln_y, curr_ln_x, retrace_corner_y,
                    exit_point_lanes, _points_lanes, _sow_flags,
                    "CustomExit_RetraceToCorner", gap_size=gap_size)
                print(f"DEBUG: Retraced to corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal to RIGHT boundary (OPPOSITE of right boundary logic)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, max_lane_idx_x, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowHorizontalToRight", gap_size=gap_size)
            print(f"DEBUG: Sowed to right boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical to opposite corner
            opposite_corner_y = max_lane_idx_y if retrace_corner_y == 0 else 0
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, opposite_corner_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalToOpposite", gap_size=gap_size)
            print(f"DEBUG: Sowed to opposite corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal back to LEFT boundary (OPPOSITE - back to exit side)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, 0, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowHorizontalToLeft", gap_size=gap_size)
            print(f"DEBUG: Sowed back to left boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Final sowing approach to actual exit
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, ex_ln_x, ex_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_FinalSowToExit", gap_size=gap_size)
            print(f"DEBUG: Final approach to exit: ({curr_ln_x}, {curr_ln_y})")
              
        elif ex_ln_y == max_lane_idx_y:  # TOP BOUNDARY EXIT
            print(f"DEBUG: VRow4 ended at ({curr_ln_x}, {curr_ln_y}), Exit at ({ex_ln_x}, {ex_ln_y})")
            
            # Step 1: Move unsown directly to exit Y-coordinate (positioning only)
            if curr_ln_y != ex_ln_y:
                _commit((curr_ln_x, ex_ln_y), False, "CustomExit_MoveToExitRow")
                curr_ln_y = ex_ln_y
                print(f"DEBUG: Moved to exit row: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 2: Calculate position just before exit (with gap) and STOP there (positioning only)
            if curr_ln_x != ex_ln_x:
                if ex_ln_x > curr_ln_x:  # Exit is to the right
                    stop_x = max(curr_ln_x, ex_ln_x - gap_size)
                else:  # Exit is to the left
                    stop_x = min(curr_ln_x, ex_ln_x + gap_size)
                
                if stop_x != curr_ln_x:
                    _commit((stop_x, curr_ln_y), False, "CustomExit_StopBeforeExit")
                    curr_ln_x = stop_x
                    print(f"DEBUG: STOPPED just before exit at: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 3: Now retrace back and sow perimeter (INCLUDING FARTHEST BOUNDARY)
            print(f"DEBUG: Starting retrace from: ({curr_ln_x}, {curr_ln_y})")
            
            # Determine which corner to go to first for retracing
            if curr_ln_x <= max_lane_idx_x / 2:
                retrace_corner_x = 0  # Go to top-left corner first
                farthest_boundary_x = max_lane_idx_x  # Farthest is right boundary
            else:
                retrace_corner_x = max_lane_idx_x  # Go to top-right corner first
                farthest_boundary_x = 0  # Farthest is left boundary
            
            # Retrace: Sow horizontally to corner
            if curr_ln_x != retrace_corner_x:
                curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                    curr_ln_x, curr_ln_y, retrace_corner_x, curr_ln_y,
                    exit_point_lanes, _points_lanes, _sow_flags,
                    "CustomExit_RetraceToCorner", gap_size=gap_size)
                print(f"DEBUG: Retraced to corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical to bottom boundary (FARTHEST HORIZONTAL BOUNDARY)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, 0,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalToFarthestBoundary", gap_size=gap_size)
            print(f"DEBUG: Sowed to farthest boundary (bottom): ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal to opposite corner (completing farthest boundary)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, farthest_boundary_x, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowFarthestBoundaryHorizontal", gap_size=gap_size)
            print(f"DEBUG: Sowed farthest boundary horizontal: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical back to top boundary
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, max_lane_idx_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalBackToTop", gap_size=gap_size)
            print(f"DEBUG: Sowed back to top boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Final sowing approach to actual exit
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, ex_ln_x, ex_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_FinalSowToExit", gap_size=gap_size)
            print(f"DEBUG: Final approach to exit: ({curr_ln_x}, {curr_ln_y})")

        else:  # BOTTOM BOUNDARY EXIT (ex_ln_y == 0)
            print(f"DEBUG: VRow4 ended at ({curr_ln_x}, {curr_ln_y}), Exit at ({ex_ln_x}, {ex_ln_y})")
            
            # Step 1: Move unsown directly to exit Y-coordinate (positioning only)
            if curr_ln_y != ex_ln_y:
                _commit((curr_ln_x, ex_ln_y), False, "CustomExit_MoveToExitRow")
                curr_ln_y = ex_ln_y
                print(f"DEBUG: Moved to exit row: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 2: Calculate position just before exit (with gap) and STOP there (positioning only)
            if curr_ln_x != ex_ln_x:
                if ex_ln_x > curr_ln_x:  # Exit is to the right
                    stop_x = max(curr_ln_x, ex_ln_x - gap_size)
                else:  # Exit is to the left
                    stop_x = min(curr_ln_x, ex_ln_x + gap_size)
                
                if stop_x != curr_ln_x:
                    _commit((stop_x, curr_ln_y), False, "CustomExit_StopBeforeExit")
                    curr_ln_x = stop_x
                    print(f"DEBUG: STOPPED just before exit at: ({curr_ln_x}, {curr_ln_y})")
            
            # Step 3: Now retrace back and sow perimeter (INCLUDING FARTHEST BOUNDARY)
            print(f"DEBUG: Starting retrace from: ({curr_ln_x}, {curr_ln_y})")
            
            # Determine which corner to go to first for retracing
            if curr_ln_x <= max_lane_idx_x / 2:
                retrace_corner_x = 0  # Go to bottom-left corner first
                farthest_boundary_x = max_lane_idx_x  # Farthest is right boundary
            else:
                retrace_corner_x = max_lane_idx_x  # Go to bottom-right corner first
                farthest_boundary_x = 0  # Farthest is left boundary
            
            # Retrace: Sow horizontally to corner
            if curr_ln_x != retrace_corner_x:
                curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                    curr_ln_x, curr_ln_y, retrace_corner_x, curr_ln_y,
                    exit_point_lanes, _points_lanes, _sow_flags,
                    "CustomExit_RetraceToCorner", gap_size=gap_size)
                print(f"DEBUG: Retraced to corner: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical to top boundary (FARTHEST HORIZONTAL BOUNDARY)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, max_lane_idx_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalToFarthestBoundary", gap_size=gap_size)
            print(f"DEBUG: Sowed to farthest boundary (top): ({curr_ln_x}, {curr_ln_y})")
            
            # Sow horizontal to opposite corner (completing farthest boundary)
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, farthest_boundary_x, curr_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowFarthestBoundaryHorizontal", gap_size=gap_size)
            print(f"DEBUG: Sowed farthest boundary horizontal: ({curr_ln_x}, {curr_ln_y})")
            
            # Sow vertical back to bottom boundary
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, curr_ln_x, 0,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_SowVerticalBackToBottom", gap_size=gap_size)
            print(f"DEBUG: Sowed back to bottom boundary: ({curr_ln_x}, {curr_ln_y})")
            
            # Final sowing approach to actual exit
            curr_ln_x, curr_ln_y = _add_headland_segment_custom_exit_with_gaps(
                curr_ln_x, curr_ln_y, ex_ln_x, ex_ln_y,
                exit_point_lanes, _points_lanes, _sow_flags,
                "CustomExit_FinalSowToExit", gap_size=gap_size)
            print(f"DEBUG: Final approach to exit: ({curr_ln_x}, {curr_ln_y})")




        
        # Ensure we end exactly at exit point
        if not _points_lanes or _points_lanes[-1] != exit_point_lanes:
            last_path_pt_ln = _points_lanes[-1]
            if last_path_pt_ln[0] != ex_ln_x:
                _commit((ex_ln_x, last_path_pt_ln[1]), True, "CustomExit_FinalNav_AlignX")
            if _points_lanes[-1] != exit_point_lanes:
                _commit((ex_ln_x, ex_ln_y), True, "CustomExit_FinalNav_AlignY_to_Exit")
            
    return {'points_lanes': _points_lanes, 'sow_flags': _sow_flags}

# --- Path Analysis (operates on lane indices) ---
def analyze_path_sequence_fixed(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_list):
    if not path_lanes_list or len(path_lanes_list) < 2: return []
    
    row_sequence = []
    v_counter = 1; h_counter = 1
    labeled_v_lanes = set(); labeled_h_lanes = set()  
    
    # FIXED: Create a mapping of lane X positions to VRow numbers
    # Get all unique X lanes that have vertical sown movements, then sort them
    all_vertical_lanes = set()
    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]; lx2, ly2 = path_lanes_list[i+1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        if lx1 == lx2 and ly1 != ly2 and is_sown:  # Vertical sown movement
            all_vertical_lanes.add(lx1)
    
    # Sort the lanes and create a mapping: lane_x -> VRow_number
    sorted_vertical_lanes = sorted(all_vertical_lanes)
    lane_to_vrow_mapping = {lane_x: f"VRow{idx+1}" for idx, lane_x in enumerate(sorted_vertical_lanes)}

    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]; lx2, ly2 = path_lanes_list[i+1]
        is_sown = i < len(sow_flags_list) and sow_flags_list[i]
        label = ""; mov_type = "Other"
        
        if lx1 == lx2 and ly1 != ly2: 
            mov_type = "VRow_Path" 
            # Use the mapping to get correct VRow label
            if is_sown and lx1 in lane_to_vrow_mapping:
                label = lane_to_vrow_mapping[lx1]
                
        elif ly1 == ly2 and lx1 != lx2: 
            mov_type = "HRow_Path"
            if is_sown and (ly1 == 0 or ly1 == max_ly_idx_val) and ly1 not in labeled_h_lanes:
                label = f"HRow{h_counter}"; labeled_h_lanes.add(ly1); h_counter += 1
            elif is_sown: 
                label = f"H-Turn{i+1}" 
                    
        row_sequence.append({'segment_path_index': i, 'movement_type': mov_type, 
                             'from_pos_lanes': (lx1, ly1), 'to_pos_lanes': (lx2, ly2),
                             'label': label, 'is_sown': is_sown})
    return row_sequence

def get_movement_analysis(path_lanes_list, seg_idx, max_lx_idx_val, max_ly_idx_val, analyzed_row_seq, sow_flags_all, rover_width_m_val, rover_length_m_val):
    if seg_idx >= len(path_lanes_list) - 1 or seg_idx >= len(sow_flags_all): return None 
    
    lx1, ly1 = path_lanes_list[seg_idx]; lx2, ly2 = path_lanes_list[seg_idx+1]

    # FIXED: Use correct dimensions for lane center calculation
    from_pos_m_center = ((lx1 + 0.5) * rover_width_m_val, (ly1 + 0.5) * rover_length_m_val)
    to_pos_m_center = ((lx2 + 0.5) * rover_width_m_val, (ly2 + 0.5) * rover_length_m_val)

    # FIXED: Use correct dimensions for distance calculation
    distance_m = abs(lx2 - lx1) * rover_width_m_val + abs(ly2 - ly1) * rover_length_m_val
    
    row_info = next((rs for rs in analyzed_row_seq if rs['segment_path_index'] == seg_idx), None)
    label = row_info['label'] if row_info and row_info['label'] else f"Segment{seg_idx + 1}" 
    
    analysis = {'from_pos_m': from_pos_m_center, 'to_pos_m': to_pos_m_center,
                'from_row_y_coord_m': from_pos_m_center[1], 'to_row_y_coord_m': to_pos_m_center[1],
                'distance_m': distance_m, 'direction': '', 'action': '', 
                'farming_type': '', 'status': '', 'row_sequence_label': label}

    if lx2 > lx1: analysis['direction'] = 'EAST'
    elif lx2 < lx1: analysis['direction'] = 'WEST'
    elif ly2 > ly1: analysis['direction'] = 'NORTH'
    elif ly2 < ly1: analysis['direction'] = 'SOUTH'
    
    is_sown = sow_flags_all[seg_idx]
    first_inner_lx = 1
    last_inner_lx = max_lx_idx_val - 1

    if not is_sown:
        analysis['farming_type'] = 'NONE'; analysis['action'] = 'NAVIGATION_UNSOWN'; analysis['status'] = 'TRAVERSING_NO_SOW'
    else:
        if lx1 == lx2 and ly1 != ly2: 
            analysis['status'] = 'SOWING_VERTICALLY'
            first_inner_lx = 1
            last_inner_lx = max_lx_idx_val - 1
            is_inner_v_sweep = (first_inner_lx <= lx1 <= last_inner_lx)
            analysis['action'] = 'INNER_VERTICAL_FARMING' if is_inner_v_sweep else 'BOUNDARY_VERTICAL_FARMING'
            analysis['farming_type'] = 'CROP_PLANTING_V' if is_inner_v_sweep else 'PERIMETER_SOWING_V'

        elif ly1 == ly2 and lx1 != lx2: 
            analysis['status'] = 'SOWING_HORIZONTALLY'
            is_headland_h = (ly1 == 0 or ly1 == max_ly_idx_val)
            analysis['action'] = 'BOUNDARY_HORIZONTAL_FARMING' if is_headland_h else 'TRANSITION_SOWING_H' 
            analysis['farming_type'] = 'PERIMETER_SOWING_H' if is_headland_h else 'TRANSITION_SOWING_H'
        else: 
            analysis['action'] = 'DIAGONAL_SOWING_ERROR'; analysis['farming_type'] = 'ERROR_SOW'; analysis['status'] = 'SOWING_ERROR_PATH'
    return analysis

# --- Telemetry Logger ---
class LiveTelemetryLogger:
    def __init__(self, farm_w_m, farm_b_m, rover_lw_m, exit_info_str):
        self.farm_w_m = farm_w_m; self.farm_b_m = farm_b_m; self.rover_lw_m = rover_lw_m
        self.sown_v_segs = 0; self.sown_h_segs = 0
        self.total_dist_m = 0; self.total_sow_dist_m = 0
        self.start_time = datetime.now(); self.csv_filename = "navigation_log.csv"
        if not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f); w.writerow(["Timestamp", "Step", "Label", "From (m)", "To (m)", "FromY (m)", "ToY (m)", "Dir", "Action", "FarmType", "Status", "SegDist (m)", "TotalDist (m)", "SownDist (m)", "V_Sown_Segs", "H_Sown_Segs"])
                print(f"💾 CSV created: {self.csv_filename}")
            except IOError as e: print(f"❌ CSV Error: {e}")
        else: print(f"📝 Appending to: {self.csv_filename}")
        hdr = "🤖 FARM ROBOT TELEMETRY 🤖"; print(f"\n{hdr}\n{'='*len(hdr)}\n📊 Farm: {farm_w_m}x{farm_b_m}m, Rover Lane: {rover_lw_m}m\n🎯 Exit: {exit_info_str}\n⏰ Start: {self.start_time:%Y-%m-%d %H:%M:%S}\n{'='*len(hdr)}\n🔴 LIVE LOG:\n{'='*len(hdr)}")

    def log_movement(self, step, analysis, time_now): 
        if not analysis: return
        self.total_dist_m += analysis['distance_m']
        if analysis['farming_type'] != 'NONE': 
            self.total_sow_dist_m += analysis['distance_m']
            if 'VERTICAL' in analysis['action'] or '_V' in analysis['farming_type']: self.sown_v_segs += 1
            elif 'HORIZONTAL' in analysis['action'] or '_H' in analysis['farming_type']: self.sown_h_segs += 1
        elapsed = (time_now - self.start_time).total_seconds()
        display_label = analysis['row_sequence_label']
        from_pos_str = f"({analysis['from_pos_m'][0]:.1f}, {analysis['from_pos_m'][1]:.1f})"
        to_pos_str = f"({analysis['to_pos_m'][0]:.1f}, {analysis['to_pos_m'][1]:.1f})"
        print(f"\n⏱️ {time_now:%H:%M:%S.%f}"[:-3] + f" [S{step:02d}] (+{elapsed:.1f}s)\n🏷️ {display_label}\n📍 {from_pos_str} → {to_pos_str} (D:{analysis['distance_m']:.1f}m)\n🧭 Act: {analysis['action']} ({analysis['status']}) | Type: {analysis['farming_type']}\n📊 TD:{self.total_dist_m:.1f}m SD:{self.total_sow_dist_m:.1f}m VS:{self.sown_v_segs} HS:{self.sown_h_segs}\n{'-'*70}")
        row = [time_now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], step, display_label, str(analysis['from_pos_m']), str(analysis['to_pos_m']), f"{analysis['from_row_y_coord_m']:.1f}", f"{analysis['to_row_y_coord_m']:.1f}", analysis['direction'], analysis['action'], analysis['farming_type'], analysis['status'], f"{analysis['distance_m']:.1f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
        except IOError as e: print(f"❌ CSV Write Err (S{step}): {e}")

    def finalize_mission(self, final_pos_m): 
        end_time = datetime.now(); duration = (end_time - self.start_time).total_seconds()
        eff = (self.total_sow_dist_m / self.total_dist_m * 100) if self.total_dist_m > 0 else 0
        final_pos_str = f"({final_pos_m[0]:.1f}, {final_pos_m[1]:.1f})m"
        summary = f"\n🏁 MISSION COMPLETE! 🏁\n{'='*25}\n📍 End: {final_pos_str}\n⏰ Time: {duration:.2f}s\n📏 TD: {self.total_dist_m:.1f}m\n🌱 SD: {self.total_sow_dist_m:.1f}m ({eff:.1f}%)\n🚜 VS: {self.sown_v_segs}\n↔️ HS: {self.sown_h_segs}\n{'='*25}"
        print(summary)
        row = [end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], "FINAL", "End", str(final_pos_m), "", "", "", "", "", "", "", f"{duration:.2f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
            print(f"💾 Final summary logged to {self.csv_filename}")
        except IOError as e: print(f"❌ CSV Final Err: {e}")


# --- Animation ---
def interpolate_path(path_pts_metric_centers, pts_per_seg=25): 
    if not path_pts_metric_centers or len(path_pts_metric_centers) < 2: return np.array([]), np.array([])
    sx_m, sy_m = [], []
    for i in range(len(path_pts_metric_centers) - 1):
        x0,y0 = path_pts_metric_centers[i]; x1,y1 = path_pts_metric_centers[i+1]
        sx_m.extend(np.linspace(x0, x1, pts_per_seg)); sy_m.extend(np.linspace(y0, y1, pts_per_seg))
    return np.array(sx_m), np.array(sy_m)

def animate_robot(n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, title_suffix_str, 
                  path_lanes_list, sow_flags_all_list, 
                  exit_vis_lanes, farm_w_m_val, farm_b_m_val, rover_width_m_val, rover_length_m_val):
    if not path_lanes_list or len(path_lanes_list) < 2: print("❌ Anim Err: Path short."); return None
    if len(sow_flags_all_list) != len(path_lanes_list) -1 : 
        print(f"❌ Anim Err: Mismatch sow_flags ({len(sow_flags_all_list)}) and segments ({len(path_lanes_list)-1}).")
        sow_flags_all_list.extend([False] * (max(0, len(path_lanes_list) - 1 - len(sow_flags_all_list))))

    # FIXED: Use correct dimensions for path metric centers
    path_metric_centers_list = [((ln_x + 0.5) * rover_width_m_val, (ln_y + 0.5) * rover_length_m_val) for ln_x, ln_y in path_lanes_list]
    exit_vis_metric_center = ((exit_vis_lanes[0] + 0.5) * rover_width_m_val, (exit_vis_lanes[1] + 0.5) * rover_length_m_val)
    
    # FIXED: Use the existing analyze_path_sequence_fixed function instead of missing function
    row_labels_info_list = analyze_path_sequence_fixed(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_all_list)    
    seg_labels_info_list = analyze_path_sequence_fixed(path_lanes_list, n_inner_x_sweeps_val, max_lx_idx_val, max_ly_idx_val, sow_flags_all_list)
    logger_obj = LiveTelemetryLogger(farm_w_m_val, farm_b_m_val, rover_width_m_val, rover_length_m_val, title_suffix_str)
    
    smooth_x_m_centers, smooth_y_m_centers = interpolate_path(path_metric_centers_list)
    if smooth_x_m_centers.size == 0: print("❌ Anim Err: Interpolated path empty."); return None
    
    # FIXED: Pass rover_length_m_val to get_movement_analysis
    analyses_metric_list = [get_movement_analysis(path_lanes_list, i, max_lx_idx_val, max_ly_idx_val, seg_labels_info_list, sow_flags_all_list, rover_width_m_val, rover_length_m_val) for i in range(len(path_lanes_list) - 1)]
    
    fig, ax = plt.subplots(figsize=(12, 10)); ax.set_aspect('equal')
    plot_padding_m = max(rover_width_m_val, rover_length_m_val) * 0.5
    ax.set_xlim(-plot_padding_m, farm_w_m_val + plot_padding_m)
    ax.set_ylim(-plot_padding_m, farm_b_m_val + plot_padding_m)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f'🤖 Farm: {farm_w_m_val}x{farm_b_m_val}m, Rover: {rover_width_m_val}x{rover_length_m_val}m ({title_suffix_str})', fontsize=14, pad=20)
    
    ax.add_patch(Rectangle((0,0), farm_w_m_val, farm_b_m_val, fill=False, edgecolor='darkgray', lw=2, zorder=1))
    
    # FIXED: Draw row labels with proper positioning using correct dimensions
    # Get unique columns for VRow labels from vertical movements (INCLUDING OUTER LANES)
    vrow_columns = set()

    for i in range(len(path_lanes_list) - 1):
        lx1, ly1 = path_lanes_list[i]
        lx2, ly2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_all_list) and sow_flags_all_list[i]
        
        # Vertical movement in ANY lane that are sown (including outer boundary lanes)
        if lx1 == lx2 and ly1 != ly2 and is_sown:
            vrow_columns.add(lx1)

    
    # Sort columns and assign VRow labels (all on bottom side)
    sorted_vrow_columns = sorted(vrow_columns)
    for i, column_x in enumerate(sorted_vrow_columns):
        tx_m = (column_x + 0.5) * rover_width_m_val
        ty_m = -plot_padding_m * 0.7  # All VRow labels on bottom side
        
        label = f'VRow{i+1}'
        ax.text(tx_m, ty_m, label, fontsize=10, color='navy', 
               ha='center', va='center', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.9, ec='navy', lw=2))
    
    # FIXED: Position HRow labels based on actual horizontal paths in the robot's route
    # Find the Y-coordinates of horizontal movements in the path
    horizontal_y_positions = set()
    for i in range(len(path_lanes_list) - 1):
        x1, y1 = path_lanes_list[i]
        x2, y2 = path_lanes_list[i + 1]
        is_sown = i < len(sow_flags_all_list) and sow_flags_all_list[i]
        
        if y1 == y2 and x1 != x2 and is_sown:  # Horizontal movement that is sown
            horizontal_y_positions.add(y1)
    
    # Sort horizontal positions and assign labels
    sorted_horizontal_positions = sorted(horizontal_y_positions)
    
    if len(sorted_horizontal_positions) >= 1:
        # HRow1 (bottom horizontal path) - positioned on left side
        hrow1_y_lane = sorted_horizontal_positions[0]
        hrow1_tx_m = -plot_padding_m * 0.7
        hrow1_ty_m = (hrow1_y_lane + 0.5) * rover_length_m_val  # FIXED: Use rover_length_m_val
        ax.text(hrow1_tx_m, hrow1_ty_m, 'HRow1', fontsize=10, color='darkgreen', 
               ha='center', va='center', weight='bold', rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.9, ec='darkgreen', lw=2))
    
    if len(sorted_horizontal_positions) >= 2:
        # HRow2 (top horizontal path) - positioned on left side
        hrow2_y_lane = sorted_horizontal_positions[-1]  # Last (topmost) horizontal position
        hrow2_tx_m = -plot_padding_m * 0.7
        hrow2_ty_m = (hrow2_y_lane + 0.5) * rover_length_m_val  # FIXED: Use rover_length_m_val
        ax.text(hrow2_tx_m, hrow2_ty_m, 'HRow2', fontsize=10, color='darkgreen', 
               ha='center', va='center', weight='bold', rotation=0,
               bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.9, ec='darkgreen', lw=2))
    
    # FIXED: Draw full path trace using rectangles with proper rover dimensions
    single_brown_color = '#8B4513'  # Consistent brown color
    for i in range(len(path_metric_centers_list) - 1):
        x1, y1 = path_metric_centers_list[i]
        x2, y2 = path_metric_centers_list[i + 1]
        
        if x1 == x2:  # Vertical movement
            rect_x = x1 - rover_width_m_val / 2
            rect_y = min(y1, y2) - rover_length_m_val / 2  # FIXED: Use rover_length_m_val
            rect_width = rover_width_m_val
            rect_height = abs(y2 - y1) + rover_length_m_val  # FIXED: Use rover_length_m_val
        else:  # Horizontal movement
            rect_x = min(x1, x2) - rover_width_m_val / 2
            rect_y = y1 - rover_length_m_val / 2
            rect_width = abs(x2 - x1) + rover_width_m_val
            rect_height = rover_length_m_val
        
        # Add rectangle for full path trace
        path_rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                            color=single_brown_color, alpha=0.4, zorder=2)
        ax.add_patch(path_rect)
    
    # FIXED: Pre-create all sown rectangles with proper rover dimensions
    sown_rectangles = []
    sown_progress_masks = []  # To track partial visibility parameters
    for i in range(len(path_lanes_list) - 1):
        if sow_flags_all_list[i]:
            x1, y1 = path_metric_centers_list[i]
            x2, y2 = path_metric_centers_list[i + 1]
            
            if x1 == x2:  # Vertical movement
                rect_x = x1 - rover_width_m_val / 2
                rect_y = min(y1, y2) - rover_length_m_val / 2  # FIXED: Use rover_length_m_val
                rect_width = rover_width_m_val
                rect_height = abs(y2 - y1) + rover_length_m_val  # FIXED: Use rover_length_m_val
            else:  # Horizontal movement
                rect_x = min(x1, x2) - rover_width_m_val / 2
                rect_y = y1 - rover_length_m_val / 2
                rect_width = abs(x2 - x1) + rover_width_m_val
                rect_height = rover_length_m_val
            
            # Create a clipping rectangle that will grow gradually
            # Start with zero size in the movement direction
            if x1 == x2:  # Vertical - start with zero height
                clip_rect = Rectangle((rect_x, rect_y), rect_width, 0, 
                                    color='#006400', alpha=0.8, zorder=3, visible=False)
            else:  # Horizontal - start with zero width
                clip_rect = Rectangle((rect_x, rect_y), 0, rect_height, 
                                    color='#006400', alpha=0.8, zorder=3, visible=False)
            
            ax.add_patch(clip_rect)
            sown_rectangles.append((i, clip_rect))
            sown_progress_masks.append({
                'full_width': rect_width, 
                'full_height': rect_height, 
                'is_vertical': x1 == x2, 
                'base_x': rect_x, 
                'base_y': rect_y,
                'start_pos': (x1, y1),
                'end_pos': (x2, y2)
            })
        else:
            sown_rectangles.append((i, None))
            sown_progress_masks.append(None)
    
    # Create dummy lines for legend with rover dimensions info
    full_path_trace_line, = ax.plot([], [], color=single_brown_color, lw=10, alpha=0.4, label=f'Unsown Path (Rover:{rover_width_m_val}x{rover_length_m_val}m)')
    sown_segments_line, = ax.plot([], [], color='#006400', lw=10, alpha=0.8, label=f'Sown Area (Rover:{rover_width_m_val}x{rover_length_m_val}m)')
    
    # UPDATED: Use actual rover dimensions for visual representation
    rover_body_width_m = rover_width_m_val 
    rover_body_height_m = rover_length_m_val  # Use rover length for visual height
    initial_rover_center_m = path_metric_centers_list[0]
    robot_body_patch = Rectangle((initial_rover_center_m[0] - rover_body_width_m/2, initial_rover_center_m[1] - rover_body_height_m/2), 
                                 rover_body_width_m, rover_body_height_m, color='orange', ec='black', lw=1.5, zorder=4)
    ax.add_patch(robot_body_patch)
    
    start_marker_center_m = path_metric_centers_list[0]; marker_radius_m = 0.4 * max(rover_width_m_val, rover_length_m_val)
    ax.add_patch(Circle(start_marker_center_m, marker_radius_m, color='blue', fill=True, lw=2.5, zorder=5, alpha=0.5))
    ax.add_patch(Circle(start_marker_center_m, marker_radius_m, color='blue', fill=False, lw=2.5, zorder=5, hatch='//'))
    
    # Create gate-style exit marker instead of circle
    gate_width = rover_width_m_val * 0.8
    gate_height = rover_length_m_val * 0.3  # FIXED: Use rover_length_m_val for gate height
    
    # Determine which border the exit is on and position gate accordingly
    exit_x, exit_y = exit_vis_lanes[0], exit_vis_lanes[1]
    
    if exit_x == 0:  # Left border
        gate_x = -gate_height/2
        gate_y = exit_vis_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_x == max_lx_idx_val:  # Right border
        gate_x = farm_w_m_val - gate_height/2
        gate_y = exit_vis_metric_center[1] - gate_width/2
        gate_w, gate_h = gate_height, gate_width
    elif exit_y == 0:  # Bottom border
        gate_x = exit_vis_metric_center[0] - gate_width/2
        gate_y = -gate_height/2
        gate_w, gate_h = gate_width, gate_height
    else:  # Top border
        gate_x = exit_vis_metric_center[0] - gate_width/2
        gate_y = farm_b_m_val - gate_height/2
        gate_w, gate_h = gate_width, gate_height
    
    # Add the gate marker
    exit_gate = Rectangle((gate_x, gate_y), gate_w, gate_h, 
                         color='red', alpha=0.8, zorder=5, 
                         edgecolor='darkred', linewidth=2)
    ax.add_patch(exit_gate)
    
    legend_handles = [
        Line2D([0],[0],c=single_brown_color,lw=10,alpha=0.4,label=f'Unsown Path (Rover:{rover_width_m_val}x{rover_length_m_val}m)'), 
        Line2D([0],[0],c='#006400',lw=10,alpha=0.8,label=f'Sown Area (Rover:{rover_width_m_val}x{rover_length_m_val}m)'), 
        Rectangle((0,0), 1, 1, fc='orange', ec='black', label=f'🤖 Rover ({rover_width_m_val}x{rover_length_m_val}m)'),
        Line2D([0],[0],marker='o',mfc='blue',mec='blue',ms=10,ls='None',label=f'🔵 START ({start_marker_center_m[0]:.1f}, {start_marker_center_m[1]:.1f})m'), 
        Rectangle((0,0), 1, 1, fc='red', ec='darkred', label=f'🚪 EXIT GATE ({exit_vis_metric_center[0]:.1f}, {exit_vis_metric_center[1]:.1f})m')
    ]
    ax.legend(handles=legend_handles,loc='upper right',bbox_to_anchor=(1.28,1.02),fontsize=8); plt.subplots_adjust(right=0.75)
    
    logged_segments_indices = set(); logger_obj.mission_finalized = False; total_animation_frames = len(smooth_x_m_centers)
    num_orig_segments = len(path_metric_centers_list) -1
    pts_per_orig_segment_approx = total_animation_frames // num_orig_segments if num_orig_segments > 0 else total_animation_frames

    def init_animation_func(): 
        robot_body_patch.set_xy((smooth_x_m_centers[0]-rover_body_width_m/2, smooth_y_m_centers[0]-rover_body_height_m/2))
        return [robot_body_patch]
    
    def update_animation_func(frame_idx):
        current_x_center_m = smooth_x_m_centers[frame_idx]
        current_y_center_m = smooth_y_m_centers[frame_idx]
        robot_body_patch.set_xy((current_x_center_m - rover_body_width_m/2, current_y_center_m - rover_body_height_m/2))
        
        current_original_segment_idx = frame_idx // pts_per_orig_segment_approx if pts_per_orig_segment_approx > 0 else num_orig_segments -1
        current_original_segment_idx = min(current_original_segment_idx, num_orig_segments - 1) 

        if current_original_segment_idx != -1 and current_original_segment_idx < len(analyses_metric_list) and current_original_segment_idx not in logged_segments_indices:
            if analyses_metric_list[current_original_segment_idx]:
                 logger_obj.log_movement(current_original_segment_idx + 1, analyses_metric_list[current_original_segment_idx], datetime.now())
            logged_segments_indices.add(current_original_segment_idx)
        
        # Show sown rectangles gradually as rover moves - FIXED VERSION
        if current_original_segment_idx >= 0:
            # Make completed segments fully visible with correct dimensions
            for i in range(min(current_original_segment_idx, len(sown_rectangles))):
                if i < len(sown_rectangles) and sown_rectangles[i][1] is not None:
                    rect = sown_rectangles[i][1]
                    mask = sown_progress_masks[i]
                    rect.set_visible(True)
                
                    # Set to full dimensions immediately for completed segments
                    if mask['is_vertical']:
                        rect.set_height(mask['full_height'])
                        rect.set_y(mask['base_y'])  # Ensure correct position
                    else:
                        rect.set_width(mask['full_width'])
                        rect.set_x(mask['base_x'])  # Ensure correct position
        
            # Gradually show current segment based on rover progress
            if (current_original_segment_idx < len(sown_rectangles) and 
                sown_rectangles[current_original_segment_idx][1] is not None and
                current_original_segment_idx < len(sow_flags_all_list) and
                sow_flags_all_list[current_original_segment_idx]):
            
                current_rect = sown_rectangles[current_original_segment_idx][1]
                current_mask = sown_progress_masks[current_original_segment_idx]
                current_rect.set_visible(True)
            
                # Calculate progress within current segment
                segment_start_frame = current_original_segment_idx * pts_per_orig_segment_approx
                frames_in_segment = min(pts_per_orig_segment_approx, total_animation_frames - segment_start_frame)
                progress_in_segment = (frame_idx - segment_start_frame) / frames_in_segment if frames_in_segment > 0 else 1
                progress_in_segment = max(0, min(1, progress_in_segment))
            
                if current_mask['is_vertical']:
                    # Vertical movement - grow height gradually
                    new_height = current_mask['full_height'] * progress_in_segment
                    current_rect.set_height(new_height)
                
                    # Determine growth direction based on start/end positions
                    start_y, end_y = current_mask['start_pos'][1], current_mask['end_pos'][1]
                    if end_y < start_y:  # Moving from top to bottom
                        # Rectangle grows from top, so adjust y position
                        new_y = current_mask['base_y'] + current_mask['full_height'] - new_height
                        current_rect.set_y(new_y)
                    else:  # Moving from bottom to top
                        # Rectangle grows from bottom, keep base_y
                        current_rect.set_y(current_mask['base_y'])
                else:
                    # Horizontal movement - grow width gradually
                    new_width = current_mask['full_width'] * progress_in_segment
                    current_rect.set_width(new_width)
                
                    # Determine growth direction based on start/end positions
                    start_x, end_x = current_mask['start_pos'][0], current_mask['end_pos'][0]
                    if end_x < start_x:  # Moving from right to left
                        # Rectangle grows from right, so adjust x position
                        new_x = current_mask['base_x'] + current_mask['full_width'] - new_width
                        current_rect.set_x(new_x)
                    else:  # Moving from left to right
                        # Rectangle grows from left, keep base_x
                        current_rect.set_x(current_mask['base_x'])
        
        # Final frame handling - ensure all sown rectangles are properly displayed
        if frame_idx >= total_animation_frames - 1 and not logger_obj.mission_finalized: 
            # Make all remaining sown rectangles fully visible with correct dimensions
            for i, (_, rect) in enumerate(sown_rectangles):
                if rect is not None:
                    rect.set_visible(True)
                    if i < len(sown_progress_masks) and sown_progress_masks[i]:
                        mask = sown_progress_masks[i]
                        if mask['is_vertical']:
                            rect.set_height(mask['full_height'])
                            rect.set_y(mask['base_y'])
                        else:
                            rect.set_width(mask['full_width'])
                            rect.set_x(mask['base_x'])
                        
            for i_log_final_check in range(len(analyses_metric_list)): 
                if i_log_final_check not in logged_segments_indices and analyses_metric_list[i_log_final_check]: 
                    logger_obj.log_movement(i_log_final_check+1, analyses_metric_list[i_log_final_check], datetime.now())
            logger_obj.finalize_mission(path_metric_centers_list[-1]) 
            logger_obj.mission_finalized = True
        
        return [robot_body_patch] + [rect for _, rect in sown_rectangles if rect is not None and rect.get_visible()]

    animation_obj = FuncAnimation(fig, update_animation_func, frames=total_animation_frames, 
                                  init_func=init_animation_func, blit=True, interval=50, repeat=False)
    plt.show()
    return animation_obj





class LiveTelemetryLogger:
    def __init__(self, farm_w_m, farm_b_m, rover_width_m, rover_length_m, exit_info_str):
        self.farm_w_m = farm_w_m; self.farm_b_m = farm_b_m; self.rover_width_m = rover_width_m; self.rover_length_m = rover_length_m
        self.sown_v_segs = 0; self.sown_h_segs = 0
        self.total_dist_m = 0; self.total_sow_dist_m = 0
        self.start_time = datetime.now(); self.csv_filename = "navigation_log.csv"
        if not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f); w.writerow(["Timestamp", "Step", "Label", "From (m)", "To (m)", "FromY (m)", "ToY (m)", "Dir", "Action", "FarmType", "Status", "SegDist (m)", "TotalDist (m)", "SownDist (m)", "V_Sown_Segs", "H_Sown_Segs"])
                print(f"💾 CSV created: {self.csv_filename}")
            except IOError as e: print(f"❌ CSV Error: {e}")
        else: print(f"📝 Appending to: {self.csv_filename}")
        hdr = "🤖 FARM ROBOT TELEMETRY 🤖"; print(f"\n{hdr}\n{'='*len(hdr)}\n📊 Farm: {farm_w_m}x{farm_b_m}m, Rover: {rover_width_m}x{rover_length_m}m\n🎯 Exit: {exit_info_str}\n⏰ Start: {self.start_time:%Y-%m-%d %H:%M:%S}\n{'='*len(hdr)}\n🔴 LIVE LOG:\n{'='*len(hdr)}")

    def log_movement(self, step, analysis, time_now): 
        if not analysis: return
        self.total_dist_m += analysis['distance_m']
        if analysis['farming_type'] != 'NONE': 
            self.total_sow_dist_m += analysis['distance_m']
            if 'VERTICAL' in analysis['action'] or '_V' in analysis['farming_type']: self.sown_v_segs += 1
            elif 'HORIZONTAL' in analysis['action'] or '_H' in analysis['farming_type']: self.sown_h_segs += 1
        elapsed = (time_now - self.start_time).total_seconds()
        display_label = analysis['row_sequence_label']
        from_pos_str = f"({analysis['from_pos_m'][0]:.1f}, {analysis['from_pos_m'][1]:.1f})"
        to_pos_str = f"({analysis['to_pos_m'][0]:.1f}, {analysis['to_pos_m'][1]:.1f})"
        print(f"\n⏱️ {time_now:%H:%M:%S.%f}"[:-3] + f" [S{step:02d}] (+{elapsed:.1f}s)\n🏷️ {display_label}\n📍 {from_pos_str} → {to_pos_str} (D:{analysis['distance_m']:.1f}m)\n🧭 Act: {analysis['action']} ({analysis['status']}) | Type: {analysis['farming_type']}\n📊 TD:{self.total_dist_m:.1f}m SD:{self.total_sow_dist_m:.1f}m VS:{self.sown_v_segs} HS:{self.sown_h_segs}\n{'-'*70}")
        row = [time_now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], step, display_label, str(analysis['from_pos_m']), str(analysis['to_pos_m']), f"{analysis['from_row_y_coord_m']:.1f}", f"{analysis['to_row_y_coord_m']:.1f}", analysis['direction'], analysis['action'], analysis['farming_type'], analysis['status'], f"{analysis['distance_m']:.1f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
        except IOError as e: print(f"❌ CSV Write Err (S{step}): {e}")

    def finalize_mission(self, final_pos_m): 
        end_time = datetime.now(); duration = (end_time - self.start_time).total_seconds()
        eff = (self.total_sow_dist_m / self.total_dist_m * 100) if self.total_dist_m > 0 else 0
        final_pos_str = f"({final_pos_m[0]:.1f}, {final_pos_m[1]:.1f})m"
        summary = f"\n🏁 MISSION COMPLETE! 🏁\n{'='*25}\n📍 End: {final_pos_str}\n⏰ Time: {duration:.2f}s\n📏 TD: {self.total_dist_m:.1f}m\n🌱 SD: {self.total_sow_dist_m:.1f}m ({eff:.1f}%)\n🚜 VS: {self.sown_v_segs}\n↔️ HS: {self.sown_h_segs}\n{'='*25}"
        print(summary)
        row = [end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], "FINAL", "End", str(final_pos_m), "", "", "", "", "", "", "", f"{duration:.2f}", f"{self.total_dist_m:.1f}", f"{self.total_sow_dist_m:.1f}", self.sown_v_segs, self.sown_h_segs]
        try:
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f: csv.writer(f).writerow(row)
            print(f"💾 Final summary logged to {self.csv_filename}")
        except IOError as e: print(f"❌ CSV Final Err: {e}")


def main():
    print("=== 🤖 ADVANCED FARM ROBOT TRAVERSAL SYSTEM (v12.9) ===")
    farm_width_m, farm_breadth_m, rover_width_m, rover_length_m = 0.0, 0.0, 0.0, 0.0

    while True: 
        try: farm_width_m = float(input("Enter farm WIDTH (X-axis, e.g., 50 meters): ")); assert farm_width_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: farm_breadth_m = float(input("Enter farm BREADTH (Y-axis, e.g., 50 meters): ")); assert farm_breadth_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: rover_width_m = float(input("Enter rover WIDTH (for lane spacing, e.g., 10 meters): ")); assert rover_width_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")
    while True: 
        try: rover_length_m = float(input("Enter rover LENGTH (for coverage depth, e.g., 8 meters): ")); assert rover_length_m > 0; break
        except (ValueError, AssertionError): print("Invalid input. Must be positive number.")

    # FIXED: Correct grid calculation logic
    # Vertical passes (columns) = Farm_Width / Rover_Width
    num_lanes_x = int(farm_width_m / rover_width_m)
    # Horizontal rows = Farm_Height / Rover_Height (rover_length_m is the rover's height)
    num_lanes_y = int(farm_breadth_m / rover_length_m)

    # FIXED: Updated minimum requirements with correct logic
        # FIXED: Updated minimum requirements with correct logic
    if num_lanes_x < 3: 
        print(f"❌ Farm width {farm_width_m}m too small for rover width {rover_width_m}m.")
        print(f"   Need at least 3 vertical passes (2 headlands + 1 inner) = {3 * rover_width_m}m minimum farm width.")
        return None
    if num_lanes_y < 1: 
        print(f"❌ Farm breadth {farm_breadth_m}m too small for rover length {rover_length_m}m.")
        print(f"   Need at least 1 horizontal row = {rover_length_m}m minimum farm breadth.")
        return None
        
    max_lane_idx_x = num_lanes_x - 1
    max_lane_idx_y = num_lanes_y - 1
    
    # FIXED: Inner sweeps calculation remains the same (based on vertical passes)
    n_inner_x_sweeps = num_lanes_x - 2 
    if n_inner_x_sweeps < 0: n_inner_x_sweeps = 0 
    
    # FIXED: Updated information display with correct grid logic
    total_coverage_area = num_lanes_x * num_lanes_y * rover_width_m * rover_length_m
    farm_area = farm_width_m * farm_breadth_m
    coverage_percentage = (total_coverage_area / farm_area * 100) if farm_area > 0 else 0
    
    print(f"\n🚀 Grid Analysis for {farm_width_m}x{farm_breadth_m}m farm, Rover: {rover_width_m}x{rover_length_m}m")
    print(f"📊 Grid Layout: {num_lanes_x} vertical passes × {num_lanes_y} horizontal rows")
    print(f"📊 Calculation:")
    print(f"   • Vertical passes = Farm_Width({farm_width_m}) / Rover_Width({rover_width_m}) = {num_lanes_x}")
    print(f"   • Horizontal rows = Farm_Breadth({farm_breadth_m}) / Rover_Length({rover_length_m}) = {num_lanes_y}")
    print(f"📊 Coverage: {total_coverage_area:.1f}m² of {farm_area:.1f}m² ({coverage_percentage:.1f}%)")
    
    print(f"📊 Productive rows breakdown:")
    print(f"   • Inner vertical sweeps: {n_inner_x_sweeps} (lanes 1 to {max_lane_idx_x-1})" if n_inner_x_sweeps > 0 else "   • No inner vertical sweeps (farm too narrow)")
    print(f"   • Outer vertical headlands: 2 (lanes 0 & {max_lane_idx_x})")
    print(f"   • Horizontal headland rows: {min(2, num_lanes_y)} (covering {min(2, num_lanes_y) * rover_length_m}m breadth)")

    print(f"\nℹ️ Lane Index Ranges:")
    print(f"   • X-lanes (vertical passes): 0 to {max_lane_idx_x}")
    print(f"   • Y-lanes (horizontal rows): 0 to {max_lane_idx_y}")

    exit_point_lanes, anim_title_suffix_str = None, ""
    print("\nSelect Exit Type:\n1. Corner Exit (select a corner lane)\n2. Custom Boundary Exit (select a boundary lane)")
    nav_choice_str = ""
    while nav_choice_str not in ["1", "2"]: nav_choice_str = input("Choice (1-2): ").strip()
    is_corner_exit_choice = (nav_choice_str == "1")

    if is_corner_exit_choice:
        exit_point_lanes = get_user_choice_corner_lanes(max_lane_idx_x, max_lane_idx_y)
        anim_title_suffix_str = f"Corner Exit (Lane: {exit_point_lanes})"
    else: 
        exit_point_lanes = get_user_defined_exit_lanes(max_lane_idx_x, max_lane_idx_y)
        anim_title_suffix_str = f"Custom Exit (Lane: {exit_point_lanes})"
    
    # FIXED: Exit position calculation using correct dimensions
    exit_metric_center_display = ((exit_point_lanes[0] + 0.5) * rover_width_m, (exit_point_lanes[1] + 0.5) * rover_length_m)
    print(f"ℹ️ Target Exit Lane {exit_point_lanes} (approx. center: {exit_metric_center_display[0]:.1f}m, {exit_metric_center_display[1]:.1f}m)")

    path_data_dict = generate_fixed_path(n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, exit_point_lanes, is_corner_exit_choice)
    path_lanes_list_gen, sow_flags_list_gen = path_data_dict['points_lanes'], path_data_dict['sow_flags']

    if not path_lanes_list_gen or len(path_lanes_list_gen) < 2: print("❌ Path generation failed. Exiting."); return None
    if len(sow_flags_list_gen) != len(path_lanes_list_gen)-1: 
        print(f"❌ CRITICAL MISMATCH: Sow_flags ({len(sow_flags_list_gen)}) vs segments ({len(path_lanes_list_gen)-1}). Exiting.")
        return None

    # FIXED: Final summary with correct grid understanding
    print(f"\n🎬 MISSION SUMMARY:")
    print(f"📊 Farm: {farm_width_m}x{farm_breadth_m}m | Rover: {rover_width_m}x{rover_length_m}m")
    print(f"📊 Grid: {num_lanes_x} vertical passes × {num_lanes_y} horizontal rows")
    print(f"📊 Path: {len(path_lanes_list_gen)} waypoints, {len(sow_flags_list_gen)} segments")
    print(f"📊 Coverage: {coverage_percentage:.1f}% of farm area")
    input("\n🎬 Press Enter to start animation & telemetry...")

    final_animation_object = None
    try: 
        final_animation_object = animate_robot(
            n_inner_x_sweeps, max_lane_idx_x, max_lane_idx_y, anim_title_suffix_str, 
            path_lanes_list_gen, sow_flags_list_gen, exit_point_lanes, 
            farm_width_m, farm_breadth_m, rover_width_m, rover_length_m
        )
    except KeyboardInterrupt: print("\n\n⚠️ Animation aborted by user.")
    except Exception as e_anim: print(f"\n❌ An error occurred during animation: {e_anim}"); import traceback; traceback.print_exc()
    return final_animation_object



if __name__ == "__main__":
    run_animation_main_obj = main()
    if run_animation_main_obj: 
        print("\n✅ Animation process completed or started successfully.")
        print("📁 Check 'navigation_log.csv' for telemetry data.")
    else: 
        print("\n🔴 Animation did not complete or was not started due to an error.")






