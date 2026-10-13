"""Excel export for Lagrangian Particle Tracking results."""

from pathlib import Path
from typing import Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, numbers
from openpyxl.utils import get_column_letter

from .config import Config
from .simulation import SimulationResult, config_to_dict


def format_worksheet(ws, df: pd.DataFrame) -> None:
    """Apply formatting to a worksheet."""
    # Bold headers
    header_font = Font(bold=True)
    for cell in ws[1]:
        cell.font = header_font
    
    # Freeze header row
    ws.freeze_panes = "A2"
    
    # Auto-filter
    ws.auto_filter.ref = ws.dimensions
    
    # Column widths and number formats
    for col_idx, col_name in enumerate(df.columns, 1):
        col_letter = get_column_letter(col_idx)
        
        # Set column width based on content
        max_len = max(
            len(str(col_name)),
            *(len(str(df[col_name].iloc[i])) for i in range(min(100, len(df))))
        )
        ws.column_dimensions[col_letter].width = min(max_len + 2, 30)
        
        # Numeric formatting for float columns
        if df[col_name].dtype in ("float64", "float32", "int64", "int32"):
            for row_idx in range(2, len(df) + 2):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.number_format = numbers.FORMAT_NUMBER_00


def create_parameters_sheet(wb: Workbook, config: Config) -> None:
    """Create Parameters sheet."""
    ws = wb.active
    ws.title = "Parameters"
    
    params = config_to_dict(config)
    params["method"] = "both"
    
    ws.append(["Parameter", "Value"])
    for key, value in params.items():
        ws.append([key, value])
    
    format_worksheet(ws, pd.DataFrame(list(params.items()), columns=["Parameter", "Value"]))


def create_method_sheet(wb: Workbook, df: pd.DataFrame, method: str) -> None:
    """Create sheet for a single method's timestep data."""
    ws = wb.create_sheet(title=method.upper())
    
    # Write headers
    ws.append(list(df.columns))
    
    # Write data
    for _, row in df.iterrows():
        ws.append(list(row))
    
    format_worksheet(ws, df)


def create_comparison_sheet(
    wb: Workbook,
    euler_result: SimulationResult,
    euler_df: pd.DataFrame,
    rk4_result: SimulationResult,
    rk4_df: pd.DataFrame
) -> None:
    """Create Comparison sheet."""
    ws = wb.create_sheet(title="Comparison")
    
    # Exit time comparison
    ws.append(["Metric", "Euler", "RK4", "Difference (RK4 - Euler)"])
    
    # Exit time
    euler_exit = euler_result.exit_time or 0
    rk4_exit = rk4_result.exit_time or 0
    ws.append(["Exit Time (s)", euler_exit, rk4_exit, rk4_exit - euler_exit])
    
    # Exit position
    if euler_result.exit_position and rk4_result.exit_position:
        ex, ey = euler_result.exit_position
        rx, ry = rk4_result.exit_position
        ws.append(["Exit X (m)", ex, rx, rx - ex])
        ws.append(["Exit Y (m)", ey, ry, ry - ey])
    
    # Exit velocity
    if euler_result.exit_velocity and rk4_result.exit_velocity:
        evx, evy = euler_result.exit_velocity
        rvx, rvy = rk4_result.exit_velocity
        ws.append(["Exit Ux (m/s)", evx, rvx, rvx - evx])
        ws.append(["Exit Uy (m/s)", evy, rvy, rvy - evy])
    
    # Steps
    ws.append(["Steps", euler_result.steps, rk4_result.steps, rk4_result.steps - euler_result.steps])
    
    # Termination reason
    ws.append(["Termination", euler_result.termination_reason, rk4_result.termination_reason, ""])
    
    # Distance between exit positions
    if euler_result.exit_position and rk4_result.exit_position:
        ex, ey = euler_result.exit_position
        rx, ry = rk4_result.exit_position
        dist = ((rx - ex)**2 + (ry - ey)**2)**0.5
        ws.append(["Exit Position Distance (m)", "", "", dist])
    
    format_worksheet(ws, pd.DataFrame())


def create_summary_sheet(
    wb: Workbook,
    euler_result: SimulationResult,
    rk4_result: SimulationResult,
    config: Config
) -> None:
    """Create Summary sheet."""
    ws = wb.create_sheet(title="Summary")
    
    ws.append(["Lagrangian Particle Tracking Simulation Summary"])
    ws.append([""])
    
    ws.append(["Configuration"])
    for key, value in config_to_dict(config).items():
        ws.append([f"  {key}", value])
    
    ws.append([""])
    ws.append(["Euler Simulation"])
    ws.append([f"  Termination: {euler_result.termination_reason}"])
    ws.append([f"  Steps: {euler_result.steps}"])
    ws.append([f"  Exit Time: {euler_result.exit_time}"])
    ws.append([f"  Exit Position: {euler_result.exit_position}"])
    ws.append([f"  Exit Velocity: {euler_result.exit_velocity}"])
    
    ws.append([""])
    ws.append(["RK4 Simulation"])
    ws.append([f"  Termination: {rk4_result.termination_reason}"])
    ws.append([f"  Steps: {rk4_result.steps}"])
    ws.append([f"  Exit Time: {rk4_result.exit_time}"])
    ws.append([f"  Exit Position: {rk4_result.exit_position}"])
    ws.append([f"  Exit Velocity: {rk4_result.exit_velocity}"])
    
    ws.append([""])
    ws.append(["Comparison"])
    if euler_result.exit_time and rk4_result.exit_time:
        ws.append([f"  Exit Time Difference: {rk4_result.exit_time - euler_result.exit_time:.6f} s"])
    if euler_result.exit_position and rk4_result.exit_position:
        ex, ey = euler_result.exit_position
        rx, ry = rk4_result.exit_position
        dist = ((rx - ex)**2 + (ry - ey)**2)**0.5
        ws.append([f"  Exit Position Distance: {dist:.6f} m"])
    
    # Format
    for col_idx in range(1, 3):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 40


def generate_excel_workbook(
    euler_result: SimulationResult,
    euler_df: pd.DataFrame,
    rk4_result: SimulationResult,
    rk4_df: pd.DataFrame,
    config: Config,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate Excel workbook with all required sheets.
    
    Sheets:
    1. Parameters - All physical parameters and settings
    2. Euler - Euler timestep data
    3. RK4 - RK4 timestep data
    4. Comparison - Method comparison metrics
    5. Summary - Concise simulation description
    """
    wb = Workbook()
    
    # Sheet 1: Parameters
    create_parameters_sheet(wb, config)
    
    # Sheet 2: Euler
    create_method_sheet(wb, euler_df, "euler")
    
    # Sheet 3: RK4
    create_method_sheet(wb, rk4_df, "rk4")
    
    # Sheet 4: Comparison
    create_comparison_sheet(wb, euler_result, euler_df, rk4_result, rk4_df)
    
    # Sheet 5: Summary
    create_summary_sheet(wb, euler_result, rk4_result, config)
    
    # Save
    if output_path is None:
        output_path = config.simulation.get_run_output_dir() / "particle_tracking_results.xlsx"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    
    return output_path


def export_from_files(config: Config, output_path: Optional[Path] = None) -> Path:
    """
    Generate Excel workbook from saved CSV files.
    
    This function loads existing results and creates the workbook
    without running the simulation.
    """
    from .data import load_results
    from .simulation import results_to_dataframe
    
    # Load data
    euler_df, euler_meta = load_results("euler", config)
    rk4_df, rk4_meta = load_results("rk4", config)
    
    # Reconstruct SimulationResult objects (minimal)
    class MinimalResult:
        def __init__(self, meta, df):
            self.method = meta["method"]
            self.steps = meta["steps"]
            self.termination_reason = meta["termination_reason"]
            self.exit_time = meta["exit_time"]
            self.exit_position = meta["exit_position"]
            self.exit_velocity = meta["exit_velocity"]
    
    euler_result = MinimalResult(euler_meta, euler_df)
    rk4_result = MinimalResult(rk4_meta, rk4_df)
    
    return generate_excel_workbook(euler_result, euler_df, rk4_result, rk4_df, config, output_path)