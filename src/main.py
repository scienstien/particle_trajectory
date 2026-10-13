"""Main entry point for Lagrangian Particle Tracking simulation."""

import argparse
import sys
from pathlib import Path

from .config import Config, get_default_config, SimulationConfig, generate_run_id
from .data import (
    check_results_exist,
    load_results,
    save_results,
    save_summary,
)
from .export import export_from_files, generate_excel_workbook
from .simulation import run_both_methods, run_simulation, results_to_dataframe
from .visualization import generate_all_plots
from .animation import animate_from_files


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run simulation."""
    base_config = get_default_config(Z=args.Z)
    run_id = generate_run_id(args.Z)
    
    config = Config(
        particle=base_config.particle,
        fluid=base_config.fluid,
        gravity=base_config.gravity,
        initial=base_config.initial,
        domain=base_config.domain,
        simulation=SimulationConfig(
            dt=args.dt,
            t_max=args.t_max,
            max_steps=args.max_steps,
            output_dir=Path(args.output),
            run_id=run_id
        )
    )
    
    print(f"Configuration: Z={config.gravity.Z}, g_y={config.gravity.g_y}")
    print(f"Run ID: {run_id}")
    print(f"Timestep: {config.simulation.dt}, Max time: {config.simulation.t_max}")
    print(f"Initial position: ({config.initial.x}, {config.initial.y})")
    print(f"Domain: [{config.domain.x_min}, {config.domain.x_max}] x [{config.domain.y_min}, {config.domain.y_max}]")
    print(f"Output directory: {config.simulation.get_run_output_dir()}")
    print()
    
    methods_to_run = []
    if args.method == "both":
        methods_to_run = ["euler", "rk4"]
    else:
        methods_to_run = [args.method]
    
    results = {}
    dataframes = {}
    metadata = {}
    
    for method in methods_to_run:
        print(f"Running {method.upper()} simulation...")
        result = run_simulation(config, method)
        results[method] = result
        
        df = results_to_dataframe(result, config)
        dataframes[method] = df
        
        csv_path, meta_path = save_results(result, config)
        metadata[method] = {
            "method": result.method,
            "steps": result.steps,
            "termination_reason": result.termination_reason,
            "exit_time": result.exit_time,
            "exit_position": result.exit_position,
            "exit_velocity": result.exit_velocity,
            "config": result.config_snapshot,
        }
        
        print(f"  Termination: {result.termination_reason}")
        print(f"  Steps: {result.steps}")
        print(f"  Exit time: {result.exit_time}")
        print(f"  Exit position: {result.exit_position}")
        print(f"  Saved: {csv_path}")
        print()
    
    # Save summary
    if "euler" in metadata and "rk4" in metadata:
        summary_path = save_summary(metadata["euler"], metadata["rk4"], config)
        print(f"Summary saved: {summary_path}")
    
    # Generate plots if both methods run
    if "euler" in dataframes and "rk4" in dataframes:
        print("Generating plots...")
        paths = generate_all_plots(
            config,
            dataframes["euler"], metadata["euler"],
            dataframes["rk4"], metadata["rk4"]
        )
        for p in paths:
            print(f"  Saved: {p}")
        print()
    
    # Generate Excel workbook
    if "euler" in results and "rk4" in results:
        print("Generating Excel workbook...")
        excel_path = generate_excel_workbook(
            results["euler"], dataframes["euler"],
            results["rk4"], dataframes["rk4"],
            config
        )
        print(f"  Saved: {excel_path}")
        print()
    
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize existing results."""
    base_config = get_default_config(Z=args.Z)
    
    # Determine run output directory
    if args.run_id:
        output_dir = Path(args.output) / args.run_id
    else:
        # Try to find the latest run directory
        output_dir = _find_latest_run_dir(Path(args.output), args.Z)
        if output_dir is None:
            print("Error: No simulation results found.")
            print(f"Run 'python -m src.main simulate' first to generate results,")
            print(f"or specify --run-id to select a specific run.")
            return 1
    
    config = Config(
        particle=base_config.particle,
        fluid=base_config.fluid,
        gravity=base_config.gravity,
        initial=base_config.initial,
        domain=base_config.domain,
        simulation=SimulationConfig(
            dt=base_config.simulation.dt,
            t_max=base_config.simulation.t_max,
            max_steps=base_config.simulation.max_steps,
            output_dir=output_dir
        )
    )
    
    exists = check_results_exist(config, ["euler", "rk4"])
    
    if not any(exists.values()):
        print(f"Error: No simulation results found in {output_dir}.")
        print(f"Run 'python -m src.main simulate' first to generate results.")
        return 1
    
    print(f"Loading results from: {output_dir}")
    print("Generating plots from existing results...")
    
    for method in ["euler", "rk4"]:
        if exists[method]:
            df, meta = load_results(method, config)
            print(f"Loaded {method}: {len(df)} steps")
    
    # Generate plots
    if exists["euler"] and exists["rk4"]:
        euler_df, euler_meta = load_results("euler", config)
        rk4_df, rk4_meta = load_results("rk4", config)
        paths = generate_all_plots(config, euler_df, euler_meta, rk4_df, rk4_meta)
        for p in paths:
            print(f"  Saved: {p}")
    elif exists["euler"]:
        euler_df, euler_meta = load_results("euler", config)
        from .visualization import create_flow_and_trajectory_plot
        path = config.simulation.output_dir / "euler_trajectory.png"
        create_flow_and_trajectory_plot(euler_df, euler_meta, config, path)
        print(f"  Saved: {path}")
    elif exists["rk4"]:
        rk4_df, rk4_meta = load_results("rk4", config)
        from .visualization import create_flow_and_trajectory_plot
        path = config.simulation.output_dir / "rk4_trajectory.png"
        create_flow_and_trajectory_plot(rk4_df, rk4_meta, config, path)
        print(f"  Saved: {path}")
    
    return 0


def _find_latest_run_dir(base_output_dir: Path, Z: int) -> Path | None:
    """Find the latest run directory for the given Z value."""
    if not base_output_dir.exists():
        return None
    
    run_dirs = list(base_output_dir.glob(f"run_Z{Z}_*"))
    if not run_dirs:
        return None
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs[0]


def cmd_animate(args: argparse.Namespace) -> int:
    """Animate existing results."""
    base_config = get_default_config(Z=args.Z)
    
    # Determine run output directory
    if args.run_id:
        output_dir = Path(args.output) / args.run_id
    else:
        output_dir = _find_latest_run_dir(Path(args.output), args.Z)
        if output_dir is None:
            print(f"Error: No {args.method} results found.")
            print(f"Run 'python -m src.main simulate --method {args.method}' first,")
            print(f"or specify --run-id to select a specific run.")
            return 1
    
    config = Config(
        particle=base_config.particle,
        fluid=base_config.fluid,
        gravity=base_config.gravity,
        initial=base_config.initial,
        domain=base_config.domain,
        simulation=SimulationConfig(
            dt=base_config.simulation.dt,
            t_max=base_config.simulation.t_max,
            max_steps=base_config.simulation.max_steps,
            output_dir=output_dir
        )
    )
    
    exists = check_results_exist(config, [args.method])
    
    if not exists[args.method]:
        print(f"Error: No {args.method} results found in {output_dir}.")
        print(f"Run 'python -m src.main simulate --method {args.method}' first,")
        print(f"or specify --run-id to select a specific run.")
        return 1
    
    output_path = None
    if args.save:
        output_path = config.simulation.output_dir / f"particle_animation_{args.method}.gif"
    
    print(f"Loading results from: {output_dir}")
    print(f"Creating {args.method.upper()} animation...")
    try:
        anim = animate_from_files(
            args.method, config, output_path,
            frame_skip=args.frame_skip,
            interval=args.interval, fps=args.fps, dpi=args.dpi
        )
        if output_path:
            print(f"Animation saved: {output_path}")
        else:
            print("Animation created (not saved). Use --save to save as GIF.")
            import matplotlib.pyplot as plt
            plt.show()
    except Exception as e:
        print(f"Error creating animation: {e}")
        return 1
    
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export existing results to Excel."""
    base_config = get_default_config(Z=args.Z)
    
    # Determine run output directory
    if args.run_id:
        output_dir = Path(args.output) / args.run_id
    else:
        output_dir = _find_latest_run_dir(Path(args.output), args.Z)
        if output_dir is None:
            print("Error: Both Euler and RK4 results required for Excel export.")
            print(f"Run 'python -m src.main simulate --method both' first,")
            print(f"or specify --run-id to select a specific run.")
            return 1
    
    config = Config(
        particle=base_config.particle,
        fluid=base_config.fluid,
        gravity=base_config.gravity,
        initial=base_config.initial,
        domain=base_config.domain,
        simulation=SimulationConfig(
            dt=base_config.simulation.dt,
            t_max=base_config.simulation.t_max,
            max_steps=base_config.simulation.max_steps,
            output_dir=output_dir
        )
    )
    
    exists = check_results_exist(config, ["euler", "rk4"])
    
    if not exists["euler"] or not exists["rk4"]:
        print(f"Error: Both Euler and RK4 results required for Excel export in {output_dir}.")
        missing = [m for m, e in exists.items() if not e]
        print(f"Missing: {', '.join(missing)}")
        print("Run 'python -m src.main simulate --method both' first,")
        print("or specify --run-id to select a specific run.")
        return 1
    
    print(f"Loading results from: {output_dir}")
    print("Generating Excel workbook from existing results...")
    try:
        excel_path = export_from_files(config)
        print(f"Excel workbook saved: {excel_path}")
    except Exception as e:
        print(f"Error generating Excel: {e}")
        return 1
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lagrangian Particle Tracking Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main simulate --method both --Z 5
  python -m src.main simulate --method euler --Z 5 --dt 0.001
  python -m src.main visualize --Z 5
  python -m src.main animate --method rk4 --Z 5 --save
  python -m src.main export --Z 5
  python -m src.main visualize --Z 5 --run-id run_Z5_20240115_143022
        """
    )
    
    parser.add_argument(
        "--Z", type=int, default=0,
        help="Last digit of roll number (0-9), used for gravity calculation"
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001,
        help="Timestep (default: 0.001)"
    )
    parser.add_argument(
        "--t-max", type=float, default=100.0,
        help="Maximum simulation time (default: 100.0)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1_000_000,
        help="Maximum number of steps (default: 1000000)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument(
        "--method", choices=["euler", "rk4", "both"], default="both",
        help="Integration method (default: both)"
    )
    sim_parser.set_defaults(func=cmd_simulate)
    
    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Generate plots from saved results")
    vis_parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID to visualize (default: latest run for given Z)"
    )
    vis_parser.set_defaults(func=cmd_visualize)
    
    # Animate command
    anim_parser = subparsers.add_parser("animate", help="Create animation from saved results")
    anim_parser.add_argument(
        "--method", choices=["euler", "rk4"], required=True,
        help="Method to animate"
    )
    anim_parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID to animate (default: latest run for given Z)"
    )
    anim_parser.add_argument(
        "--save", action="store_true",
        help="Save animation as GIF"
    )
    anim_parser.add_argument(
        "--interval", type=int, default=20,
        help="Frame interval in ms (default: 20)"
    )
    anim_parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for saved animation (default: 30)"
    )
    anim_parser.add_argument(
        "--dpi", type=int, default=100,
        help="DPI for saved animation (default: 100)"
    )
    anim_parser.add_argument(
        "--frame-skip", type=int, default=20,
        help="Skip frames for faster animation (default: 20)"
    )
    anim_parser.set_defaults(func=cmd_animate)
    
    # Export command
    exp_parser = subparsers.add_parser("export", help="Generate Excel workbook from saved results")
    exp_parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID to export (default: latest run for given Z)"
    )
    exp_parser.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())