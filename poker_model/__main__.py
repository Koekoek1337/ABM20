import json
import argparse
import pathlib

from simulation.running import run_multiple_simulations
from simulation.analysis import create_comprehensive_analysis

# TODO: Parameter commandline argument
# TODO: Add new model parameters
def main(jobfilepath):
    print("Starting comprehensive poker simulation analysis...")
    print("This will run multiple scenarios with different risk aversion bounds.")
    
    # Run the comprehensive analysis
    simulation_parms = None
    with open(jobfilepath) as jsonFile:
        simulation_parms = json.load(jsonFile)
    
    if "jobType" not in simulation_parms or simulation_parms["jobType"] == "batchrun":
        evolution_df, final_agents_df = run_multiple_simulations(**simulation_parms)
    elif simulation_parms["jobType"] == "animate":
        raise NotImplementedError() # TODO batch animations 
    
    # Create all the analysis plots
    create_comprehensive_analysis(simulation_parms["scenarios"], evolution_df, final_agents_df)
    
    print("\nAnalysis complete! Check the plots for insights into risk aversion effects.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ABM Group 20 poker model", description="TODO")
    parser.add_argument("jobfilepath", type=pathlib.Path, nargs='?', default=pathlib.Path("./jobs/job.json"))
    args = parser.parse_args()
    
    main(args.jobfilepath)