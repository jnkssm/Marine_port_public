"""
Main entry point for the marine port optimization simulation
Choose between web GUI or batch mode
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Marine Port Optimization Simulation')
    parser.add_argument('--mode', type=str, choices=['web', 'batch'], default='web',
                       help='Run mode: web GUI or batch simulation')
    parser.add_argument('--headless', action='store_true',
                       help='Run batch mode without plots')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        print("🚀 Starting Web GUI...")
        from web_gui import app
        # Web GUI will run in its own module
        print("Web server starting...")
    else:
        print("🚀 Running Batch Simulation...")
        # You can implement batch mode here
        print("Batch mode not yet implemented")
        print("Use --mode web for the web interface")

if __name__ == "__main__":
    main()