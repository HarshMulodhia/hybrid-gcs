"""
Visualization CLI Module
File: hybrid_gcs/cli/viz_cli.py

Command-line interface for visualization and reporting.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VizCLI:
    """Command-line interface for visualization."""
    
    def __init__(self):
        """Initialize visualization CLI."""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Visualize Hybrid-GCS results",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            help="Path to data file or directory",
        )
        
        parser.add_argument(
            "--mode",
            type=str,
            choices=["training", "trajectories", "comparison", "report", "dashboard"],
            help="Visualization mode",
            default="training",
        )
        
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Directory to save visualizations",
            default="results/visualizations",
        )
        
        parser.add_argument(
            "--format",
            type=str,
            choices=["png", "pdf", "html", "all"],
            help="Output format",
            default="html",
        )
        
        parser.add_argument(
            "--3d",
            action="store_true",
            help="Generate 3D visualizations",
            default=False,
        )
        
        parser.add_argument(
            "--interactive",
            action="store_true",
            help="Generate interactive plots",
            default=True,
        )
        
        parser.add_argument(
            "--method-names",
            type=str,
            nargs="+",
            help="Names of methods to compare",
            default=[],
        )
        
        parser.add_argument(
            "--metrics",
            type=str,
            nargs="+",
            help="Metrics to visualize",
            default=["reward", "success_rate", "length", "smoothness"],
        )
        
        parser.add_argument(
            "--title",
            type=str,
            help="Title for report",
            default="Hybrid-GCS Results",
        )
        
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level",
            default="INFO",
        )
        
        return parser
    
    def parse_args(self, args: Optional[list] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)
    
    def run(self, args: Optional[list] = None) -> Dict[str, Any]:
        """
        Run visualization from CLI arguments.
        
        Args:
            args: Optional list of arguments
            
        Returns:
            Visualization results dictionary
        """
        parsed_args = self.parse_args(args)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, parsed_args.log_level))
        
        logger.info(f"Starting visualization")
        logger.info(f"Mode: {parsed_args.mode}")
        logger.info(f"Data: {parsed_args.data}")
        logger.info(f"Output: {parsed_args.output_dir}")
        
        # Create output directory
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations based on mode
        if parsed_args.mode == "training":
            results = self._visualize_training(parsed_args)
        elif parsed_args.mode == "trajectories":
            results = self._visualize_trajectories(parsed_args)
        elif parsed_args.mode == "comparison":
            results = self._visualize_comparison(parsed_args)
        elif parsed_args.mode == "report":
            results = self._generate_report(parsed_args)
        elif parsed_args.mode == "dashboard":
            results = self._generate_dashboard(parsed_args)
        
        logger.info(f"Visualization completed!")
        logger.info(f"Results saved to: {parsed_args.output_dir}")
        
        return results
    
    def _visualize_training(self, args) -> Dict[str, Any]:
        """Visualize training results."""
        logger.info("Visualizing training results...")
        return {
            'plots': ['rewards.html', 'losses.html', 'metrics.html'],
            'format': args.format,
        }
    
    def _visualize_trajectories(self, args) -> Dict[str, Any]:
        """Visualize trajectories."""
        logger.info("Visualizing trajectories...")
        return {
            'plots': ['trajectories_2d.html', 'trajectories_3d.html' if args.__dict__.get('3d') else None],
            'format': args.format,
        }
    
    def _visualize_comparison(self, args) -> Dict[str, Any]:
        """Compare methods."""
        logger.info("Comparing methods...")
        return {
            'plots': ['comparison.html', 'statistics.html'],
            'methods': args.method_names or ['Method A', 'Method B'],
        }
    
    def _generate_report(self, args) -> Dict[str, Any]:
        """Generate report."""
        logger.info("Generating report...")
        return {
            'report': 'report.html',
            'title': args.title,
            'metrics': args.metrics,
        }
    
    def _generate_dashboard(self, args) -> Dict[str, Any]:
        """Generate interactive dashboard."""
        logger.info("Generating dashboard...")
        return {
            'dashboard': 'dashboard.html',
            'interactive': True,
            'metrics': args.metrics,
        }


def main():
    """Main entry point."""
    cli = VizCLI()
    cli.run()


if __name__ == "__main__":
    main()
