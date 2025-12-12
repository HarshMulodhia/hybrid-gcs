"""
HTML Report Generation
File: hybrid_gcs/visualization/report_generator.py

Generates comprehensive HTML reports with analysis and visualizations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive HTML reports.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized ReportGenerator: output_dir={output_dir}")
    
    def generate_training_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        plots: Optional[Dict[str, str]] = None,
        output_file: str = "training_report.html",
    ) -> str:
        """
        Generate training report.
        
        Args:
            config: Training configuration
            metrics: Training metrics
            plots: Dictionary of {plot_name: plot_html_path}
            output_file: Output file name
            
        Returns:
            Path to generated report
        """
        html = self._generate_training_report_html(config, metrics, plots)
        
        output_path = f"{self.output_dir}/{output_file}"
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Saved training report to {output_path}")
        return output_path
    
    def generate_evaluation_report(
        self,
        results: Dict[str, Any],
        comparisons: Optional[Dict[str, Any]] = None,
        output_file: str = "evaluation_report.html",
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results
            comparisons: Optional comparison results
            output_file: Output file name
            
        Returns:
            Path to generated report
        """
        html = self._generate_evaluation_report_html(results, comparisons)
        
        output_path = f"{self.output_dir}/{output_file}"
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Saved evaluation report to {output_path}")
        return output_path
    
    def _generate_training_report_html(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        plots: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate HTML for training report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid-GCS Training Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.5em;
            color: #667eea;
            margin-top: 8px;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .config-table th, .config-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .config-table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
        .plot-container {{
            margin: 20px 0;
            border-radius: 5px;
            overflow: hidden;
        }}
        .plot-container iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Hybrid-GCS Training Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>📊 Training Metrics</h2>
            <div class="metrics-grid">
"""
        
        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)
                
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{key}</div>
                    <div class="metric-value">{display_value}</div>
                </div>
"""
        
        html += """
            </div>
        </div>
        
        <div class="section">
            <h2>⚙️ Configuration</h2>
            <table class="config-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for key, value in config.items():
            html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
"""
        
        # Add plots
        if plots:
            html += """
        <div class="section">
            <h2>📈 Visualizations</h2>
"""
            for plot_name, plot_path in plots.items():
                html += f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <iframe src="{plot_path}"></iframe>
            </div>
"""
            html += """
        </div>
"""
        
        html += """
        <div class="footer">
            <p>Hybrid-GCS: Combining GCS Decomposition with Reinforcement Learning</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_evaluation_report_html(
        self,
        results: Dict[str, Any],
        comparisons: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate HTML for evaluation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hybrid-GCS Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background: white; padding: 20px; margin: 10px 0; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Evaluation Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Results</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        
        for key, value in results.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html
