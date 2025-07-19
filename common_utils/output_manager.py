#!/usr/bin/env python3
"""
Output Manager for Multi-Agent Data Analysis Framework
Manages timestamped output folders for storing analysis results, logs, and reports
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging


class OutputManager:
    """Manages output folders and file organization for analysis results"""
    
    def __init__(self, base_output_dir: str = "outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.current_session_dir: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        
        # Ensure base output directory exists
        self.base_output_dir.mkdir(exist_ok=True)
        
    def create_session_folder(self, dataset_name: str, analysis_query: str = "") -> Path:
        """
        Create a timestamped session folder for a dataset analysis
        
        Args:
            dataset_name: Name of the dataset being analyzed
            analysis_query: The analysis query/description
            
        Returns:
            Path to the created session folder
        """
        # Clean dataset name for folder creation
        clean_dataset_name = self._clean_filename(dataset_name)
        if clean_dataset_name.endswith('.csv'):
            clean_dataset_name = clean_dataset_name[:-4]
        elif clean_dataset_name.endswith('.tdsx'):
            clean_dataset_name = clean_dataset_name[:-5]
        elif clean_dataset_name.endswith('.json'):
            clean_dataset_name = clean_dataset_name[:-5]
            
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create dataset folder if it doesn't exist
        dataset_dir = self.base_output_dir / clean_dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Create timestamped session folder
        session_dir = dataset_dir / timestamp
        session_dir.mkdir(exist_ok=True)
        
        # Create subfolders
        (session_dir / "logs").mkdir(exist_ok=True)
        (session_dir / "reports").mkdir(exist_ok=True)
        (session_dir / "data").mkdir(exist_ok=True)
        (session_dir / "analysis_results").mkdir(exist_ok=True)
        
        # Save session metadata
        session_metadata = {
            "dataset_name": dataset_name,
            "analysis_query": analysis_query,
            "created_at": datetime.now().isoformat(),
            "session_id": timestamp,
            "folder_structure": {
                "logs": "Agent execution logs and debug information",
                "reports": "Generated reports (HTML, PDF, markdown)",
                "data": "Processed data files and intermediates",
                "analysis_results": "Raw analysis results and insights"
            }
        }
        
        self._save_json(session_dir / "session_metadata.json", session_metadata)
        
        # Set as current session
        self.current_session_dir = session_dir
        
        self.logger.info(f"Created analysis session folder: {session_dir}")
        return session_dir
    
    def save_analysis_results(self, results: Dict[str, Any], filename: str = "pipeline_results.json") -> Path:
        """Save analysis pipeline results to the current session"""
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session_folder first.")
            
        results_file = self.current_session_dir / "analysis_results" / filename
        self._save_json(results_file, results)
        
        self.logger.info(f"Saved analysis results to: {results_file}")
        return results_file
    
    def save_log(self, log_content: str, log_name: str = "analysis_log.txt") -> Path:
        """Save log content to the current session"""
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session_folder first.")
            
        log_file = self.current_session_dir / "logs" / log_name
        
        # Append timestamp to log content
        timestamped_content = f"[{datetime.now().isoformat()}] {log_content}\n"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(timestamped_content)
            
        return log_file
    
    def save_report(self, report_content: str, report_name: str, report_format: str = "html") -> Path:
        """Save a generated report to the current session"""
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session_folder first.")
            
        if not report_name.endswith(f".{report_format}"):
            report_name = f"{report_name}.{report_format}"
            
        report_file = self.current_session_dir / "reports" / report_name
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        self.logger.info(f"Saved report to: {report_file}")
        return report_file
    
    def save_data_file(self, data_content: Any, filename: str) -> Path:
        """Save processed data to the current session"""
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session_folder first.")
            
        data_file = self.current_session_dir / "data" / filename
        
        if filename.endswith('.json'):
            self._save_json(data_file, data_content)
        else:
            # For other file types, assume it's already formatted content
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(str(data_content))
                
        return data_file
    
    def create_executive_summary(self, analysis_results: Dict[str, Any], 
                               insights: List[str], 
                               recommendations: List[str]) -> Path:
        """Create an executive summary document"""
        if not self.current_session_dir:
            raise ValueError("No active session. Call create_session_folder first.")
            
        # Load session metadata
        metadata_file = self.current_session_dir / "session_metadata.json"
        with open(metadata_file, "r") as f:
            session_metadata = json.load(f)
        
        # Create executive summary
        summary = {
            "executive_summary": {
                "dataset": session_metadata["dataset_name"],
                "analysis_query": session_metadata["analysis_query"],
                "analysis_date": session_metadata["created_at"],
                "session_id": session_metadata["session_id"],
                "key_insights": insights,
                "recommendations": recommendations,
                "pipeline_status": analysis_results.get("status", "unknown"),
                "stages_completed": list(analysis_results.get("stages", {}).keys()) if "stages" in analysis_results else [],
                "data_quality_score": self._calculate_data_quality_score(analysis_results),
                "confidence_level": self._calculate_confidence_level(analysis_results)
            }
        }
        
        # Save JSON version
        summary_file = self.save_analysis_results(summary, "executive_summary.json")
        
        # Create markdown version
        markdown_summary = self._create_markdown_summary(summary["executive_summary"])
        markdown_file = self.save_report(markdown_summary, "executive_summary", "md")
        
        return summary_file
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session contents"""
        if not self.current_session_dir:
            return {}
            
        summary = {
            "session_path": str(self.current_session_dir),
            "files_created": {},
            "total_size_mb": 0
        }
        
        for subfolder in ["logs", "reports", "data", "analysis_results"]:
            folder_path = self.current_session_dir / subfolder
            files = list(folder_path.glob("*"))
            
            summary["files_created"][subfolder] = []
            for file_path in files:
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    summary["files_created"][subfolder].append({
                        "name": file_path.name,
                        "size_mb": round(size_mb, 2),
                        "path": str(file_path)
                    })
                    summary["total_size_mb"] += size_mb
        
        summary["total_size_mb"] = round(summary["total_size_mb"], 2)
        return summary
    
    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """List all previous analysis sessions"""
        sessions = []
        
        for dataset_dir in self.base_output_dir.iterdir():
            if dataset_dir.is_dir():
                for session_dir in dataset_dir.iterdir():
                    if session_dir.is_dir():
                        metadata_file = session_dir / "session_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                                metadata["session_path"] = str(session_dir)
                                sessions.append(metadata)
        
        # Sort by creation date, newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions
    
    def cleanup_old_sessions(self, keep_recent: int = 10) -> int:
        """Remove old session folders, keeping only the most recent ones per dataset"""
        deleted_count = 0
        
        for dataset_dir in self.base_output_dir.iterdir():
            if dataset_dir.is_dir():
                sessions = []
                for session_dir in dataset_dir.iterdir():
                    if session_dir.is_dir():
                        sessions.append(session_dir)
                
                # Sort by modification time, newest first
                sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove old sessions
                for session_dir in sessions[keep_recent:]:
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    self.logger.info(f"Deleted old session: {session_dir}")
        
        return deleted_count
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for safe folder creation"""
        # Remove or replace unsafe characters
        unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        return filename.strip()
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data as JSON with proper formatting"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _calculate_data_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate a data quality score based on pipeline results"""
        # Simple scoring based on completed stages
        if "stages" not in results:
            return 0.5
            
        stages = results["stages"]
        completed_stages = sum(1 for stage in stages.values() 
                             if stage.get("status") == "completed")
        total_stages = len(stages)
        
        return round(completed_stages / total_stages if total_stages > 0 else 0, 2)
    
    def _calculate_confidence_level(self, results: Dict[str, Any]) -> str:
        """Calculate confidence level based on analysis results"""
        quality_score = self._calculate_data_quality_score(results)
        
        if quality_score >= 0.9:
            return "High"
        elif quality_score >= 0.7:
            return "Medium"
        elif quality_score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _create_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Create a markdown version of the executive summary"""
        md = f"""# Executive Summary - Data Analysis

## Analysis Overview
- **Dataset**: {summary['dataset']}
- **Query**: {summary['analysis_query']}
- **Date**: {summary['analysis_date']}
- **Session ID**: {summary['session_id']}

## Results
- **Pipeline Status**: {summary['pipeline_status']}
- **Data Quality Score**: {summary['data_quality_score']}
- **Confidence Level**: {summary['confidence_level']}
- **Stages Completed**: {', '.join(summary['stages_completed'])}

## Key Insights
"""
        for i, insight in enumerate(summary['key_insights'], 1):
            md += f"{i}. {insight}\n"
        
        md += "\n## Recommendations\n"
        for i, recommendation in enumerate(summary['recommendations'], 1):
            md += f"{i}. {recommendation}\n"
        
        md += f"\n---\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return md


# Global instance
output_manager = OutputManager() 