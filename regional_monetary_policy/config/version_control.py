"""
Version control system for reproducible analysis settings.
Tracks configuration changes, analysis runs, and results for reproducibility.
"""

import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import shutil
import pickle
import git
from git.exc import InvalidGitRepositoryError


@dataclass
class AnalysisRun:
    """Record of a single analysis run."""
    
    run_id: str
    timestamp: datetime.datetime
    config_hash: str
    config_version: str
    parameters: Dict[str, Any]
    results_path: Optional[str] = None
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisRun':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConfigVersion:
    """Version record for configuration."""
    
    version_id: str
    timestamp: datetime.datetime
    config_hash: str
    config_data: Dict[str, Any]
    description: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigVersion':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class VersionControlManager:
    """Manages version control for analysis configurations and results."""
    
    def __init__(self, base_directory: str = ".analysis_versions"):
        self.base_dir = Path(base_directory)
        self.config_dir = self.base_dir / "configs"
        self.runs_dir = self.base_dir / "runs"
        self.results_dir = self.base_dir / "results"
        
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        for directory in [self.base_dir, self.config_dir, self.runs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking files
        self.config_versions_file = self.base_dir / "config_versions.json"
        self.analysis_runs_file = self.base_dir / "analysis_runs.json"
        
        # Load existing versions and runs
        self.config_versions = self._load_config_versions()
        self.analysis_runs = self._load_analysis_runs()
        
        # Try to initialize git repository
        self.git_repo = self._init_git_repo()
    
    def _init_git_repo(self) -> Optional[git.Repo]:
        """Initialize git repository for version control."""
        try:
            # Try to open existing repository
            repo = git.Repo(self.base_dir)
            self.logger.info("Using existing git repository for version control")
            return repo
        except InvalidGitRepositoryError:
            try:
                # Initialize new repository
                repo = git.Repo.init(self.base_dir)
                
                # Create .gitignore
                gitignore_path = self.base_dir / ".gitignore"
                with open(gitignore_path, 'w') as f:
                    f.write("*.pyc\n__pycache__/\n*.log\n.DS_Store\n")
                
                # Initial commit
                repo.index.add([".gitignore"])
                repo.index.commit("Initial commit")
                
                self.logger.info("Initialized git repository for version control")
                return repo
            except Exception as e:
                self.logger.warning(f"Could not initialize git repository: {e}")
                return None
    
    def _load_config_versions(self) -> Dict[str, ConfigVersion]:
        """Load configuration versions from file."""
        if not self.config_versions_file.exists():
            return {}
        
        try:
            with open(self.config_versions_file, 'r') as f:
                data = json.load(f)
            
            return {
                version_id: ConfigVersion.from_dict(version_data)
                for version_id, version_data in data.items()
            }
        except Exception as e:
            self.logger.error(f"Error loading config versions: {e}")
            return {}
    
    def _load_analysis_runs(self) -> Dict[str, AnalysisRun]:
        """Load analysis runs from file."""
        if not self.analysis_runs_file.exists():
            return {}
        
        try:
            with open(self.analysis_runs_file, 'r') as f:
                data = json.load(f)
            
            return {
                run_id: AnalysisRun.from_dict(run_data)
                for run_id, run_data in data.items()
            }
        except Exception as e:
            self.logger.error(f"Error loading analysis runs: {e}")
            return {}
    
    def _save_config_versions(self):
        """Save configuration versions to file."""
        data = {
            version_id: version.to_dict()
            for version_id, version in self.config_versions.items()
        }
        
        with open(self.config_versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_analysis_runs(self):
        """Save analysis runs to file."""
        data = {
            run_id: run.to_dict()
            for run_id, run in self.analysis_runs.items()
        }
        
        with open(self.analysis_runs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Compute hash of configuration data."""
        # Convert to JSON string with sorted keys for consistent hashing
        config_str = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"run_{timestamp}"
    
    def save_config_version(self, config_data: Dict[str, Any], description: str = "", 
                           tags: Optional[List[str]] = None, parent_version: Optional[str] = None) -> str:
        """
        Save a new configuration version.
        
        Args:
            config_data: Configuration dictionary
            description: Description of changes
            tags: Optional tags for the version
            parent_version: Parent version ID
            
        Returns:
            Version ID of saved configuration
        """
        config_hash = self._compute_config_hash(config_data)
        
        # Check if this exact configuration already exists
        for version_id, version in self.config_versions.items():
            if version.config_hash == config_hash:
                self.logger.info(f"Configuration already exists as version {version_id}")
                return version_id
        
        # Create new version
        version_id = self._generate_version_id()
        version = ConfigVersion(
            version_id=version_id,
            timestamp=datetime.datetime.now(),
            config_hash=config_hash,
            config_data=config_data,
            description=description,
            parent_version=parent_version,
            tags=tags or []
        )
        
        # Save configuration file
        config_file = self.config_dir / f"{version_id}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Update versions registry
        self.config_versions[version_id] = version
        self._save_config_versions()
        
        # Git commit if available
        if self.git_repo:
            try:
                self.git_repo.index.add([str(config_file), str(self.config_versions_file)])
                commit_message = f"Add config version {version_id}: {description}"
                self.git_repo.index.commit(commit_message)
                
                # Add tags if specified
                for tag in (tags or []):
                    try:
                        self.git_repo.create_tag(f"config_{tag}_{version_id}")
                    except Exception as e:
                        self.logger.warning(f"Could not create git tag {tag}: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Could not commit to git: {e}")
        
        self.logger.info(f"Saved configuration version {version_id}")
        return version_id
    
    def load_config_version(self, version_id: str) -> Dict[str, Any]:
        """
        Load configuration by version ID.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Configuration dictionary
        """
        if version_id not in self.config_versions:
            raise ValueError(f"Configuration version {version_id} not found")
        
        config_file = self.config_dir / f"{version_id}.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file for version {version_id} not found")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def start_analysis_run(self, config_data: Dict[str, Any], parameters: Dict[str, Any] = None) -> str:
        """
        Start a new analysis run.
        
        Args:
            config_data: Configuration for the run
            parameters: Additional parameters for the run
            
        Returns:
            Run ID
        """
        # Save configuration version if needed
        config_hash = self._compute_config_hash(config_data)
        config_version = None
        
        for version_id, version in self.config_versions.items():
            if version.config_hash == config_hash:
                config_version = version_id
                break
        
        if config_version is None:
            config_version = self.save_config_version(
                config_data, 
                description=f"Auto-saved for run at {datetime.datetime.now()}"
            )
        
        # Create analysis run record
        run_id = self._generate_run_id()
        run = AnalysisRun(
            run_id=run_id,
            timestamp=datetime.datetime.now(),
            config_hash=config_hash,
            config_version=config_version,
            parameters=parameters or {},
            status="running"
        )
        
        # Save run record
        self.analysis_runs[run_id] = run
        self._save_analysis_runs()
        
        self.logger.info(f"Started analysis run {run_id} with config version {config_version}")
        return run_id
    
    def complete_analysis_run(self, run_id: str, results_path: Optional[str] = None, 
                            execution_time: Optional[float] = None):
        """
        Mark analysis run as completed.
        
        Args:
            run_id: Run ID to complete
            results_path: Path to results
            execution_time: Execution time in seconds
        """
        if run_id not in self.analysis_runs:
            raise ValueError(f"Analysis run {run_id} not found")
        
        run = self.analysis_runs[run_id]
        run.status = "completed"
        run.results_path = results_path
        run.execution_time = execution_time
        
        # Save results if path provided
        if results_path and Path(results_path).exists():
            run_results_dir = self.results_dir / run_id
            run_results_dir.mkdir(exist_ok=True)
            
            # Copy results
            if Path(results_path).is_file():
                shutil.copy2(results_path, run_results_dir)
            else:
                shutil.copytree(results_path, run_results_dir / Path(results_path).name, dirs_exist_ok=True)
        
        self._save_analysis_runs()
        
        # Git commit if available
        if self.git_repo:
            try:
                self.git_repo.index.add([str(self.analysis_runs_file)])
                if results_path:
                    # Add results to git (be careful with large files)
                    result_files = list((self.results_dir / run_id).rglob("*"))
                    if len(result_files) < 100:  # Avoid adding too many files
                        self.git_repo.index.add([str(f) for f in result_files if f.is_file()])
                
                commit_message = f"Complete analysis run {run_id}"
                self.git_repo.index.commit(commit_message)
            except Exception as e:
                self.logger.warning(f"Could not commit run completion to git: {e}")
        
        self.logger.info(f"Completed analysis run {run_id}")
    
    def fail_analysis_run(self, run_id: str, error_message: str):
        """
        Mark analysis run as failed.
        
        Args:
            run_id: Run ID to fail
            error_message: Error message
        """
        if run_id not in self.analysis_runs:
            raise ValueError(f"Analysis run {run_id} not found")
        
        run = self.analysis_runs[run_id]
        run.status = "failed"
        run.error_message = error_message
        
        self._save_analysis_runs()
        self.logger.error(f"Analysis run {run_id} failed: {error_message}")
    
    def get_config_history(self, limit: Optional[int] = None) -> List[ConfigVersion]:
        """
        Get configuration version history.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of configuration versions, sorted by timestamp
        """
        versions = sorted(self.config_versions.values(), key=lambda v: v.timestamp, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def get_run_history(self, config_version: Optional[str] = None, 
                       status: Optional[str] = None, limit: Optional[int] = None) -> List[AnalysisRun]:
        """
        Get analysis run history.
        
        Args:
            config_version: Filter by configuration version
            status: Filter by run status
            limit: Maximum number of runs to return
            
        Returns:
            List of analysis runs, sorted by timestamp
        """
        runs = list(self.analysis_runs.values())
        
        # Apply filters
        if config_version:
            runs = [run for run in runs if run.config_version == config_version]
        
        if status:
            runs = [run for run in runs if run.status == status]
        
        # Sort by timestamp
        runs = sorted(runs, key=lambda r: r.timestamp, reverse=True)
        
        if limit:
            runs = runs[:limit]
        
        return runs
    
    def compare_configs(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two configuration versions.
        
        Args:
            version1: First version ID
            version2: Second version ID
            
        Returns:
            Dictionary with comparison results
        """
        config1 = self.load_config_version(version1)
        config2 = self.load_config_version(version2)
        
        def find_differences(dict1, dict2, path=""):
            """Recursively find differences between dictionaries."""
            differences = []
            
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in dict1:
                    differences.append({
                        'type': 'added',
                        'path': current_path,
                        'value': dict2[key]
                    })
                elif key not in dict2:
                    differences.append({
                        'type': 'removed',
                        'path': current_path,
                        'value': dict1[key]
                    })
                elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    differences.extend(find_differences(dict1[key], dict2[key], current_path))
                elif dict1[key] != dict2[key]:
                    differences.append({
                        'type': 'changed',
                        'path': current_path,
                        'old_value': dict1[key],
                        'new_value': dict2[key]
                    })
            
            return differences
        
        differences = find_differences(config1, config2)
        
        return {
            'version1': version1,
            'version2': version2,
            'differences': differences,
            'summary': {
                'total_changes': len(differences),
                'added': len([d for d in differences if d['type'] == 'added']),
                'removed': len([d for d in differences if d['type'] == 'removed']),
                'changed': len([d for d in differences if d['type'] == 'changed'])
            }
        }
    
    def create_reproducibility_report(self, run_id: str) -> Dict[str, Any]:
        """
        Create a reproducibility report for an analysis run.
        
        Args:
            run_id: Analysis run ID
            
        Returns:
            Reproducibility report
        """
        if run_id not in self.analysis_runs:
            raise ValueError(f"Analysis run {run_id} not found")
        
        run = self.analysis_runs[run_id]
        config_version = self.config_versions[run.config_version]
        
        report = {
            'run_info': {
                'run_id': run_id,
                'timestamp': run.timestamp.isoformat(),
                'status': run.status,
                'execution_time': run.execution_time
            },
            'configuration': {
                'version_id': run.config_version,
                'config_hash': run.config_hash,
                'description': config_version.description,
                'tags': config_version.tags
            },
            'reproducibility': {
                'config_file': str(self.config_dir / f"{run.config_version}.json"),
                'results_path': run.results_path,
                'git_commit': None
            },
            'environment': {
                'python_version': None,  # Could be captured during run
                'package_versions': None,  # Could be captured during run
                'system_info': None  # Could be captured during run
            }
        }
        
        # Add git information if available
        if self.git_repo:
            try:
                # Find commit for this run
                commits = list(self.git_repo.iter_commits(paths=[str(self.analysis_runs_file)]))
                for commit in commits:
                    if run_id in commit.message:
                        report['reproducibility']['git_commit'] = commit.hexsha
                        break
            except Exception as e:
                self.logger.warning(f"Could not get git information: {e}")
        
        return report
    
    def export_version_history(self, output_path: str):
        """
        Export complete version history to file.
        
        Args:
            output_path: Path to export file
        """
        export_data = {
            'config_versions': {
                version_id: version.to_dict()
                for version_id, version in self.config_versions.items()
            },
            'analysis_runs': {
                run_id: run.to_dict()
                for run_id, run in self.analysis_runs.items()
            },
            'export_timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported version history to {output_path}")
    
    def cleanup_old_versions(self, keep_days: int = 30, keep_tagged: bool = True):
        """
        Clean up old configuration versions and runs.
        
        Args:
            keep_days: Number of days to keep
            keep_tagged: Whether to keep tagged versions
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        # Clean up config versions
        versions_to_remove = []
        for version_id, version in self.config_versions.items():
            if version.timestamp < cutoff_date:
                if keep_tagged and version.tags:
                    continue
                versions_to_remove.append(version_id)
        
        for version_id in versions_to_remove:
            # Remove config file
            config_file = self.config_dir / f"{version_id}.json"
            if config_file.exists():
                config_file.unlink()
            
            # Remove from registry
            del self.config_versions[version_id]
        
        # Clean up analysis runs
        runs_to_remove = []
        for run_id, run in self.analysis_runs.items():
            if run.timestamp < cutoff_date:
                runs_to_remove.append(run_id)
        
        for run_id in runs_to_remove:
            # Remove results directory
            run_results_dir = self.results_dir / run_id
            if run_results_dir.exists():
                shutil.rmtree(run_results_dir)
            
            # Remove from registry
            del self.analysis_runs[run_id]
        
        # Save updated registries
        self._save_config_versions()
        self._save_analysis_runs()
        
        self.logger.info(f"Cleaned up {len(versions_to_remove)} config versions and {len(runs_to_remove)} analysis runs")