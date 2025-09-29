"""
Command-line interface for configuration management.
Provides easy access to configuration, extensibility, and version control features.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from regional_monetary_policy.config.config_manager import ConfigManager
from regional_monetary_policy.config.extensibility import (
    ExtensibilityConfig, SpatialWeightMethod, IdentificationStrategy, 
    ModelExtension, RobustnessCheck, ParameterRestriction
)


@click.group()
@click.option('--config-file', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config_file: Optional[str], verbose: bool):
    """Regional Monetary Policy Analysis Configuration Manager."""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize config manager
    try:
        config_manager = ConfigManager(config_file)
        ctx.ensure_object(dict)
        ctx.obj['config_manager'] = config_manager
        ctx.obj['config_file'] = config_file
    except Exception as e:
        click.echo(f"Error initializing configuration manager: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='config_template.json', help='Output file path')
@click.option('--include-extensibility', is_flag=True, default=True, help='Include extensibility options')
@click.pass_context
def create_template(ctx, output: str, include_extensibility: bool):
    """Create a configuration template file."""
    
    config_manager = ctx.obj['config_manager']
    
    try:
        config_manager.export_configuration_template(output, include_extensibility)
        click.echo(f"Configuration template created: {output}")
    except Exception as e:
        click.echo(f"Error creating template: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config-file', '-c', help='Configuration file to load')
@click.pass_context
def load(ctx, config_file: Optional[str]):
    """Load and validate a configuration file."""
    
    config_manager = ctx.obj['config_manager']
    
    try:
        if config_file:
            ctx.obj['config_file'] = config_file
        
        settings = config_manager.load_config(config_file)
        
        click.echo("Configuration loaded successfully!")
        click.echo(f"Analysis name: {settings.analysis_name}")
        click.echo(f"Output directory: {settings.output_directory}")
        click.echo(f"Regions: {', '.join(settings.data.regions)}")
        
        # Show extensibility features if available
        if config_manager.has_extensibility:
            ext_config = config_manager.extensibility_config
            click.echo(f"Spatial weight methods: {len(ext_config.spatial_weight_methods)}")
            click.echo(f"Model extensions: {len(ext_config.model_extensions)}")
            click.echo(f"Robustness checks: {len(ext_config.robustness_checks)}")
        
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def save(ctx, output: Optional[str]):
    """Save current configuration to file."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.is_configured:
        click.echo("No configuration loaded to save", err=True)
        sys.exit(1)
    
    try:
        save_path = output or ctx.obj.get('config_file') or 'config.json'
        config_manager.save_config(save_path)
        click.echo(f"Configuration saved to: {save_path}")
    except Exception as e:
        click.echo(f"Error saving configuration: {e}", err=True)
        sys.exit(1)


@cli.group()
def version():
    """Version control commands."""
    pass


@version.command('save')
@click.option('--description', '-d', default='', help='Version description')
@click.option('--tag', '-t', multiple=True, help='Tags for this version')
@click.pass_context
def save_version(ctx, description: str, tag: List[str]):
    """Save current configuration as a new version."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_version_control:
        click.echo("Version control not available", err=True)
        sys.exit(1)
    
    if not config_manager.is_configured:
        click.echo("No configuration loaded to save", err=True)
        sys.exit(1)
    
    try:
        version_id = config_manager.save_config_version(description, list(tag))
        click.echo(f"Configuration saved as version: {version_id}")
    except Exception as e:
        click.echo(f"Error saving version: {e}", err=True)
        sys.exit(1)


@version.command('load')
@click.argument('version_id')
@click.pass_context
def load_version(ctx, version_id: str):
    """Load a specific configuration version."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_version_control:
        click.echo("Version control not available", err=True)
        sys.exit(1)
    
    try:
        config_manager.load_config_version(version_id)
        click.echo(f"Loaded configuration version: {version_id}")
    except Exception as e:
        click.echo(f"Error loading version: {e}", err=True)
        sys.exit(1)


@version.command('list')
@click.option('--limit', '-l', type=int, help='Maximum number of versions to show')
@click.pass_context
def list_versions(ctx, limit: Optional[int]):
    """List configuration versions."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_version_control:
        click.echo("Version control not available", err=True)
        sys.exit(1)
    
    try:
        versions = config_manager.get_config_history(limit)
        
        if not versions:
            click.echo("No configuration versions found")
            return
        
        click.echo(f"Configuration versions ({len(versions)} total):")
        click.echo("-" * 80)
        
        for version in versions:
            tags_str = f" [{', '.join(version.tags)}]" if version.tags else ""
            click.echo(f"{version.version_id}{tags_str}")
            click.echo(f"  Date: {version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo(f"  Description: {version.description}")
            click.echo(f"  Hash: {version.config_hash}")
            click.echo()
        
    except Exception as e:
        click.echo(f"Error listing versions: {e}", err=True)
        sys.exit(1)


@version.command('compare')
@click.argument('version1')
@click.argument('version2')
@click.pass_context
def compare_versions(ctx, version1: str, version2: str):
    """Compare two configuration versions."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_version_control:
        click.echo("Version control not available", err=True)
        sys.exit(1)
    
    try:
        comparison = config_manager.version_control.compare_configs(version1, version2)
        
        click.echo(f"Comparing {version1} -> {version2}")
        click.echo("-" * 50)
        
        summary = comparison['summary']
        click.echo(f"Total changes: {summary['total_changes']}")
        click.echo(f"Added: {summary['added']}, Removed: {summary['removed']}, Changed: {summary['changed']}")
        click.echo()
        
        if comparison['differences']:
            click.echo("Differences:")
            for diff in comparison['differences']:
                if diff['type'] == 'added':
                    click.echo(f"  + {diff['path']}: {diff['value']}")
                elif diff['type'] == 'removed':
                    click.echo(f"  - {diff['path']}: {diff['value']}")
                elif diff['type'] == 'changed':
                    click.echo(f"  ~ {diff['path']}: {diff['old_value']} -> {diff['new_value']}")
        
    except Exception as e:
        click.echo(f"Error comparing versions: {e}", err=True)
        sys.exit(1)


@cli.group()
def extensions():
    """Model extension commands."""
    pass


@extensions.command('list')
@click.pass_context
def list_extensions(ctx):
    """List available model extensions."""
    
    click.echo("Available model extensions:")
    click.echo("-" * 30)
    
    for extension in ModelExtension:
        click.echo(f"  {extension.value}")
    
    click.echo("\nAvailable spatial weight methods:")
    click.echo("-" * 35)
    
    for method in SpatialWeightMethod:
        click.echo(f"  {method.value}")
    
    click.echo("\nAvailable identification strategies:")
    click.echo("-" * 38)
    
    for strategy in IdentificationStrategy:
        click.echo(f"  {strategy.value}")


@extensions.command('enable')
@click.argument('extension')
@click.option('--config', '-c', help='Extension configuration (JSON string)')
@click.pass_context
def enable_extension(ctx, extension: str, config: Optional[str]):
    """Enable a model extension."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_extensibility:
        click.echo("Extensibility framework not available", err=True)
        sys.exit(1)
    
    try:
        # Parse extension
        try:
            ext_enum = ModelExtension(extension)
        except ValueError:
            click.echo(f"Unknown extension: {extension}", err=True)
            click.echo(f"Available extensions: {[e.value for e in ModelExtension]}")
            sys.exit(1)
        
        # Parse configuration
        ext_config = {}
        if config:
            try:
                ext_config = json.loads(config)
            except json.JSONDecodeError as e:
                click.echo(f"Invalid JSON configuration: {e}", err=True)
                sys.exit(1)
        
        # Enable extension
        if not config_manager.extensibility_config:
            config_manager.load_extensibility_config()
        
        config_manager.extensibility_config.enable_model_extension(ext_enum, ext_config)
        
        click.echo(f"Enabled extension: {extension}")
        if ext_config:
            click.echo(f"Configuration: {ext_config}")
        
    except Exception as e:
        click.echo(f"Error enabling extension: {e}", err=True)
        sys.exit(1)


@extensions.command('add-robustness-check')
@click.argument('name')
@click.argument('description')
@click.option('--parameters', '-p', help='Parameters (JSON string)')
@click.pass_context
def add_robustness_check(ctx, name: str, description: str, parameters: Optional[str]):
    """Add a robustness check."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.has_extensibility:
        click.echo("Extensibility framework not available", err=True)
        sys.exit(1)
    
    try:
        # Parse parameters
        params = {}
        if parameters:
            try:
                params = json.loads(parameters)
            except json.JSONDecodeError as e:
                click.echo(f"Invalid JSON parameters: {e}", err=True)
                sys.exit(1)
        
        # Add robustness check
        if not config_manager.extensibility_config:
            config_manager.load_extensibility_config()
        
        config_manager.extensibility_config.add_robustness_check(name, description, params)
        
        click.echo(f"Added robustness check: {name}")
        click.echo(f"Description: {description}")
        if params:
            click.echo(f"Parameters: {params}")
        
    except Exception as e:
        click.echo(f"Error adding robustness check: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate current configuration."""
    
    config_manager = ctx.obj['config_manager']
    
    if not config_manager.is_configured:
        click.echo("No configuration loaded to validate", err=True)
        sys.exit(1)
    
    try:
        # Validate main configuration
        errors = config_manager.settings.validate()
        
        # Validate extensibility configuration
        if config_manager.has_extensibility and config_manager.extensibility_config:
            ext_errors = config_manager.extensibility_config.validate()
            errors.extend(ext_errors)
        
        if errors:
            click.echo("Configuration validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)
        else:
            click.echo("Configuration is valid!")
        
    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show configuration status and information."""
    
    config_manager = ctx.obj['config_manager']
    
    click.echo("Configuration Manager Status")
    click.echo("=" * 30)
    click.echo(f"Configured: {config_manager.is_configured}")
    click.echo(f"Config file: {ctx.obj.get('config_file', 'None')}")
    click.echo(f"Version control: {config_manager.has_version_control}")
    click.echo(f"Extensibility: {config_manager.has_extensibility}")
    
    if config_manager.is_configured:
        settings = config_manager.settings
        click.echo(f"\nAnalysis Settings:")
        click.echo(f"  Name: {settings.analysis_name}")
        click.echo(f"  Output directory: {settings.output_directory}")
        click.echo(f"  Log level: {settings.log_level}")
        click.echo(f"  Random seed: {settings.random_seed}")
        
        click.echo(f"\nData Settings:")
        click.echo(f"  Regions: {len(settings.data.regions)}")
        click.echo(f"  Start date: {settings.data.start_date}")
        click.echo(f"  End date: {settings.data.end_date}")
        click.echo(f"  Frequency: {settings.data.frequency}")
        
        if config_manager.has_extensibility and config_manager.extensibility_config:
            ext_config = config_manager.extensibility_config
            click.echo(f"\nExtensibility Settings:")
            click.echo(f"  Spatial weight methods: {len(ext_config.spatial_weight_methods)}")
            click.echo(f"  Identification strategies: {len(ext_config.identification_strategies)}")
            click.echo(f"  Model extensions: {len(ext_config.model_extensions)}")
            click.echo(f"  Parameter restrictions: {len(ext_config.parameter_restrictions)}")
            click.echo(f"  Robustness checks: {len(ext_config.robustness_checks)}")


if __name__ == '__main__':
    cli()