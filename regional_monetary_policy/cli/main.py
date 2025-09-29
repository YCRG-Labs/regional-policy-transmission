"""
Main CLI entry point for regional monetary policy analysis.
"""

import click
import logging
from pathlib import Path
from ..config import ConfigManager


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.pass_context
def main(ctx, config, verbose):
    """Regional Monetary Policy Analysis System."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Initialize configuration manager
    config_manager = ConfigManager(config)
    ctx.obj['config_manager'] = config_manager
    
    if config:
        click.echo(f"Using configuration file: {config}")


@main.command()
@click.option('--output', '-o', default='config.json',
              help='Output path for configuration template')
@click.pass_context
def init_config(ctx, output):
    """Create a configuration template file."""
    config_manager = ctx.obj['config_manager']
    config_manager.create_config_template(output)
    click.echo(f"Configuration template created at: {output}")


@main.command()
@click.pass_context
def validate_config(ctx):
    """Validate the current configuration."""
    config_manager = ctx.obj['config_manager']
    
    try:
        settings = config_manager.load_config()
        click.echo("✓ Configuration is valid")
        click.echo(f"Analysis name: {settings.analysis_name}")
        click.echo(f"Output directory: {settings.output_directory}")
        click.echo(f"Number of regions: {len(settings.data.regions)}")
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()