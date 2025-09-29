# Scripts Directory

This directory contains utility scripts for development, validation, and paper generation.

## Available Scripts

### `run_full_pipeline.py`
Complete end-to-end analysis pipeline for regional monetary policy analysis.

```bash
# Basic usage
python scripts/run_full_pipeline.py

# With custom parameters
python scripts/run_full_pipeline.py --start-date 2010-01-01 --end-date 2020-12-31 --regions CA,TX,NY

# Quick analysis for testing
python scripts/run_full_pipeline.py --quick

# Validation only
python scripts/run_full_pipeline.py --validate-only
```

**Purpose**: Runs the complete analysis workflow including:
1. System validation and setup
2. Data retrieval and validation  
3. Spatial modeling and parameter estimation
4. Policy analysis and mistake decomposition
5. Counterfactual analysis
6. Visualization and reporting

**Options**:
- `--start-date`: Analysis start date (default: 2000-01-01)
- `--end-date`: Analysis end date (default: 2023-12-31)
- `--regions`: Comma-separated regions (default: from config)
- `--config`: Configuration file path
- `--analysis-name`: Custom analysis name
- `--quick`: Run quick analysis for testing
- `--validate-only`: Only run system validation
- `--output-dir`: Output directory (default: output/)
- `--verbose`: Enable detailed logging

### `test_basic_functionality.py`
Quick validation test to verify core system components are working properly.

```bash
python scripts/test_basic_functionality.py
```

**Purpose**: Run after installation to ensure the system is properly set up.

### `final_publication_check.py`
Comprehensive publication readiness assessment script.

```bash
python scripts/final_publication_check.py
```

**Purpose**: Evaluates system completeness across multiple criteria:
- Core functionality (30% weight)
- Documentation (25% weight)  
- Project structure (20% weight)
- Dependencies (10% weight)
- Examples (10% weight)
- Tests (5% weight)

### `generate_paper_figures.py`
Generates all figures for the academic paper.

```bash
python scripts/generate_paper_figures.py
```

**Purpose**: Creates publication-quality figures:
- Figure 1: Regional Heterogeneity in Structural Parameters
- Figure 2: Policy Mistake Decomposition Over Time
- Figure 3: Welfare Comparison Across Policy Scenarios
- Figure 4: Spatial Spillover Effects
- Figure 5: Estimation Diagnostics

**Output**: Saves figures to `paper/figures/` in both PDF and PNG formats.

## Usage from Makefile

These scripts are integrated into the project Makefile:

```bash
# Run basic validation
make validate

# Generate paper figures
make figures

# Compile complete paper
make paper
```

## Development Workflow

1. **After Installation**: Run `test_basic_functionality.py`
2. **Before Committing**: Run `final_publication_check.py`
3. **Paper Updates**: Run `generate_paper_figures.py` then compile paper
4. **Full Validation**: Use `make all` to run everything

## Script Dependencies

All scripts use the main package dependencies. No additional requirements needed.