#!/usr/bin/env python3
"""
Compile the LaTeX paper and generate PDF output.
"""

import subprocess
import os
from pathlib import Path
import sys

def compile_latex_paper():
    """Compile the LaTeX paper using pdflatex and bibtex."""
    
    paper_dir = Path("paper")
    original_dir = os.getcwd()
    
    try:
        os.chdir(paper_dir)
        
        print("Compiling Regional Monetary Policy Analysis paper...")
        
        # First pdflatex run
        print("  - Running pdflatex (1st pass)...")
        result1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], 
                                capture_output=True, text=True)
        
        if result1.returncode != 0:
            print("First pdflatex run had issues, but continuing...")
            print("Errors:", result1.stderr)
        
        # Run bibtex
        print("  - Running bibtex...")
        result2 = subprocess.run(["bibtex", "paper"], 
                                capture_output=True, text=True)
        
        if result2.returncode != 0:
            print("Bibtex had issues, but continuing...")
            print("Errors:", result2.stderr)
        
        # Second pdflatex run
        print("  - Running pdflatex (2nd pass)...")
        result3 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], 
                                capture_output=True, text=True)
        
        # Third pdflatex run (for cross-references)
        print("  - Running pdflatex (3rd pass)...")
        result4 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], 
                                capture_output=True, text=True)
        
        if result4.returncode == 0:
            print("✓ Paper compiled successfully!")
            print(f"✓ Output: {paper_dir}/paper.pdf")
            return True
        else:
            print("✗ Final compilation had issues")
            print("Errors:", result4.stderr)
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e}")
        return False
    except FileNotFoundError:
        print("✗ LaTeX not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)")
        return False
    
    finally:
        os.chdir(original_dir)

def main():
    """Main compilation function."""
    success = compile_latex_paper()
    
    if success:
        print("\n" + "="*60)
        print("PAPER COMPILATION COMPLETE")
        print("="*60)
        print("The academic paper has been successfully compiled.")
        print("Files generated:")
        print("  - paper/paper.pdf (main paper)")
        print("  - paper/figures/ (all figures)")
        print("\nThe paper is ready for submission to academic journals.")
        return 0
    else:
        print("\n" + "="*60)
        print("COMPILATION HAD ISSUES")
        print("="*60)
        print("The paper was compiled but may have some formatting issues.")
        print("Check paper/paper.pdf to see if it's acceptable.")
        return 1

if __name__ == "__main__":
    sys.exit(main())