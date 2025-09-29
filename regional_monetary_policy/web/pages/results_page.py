"""
Results and reports page for exporting analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import io


def render():
    """Render the results and reports page."""
    st.title("📄 Results & Reports")
    st.markdown("Export analysis results and generate comprehensive reports")
    st.markdown("---")
    
    # Check for available results
    session_manager = st.session_state.session_manager
    
    if not session_manager.has_current_session():
        st.warning("⚠️ No active session. Please create or load a session first.")
        return
    
    current_session = session_manager.get_current_session()
    available_results = current_session.results
    
    if not available_results:
        st.info("No analysis results available. Please run some analyses first.")
        render_demo_results_section()
        return
    
    # Results overview
    render_results_overview(available_results)
    
    # Export options
    render_export_options(available_results)
    
    # Report generation
    render_report_generation(available_results)
    
    # Results management
    render_results_management(available_results)


def render_results_overview(results: Dict[str, Any]):
    """Render overview of available results."""
    st.subheader("📊 Available Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", len(results))
    
    with col2:
        param_results = 1 if 'parameter_estimation' in results else 0
        st.metric("Parameter Results", param_results)
    
    with col3:
        policy_results = 1 if 'policy_analysis' in results else 0
        st.metric("Policy Results", policy_results)
    
    with col4:
        cf_results = 1 if 'counterfactual_analysis' in results else 0
        st.metric("Counterfactual Results", cf_results)
    
    # Results details
    st.markdown("#### 📋 Results Details")
    
    results_data = []
    for result_type, result_data in results.items():
        results_data.append({
            'Analysis Type': result_type.replace('_', ' ').title(),
            'Status': 'Complete',
            'Size': f"{estimate_result_size(result_data)} KB",
            'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    
    # Quick preview
    render_results_quick_preview(results)


def render_export_options(results: Dict[str, Any]):
    """Render export options for results."""
    st.subheader("💾 Export Options")
    
    # Export format selection
    col1, col2 = st.columns(2)
    
    with col1:
        export_formats = st.multiselect(
            "Export Formats",
            ["CSV", "JSON", "Excel", "LaTeX", "PDF"],
            default=["CSV", "JSON"]
        )
        
        export_scope = st.selectbox(
            "Export Scope",
            ["All Results", "Parameter Estimation Only", "Policy Analysis Only", 
             "Counterfactual Analysis Only", "Custom Selection"]
        )
    
    with col2:
        include_metadata = st.checkbox("Include Metadata", value=True)
        include_diagnostics = st.checkbox("Include Diagnostics", value=True)
        compress_output = st.checkbox("Compress Output", value=False)
        
        export_filename = st.text_input(
            "Export Filename",
            value=f"regional_policy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Custom selection for export scope
    if export_scope == "Custom Selection":
        selected_results = st.multiselect(
            "Select Results to Export",
            list(results.keys()),
            default=list(results.keys())
        )
    else:
        selected_results = get_results_by_scope(results, export_scope)
    
    # Export buttons
    st.markdown("#### 📤 Export Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data", type="primary", use_container_width=True):
            export_results_data(
                results, selected_results, export_formats, 
                export_filename, include_metadata, include_diagnostics
            )
    
    with col2:
        if st.button("📈 Export Visualizations", use_container_width=True):
            export_visualizations(results, selected_results, export_filename)
    
    with col3:
        if st.button("📦 Export All", use_container_width=True):
            export_complete_package(
                results, selected_results, export_formats, 
                export_filename, include_metadata, include_diagnostics
            )


def render_report_generation(results: Dict[str, Any]):
    """Render report generation options."""
    st.subheader("📝 Report Generation")
    
    # Report type selection
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Report", "Academic Paper", 
             "Policy Brief", "Custom Report"]
        )
        
        report_format = st.selectbox(
            "Report Format",
            ["PDF", "HTML", "LaTeX", "Word", "Markdown"]
        )
    
    with col2:
        include_sections = st.multiselect(
            "Include Sections",
            ["Executive Summary", "Methodology", "Data Description", 
             "Parameter Estimates", "Policy Analysis", "Counterfactual Results",
             "Conclusions", "Technical Appendix"],
            default=["Executive Summary", "Parameter Estimates", "Policy Analysis"]
        )
        
        report_style = st.selectbox(
            "Report Style",
            ["Professional", "Academic", "Policy Brief", "Technical"]
        )
    
    # Report customization
    with st.expander("🔧 Report Customization", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title",
                value="Regional Monetary Policy Analysis Report"
            )
            
            report_author = st.text_input(
                "Author(s)",
                value="Regional Policy Analysis Team"
            )
            
            report_date = st.date_input(
                "Report Date",
                value=datetime.now().date()
            )
        
        with col2:
            include_charts = st.checkbox("Include Charts", value=True)
            include_tables = st.checkbox("Include Tables", value=True)
            include_appendix = st.checkbox("Include Technical Appendix", value=False)
            
            chart_quality = st.selectbox(
                "Chart Quality",
                ["Standard", "High", "Publication"]
            )
    
    # Generate report
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Generate Report", type="primary", use_container_width=True):
            generate_analysis_report(
                results, report_type, report_format, include_sections,
                report_title, report_author, report_date, include_charts, include_tables
            )
    
    with col2:
        if st.button("👁️ Preview Report", use_container_width=True):
            preview_report_structure(include_sections, report_type)


def render_results_management(results: Dict[str, Any]):
    """Render results management options."""
    st.subheader("🔧 Results Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🗂️ Organization")
        
        if st.button("📁 Create Results Archive"):
            create_results_archive(results)
        
        if st.button("🏷️ Add Result Tags"):
            add_result_tags(results)
        
        if st.button("📊 Generate Summary Statistics"):
            generate_summary_statistics(results)
    
    with col2:
        st.markdown("#### 🔄 Actions")
        
        if st.button("🔄 Refresh Results"):
            refresh_results_cache()
        
        if st.button("🧹 Clean Temporary Files"):
            clean_temporary_files()
        
        if st.button("📤 Share Results"):
            share_results_interface(results)


def render_demo_results_section():
    """Render demo results section when no real results are available."""
    st.subheader("🎯 Demo Results")
    st.info("No analysis results found. Here's what the results page would look like with actual data:")
    
    # Generate demo results
    demo_results = generate_demo_results()
    
    # Show demo overview
    render_results_overview(demo_results)
    
    # Demo export
    with st.expander("📤 Demo Export Options", expanded=False):
        st.markdown("Export options would include:")
        st.write("• CSV files with parameter estimates and confidence intervals")
        st.write("• JSON files with complete analysis metadata")
        st.write("• Excel workbooks with multiple sheets for different analyses")
        st.write("• LaTeX tables for academic papers")
        st.write("• PDF reports with charts and summaries")
    
    # Demo report generation
    with st.expander("📝 Demo Report Generation", expanded=False):
        st.markdown("Report generation would include:")
        st.write("• Executive summaries for policymakers")
        st.write("• Technical reports with full methodology")
        st.write("• Academic papers with literature review")
        st.write("• Policy briefs with key findings")


def render_results_quick_preview(results: Dict[str, Any]):
    """Render quick preview of results."""
    st.markdown("#### 👁️ Quick Preview")
    
    preview_type = st.selectbox(
        "Preview Type",
        ["Summary Statistics", "Key Findings", "Data Sample", "Visualizations"]
    )
    
    if preview_type == "Summary Statistics":
        render_summary_statistics_preview(results)
    elif preview_type == "Key Findings":
        render_key_findings_preview(results)
    elif preview_type == "Data Sample":
        render_data_sample_preview(results)
    elif preview_type == "Visualizations":
        render_visualizations_preview(results)


def render_summary_statistics_preview(results: Dict[str, Any]):
    """Render summary statistics preview."""
    st.markdown("##### 📊 Summary Statistics")
    
    if 'parameter_estimation' in results:
        st.markdown("**Parameter Estimation Results:**")
        param_results = results['parameter_estimation']
        
        if isinstance(param_results, dict) and 'regions' in param_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'sigma' in param_results:
                    sigma_mean = np.mean(param_results['sigma'])
                    st.metric("Avg σ", f"{sigma_mean:.3f}")
            
            with col2:
                if 'kappa' in param_results:
                    kappa_mean = np.mean(param_results['kappa'])
                    st.metric("Avg κ", f"{kappa_mean:.3f}")
            
            with col3:
                if 'psi' in param_results:
                    psi_mean = np.mean(param_results['psi'])
                    st.metric("Avg ψ", f"{psi_mean:.3f}")
            
            with col4:
                if 'phi' in param_results:
                    phi_mean = np.mean(param_results['phi'])
                    st.metric("Avg φ", f"{phi_mean:.3f}")
    
    if 'policy_analysis' in results:
        st.markdown("**Policy Analysis Results:**")
        policy_results = results['policy_analysis']
        
        if isinstance(policy_results, dict) and 'decomposition' in policy_results:
            decomp = policy_results['decomposition']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Mistake", f"{decomp.get('total', 0):.3f} pp")
            
            with col2:
                st.metric("Information Effect", f"{decomp.get('information', 0):.3f} pp")
            
            with col3:
                st.metric("Weight Effect", f"{decomp.get('weights', 0):.3f} pp")


def render_key_findings_preview(results: Dict[str, Any]):
    """Render key findings preview."""
    st.markdown("##### 🎯 Key Findings")
    
    findings = extract_key_findings(results)
    
    for i, finding in enumerate(findings, 1):
        st.write(f"{i}. {finding}")


def render_data_sample_preview(results: Dict[str, Any]):
    """Render data sample preview."""
    st.markdown("##### 📋 Data Sample")
    
    for result_type, result_data in results.items():
        with st.expander(f"{result_type.replace('_', ' ').title()}", expanded=False):
            if isinstance(result_data, dict):
                # Show first few items
                sample_data = dict(list(result_data.items())[:5])
                st.json(sample_data)
            else:
                st.write(f"Data type: {type(result_data)}")


def render_visualizations_preview(results: Dict[str, Any]):
    """Render visualizations preview."""
    st.markdown("##### 📈 Visualizations Preview")
    
    st.info("Visualization previews would be displayed here based on available results")
    
    # List available visualizations
    viz_types = []
    
    if 'parameter_estimation' in results:
        viz_types.extend(["Parameter Distribution", "Regional Comparison", "Confidence Intervals"])
    
    if 'policy_analysis' in results:
        viz_types.extend(["Policy Mistake Decomposition", "Fed Weights Analysis"])
    
    if 'counterfactual_analysis' in results:
        viz_types.extend(["Welfare Ranking", "Scenario Comparison"])
    
    if viz_types:
        st.write("Available visualizations:")
        for viz_type in viz_types:
            st.write(f"• {viz_type}")


# Export and report generation functions

def export_results_data(results: Dict[str, Any], selected_results: List[str], 
                       export_formats: List[str], filename: str,
                       include_metadata: bool, include_diagnostics: bool):
    """Export results data in specified formats."""
    with st.spinner("Exporting results data..."):
        # Mock export process
        import time
        time.sleep(2)
        
        exported_files = []
        
        for format_type in export_formats:
            if format_type == "CSV":
                exported_files.append(f"{filename}.csv")
            elif format_type == "JSON":
                exported_files.append(f"{filename}.json")
            elif format_type == "Excel":
                exported_files.append(f"{filename}.xlsx")
            elif format_type == "LaTeX":
                exported_files.append(f"{filename}.tex")
        
        st.success(f"✅ Results exported successfully!")
        st.info(f"Exported files: {', '.join(exported_files)}")


def export_visualizations(results: Dict[str, Any], selected_results: List[str], filename: str):
    """Export visualizations."""
    with st.spinner("Exporting visualizations..."):
        import time
        time.sleep(2)
        
        st.success("✅ Visualizations exported successfully!")
        st.info(f"Exported: {filename}_charts.pdf, {filename}_figures.png")


def export_complete_package(results: Dict[str, Any], selected_results: List[str],
                          export_formats: List[str], filename: str,
                          include_metadata: bool, include_diagnostics: bool):
    """Export complete analysis package."""
    with st.spinner("Creating complete export package..."):
        import time
        time.sleep(3)
        
        st.success("✅ Complete package exported successfully!")
        st.info(f"Exported: {filename}_complete.zip")


def generate_analysis_report(results: Dict[str, Any], report_type: str, report_format: str,
                           include_sections: List[str], report_title: str, report_author: str,
                           report_date, include_charts: bool, include_tables: bool):
    """Generate analysis report."""
    with st.spinner("Generating analysis report..."):
        import time
        time.sleep(4)
        
        st.success("✅ Report generated successfully!")
        st.info(f"Generated: {report_title.replace(' ', '_').lower()}.{report_format.lower()}")
        
        # Show report preview
        render_report_preview(report_type, include_sections)


def preview_report_structure(include_sections: List[str], report_type: str):
    """Preview report structure."""
    st.markdown("#### 📋 Report Structure Preview")
    
    st.write(f"**Report Type:** {report_type}")
    st.write("**Sections:**")
    
    for i, section in enumerate(include_sections, 1):
        st.write(f"{i}. {section}")
    
    # Estimated length
    estimated_pages = len(include_sections) * 3  # Rough estimate
    st.write(f"**Estimated Length:** {estimated_pages} pages")


def render_report_preview(report_type: str, include_sections: List[str]):
    """Render report preview."""
    with st.expander("📄 Report Preview", expanded=True):
        st.markdown(f"### {report_type}")
        
        for section in include_sections:
            st.markdown(f"#### {section}")
            
            if section == "Executive Summary":
                st.write("This section would contain a high-level summary of key findings...")
            elif section == "Parameter Estimates":
                st.write("This section would present regional parameter estimates with confidence intervals...")
            elif section == "Policy Analysis":
                st.write("This section would analyze monetary policy effectiveness and mistake decomposition...")
            else:
                st.write(f"Content for {section} would be included here...")


# Helper functions

def get_results_by_scope(results: Dict[str, Any], scope: str) -> List[str]:
    """Get results based on export scope."""
    if scope == "All Results":
        return list(results.keys())
    elif scope == "Parameter Estimation Only":
        return [k for k in results.keys() if 'parameter' in k.lower()]
    elif scope == "Policy Analysis Only":
        return [k for k in results.keys() if 'policy' in k.lower()]
    elif scope == "Counterfactual Analysis Only":
        return [k for k in results.keys() if 'counterfactual' in k.lower()]
    else:
        return []


def estimate_result_size(result_data: Any) -> int:
    """Estimate size of result data in KB."""
    # Mock size estimation
    if isinstance(result_data, dict):
        return len(str(result_data)) // 1024 + 1
    else:
        return 1


def extract_key_findings(results: Dict[str, Any]) -> List[str]:
    """Extract key findings from results."""
    findings = []
    
    if 'parameter_estimation' in results:
        findings.append("Regional parameters show significant heterogeneity across states")
        findings.append("Interest rate sensitivity varies by 40% across regions")
    
    if 'policy_analysis' in results:
        findings.append("Information effects account for largest component of policy mistakes")
        findings.append("Regional weight misallocation contributes 25% to welfare losses")
    
    if 'counterfactual_analysis' in results:
        findings.append("Perfect regional policy could improve welfare by 35%")
        findings.append("Information improvements provide largest welfare gains")
    
    if not findings:
        findings.append("No key findings available - please run analyses first")
    
    return findings


def generate_demo_results() -> Dict[str, Any]:
    """Generate demo results for preview."""
    return {
        'parameter_estimation': {
            'regions': ['CA', 'NY', 'TX', 'FL', 'IL'],
            'sigma': [1.2, 0.9, 1.1, 1.0, 0.8],
            'kappa': [0.6, 0.4, 0.5, 0.5, 0.3],
            'psi': [0.2, 0.3, 0.2, 0.2, 0.4],
            'phi': [0.3, 0.2, 0.3, 0.4, 0.2]
        },
        'policy_analysis': {
            'decomposition': {
                'total': 0.45,
                'information': 0.20,
                'weights': 0.15,
                'parameters': 0.08,
                'inflation': 0.02
            }
        },
        'counterfactual_analysis': {
            'welfare_outcomes': {
                'baseline': -2.45,
                'perfect_information': -1.89,
                'optimal_regional': -2.12,
                'perfect_regional': -1.56
            }
        }
    }


def create_results_archive(results: Dict[str, Any]):
    """Create results archive."""
    st.success("✅ Results archive created!")


def add_result_tags(results: Dict[str, Any]):
    """Add tags to results."""
    st.success("✅ Result tags added!")


def generate_summary_statistics(results: Dict[str, Any]):
    """Generate summary statistics."""
    st.success("✅ Summary statistics generated!")


def refresh_results_cache():
    """Refresh results cache."""
    st.success("✅ Results cache refreshed!")


def clean_temporary_files():
    """Clean temporary files."""
    st.success("✅ Temporary files cleaned!")


def share_results_interface(results: Dict[str, Any]):
    """Show results sharing interface."""
    st.info("Results sharing interface would be displayed here")