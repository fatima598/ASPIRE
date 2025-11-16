# streamlit_app.py
import streamlit as st
import requests
import base64
from PIL import Image
import io
import time
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Smart Damage Detection Dashboard",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #ffffff;
        padding: 0;
    }
    
    /* Header styling */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Feature cards */
    .features-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        flex: 1;
        text-align: center;
        padding: 1.5rem 1rem;
        background: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.25rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Upload area - REMOVED the custom upload section styling */
    .upload-instruction {
        color: #666;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    
    /* Results styling */
    .results-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Comparison styling */
    .comparison-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .comparison-column {
        flex: 1;
        text-align: center;
    }
    
    .comparison-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def generate_pdf_report(results):
    """
    Generate a text-based PDF report
    """
    # Create a professional text report
    report_content = f"""
VEHICLE DAMAGE ASSESSMENT REPORT
================================

Assessment Details:
------------------
Assessment ID: {results.get('assessment_id', 'N/A')}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Processed Images: {results.get('images_processed', {}).get('pickup_images', 0)} pickup, {results.get('images_processed', {}).get('return_images', 0)} return

Summary:
--------
• New Damages Found: {results.get('summary', {}).get('new_damages_count', 0)}
• Total Repair Cost: ${results.get('summary', {}).get('total_repair_cost', 0):.2f}
• Severity Score: {results.get('summary', {}).get('severity_score', 0)}/10

Damage Analysis:
---------------
"""
    
    # Add damage details
    detailed_damages = results.get('detailed_damages', [])
    if detailed_damages:
        for i, damage in enumerate(detailed_damages, 1):
            report_content += f"""
Damage #{i}:
• Type: {damage.get('type', 'Unknown')}
• Confidence: {damage.get('confidence', 0):.1%}
• Location: {damage.get('image_source', 'N/A')}
• Bounding Box: {damage.get('bbox', [])}

"""
    else:
        report_content += "• No new damages detected - vehicle condition unchanged\n\n"
    
    # Add cost breakdown if available
    cost_breakdown = results.get('cost_breakdown', [])
    if cost_breakdown:
        report_content += "Cost Breakdown:\n----------------\n"
        total_cost = 0
        if isinstance(cost_breakdown, list):
            for item in cost_breakdown:
                cost = item.get('final_cost', 0)
                total_cost += cost
                report_content += f"• {item.get('type', 'Unknown')}: ${cost:.2f}\n"
        report_content += f"• TOTAL: ${total_cost:.2f}\n\n"
    
    report_content += """
Report Generated by:
AutoInspect Pro - AI Vehicle Damage Assessment System
Powered by YOLOv8 Technology

Note: This is an AI-generated assessment. For official claims, please consult with certified automotive professionals.
"""
    
    return report_content.encode('utf-8')

def main():
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Smart Damage Detection Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="dashboard-subtitle">
        Welcome to our cutting-edge vehicle damage detection system. Powered by advanced YOLOv8 AI technology, 
        this tool revolutionizes the way we assess and estimate vehicle repairs. Whether you're an insurance adjuster, 
        a car owner, or a repair shop manager, our system provides quick, accurate, and reliable damage assessments.
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
    <div class="features-container">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Analysis</div>
            <div class="feature-desc">Get results in seconds</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">Powered by YOLOv8 AI</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Cost Estimation</div>
            <div class="feature-desc">Instant repair quotes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'pickup_images' not in st.session_state:
        st.session_state.pickup_images = []
    if 'return_images' not in st.session_state:
        st.session_state.return_images = []
    
    # Check backend connection
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.error("❌ Backend Offline")
    except:
        st.sidebar.error("❌ Service Unavailable")
    
    # Main interface
    if st.session_state.results is None:
        show_upload_interface()
    else:
        show_results_interface()

def show_upload_interface():
    # Upload sections in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📸 Pickup Images")
        st.markdown("Upload images from vehicle pickup")
        st.markdown('<div class="upload-instruction">Select images taken when vehicle was picked up</div>', unsafe_allow_html=True)
        
        pickup_files = st.file_uploader(
            "Choose pickup images",
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            key="pickup_uploader"
        )
        
        if pickup_files:
            st.session_state.pickup_images = pickup_files
            st.success(f"✅ {len(pickup_files)} pickup images selected")
            
            # Show previews
            st.markdown("**Selected Images:**")
            for file in pickup_files:
                image = Image.open(file)
                st.image(image, use_container_width=True, caption=file.name)
    
    with col2:
        st.markdown("#### 📸 Return Images")
        st.markdown("Upload images from vehicle return")
        st.markdown('<div class="upload-instruction">Select images taken when vehicle was returned</div>', unsafe_allow_html=True)
        
        return_files = st.file_uploader(
            "Choose return images",
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            key="return_uploader"
        )
        
        if return_files:
            st.session_state.return_images = return_files
            st.success(f"✅ {len(return_files)} return images selected")
            
            # Show previews
            st.markdown("**Selected Images:**")
            for file in return_files:
                image = Image.open(file)
                st.image(image, use_container_width=True, caption=file.name)
    
    # Analyze button
    st.markdown("---")
    if st.button("🚀 Analyze Damage", use_container_width=True):
        if not st.session_state.pickup_images or not st.session_state.return_images:
            st.error("Please upload both pickup and return images")
        else:
            assess_damage_with_visualization()

def assess_damage_with_visualization():
    try:
        with st.spinner("🔍 Analyzing vehicle damages..."):
            # Prepare files for API
            files = []
            
            for img in st.session_state.pickup_images:
                files.append(("pickup_images", (img.name, img.getvalue(), img.type)))
            
            for img in st.session_state.return_images:
                files.append(("return_images", (img.name, img.getvalue(), img.type)))
            
            # Call the API endpoint
            response = requests.post(
                f"{API_BASE_URL}/api/compare-with-visualization",
                files=files,
                timeout=300
            )
            
            if response.status_code == 200:
                st.session_state.results = response.json()
                st.rerun()
            else:
                st.error(f"API Error: {response.text}")
                
    except Exception as e:
        st.error(f"Failed to assess damage: {str(e)}")

def show_results_interface():
    results = st.session_state.results
    summary = results['summary']
    
    # Header with action button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📊 Damage Assessment Report")
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            reset_assessment()
    
    st.markdown("---")
    
    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">New Damages</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(summary['new_damages_count']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Repair Cost</div>
            <div class="metric-value">${:,.2f}</div>
        </div>
        """.format(summary['total_repair_cost']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Severity</div>
            <div class="metric-value">{}/10</div>
        </div>
        """.format(summary['severity_score']), unsafe_allow_html=True)
    
    # Show visualizations - SIDE BY SIDE
    if 'visualizations' in results:
        show_side_by_side_comparison(results['visualizations'])
    
    # Clean detailed analysis section with PDF download
    st.markdown("## 🔍 Detailed Analysis")
    
    # PDF download button in detailed section
    pdf_report = generate_pdf_report(results)
    st.download_button(
        label="📥 Download Detailed Report",
        data=pdf_report,
        file_name=f"damage_report_{results['assessment_id']}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Show cost breakdown only
    if 'cost_breakdown' in results and results['cost_breakdown']:
        show_cost_breakdown(results['cost_breakdown'])

def show_side_by_side_comparison(visualizations):
    """
    Show pickup and return images side by side for better comparison
    """
    st.subheader("🔄 Side-by-Side Comparison")
    
    pickup_images = visualizations['pickup_images']
    return_images = visualizations['return_images']
    
    # Determine the number of comparisons to show (min of both sets)
    num_comparisons = min(len(pickup_images), len(return_images))
    
    for i in range(num_comparisons):
        st.markdown(f"**Comparison #{i+1}**")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pickup Image**")
            if pickup_images[i]['image_with_boxes']:
                st.image(pickup_images[i]['image_with_boxes'], use_container_width=True)
                st.caption(f"File: {pickup_images[i]['filename']}")
                damage_count = len(pickup_images[i]['damages'])
                if damage_count > 0:
                    st.error(f"🚨 {damage_count} pre-existing damage(s)")
                else:
                    st.success("✅ No damages at pickup")
        
        with col2:
            st.markdown("**Return Image**")
            if return_images[i]['image_with_boxes']:
                st.image(return_images[i]['image_with_boxes'], use_container_width=True)
                st.caption(f"File: {return_images[i]['filename']}")
                damage_count = len(return_images[i]['damages'])
                if damage_count > 0:
                    st.error(f"🚨 {damage_count} damage(s) detected")
                else:
                    st.success("✅ No new damages")
        
        st.markdown("---")
    
    # Show any remaining images that couldn't be paired
    remaining_pickup = len(pickup_images) - num_comparisons
    remaining_return = len(return_images) - num_comparisons
    
    if remaining_pickup > 0 or remaining_return > 0:
        st.subheader("Additional Images")
        
        if remaining_pickup > 0:
            st.markdown("**Additional Pickup Images**")
            cols = st.columns(min(3, remaining_pickup))
            for i in range(num_comparisons, len(pickup_images)):
                if pickup_images[i]['image_with_boxes']:
                    with cols[i % len(cols)]:
                        st.image(pickup_images[i]['image_with_boxes'], use_container_width=True)
                        st.caption(f"Pickup: {pickup_images[i]['filename']}")
        
        if remaining_return > 0:
            st.markdown("**Additional Return Images**")
            cols = st.columns(min(3, remaining_return))
            for i in range(num_comparisons, len(return_images)):
                if return_images[i]['image_with_boxes']:
                    with cols[i % len(cols)]:
                        st.image(return_images[i]['image_with_boxes'], use_container_width=True)
                        st.caption(f"Return: {return_images[i]['filename']}")

def show_cost_breakdown(cost_breakdown):
    """
    Show only cost breakdown without the removed sections
    """
    st.subheader("💰 Cost Estimation")
    
    if not cost_breakdown:
        st.info("No cost breakdown available")
        return
    
    # Extract total cost
    total_cost = 0
    if isinstance(cost_breakdown, list):
        for item in cost_breakdown:
            total_cost += item.get('final_cost', 0)
    elif isinstance(cost_breakdown, dict):
        total_cost = cost_breakdown.get('total_cost', 0)
    
    st.metric("Total Estimated Repair Cost", f"${total_cost:.2f}")
    
    # Show simple breakdown if available
    if isinstance(cost_breakdown, list) and len(cost_breakdown) > 0:
        st.markdown("**Cost Breakdown:**")
        for item in cost_breakdown:
            st.write(f"- **{item.get('type', 'Unknown')}**: ${item.get('final_cost', 0):.2f}")

def reset_assessment():
    st.session_state.results = None
    st.session_state.pickup_images = []
    st.session_state.return_images = []
    st.rerun()

if __name__ == "__main__":
    main()
