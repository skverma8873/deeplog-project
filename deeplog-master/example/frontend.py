import streamlit as st
import os
import re
import sys
sys.path.append('../')
import pandas as pd
import json
import logging
from spellpy import spell
from deeplog.deeplog import model_fn, input_fn, predict_fn
import threading
import multiprocessing
import openai  # Add OpenAI library
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.colors import black
from streamlit.components.v1 import html

# Configure OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Use Streamlit secrets for API key

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Function to display the Mermaid chart
def mermaid_chart(mermaid_code):
    html_code = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <div class="mermaid">{mermaid_code}</div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true}});</script>
    """
    return html_code

# Utility function to extract unique IDs
def extract_unique_ids(logs):
    pattern = r"req-[a-f0-9\-]+"
    return list(set(re.findall(pattern, "\n".join(logs))))

# Function to fetch logs for a specific unique ID
def filter_logs_by_id(logs, unique_id):
    return [line for line in logs if unique_id in line]

def get_valid_unique_ids(logs):
    """Return unique IDs that appear in more than one log line."""
    pattern = r"req-[a-f0-9\-]+"
    unique_id_counts = {}
    
    # Count occurrences of each unique ID
    for log in logs:
        matches = re.findall(pattern, log)
        for match in matches:
            unique_id_counts[match] = unique_id_counts.get(match, 0) + 1
    
    # Filter IDs that appear in more than one line
    valid_unique_ids = [uid for uid, count in unique_id_counts.items() if count > 1]
    return valid_unique_ids


# Function to call OpenAI to generate Mermaid code
def generate_mermaid_code(log_lines, unique_id):
    log_text = "\n".join(log_lines)
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that converts log lines into Mermaid sequence diagram code."
        },
        {
            "role": "user",
            "content": f"""
            Based on the following log lines associated with unique ID '{unique_id}', generate a detailed Mermaid sequence diagram code:

            Log lines:
            {log_text}

            Please provide only the Mermaid code.
            """
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or "gpt-3.5-turbo" if GPT-4 is unavailable
        messages=messages,
        max_tokens=500,
        temperature=0
    )
    response = response['choices'][0]['message']['content'].strip()
    if "```mermaid" in response:
            response = (
                response.replace("```mermaid", "").replace("```", "")
            )
    return response

def generate_pdf_report(log_input, anomaly_result, ai_analysis):
    """
    Generate a PDF report of the log analysis
    
    Args:
        log_input (str): Original log input
        anomaly_result (dict): Anomaly detection results
        ai_analysis (dict): AI-generated analysis
    
    Returns:
        bytes: PDF report as bytes
    """
    # Create a buffer to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Prepare styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=16,
        textColor=black,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        textColor=black,
        alignment=TA_LEFT
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        textColor=black,
        alignment=TA_JUSTIFY
    )
    
    # Prepare content
    content = []
    
    # Title
    content.append(Paragraph("Log Anomaly Analysis Report", title_style))
    content.append(Spacer(1, 12))
    
    # Log Input Section
    content.append(Paragraph("Log Input Summary", heading_style))
    content.append(Paragraph(f"Original Log Messages (first 500 chars):<br/>{log_input[:500]}...", normal_style))
    content.append(Spacer(1, 12))
    
    # Anomaly Metrics
    content.append(Paragraph("Anomaly Metrics", heading_style))
    content.append(Paragraph(f"Total Anomalies Detected: {sum(anomaly_result['has_anomaly'])}", normal_style))
    content.append(Paragraph(f"Anomaly Counts: {anomaly_result['anomaly_count']}", normal_style))
    content.append(Spacer(1, 12))
    
    # AI Analysis
    content.append(Paragraph("AI-Powered Analysis", heading_style))
    
    # Split AI analysis into paragraphs for better readability
    ai_analysis_paragraphs = ai_analysis['analysis'].split('\n')
    for paragraph in ai_analysis_paragraphs:
        content.append(Paragraph(paragraph, normal_style))
    
    content.append(Spacer(1, 12))
    
    # Detailed Anomaly Result
    content.append(Paragraph("Detailed Anomaly Results", heading_style))
    content.append(Paragraph(json.dumps(anomaly_result, indent=2), normal_style))
    
    # Build PDF
    doc.build(content)
    
    # Get the value of the BytesIO buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def get_ai_analysis(log_messages, anomaly_details, settings):
    """
    Use OpenAI to analyze log anomalies and provide root cause and recommendations
    with configurable settings

    Args:
        log_messages (str): Original log messages
        anomaly_details (dict): Details of detected anomalies
        settings: Configuration settings
    
    Returns:
        dict: AI-generated analysis
    """
    try:
        # Use sidebar settings for AI analysis
        ai_model = settings['ai_model']# .lower().replace('-', '')

        # Mapping to correct OpenAI model names
        model_mapping = {
            "GPT-4": "gpt-4",
            "GPT-3.5-Turbo": "gpt-3.5-turbo",
            "GPT-4 Turbo": "gpt-4-turbo",
            "GPT-4o": "gpt-4o"
        }
        
        # Get the correct model name, default to gpt-3.5-turbo if not found
        model = model_mapping.get(ai_model, "gpt-3.5-turbo")

        temperature = settings['temperature']

        # Construct a detailed prompt for GPT
        prompt = f"""
        Perform a comprehensive root cause analysis for the following log anomalies:

        Log Messages:
        {log_messages}

        Anomaly Details:
        {json.dumps(anomaly_details, indent=2)}

        Please provide:
        1. Potential Root Cause Analysis
        2. Recommended Immediate Actions
        3. Long-term Prevention Strategies
        4. Severity Assessment

        Format the response in a clear, structured manner.
        """

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=model,  # Use the latest GPT model
            messages=[
                {"role": "system", "content": "You are an expert system log analyzer and IT operations specialist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=temperature
        )

        # Extract the AI's analysis
        ai_analysis = response.choices[0].message.content
        return {
            "analysis": ai_analysis,
            "raw_response": response,
            "model_used": model
        }
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return None

def parse_log_string(log_string, input_dir, output_dir):
    """
    Parse log string using SpellParser
    
    Args:
        log_string (str): Raw log messages
    
    Returns:
        pd.DataFrame: Structured log dataframe
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    
    # Create a temporary file from log string
    temp_log_file = os.path.join(input_dir, 'temp_log.log')
    
    try:
        # Write log string to the file
        with open(temp_log_file, 'w') as f:
            f.write(log_string)
        
        return temp_log_file, 'temp_log.log'
    
    except Exception as e:
        logger.error(f"Error creating log: {e}")
        
        # Additional error handling
        if os.path.exists(temp_log_file):
            os.remove(temp_log_file)
        
        return None, None

def deeplog_df_transfer(df, event_id_map):
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'EventId']]
    df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df

def _custom_resampler(array_like):
    return list(array_like)

def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')

def detect_anomalies(log_string):
    print(f"Running in thread: {threading.current_thread().name}")
    try:
        # Directories
        input_dir = './data/'
        output_dir = './log_result/'
        trained_log_csv_dir = "./openstack_result/"
        
        # Log parsing parameters
        log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
        log_main = 'log_test'
        tau = 0.5

        # Initialize parser
        parser = spell.LogParser(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            logmain=log_main,
            tau=tau,
        )

        # Parse log string
        temp_log_file, file_name = parse_log_string(log_string, input_dir, output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        parser.parse(file_name)

        # Read structured CSV
        df_test = pd.read_csv(f'{output_dir}/temp_log.log_structured.csv')
        df_trained = pd.read_csv(f'{trained_log_csv_dir}/openstack_normal1.log_structured.csv')
        print(df_test.head())

        # Create event ID mapping
        trained_event_id_map = dict()
        for i, event_id in enumerate(df_trained['EventId'].unique(), 1):
            trained_event_id_map[event_id] = i

        # Transform data for DeepLog
        deeplog_test_normal = deeplog_df_transfer(df_test, trained_event_id_map)
        deeplog_file_generator('test_msg', deeplog_test_normal)

        # Load model
        model_dir = './model'
        model_info = model_fn(model_dir)

        # Predict
        test_msg_list = []
        with open('test_msg', 'r') as f:
            for line in f.readlines():
                line = list(map(lambda n: n, map(int, line.strip().split())))
                request = json.dumps({'line': line})
                input_data = input_fn(request, 'application/json')
                response = predict_fn(input_data, model_info)
                print(response)
                test_msg_list.append(response)

        # Evaluate anomalies
        thres = 3
        test_msg_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_msg_list]
        test_msg_cnt_anomaly = [t['anomaly_cnt'] for t in test_msg_list]
        
        return {
            'has_anomaly': test_msg_has_anomaly,
            'anomaly_count': test_msg_cnt_anomaly
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise

def setup_sidebar():
    """
    Configure and populate the Streamlit sidebar with additional information and controls
    """
    st.sidebar.title("🔍 Log Anomaly Detection Dashboard")
    
    # System Information Section
    st.sidebar.header("🖥️ System Overview")
    
    # Detection Model Details
    st.sidebar.subheader("Detection Model")
    st.sidebar.write("Model: DeepLog Anomaly Detection")
    st.sidebar.write("Version: 1.0.0")
    
    # AI Analysis Configuration
    st.sidebar.header("🤖 AI Analysis Settings")
    
    # AI Model Selection
    ai_model = st.sidebar.selectbox(
        "Select AI Analysis Model",
        ["GPT-4", "GPT-3.5-Turbo", "GPT-4o"],
        help="Choose the AI model for detailed log analysis"
    )
    
    # Temperature (Creativity) Slider
    temperature = st.sidebar.slider(
        "AI Analysis Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Higher values make the AI more creative, lower values more focused"
    )
    
    # Anomaly Sensitivity
    # anomaly_threshold = st.sidebar.slider(
    #     "Anomaly Detection Sensitivity",
    #     min_value=0.1,
    #     max_value=1.0,
    #     value=0.5,
    #     step=0.1,
    #     help="Adjust the threshold for detecting log anomalies"
    # )
    
    # Recent Analysis History
    # st.sidebar.header("📊 Recent Analyses")
    
    # # Placeholder for tracking recent analyses
    # if 'analysis_history' not in st.session_state:
    #     st.session_state.analysis_history = []
    
    # if st.session_state.analysis_history:
    #     for i, analysis in enumerate(st.session_state.analysis_history[-5:], 1):
    #         st.sidebar.text(f"{i}. {analysis[:20]}...")
    # else:
    #     st.sidebar.text("No recent analyses")
    
    # Additional Controls
    st.sidebar.header("⚙️ Additional Controls")
    
    # Log Source Selection
    log_source = st.sidebar.selectbox(
        "Select Log Source",
        ["Custom Logs", "File Upload"],
        help="Choose the source of log messages"
    )
    
    # Export Preferences
    export_format = st.sidebar.multiselect(
        "Export Formats",
        ["JSON", "CSV", "Markdown", "PDF"],
        default=["JSON"],
        help="Select preferred export formats for analysis results"
    )
    
    return {
        "ai_model": ai_model,
        "temperature": temperature,
        # "anomaly_threshold": anomaly_threshold,
        "log_source": log_source,
        "export_format": export_format
    }


def generate_incident_report(log_input, anomaly_result, ai_analysis):
    """
    Generate a structured incident report
    """
    report = f"""
    ## Incident Report

    ### Log Input Summary
    ```
    {log_input[:500]}...  # Truncate for brevity
    ```

    ### Anomaly Metrics
    - Total Anomalies Detected: {sum(anomaly_result['has_anomaly'])}
    - Anomaly Counts: {anomaly_result['anomaly_count']}

    ### AI Analysis
    {ai_analysis['analysis']}
    """
    return report

def export_analysis(log_input, anomaly_result, ai_analysis, export_formats):
    """
    Prepare export data in multiple formats
    
    Args:
        log_input (str): Original log input
        anomaly_result (dict): Anomaly detection results
        ai_analysis (dict): AI-generated analysis
        export_formats (list): List of desired export formats
    
    Returns:
        dict: Exported files in different formats
    """
    exports = {}
    
    # JSON Export
    if "JSON" in export_formats:
        json_export = json.dumps({
            "log_input": log_input,
            "anomaly_result": anomaly_result,
            "ai_analysis": ai_analysis['analysis']
        }, indent=2)
        exports['json'] = json_export
    
    # CSV Export
    if "CSV" in export_formats:
        # Convert anomaly result and analysis to CSV
        df = pd.DataFrame({
            "Log Input": [log_input],
            "Total Anomalies": [sum(anomaly_result['has_anomaly'])],
            "Anomaly Counts": [str(anomaly_result['anomaly_count'])],
            "AI Analysis": [ai_analysis['analysis']]
        })
        csv_export = df.to_csv(index=False)
        exports['csv'] = csv_export
    
    # Markdown Export
    if "Markdown" in export_formats:
        markdown_export = f"""
        # Log Anomaly Analysis Report

        ## Log Input
        ```
        {log_input[:500]}...
        ```

        ## Anomaly Metrics
        - Total Anomalies: {sum(anomaly_result['has_anomaly'])}
        - Anomaly Counts: {anomaly_result['anomaly_count']}

        ## AI Analysis
        {ai_analysis['analysis']}
        """
        exports['md'] = markdown_export
    
    # PDF Export
    if "PDF" in export_formats:
        pdf_export = generate_pdf_report(log_input, anomaly_result, ai_analysis)
        exports['pdf'] = pdf_export
    
    return exports

def main():
    # Page configuration
    st.set_page_config(page_title="Advanced Log Anomaly Detection & Analysis", layout="wide")
    # Setup sidebar and get configuration settings
    sidebar_settings = setup_sidebar()

    st.title("Advanced Log Anomaly Detection & Analysis")

    # Initialize session state if not exists
    if 'anomaly_detected' not in st.session_state:
        st.session_state.anomaly_detected = False
        st.session_state.log_input = None
        st.session_state.anomaly_result = None
        st.session_state.ai_analysis = None

    # Log input text area
    # log_input = st.text_area("Enter Log Messages:", height=300)

    # Conditional log input based on sidebar selection
    if sidebar_settings['log_source'] == "Custom Input":
        log_input = st.text_area("Enter Log Messages:", height=300)
    elif sidebar_settings['log_source'] == "File Upload":
        uploaded_file = st.file_uploader("Upload Log File", type=['log', 'txt'])
        log_input = uploaded_file.read().decode() if uploaded_file else ""
    else:
        # Placeholder for system logs
        log_input = st.text_area("Enter System Log Messages:", height=300)


    # Detect button
    if st.button("Detect and Analyze Anomalies"):
        if log_input:
            with st.spinner("Detecting and analyzing anomalies..."):
                try:
                    # Detect anomalies
                    with multiprocessing.Pool(processes=1) as pool:
                        anomaly_result = pool.apply(detect_anomalies, (log_input,))

                    # Store results in session state
                    st.session_state.anomaly_detected = True
                    st.session_state.log_input = log_input
                    st.session_state.anomaly_result = anomaly_result

                    # # Update analysis history
                    # analysis_summary = f"Anomalies: {sum(anomaly_result['has_anomaly'])}"
                    # st.session_state.analysis_history.append(analysis_summary)

                    # Process results
                    if st.session_state.anomaly_result:
                        st.subheader("🔍 Anomaly Detection Results")
                        
                        if any(anomaly_result['has_anomaly']):
                            st.error("🚨 Anomalies Detected!")
                            for i, (has_anomaly, anomaly_count) in enumerate(zip(anomaly_result['has_anomaly'], anomaly_result['anomaly_count']), 1):
                                if has_anomaly:
                                    st.warning(f"Log Sequence {i}: Anomaly Count = {anomaly_count}")
                            
                            # AI-Powered Root Cause Analysis
                            with st.spinner("Generating AI-powered analysis..."):
                                ai_analysis = get_ai_analysis(log_input, anomaly_result, sidebar_settings)
                                st.session_state.ai_analysis = ai_analysis
                                
                                if ai_analysis:
                                    with st.expander("🤖 AI-Powered Anomaly Analysis"):
                                        st.markdown("### Comprehensive Anomaly Insights")
                                        st.write(ai_analysis['analysis'])
                        else:
                            st.success("✅ No Anomalies Detected")
                        
                        # Raw Anomaly Details
                        with st.expander("Detailed Anomaly Metrics"):
                            st.json(anomaly_result)
                    
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
        else:
            st.warning("Please enter log messages to detect anomalies.")

    # Display results and action buttons only if anomalies were detected
    if st.session_state.anomaly_detected and st.session_state.anomaly_result and st.session_state.ai_analysis:
        st.markdown("---")
        st.subheader("Anomaly Analysis Actions")
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        if st.button("Generate Incident Report", key="generate_report_btn"):
            with st.expander("Incident Report"):
                # Use stored session state values
                report = generate_incident_report(
                    st.session_state.log_input, 
                    st.session_state.anomaly_result, 
                    st.session_state.ai_analysis
                )
                st.subheader("🗒️ Incident Report")
                st.markdown(report)

        # with col2:
        if st.button("Export Detailed Analysis", key="export_analysis_btn"):
            # Use stored session state values
            export_formats = sidebar_settings.get('export_format', ['JSON'])
            
            # Perform exports
            exports = export_analysis(
                st.session_state.log_input, 
                st.session_state.anomaly_result, 
                st.session_state.ai_analysis,
                export_formats
            )
            
            # Create download buttons for each selected format
            st.subheader("📥 Download Analysis")
            
            # JSON Export
            if 'json' in exports:
                st.download_button(
                    label="Download JSON Report",
                    data=exports['json'],
                    file_name="anomaly_analysis_report.json",
                    mime="application/json",
                    key="download_json_btn"
                )
            
            # CSV Export
            if 'csv' in exports:
                st.download_button(
                    label="Download CSV Report",
                    data=exports['csv'],
                    file_name="anomaly_analysis_report.csv",
                    mime="text/csv",
                    key="download_csv_btn"
                )
            
            # Markdown Export
            if 'md' in exports:
                st.download_button(
                    label="Download Markdown Report",
                    data=exports['md'],
                    file_name="anomaly_analysis_report.md",
                    mime="text/markdown",
                    key="download_md_btn"
                )
            
            # PDF Export
            if 'pdf' in exports:
                st.download_button(
                    label="Download PDF Report",
                    data=exports['pdf'],
                    file_name="anomaly_analysis_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_btn"
                )
    # Extract unique IDs
    # unique_ids = extract_unique_ids(combined_logs)
    log_input = log_input.splitlines()
    unique_ids = get_valid_unique_ids(log_input)
    st.sidebar.header("Select Unique ID")
    selected_id = st.sidebar.selectbox("Unique IDs", unique_ids)

    if selected_id:
        # Fetch related log lines
        related_logs = filter_logs_by_id(log_input, selected_id)

        # Generate Mermaid Code
        if st.button("Generate Mermaid Diagram"):
            with st.spinner("Generating Mermaid diagram..."):
                mermaid_code = generate_mermaid_code(related_logs, selected_id)
                print(mermaid_code)

                st.markdown(f"### Flow Diagram for {selected_id}")
                html(mermaid_chart(mermaid_code), width=1500, height=1500) # 
             
if __name__ == "__main__":
    main()



# def main():
#     st.title("Log Anomaly Detection")

#     # Log input text area
#     log_input = st.text_area("Enter Log Messages:", height=300)

#     # Detect button
#     if st.button("Detect Anomalies"):
#         if log_input:
#             with st.spinner("Detecting anomalies..."):
#                 try:
#                     # Use multiprocessing to run detect_anomalies
#                      with multiprocessing.Pool(processes=1) as pool:
#                         # Detect anomalies
#                         # result = detect_anomalies(log_input)
#                         result = pool.apply(detect_anomalies, (log_input,))

#                         # # Run detect_anomalies in a separate thread to avoid blocking
#                         # result = threading.Thread(target=detect_anomalies, args=(log_input,))
#                         # result.start()
#                         # result.join()  # Wait for the thread to complete
#                         # Process result if no exception occurs
#                         if result:
#                             # Display results
#                             st.subheader("Anomaly Detection Results")
                            
#                             # Check if any anomalies detected
#                             if any(result['has_anomaly']):
#                                 st.error("🚨 Anomalies Detected!")
#                                 st.write("Anomaly Details:")
#                                 for i, (has_anomaly, anomaly_count) in enumerate(zip(result['has_anomaly'], result['anomaly_count']), 1):
#                                     if has_anomaly:
#                                         st.warning(f"Log Sequence {i}: Anomaly Count = {anomaly_count}")
#                             else:
#                                 st.success("✅ No Anomalies Detected")
                            
#                             # Display detailed anomaly counts
#                             st.subheader("Detailed Anomaly Counts")
#                             st.json(result)
#                         else:
#                             st.error("An error occurred during anomaly detection.")
                    
#                 except Exception as e:
#                     st.error(f"An error occurred: {str(e)}")
#         else:
#             st.warning("Please enter log messages to detect anomalies.")

# if __name__ == "__main__":
#     main()
