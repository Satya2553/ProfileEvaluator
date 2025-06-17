import streamlit as st
import pandas as pd
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import io
from datetime import datetime
import traceback
from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
import xlsxwriter
import math
from collections import deque
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from requests import session
from streamlit import header

load_dotenv()

def create_excel_output(df: pd.DataFrame, filename: str = "candidate_analysis_results.xlsx") -> bytes:
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Analysis Results', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Analysis Results']

            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            for i, col in enumerate(df.columns):
                max_len = 0
                if not df[col].empty:
                    max_len = max(df[col].astype(str).apply(len).max(), len(str(col)))
                worksheet.set_column(i, i, min(max_len + 2, 50))

        output.seek(0)
        return output.getvalue()

    except Exception as e:
        st.error(f"Error creating Excel output: {str(e)}")
        return b""

def send_interview_emails(recipient_email: str, selected_candidates: pd.DataFrame, role: str) -> bool:
    try:
        sender_email = os.getenv('EMAIL_USER')
        sender_password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))

        if not all([sender_email, sender_password]):
            st.error("Email credentials not configured. Please set EMAIL_USER and EMAIL_PASSWORD environment variables.")
            return False

        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Interview Candidates for {role} Position"

        html_table = selected_candidates.to_html(
            index=False,
            classes='table table-striped',
            border=1,
            justify='left',
            na_rep='N/A'
        )

        html_body = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #ffffff;
                }}
                .header {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border: 1px solid #e9ecef;
                }}
                .header h2 {{
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 14px;
                }}
                .table th, .table td {{
                    padding: 12px;
                    border: 1px solid #dee2e6;
                    text-align: left;
                }}
                .table th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .table tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .table tr:hover {{
                    background-color: #f1f3f5;
                }}
                .score-cell {{
                    font-weight: bold;
                }}
                .rank-cell {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .remarks-cell {{
                    font-style: italic;
                    color: #495057;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                    font-size: 0.9em;
                    color: #6c757d;
                }}
                .summary {{
                    background-color: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Interview Candidates for {role} Position</h2>
                    <p>Please find below the list of selected candidates for the {role} position.</p>
                </div>
                
                <div class="summary">
                    <p><strong>Total Candidates Selected:</strong> {len(selected_candidates)}</p>
                    <p><strong>Score Range:</strong> {selected_candidates['score'].min():.1f} - {selected_candidates['score'].max():.1f}</p>
                </div>
                
                {html_table}
                
                <div class="footer">
                    <p>Best regards,<br>AI Candidate Analysis System</p>
                    <p><small>This is an automated email. Please do not reply directly to this message.</small></p>
                </div>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, 'html'))

        excel_data = create_excel_output(selected_candidates)
        if excel_data:
            excel_attachment = MIMEApplication(excel_data, _subtype='xlsx')
            excel_attachment.add_header('Content-Disposition', 'attachment', filename='selected_candidates.xlsx')
            msg.attach(excel_attachment)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return True

    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

class CandidateAnalysisSystem:
    def __init__(self):
        self.gemini_client = None
        self.initialize_gemini_client()
        self.parallel_batches_limit = 5
        self.failed_batches_data = []
        self.executed_batches = deque(maxlen=3)
        self.not_executed_batches = deque(maxlen=3)
        self.failed_batches = deque(maxlen=3)

    def initialize_gemini_client(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("GEMINI_API_KEY missing.")
            return

        try:
            self.gemini_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            st.success("Gemini API client initialized.")
        except Exception as e:
            st.error(f"Gemini init failed: {str(e)}")

    def read_excel_file(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            engine_map = {
                'xlsx': 'openpyxl',
                'xlsm': 'openpyxl',
                'xls': 'xlrd',
                'xlsb': 'pyxlsb'
            }
            engine = engine_map.get(file_extension)

            if not engine:
                raise ValueError(f"Unsupported file format: {file_extension}")

            excel_file = pd.ExcelFile(uploaded_file, engine=engine)

            sheets_data = {}
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if not df.empty:
                        df.columns = df.columns.astype(str).str.strip().str.replace(r'[^\w\s]', '', regex=True)
                        sheets_data[sheet_name] = df
                except Exception as e:
                    st.warning(f"Sheet '{sheet_name}' read error: {str(e)}")

            return sheets_data

        except Exception as e:
            st.error(f"Excel file read error: {str(e)}")
            return {}

    def merge_duplicate_records(self, df: pd.DataFrame, merge_identifier: Optional[str]) -> Tuple[pd.DataFrame, int]:
        if not merge_identifier:
            initial_count = len(df)
            df = df.drop_duplicates(keep='first')
            merged_count = initial_count - len(df)
            return df, merged_count

        if merge_identifier not in df.columns:
            st.warning(f"Merge identifier '{merge_identifier}' not found in DataFrame.")
            return df, 0

        initial_count = len(df)
        df = df.groupby(merge_identifier, as_index=False).first()
        merged_count = initial_count - len(df)
        return df, merged_count

    def remove_records_by_status(self, df: pd.DataFrame, status_identifier: Optional[str], status_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, int]:
        if status_df is None or status_identifier is None:
            return df, 0

        if status_identifier not in df.columns or status_identifier not in status_df.columns:
            st.warning(f"Status identifier '{status_identifier}' not found in DataFrames.")
            return df, 0

        initial_len = len(df)
        status_values = status_df[status_identifier].astype(str).tolist()
        df[status_identifier] = df[status_identifier].astype(str)
        df = df[~df[status_identifier].isin(status_values)]
        removed_count = initial_len - len(df)
        return df, removed_count

    def select_valid_columns(self, df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:

        valid_columns = [col for col in selected_columns if col in df.columns]
        return df[valid_columns]

    def consolidate_data(self, dataframes: List[pd.DataFrame], selected_columns: List[str],
                         merge_identifier: Optional[str], status_identifier: Optional[str],
                         status_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, int, int]:
        try:
            if not dataframes:
                return pd.DataFrame(), 0, 0

            consolidated_df = pd.concat(dataframes, ignore_index=True)
            initial_len = len(consolidated_df)

            consolidated_df, merged_count = self.merge_duplicate_records(consolidated_df, merge_identifier)

            consolidated_df, removed_count = self.remove_records_by_status(consolidated_df, status_identifier, status_df)

            consolidated_df = self.select_valid_columns(consolidated_df, selected_columns)

            consolidated_df['unique_id'] = range(1, len(consolidated_df)+1)

            return consolidated_df, merged_count, removed_count

        except Exception as e:
            st.error(f"Data consolidation error: {str(e)}")
            st.text(f"Full error: {traceback.format_exc()}")
            return pd.DataFrame(), 0, 0


    def prepare_batch_data(self, df: pd.DataFrame, unique_id_col: str, batch_size: int = 50) -> List[Dict]:
        batches = []
        total_candidates = len(df)

        df_for_json = df.copy()
        records = df_for_json.fillna("Not specified").astype(str).to_dict(orient='records')
        

        for i in range(0, total_candidates, batch_size):
            batch_records = records[i:i + batch_size]
            batch_data = {
                "batch_id": len(batches) + 1,
                "batch_size": len(batch_records),
                "unique_identifier_column": unique_id_col,
                "candidates": batch_records
            }
            batches.append(batch_data)
        return batches

    async def analyze_batch_with_gemini(self, batch_info: Dict, role: str, total_batches: int, semaphore: asyncio.Semaphore) -> Tuple[int, Optional[List[Dict]]]:
        batch_id = batch_info['batch_id']
        batch_data_str = json.dumps(batch_info, indent=2)
        unique_id_col = batch_info['unique_identifier_column']


        async with semaphore:
            if not self.gemini_client:
                st.error(f"Batch {batch_id}: Gemini client not initialized.")
                return batch_id, None

            st.info(f"⏳ Batch {batch_id}/{total_batches}: Starting analysis for {batch_info['batch_size']} candidates.")

            prompt = f"""
You are an expert HR analyst. Analyze this batch of candidate data for the role of "{role}".

IMPORTANT: This is batch {batch_id} of {total_batches}. You need to evaluate and score each candidate in this batch.

Your task:
1. Evaluate each candidate based on their qualifications, experience, and suitability for the "{role}" position.
    Prioritize skills along with experience, and finally availability or other factors.
2. Give each candidate a score from 0-500 (500 being perfect fit, 0 being completely unsuitable). 
3. Give the score accurately instead of continuosly giving in multiple of 5 and 10.
4. Provide specific remarks for each candidate explaining their evaluation

Data:
{batch_data_str}

CRITICAL REQUIREMENTS:
- Return ONLY a valid JSON array with NO additional text, no markdown code block delimiters (```json, ```)
- Each object must have exactly these keys: "{unique_id_col}", "score", "remarks"
- The "{unique_id_col}" value must EXACTLY match the values from the input data
- Score must be a number from 0 to 500
- Every candidate in this batch must be included in the response
- Remarks should be 1-2 sentences explaining the evaluation

Example format:
[
  {{
    "{unique_id_col}": "value_from_data",
    "score": 85,
    "remarks": "Strong technical background with relevant experience for this role."
  }},
  {{
    "{unique_id_col}": "another_value_from_data",
    "score": 0,
    "remarks": "Completely out of scope for this role."
  }}
]
"""
            try:
                response = await self.gemini_client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10000
                )

                response_text = response.choices[0].message.content.strip()

                if response_text.startswith("```json"):
                    response_text = response_text[len("```json"):].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[len("```"):].strip()

                if response_text.endswith("```"):
                    response_text = response_text[:-len("```")].strip()

                batch_results = json.loads(response_text)

                if not isinstance(batch_results, list):
                    st.error(f"Batch {batch_id} FAILED: Response is not a list. Response: {response_text[:100]}...")
                    self.failed_batches.append(batch_id)
                    return batch_id, None

                valid_results = []
                for result in batch_results:
                    if not isinstance(result, dict):
                        continue

                    required_keys = [unique_id_col, 'score', 'remarks']
                    if not all(key in result for key in required_keys):
                        continue

                    try:
                        result['score'] = float(result['score'])
                    except (ValueError, TypeError):
                        continue
                    valid_results.append(result)
                st.success(f"Batch {batch_id}/{total_batches}: Successfully executed {batch_info['batch_size']} candidates.")
                self.executed_batches.append(batch_id)
                return batch_id, valid_results

            except (RateLimitError, APIError) as e:
                st.error(f"Batch {batch_id} FAILED: API Error: {e}. Will mark for retry.")
                self.failed_batches.append(batch_id)
                return batch_id, None
            except json.JSONDecodeError as e:
                st.error(f"atch {batch_id} FAILED: Failed to parse JSON: {e}. Raw response: {response_text[:500]}...")
                self.failed_batches.append(batch_id)
                return batch_id, None
            except Exception as e:
                st.error(f"Batch {batch_id} FAILED: Unexpected error: {str(e)}. Trace: {traceback.format_exc()}")
                self.failed_batches.append(batch_id)
                return batch_id, None

    async def run_analysis(self, batches: List[Dict], role: str, total_batches: int, progress_bar, status_text) -> Tuple[List[Dict], List[Dict]]:
        all_results = []
        failed_current_run = []
        semaphore = asyncio.Semaphore(self.parallel_batches_limit)

        tasks = []

        for batch_info in batches:
            tasks.append(
                self.analyze_batch_with_gemini(
                    batch_info, role, total_batches, semaphore
                )
            )

        results_from_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        processed_count = 0
        for i, result in enumerate(results_from_tasks):
            batch_id = batches[i]['batch_id']
            if isinstance(result, tuple) and len(result) == 2:
                current_batch_id, data = result
                if data is not None:
                    all_results.extend(data)
                else:
                    failed_current_run.append(batches[i])
                    self.not_executed_batches.append(batch_id)
            elif isinstance(result, Exception):
                st.error(f"Batch {batch_id} FAILED: An unexpected exception occurred: {result}")
                failed_current_run.append(batches[i])
                self.not_executed_batches.append(batch_id)
            else:
                st.error(f"Batch {batch_id} FAILED: Unexpected result type from task: {result}")
                failed_current_run.append(batches[i])
                self.not_executed_batches.append(batch_id)

            processed_count += 1
            progress_bar.progress(processed_count / total_batches)
            status_text.text(f"Completed {processed_count} of {total_batches} batches...")

        return all_results, failed_current_run

    async def analyze_all_candidates(self, df: pd.DataFrame, role: str) -> Optional[List[Dict]]:
        if not self.gemini_client:
            return None

        self.failed_batches_data = []
        self.executed_batches.clear()
        self.not_executed_batches.clear()
        self.failed_batches.clear()

        try:
            batches = self.prepare_batch_data(df, 'unique_id')
            if not batches:
                st.error("Failed to prepare batches for analysis.")
                return None

            total_batches = len(batches)
            st.info(f"Initial analysis: Processing {total_batches} batches of candidates with {self.parallel_batches_limit} parallel requests...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            

            all_results, failed_batches = await self.run_analysis(batches, role, total_batches, progress_bar, status_text)

            self.failed_batches_data.extend(failed_batches)

            progress_bar.empty()
            status_text.empty()

            if not all_results and not self.failed_batches_data:
                st.error(" o results obtained from any batch, and no batches failed (unusual).")
                return None

            return all_results

        except Exception as e:
            st.error(f"Error during complete AI analysis setup: {str(e)}")
            st.text(f"Full error: {traceback.format_exc()}")
            return None

    async def retry_failed_batches(self, role: str) -> Optional[List[Dict]]:
        if not self.failed_batches_data:
            st.info("No failed batches to retry.")
            return []

        st.warning(f" Retrying {len(self.failed_batches_data)} failed batches...")

        batches_to_retry = self.failed_batches_data[:]
        self.failed_batches_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        retry_results, newly_failed_batches = await self.run_analysis(batches_to_retry, role, len(batches_to_retry), progress_bar, status_text)
        self.failed_batches_data.extend(newly_failed_batches)

        progress_bar.empty()
        status_text = st.empty()

        if retry_results:
            st.success(f"Retried batches: Successfully processed {len(retry_results)} candidates from retries.")
        else:
            st.info("No candidates processed in this retry attempt.")
        return retry_results

    def convert_scores_to_ranks(self, results: List[Dict], unique_id_col: str) -> List[Dict]:
        if not results:
            return []
        try:
            results_df = pd.DataFrame(results)
            results_df['score'] = pd.to_numeric(results_df['score'], errors='coerce')

            results_df.dropna(subset=['score'], inplace=True)

            if results_df.empty:
                return []

            results_df['rank'] = results_df['score'].rank(method='dense', ascending=False).astype(int)

            results_df = results_df.sort_values(by=['rank', 'score'], ascending=[True, False])
            return results_df.to_dict(orient='records')

        except Exception as e:
            st.error(f"Error converting scores to ranks: {str(e)}")
            return results

    def merge_ai_results(self, original_df: pd.DataFrame, ai_results: List[Dict], unique_id_col: str) -> pd.DataFrame:
        try:
            
            ai_df = pd.DataFrame(ai_results)
            ai_df['score'] = pd.to_numeric(ai_df['score'], errors='coerce')
            
            original_df[unique_id_col] = original_df[unique_id_col].astype(str)
            ai_df[unique_id_col] = ai_df[unique_id_col].astype(str)

            original_ids = set(original_df[unique_id_col])
            ai_ids = set(ai_df[unique_id_col])
            common_ids = original_ids & ai_ids


            if len(common_ids) == 0:
                print("⚠️ Warning: No matching unique_id values found between original and AI results! Merge will result in all NaNs.")

            df_to_merge = original_df.copy()
            df_to_merge = df_to_merge.drop(columns=['score', 'remarks', 'rank'], errors='ignore')

            ranked_ai_results = self.convert_scores_to_ranks(ai_df.to_dict(orient='records'), unique_id_col)
            ranked_ai_df = pd.DataFrame(ranked_ai_results)

            merged_df = df_to_merge.merge(ranked_ai_df, on=unique_id_col, how='left')

            merged_df['score'] = pd.to_numeric(merged_df['score'], errors='coerce').fillna(0)
            merged_df['remarks'] = merged_df['remarks'].fillna("Not analyzed by AI")
            merged_df['rank'] = pd.to_numeric(merged_df['rank'], errors='coerce').fillna(len(merged_df) + 1)

            merged_df = merged_df.sort_values('rank', ascending=True)

            analyzed_count = merged_df['rank'].apply(lambda x: x <= len(ai_results)).sum()
            return merged_df

        except Exception as e:
            original_df['rank'] = None
            original_df['score'] = None
            original_df['remarks'] = "Error in analysis or merge"
            return original_df

async def run_full_analysis_workflow(system, consolidated_df, role):
    st.session_state.analysis_in_progress = True
    st.session_state.ai_results = []
    st.session_state.final_df = pd.DataFrame()
    st.session_state.failed_batches_count = 0


    ai_results = await system.analyze_all_candidates(consolidated_df, role)
    if ai_results:
        st.session_state.ai_results.extend(ai_results)
        st.session_state.final_df = system.merge_ai_results(st.session_state.original_df, st.session_state.ai_results, 'unique_id')

    else:
        st.session_state.final_df = st.session_state.original_df.copy()

    st.session_state.failed_batches_count = len(system.failed_batches_data)
    st.session_state.analysis_complete = True
    st.session_state.analysis_in_progress = False
    st.rerun()

async def run_retry_workflow(system, consolidated_df, role, original_df):
    st.session_state.analysis_in_progress = True
    retry_results = await system.retry_failed_batches(role)

    if retry_results:
        st.session_state.ai_results.extend(retry_results)
        st.session_state.final_df = system.merge_ai_results(original_df, st.session_state.ai_results, 'unique_id')

    st.session_state.failed_batches_count = len(system.failed_batches_data)
    st.session_state.analysis_in_progress = False
    st.rerun()

def main():
    st.set_page_config(
        page_title="Candidate Analysis System",
        page_icon="",
        layout="wide",
    )

    st.title(" AI-Powered Candidate Analysis System")
    st.markdown("Upload Excel files, consolidate candidate data, and get AI-powered rankings.")

    if 'analysis_system' not in st.session_state:
        st.session_state.analysis_system = CandidateAnalysisSystem()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_in_progress' not in st.session_state:
        st.session_state.analysis_in_progress = False
    if 'ai_results' not in st.session_state:
        st.session_state.ai_results = []
    if 'final_df' not in st.session_state:
        st.session_state.final_df = pd.DataFrame()
    if 'failed_batches_count' not in st.session_state:
        st.session_state.failed_batches_count = 0
    if 'consolidated_df' not in st.session_state:
        st.session_state.consolidated_df = pd.DataFrame()
    if 'original_df' not in st.session_state:
        st.session_state.original_df = pd.DataFrame()
    if 'unique_id_col' not in st.session_state:
        st.session_state.unique_id_col = None
    if 'role_input' not in st.session_state:
        st.session_state.role_input = ""
    if 'all_dataframes' not in st.session_state:
        st.session_state.all_dataframes = []
    if 'merge_identifier' not in st.session_state:
        st.session_state.merge_identifier = None
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []
    if 'merged_count' not in st.session_state:
        st.session_state.merged_count = 0
    if 'status_df' not in st.session_state:
        st.session_state.status_df = None
    if 'status_identifier' not in st.session_state:
        st.session_state.status_identifier = None
    if 'removed_count' not in st.session_state:
        st.session_state.removed_count = 0

    system = st.session_state.analysis_system

    if not system.gemini_client:
        st.error("GEMINI_API_KEY missing.")
        st.stop()

    st.subheader(" Upload Excel Files")
    uploaded_files = st.file_uploader(
        "Choose Excel files",
        type=['xlsx', 'xls', 'xlsb', 'xlsm'],
        accept_multiple_files=True,
        help="Upload one or more Excel files containing candidate data."
    )

    all_sheets_data = {}
    if uploaded_files:
        st.session_state.all_dataframes = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                sheets_data = system.read_excel_file(uploaded_file)
                for sheet_name, df in sheets_data.items():
                    st.session_state.all_dataframes.append(df)

    if not st.session_state.all_dataframes:
        st.info("Upload Excel files to begin analysis.")
        return

    all_columns = []
    for df in st.session_state.all_dataframes:
        all_columns.extend(df.columns.tolist())
    all_columns = list(set(all_columns))

    st.subheader(" Configure Data Selection")

    st.subheader("Merge Duplicates")
    merge_identifier = st.selectbox(
        "Select column to merge duplicate records (e.g., Employee ID):",
        [None] + all_columns,
        index=0,
        key="merge_column",
        help="Choose a column to identify and merge duplicate records. Leave as None to skip duplicate merging."
    )

    st.session_state.merge_identifier = merge_identifier
    consolidated_df = pd.concat(st.session_state.all_dataframes, ignore_index=True)
    if st.session_state.merge_identifier:
        with st.spinner("Merging data..."):
            consolidated_df, merged_count = system.merge_duplicate_records(consolidated_df, st.session_state.merge_identifier)
            st.session_state.consolidated_df = consolidated_df
            st.session_state.unique_id_col = 'unique_id'
            st.session_state.merged_count = merged_count

        if st.session_state.merged_count > 0:
            st.info(f"Merged {st.session_state.merged_count} duplicate records based on '{st.session_state.merge_identifier}'.")
        else:
            st.info("No duplicate records found based on the selected merge identifier.")
    else:
        st.session_state.consolidated_df = consolidated_df
        st.session_state.merged_count = 0
        st.info("No merge identifier selected. Duplicates may exist.")

    st.subheader(" Upload Status Sheet (Optional)")
    status_file = st.file_uploader(
        "Choose Status Excel file",
        type=['xlsx', 'xls', 'xlsb', 'xlsm'],
        help="Upload an Excel file containing candidate status information.",
        key="status_file_uploader"
    )

    if status_file:
        with st.spinner("Reading status file..."):
            status_sheets_data = system.read_excel_file(status_file)
            if status_sheets_data:
                status_sheet_name = list(status_sheets_data.keys())[0]
                st.session_state.status_df = status_sheets_data[status_sheet_name]

                st.subheader(" Configure Status Sheet")
                st.session_state.status_identifier = st.selectbox(
                    "Select unique identifier column in status sheet:",
                    st.session_state.status_df.columns,
                    index=0 if len(st.session_state.status_df.columns) > 0 else None,
                    key="status_unique_id"
                )

                if st.session_state.status_identifier:
                    st.success("Status sheet configured.")
                    st.dataframe(st.session_state.status_df.head(3), use_container_width=True)
                    consolidated_df, removed_count = system.remove_records_by_status(st.session_state.consolidated_df,
                                                                 st.session_state.status_identifier,
                                                                 st.session_state.status_df)
                    st.session_state.consolidated_df = consolidated_df
                    st.session_state.unique_id_col = 'unique_id'
                    st.session_state.removed_count = removed_count
                    st.info("Removed %d records based on status." % removed_count)
                else:
                    st.warning("Configure the status sheet.")
                    st.session_state.status_df = None
            else:
                st.error("Could not read status file.")
                st.session_state.status_df = None
    else:
        st.session_state.status_df = None
        st.session_state.status_identifier = None

    st.session_state.unique_id_col = 'unique_id'
    st.session_state.consolidated_df['unique_id'] = range(1, len(consolidated_df)+1)

    st.session_state.original_df = st.session_state.consolidated_df.copy()

    st.subheader("Select Columns For Analysis")
    available_columns = all_columns
    available_columns.append('unique_id')

    with st.form("select_columns_form"):
        selected_columns = st.multiselect(
            "Choose columns to include in analysis:",
            available_columns,
            default=st.session_state.selected_columns,
            key="columns_all",
        )

        submitted = st.form_submit_button("Apply Column Selection")

    if submitted or not st.session_state.consolidated_df.empty:
        if submitted:
            if('unique_id' not in selected_columns):
                selected_columns.append('unique_id')
            st.session_state.selected_columns = selected_columns

        sensitive_patterns = [
            r"(phone|mobile|contact)\s*number",
            r"phone|mobile|contact",
            r"(employee)\s*(id|number)",
            r"(employeeid|employeenumber)"
        ]
        if submitted:
            invalid_columns = [
                col for col in selected_columns
                if any(re.search(pattern, col, re.IGNORECASE) for pattern in sensitive_patterns)
            ]

            if invalid_columns:
                is_sensitive_selected = True
                st.error(f"Sensitive columns {', '.join(invalid_columns)} are selected. Please deselect them.")
                return
            else:
                st.session_state.selected_columns = selected_columns
                is_sensitive_selected = False
                error_message = None
        else:
            selected_columns = st.session_state.selected_columns

        if selected_columns:
            st.subheader(" Data Consolidation")

            with st.spinner("Consolidating data with selected columns..."):
                consolidated_df = system.select_valid_columns(st.session_state.consolidated_df, selected_columns)
                st.session_state.consolidated_df = consolidated_df


            if consolidated_df.empty:
                st.error("Failed to consolidate data.")
                return

            st.success(f"Consolidated {len(consolidated_df)} candidate records.")
            st.info(f" Using **'unique_id'** as the unique identifier across all data.")

            st.subheader("Preview of Consolidated Data")
            st.dataframe(consolidated_df.head(10), use_container_width=True)

            st.subheader(" Analysis Configuration")

            role = st.text_input(
                "Role/Position being analyzed:",
                placeholder="e.g., Senior Software Engineer, Data Scientist, or Selenium with Java,etc..",
                help="Specify the role you're hiring for. This helps the AI tailor its evaluation.",
                value=st.session_state.role_input,
                key="role_input_text"
            )
            st.session_state.role_input = role

            st.info(f" The system will analyze and rank **ALL {len(consolidated_df)} candidates** using concurrent batch processing with **{system.parallel_batches_limit} parallel AI requests**.")

            if not role:
                st.warning("Specify the role being analyzed to proceed.")
                return

            st.subheader(" AI Analysis")

            if st.button(" Start Analysis", type="secondary", disabled=st.session_state.analysis_in_progress):
                st.info("Initiating AI analysis for all candidates...")
                asyncio.run(run_full_analysis_workflow(system, consolidated_df, role))

            if st.session_state.analysis_in_progress:
                st.info("Analysis is currently in progress. Please wait...")

            if st.session_state.analysis_complete:
                if st.session_state.ai_results:
                    st.success("Complete AI analysis finished!")

                    ranked_df = st.session_state.final_df[st.session_state.final_df['rank'].notna()].copy()
                    ranked_df['rank'] = ranked_df['rank'].astype(int)

                    if not ranked_df.empty:
                        if st.session_state.failed_batches_count > 0:
                            st.warning(
                                f" {st.session_state.failed_batches_count} batches failed during analysis. You can retry them.")
                            if st.button(f"Retry {st.session_state.failed_batches_count} Failed Batches",
                                         disabled=st.session_state.analysis_in_progress,key="Retry Batches"):
                                asyncio.run(
                                    run_retry_workflow(system, st.session_state.consolidated_df, st.session_state.role_input))
                        
                        st.subheader(" Complete Ranking Table")
                        display_columns = [col for col in ['rank', 'score', 'unique_id', 'remarks'] if
                                           col in ranked_df.columns] + \
                                          [col for col in ranked_df.columns if
                                           col not in ['rank', 'score', 'unique_id', 'remarks', 'data_source']]

                        st.dataframe(
                            ranked_df[display_columns],
                            use_container_width=True,
                            hide_index=True
                        )

                        st.subheader(" Download Results")

                        excel_data = create_excel_output(st.session_state.final_df)

                        if excel_data:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"complete_candidate_analysis_{timestamp}.xlsx"

                            st.download_button(
                                label = " Download Complete Analysis Report",
                                data = excel_data,
                                file_name = filename,
                                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            st.success("Your complete analysis report is ready for download!")

                        st.subheader("Send Interview Emails")
                        st.info("Select a range of candidates to send interview emails.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            email_min_score = st.number_input(
                                "Minimum Score for Emails:",
                                min_value=0,
                                max_value=500,
                                value=None,
                                placeholder="Enter minimum score for email selection",
                                help="Only candidates with a score greater than or equal to this value will be included in the email.",
                                format="%d",
                                key="email_min_score"
                            )
                        with col2:
                            email_max_score = st.number_input(
                                "Maximum Score for Emails:",
                                min_value=0,
                                max_value=500,
                                value=None,
                                placeholder="Enter maximum score for email selection",
                                help="Only candidates with a score less than or equal to this value will be included in the email.",
                                format="%d",
                                key="email_max_score"
                            )

                        email_candidates = ranked_df.copy()
                        if email_min_score is not None:
                            email_candidates = email_candidates[email_candidates['score'] >= email_min_score]
                        if email_max_score is not None:
                            email_candidates = email_candidates[email_candidates['score'] <= email_max_score]

                        if not email_candidates.empty:
                            st.info(f"Selected {len(email_candidates)} candidates for email.")
                            st.dataframe(
                                email_candidates[display_columns],
                                use_container_width=True,
                                hide_index=True
                            )

                            recipient_email = st.text_input(
                                "Recipient Email:",
                                placeholder="Enter email address to send selected candidates",
                                help="The email address where the selected candidates will be sent."
                            )

                            if recipient_email:
                                if st.button("Send Interview Emails", type="primary"):
                                    if send_interview_emails(recipient_email, email_candidates, st.session_state.role_input):
                                        st.success(f"Email sent successfully to {recipient_email}")
                                    else:
                                        st.error("Failed to send email. Please check the email configuration.")
                        else:
                            st.warning("No candidates match the selected score range for email.")
                    else:
                        st.error(
                            "AI analysis failed or returned no results. Please check your API key, role input, and data consistency.")

if __name__ == "__main__":
    main()