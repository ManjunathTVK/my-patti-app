import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Business Report Dashboard")

st.title("📊 Google Sheets Data Processor & Reporter")

# --- CONNECTION ---
SHEET_ID = "19DdEF2bPLOY1cJ2wHJtuG9JkPrLlRbb2I37CvnOCqYg"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

@st.cache_data
def load_and_clean_data(url):
    # 1. Load raw CSV
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()

    # 2. INTENSE DATE CLEANING (Fixes "Wrong Dates" and "None" issues)
    if 'Arrival Date' in df.columns:
        # Convert to string and clean up characters
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/')
        
        # Handle empty/merged cells
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        df['Arrival Date'] = df['Arrival Date'].ffill()
        
        # FORCE PARSING using the specific Indian/UK format DD/MM/YYYY
        # errors='coerce' will turn truly unreadable text into NaT (Not a Time)
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], format='%d/%m/%Y', errors='coerce')
        
        # Second pass ffill: if a specific row failed, it takes the date from the entry above
        df['Arrival Date'] = df['Arrival Date'].ffill()

    # 3. FORWARD FILL TEXT (Supplier, Patty date, etc.)
    text_cols = ['Patty date', 'Supplier Name', 'Lorry No.']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace(['nan', 'None', '', ' '], np.nan).ffill()

    # 4. NUMERIC CLEANING
    numeric_cols = ['Sale Amt', 'Patty Amt', 'Payment amt', 'Cleaning coolie', 'Un Coolie', 'Wei Coolie']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 5. GENERATE MONTHLY FEATURES
    # Drop rows that still have no date after all cleaning attempts
    df = df.dropna(subset=['Arrival Date'])
    
    # Sort by date so charts look correct
    df = df.sort_values('Arrival Date')
    
    df['Month'] = df['Arrival Date'].dt.strftime('%B %Y') 
    df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
    
    return df

try:
    # --- LOAD DATA ---
    raw_data = load_and_clean_data(SHEET_URL)
    
    # --- SIDEBAR & DEBUGGER ---
    st.sidebar.header("Control Panel")
    
    # Debugging Info
    with st.sidebar.expander("🛠 Data Health Check"):
        st.write(f"Total Rows Found: {len(raw_data)}")
        st.write(f"Unique Suppliers: {raw_data['Supplier Name'].nunique()}")
        st.write("Date Range detected:")
        st.info(f"{raw_data['Arrival Date'].min().date()} to {raw_data['Arrival Date'].max().date()}")

    all_suppliers = sorted(raw_data['Supplier Name'].unique())
    selected_suppliers = st.sidebar.multiselect("Filter by Supplier", all_suppliers, default=all_suppliers)
    
    filtered_df = raw_data[raw_data['Supplier Name'].isin(selected_suppliers)]

    # --- TOP KPI METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales", f"₹{filtered_df['Sale Amt'].sum():,.2f}")
    m2.metric("Total Patty Value", f"₹{filtered_df['Patty Amt'].sum():,.2f}")
    m3.metric("Total Payments", f"₹{filtered_df['Payment amt'].sum():,.2f}")
    balance = filtered_df['Patty Amt'].sum() - filtered_df['Payment amt'].sum()
    m4.metric("Pending Balance", f"₹{balance:,.2f}", delta_color="inverse")

    st.markdown("---")

    # --- MONTHLY SALES REPORT ---
    st.subheader("📅 Monthly Sales Performance")
    
    # Group and aggregate
    monthly_summary = filtered_df.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
    monthly_summary = monthly_summary.sort_values('Month_Sort')

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            monthly_summary[['Month', 'Sale Amt']].style.format({'Sale Amt': '₹{:,.2f}'}),
            hide_index=True, use_container_width=True
        )

    with c2:
        # Use Plotly for an interactive trend line
        fig = px.line(monthly_summary, x='Month', y='Sale Amt', markers=True, 
                     title="Revenue Trend (Based on Arrival Date)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- BREAKDOWN ANALYSIS ---
    st.subheader("📊 Supplier & Expense Breakdown")
    b1, b2 = st.columns(2)
    with b1:
        sup_chart = filtered_df.groupby('Supplier Name')['Sale Amt'].sum().sort_values(ascending=False).head(15)
        st.bar_chart(sup_chart)
    with b2:
        exp_sum = filtered_df[['Cleaning coolie', 'Un Coolie', 'Wei Coolie']].sum()
        st.bar_chart(exp_sum)

    # --- DETAILED DATA VIEW ---
    with st.expander("📄 Click to View All Processed Records"):
        # Format date for display
        final_display = filtered_df.copy()
        final_display['Arrival Date'] = final_display['Arrival Date'].dt.strftime('%d/%m/%Y')
        st.dataframe(final_display.drop(columns=['Month_Sort']), use_container_width=True)
        
        csv_data = final_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report (CSV)", data=csv_data, file_name="cleaned_business_report.csv")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if your Arrival Date column uses the format DD/MM/YYYY.")