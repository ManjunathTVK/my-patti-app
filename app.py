import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Business Report Dashboard")

st.title("📊 Google Sheets Data Processor & Reporter")

# --- CONNECTION ---
SHEET_ID = "1I-YGm6Lv4BGDOVUzfoNJQcRx2ro_FmxPuYByMgJ0AyI"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# --- SECOND SHEET (COMPARISON) ---
SHEET_ID_2 = "1YIIWKvAKEsleEDeucvcCa30VdKzKdwvftNg59Bn6rDc"  # Replace with actual ID
SHEET_URL_2 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_2}/export?format=csv"

@st.cache_data
def load_and_clean_data(url):
    # 1. Load raw CSV
    df = pd.read_csv(url)
    
    # --- ROW TRACKING ---
    # We add 2: +1 for 0-based indexing and +1 for the Google Sheets header row
    df['Sheet_Row_Number'] = df.index + 2
    last_row_fetched = int(df['Sheet_Row_Number'].max())
    
    df.columns = df.columns.str.strip()

    # 2. INTENSE DATE CLEANING
    if 'Arrival Date' in df.columns:
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/')
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        
        # Force parsing DD/MM/YYYY
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], format='%d/%m/%Y', errors='coerce')

    # 3. FORWARD FILL TEXT
    text_cols = ['Supplier Name']
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
    # We keep a copy of the 'raw' count before dropping NaNs for the health check
    df = df.dropna(subset=['Arrival Date'])
    # df = df.sort_values('Arrival Date')
    df = df.sort_values('Patti No')
    df['Month'] = df['Arrival Date'].dt.strftime('%B %Y') 
    df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
    
    return df, last_row_fetched

@st.cache_data
def load_comparison_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    
    # 1. Clean Date Column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Generate Sorting Columns
        df['Month'] = df['Date'].dt.strftime('%B %Y') 
        df['Month_Sort'] = df['Date'].dt.to_period('M')
        
    # 2. Clean Sale Amount
    col_name = 'Sale Amount'
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        
    return df

try:
    # --- LOAD DATA ---
    raw_data, last_sheet_row = load_and_clean_data(SHEET_URL)
    
    # --- SIDEBAR & DEBUGGER ---
    st.sidebar.header("Control Panel")
    
    with st.sidebar.expander("🛠 Data Health Check", expanded=True):
        st.write(f"**Last Row in Google Sheet:** {last_sheet_row}")
        st.write(f"Valid Rows Processed: {len(raw_data)}")
        st.write(f"Unique Suppliers: {raw_data['Supplier Name'].nunique()}")
        
        if not raw_data.empty:
            st.write("Date Range detected:")
            st.info(f"{raw_data['Arrival Date'].min().date()} to {raw_data['Arrival Date'].max().date()}")

    all_suppliers = sorted(raw_data['Supplier Name'].unique())
    select_all = st.sidebar.checkbox("Select All Suppliers", value=True)
    if select_all:
        selected_suppliers = all_suppliers
    else:
        supplier = st.sidebar.selectbox("Filter by Supplier", all_suppliers)
        selected_suppliers = [supplier]
    
    filtered_df = raw_data[raw_data['Supplier Name'].isin(selected_suppliers)]

    # --- TABS FOR ANALYSIS ---
    tab_main, tab_comp = st.tabs(["📊 Main Report", "📈 Sales Comparison"])

    with tab_main:
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
        
        monthly_summary = filtered_df.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        monthly_summary = monthly_summary.sort_values('Month_Sort')

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(
                monthly_summary[['Month', 'Sale Amt']].style.format({'Sale Amt': '₹{:,.2f}'}),
                hide_index=True, use_container_width=True
            )

        with c2:
            fig = px.bar(monthly_summary, x='Month', y='Sale Amt', 
                          title="Revenue Trend (Based on Arrival Date)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- DETAILED DATA VIEW ---
        with st.expander("📄 Click to View All Processed Records"):
            final_display = filtered_df.copy()
            final_display['Arrival Date'] = final_display['Arrival Date'].dt.strftime('%d/%m/%Y')
            st.dataframe(final_display.drop(columns=['Month_Sort']), use_container_width=True)
            
            csv_data = final_display.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report (CSV)", data=csv_data, file_name="cleaned_business_report.csv")

    with tab_comp:
        st.subheader("🔄 Monthly Total Sales Comparison")
        st.info("Comparing Total Sales (All Suppliers) from Sheet 1 vs Sheet 2")
        
        # Load Sheet 2 Data
        raw_data_2 = load_comparison_data(SHEET_URL_2)
        
        # 1. Aggregate SHEET 1 (Total, Unfiltered)
        s1_monthly = raw_data.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        s1_monthly.rename(columns={'Sale Amt': 'Main Sheet Sales'}, inplace=True)
        
        # 2. Aggregate SHEET 2 (Using 'Sale Amount')
        s2_monthly = raw_data_2.groupby(['Month_Sort', 'Month'])['Sale Amount'].sum().reset_index()
        s2_monthly.rename(columns={'Sale Amount': 'Comparison Sheet Sales'}, inplace=True)
        
        # 3. Merge
        comparison_df = pd.merge(s1_monthly, s2_monthly, on=['Month_Sort', 'Month'], how='outer').fillna(0)
        comparison_df = comparison_df.sort_values('Month_Sort')
        
        # --- ADD TOTAL ROW ---
        total_s1 = comparison_df['Main Sheet Sales'].sum()
        total_s2 = comparison_df['Comparison Sheet Sales'].sum()
        
        # Create a DataFrame for the Total row
        total_row = pd.DataFrame([{
            'Month': 'TOTAL', 
            'Month_Sort': '',  # Empty or distinct to avoid sorting issues 
            'Main Sheet Sales': total_s1, 
            'Comparison Sheet Sales': total_s2
        }])
        
        # Concatenate: Original Data + Total Row
        comparison_table = pd.concat([comparison_df, total_row], ignore_index=True)
        
        # 4. Visualize
        col_A, col_B = st.columns([1, 2])
        
        with col_A:
            st.dataframe(
                comparison_table[['Month', 'Main Sheet Sales', 'Comparison Sheet Sales']].style.format(
                    {'Main Sheet Sales': '₹{:,.2f}', 'Comparison Sheet Sales': '₹{:,.2f}'}
                ).apply(lambda x: ['font-weight: bold' if x.name == len(comparison_table)-1 else '' for i in x], axis=1),
                hide_index=True, use_container_width=True
            )
            
        with col_B:
            # Reshape for Plotly
            comp_plot = comparison_df.melt(
                id_vars=['Month', 'Month_Sort'], 
                value_vars=['Main Sheet Sales', 'Comparison Sheet Sales'],
                var_name='Source', value_name='Sales'
            )
            fig_comp = px.line(comp_plot, x='Month', y='Sales', color='Source', markers=True, title="Sales Comparison Trend")
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # --- CUMULATIVE COMPARISON ---
        st.subheader("📈 Cumulative Sales Growth")
        
        # Calculate Cumulative Sum
        comparison_df['Cumulative (Sheet 1)'] = comparison_df['Main Sheet Sales'].cumsum()
        comparison_df['Cumulative (Sheet 2)'] = comparison_df['Comparison Sheet Sales'].cumsum()
        
        col_C, col_D = st.columns([1, 2])
        
        with col_C:
             st.caption("Cumulative Progress Table")
             st.dataframe(
                comparison_df[['Month', 'Cumulative (Sheet 1)', 'Cumulative (Sheet 2)']].style.format(
                    {'Cumulative (Sheet 1)': '₹{:,.2f}', 'Cumulative (Sheet 2)': '₹{:,.2f}'}
                ),
                hide_index=True, use_container_width=True
            )

        with col_D:
            # Reshape for Plotly
            cum_plot = comparison_df.melt(
                id_vars=['Month', 'Month_Sort'], 
                value_vars=['Cumulative (Sheet 1)', 'Cumulative (Sheet 2)'],
                var_name='Source', value_name='Cumulative Sales'
            )
            fig_cum = px.line(cum_plot, x='Month', y='Cumulative Sales', color='Source', markers=True, title="Cumulative Growth Trail")
            st.plotly_chart(fig_cum, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if your Arrival Date column uses the format DD/MM/YYYY.")


