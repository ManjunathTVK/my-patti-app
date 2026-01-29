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
SHEET_ID_2 = "1YIIWKvAKEsleEDeucvcCa30VdKzKdwvftNg59Bn6rDc" 
SHEET_URL_2 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_2}/export?format=csv"

@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url)
    df['Sheet_Row_Number'] = df.index + 2
    last_row_fetched = int(df['Sheet_Row_Number'].max())
    df.columns = df.columns.str.strip()

    if 'Arrival Date' in df.columns:
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/', regex=False)
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], format='%d/%m/%Y', errors='coerce')

    text_cols = ['Supplier Name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace(['nan', 'None', '', ' '], np.nan).ffill()

    numeric_cols = ['Sale Amt', 'Patty Amt', 'Payment amt', 'Cleaning coolie', 'Un Coolie', 'Wei Coolie']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.dropna(subset=['Arrival Date'])
    # Using 'Patti No' if it exists, otherwise skip sort
    if 'Patti No' in df.columns:
        df = df.sort_values('Patti No')
    
    df['Month'] = df['Arrival Date'].dt.strftime('%B %Y') 
    df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
    return df, last_row_fetched

@st.cache_data
def load_comparison_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.strftime('%B %Y') 
        df['Month_Sort'] = df['Date'].dt.to_period('M')
        
    col_name = 'Sale Amount'
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df

try:
    raw_data, last_sheet_row = load_and_clean_data(SHEET_URL)
    
    st.sidebar.header("Control Panel")
    with st.sidebar.expander("🛠 Data Health Check", expanded=True):
        st.write(f"**Last Row in Google Sheet:** {last_sheet_row}")
        st.write(f"Valid Rows Processed: {len(raw_data)}")
        st.write(f"Unique Suppliers: {raw_data['Supplier Name'].nunique()}")
        if not raw_data.empty:
            st.info(f"Date Range: {raw_data['Arrival Date'].min().date()} to {raw_data['Arrival Date'].max().date()}")

    all_suppliers = sorted(raw_data['Supplier Name'].unique())
    select_all = st.sidebar.checkbox("Select All Suppliers", value=True)
    if select_all:
        selected_suppliers = all_suppliers
    else:
        supplier = st.sidebar.selectbox("Filter by Supplier", all_suppliers)
        selected_suppliers = [supplier]
    
    filtered_df = raw_data[raw_data['Supplier Name'].isin(selected_suppliers)]

    tab_main, tab_comp = st.tabs(["📊 Main Report", "📈 Sales Comparison"])

    with tab_main:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sales", f"₹{filtered_df['Sale Amt'].sum():,.2f}")
        m2.metric("Total Patty Value", f"₹{filtered_df['Patty Amt'].sum():,.2f}")
        m3.metric("Total Payments", f"₹{filtered_df['Payment amt'].sum():,.2f}")
        balance = filtered_df['Patty Amt'].sum() - filtered_df['Payment amt'].sum()
        m4.metric("Pending Balance", f"₹{balance:,.2f}")

        st.subheader("📅 Monthly Sales Performance")
        monthly_summary = filtered_df.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        monthly_summary = monthly_summary.sort_values('Month_Sort')

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(monthly_summary[['Month', 'Sale Amt']].style.format({'Sale Amt': '₹{:,.2f}'}), hide_index=True)
        with c2:
            fig = px.bar(monthly_summary, x='Month', y='Sale Amt', title="Arrival Trend")
            st.plotly_chart(fig, use_container_width=True)

    with tab_comp:
        st.subheader("🔄 Monthly Total Sales Comparison")
        raw_data_2 = load_comparison_data(SHEET_URL_2)
        
        s1_monthly = raw_data.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        s1_monthly.rename(columns={'Sale Amt': 'Patti Sales'}, inplace=True)
        
        s2_monthly = raw_data_2.groupby(['Month_Sort', 'Month'])['Sale Amount'].sum().reset_index()
        s2_monthly.rename(columns={'Sale Amount': 'Bill Sales'}, inplace=True)
        
        comparison_df = pd.merge(s1_monthly, s2_monthly, on=['Month_Sort', 'Month'], how='outer').fillna(0)
        comparison_df = comparison_df.sort_values('Month_Sort')
        
        # FIXED COLUMN NAMES HERE
        total_row = pd.DataFrame([{
            'Month': 'TOTAL', 
            'Month_Sort': '', 
            'Patti Sales': comparison_df['Patti Sales'].sum(), 
            'Bill Sales': comparison_df['Bill Sales'].sum()
        }])
        
        comparison_table = pd.concat([comparison_df, total_row], ignore_index=True)
        
        col_A, col_B = st.columns([1, 2])
        with col_A:
            st.dataframe(
                comparison_table[['Month', 'Patti Sales', 'Bill Sales']].style.format(
                    {'Patti Sales': '₹{:,.2f}', 'Bill Sales': '₹{:,.2f}'}
                ), hide_index=True
            )
            
        with col_B:
            comp_plot = comparison_df.melt(id_vars=['Month'], value_vars=['Patti Sales', 'Bill Sales'], var_name='Source', value_name='Sales')
            fig_comp = px.line(comp_plot, x='Month', y='Sales', color='Source', markers=True)
            st.plotly_chart(fig_comp, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
