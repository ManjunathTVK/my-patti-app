import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Business Report Dashboard")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    h1 { font-size: 2.2rem !important; }
    button[data-baseweb="tab"] { margin-right: 2rem !important; }
    button[data-baseweb="tab"] > div { font-size: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Jothi Traders : Patti Report")

# --- CONNECTION ---
SHEET_ID = "1I-YGm6Lv4BGDOVUzfoNJQcRx2ro_FmxPuYByMgJ0AyI"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"

SHEET_ID_2 = "1YIIWKvAKEsleEDeucvcCa30VdKzKdwvftNg59Bn6rDc"
SHEET_URL_2 = f"https://docs.google.com/spreadsheets/d/{SHEET_ID_2}/gviz/tq?tqx=out:csv"

@st.cache_data(ttl=600)
def load_and_clean_data(url):
    df = pd.read_csv(url)
    df['Sheet_Row_Number'] = df.index + 2
    last_row_fetched = int(df['Sheet_Row_Number'].max())
    df.columns = df.columns.str.strip()
    
    def get_fuzzy_col(df_cols, target):
        map_cols = {c.lower().replace(' ', '').replace('(', '').replace(')', '').replace('_', ''): c for c in df_cols}
        normalized_target = target.lower().replace(' ', '').replace('(', '').replace(')', '').replace('_', '')
        return map_cols.get(normalized_target)

    # 1. FLEXIBLE DATE CLEANING (Sheet 1)
    date_col = get_fuzzy_col(df.columns, 'arrivaldate')
    if date_col:
        df.rename(columns={date_col: 'Arrival Date'}, inplace=True)
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/')
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        # Fix: Priority to day-first, but handles MM/DD/YYYY for 13th+ onwards
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], dayfirst=True, errors='coerce')

    col_mapping = {}
    supp_col = get_fuzzy_col(df.columns, 'suppliername')
    if supp_col:
        df[supp_col] = df[supp_col].astype(str).str.strip().replace(['nan', 'None', '', ' '], np.nan).ffill()
        col_mapping[supp_col] = 'Supplier Name'

    targets_numeric = {
        'saleamt': 'Sale Amt', 'pattyamt': 'Patty Amt', 'paymentamt': 'Payment amt',
        'cleaningcoolie': 'Cleaning coolie', 'uncoolie': 'Un Coolie', 'weicoolie': 'Wei Coolie',
        'lorryhirebank': 'Lorry Hire', 'lorryhire': 'Lorry Hire', 'lorryhirecash': 'Lorry Hire ( Cash)'
    }
    
    for key, std_name in targets_numeric.items():
        actual_col = get_fuzzy_col(df.columns, key)
        if actual_col:
            df[actual_col] = df[actual_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0)
            col_mapping[actual_col] = std_name
            
    df.rename(columns=col_mapping, inplace=True)
    
    if 'Arrival Date' in df.columns:
        df['Month'] = df['Arrival Date'].dt.strftime('%b-%Y') 
        df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
        
    return df, last_row_fetched

@st.cache_data(ttl=600)
def load_comparison_data(url):
    df = pd.read_csv(url)
    df['Sheet_Row_Number'] = df.index + 2
    last_row_fetched = int(df['Sheet_Row_Number'].max()) if not df.empty else 0
    df.columns = df.columns.str.strip()
    
    # 2. FLEXIBLE DATE CLEANING (Sheet 2 - Fixing Feb 13th issue)
    latest_date = None
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str).str.strip()
        df['Date'] = df['Date'].str.replace('-', '/').str.replace('.', '/')
        df['Date'] = df['Date'].replace(['nan', 'None', '', ' '], np.nan)
        # Using dayfirst=True to correctly parse the 02/13/2026 data
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        latest_date = df['Date'].max()
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.strftime('%b-%Y') 
        df['Month_Sort'] = df['Date'].dt.to_period('M')
        
    total_amt = 0
    if 'Sale Amount' in df.columns:
        df['Sale Amount'] = df['Sale Amount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['Sale Amount'] = pd.to_numeric(df['Sale Amount'], errors='coerce').fillna(0)
        total_amt = df['Sale Amount'].sum()
        
    return df, last_row_fetched, latest_date, total_amt

try:
    # --- LOAD DATA ---
    raw_data, last_sheet_row = load_and_clean_data(SHEET_URL)
    raw_data_2, last_row_2, last_date_2, total_amt_2 = load_comparison_data(SHEET_URL_2)
    
    # Filtering for valid dates in Sheet 1
    if 'Arrival Date' in raw_data.columns:
        valid_data = raw_data.dropna(subset=['Arrival Date']).copy()
    else:
        valid_data = raw_data.copy()
    
    # --- SIDEBAR HEALTH CHECK & ALERTS ---
    st.sidebar.header("Control Panel")
    with st.sidebar.expander("🛠 Data Health Check", expanded=True):
        st.write("**Sheet 1 (Patti)**")
        st.write(f"Last Row: {last_sheet_row}")
        st.write(f"Valid Rows: {len(valid_data)}")
        if len(raw_data) > len(valid_data):
            st.error(f"⚠️ {len(raw_data) - len(valid_data)} rows hidden due to date errors.")
        
        st.markdown("---")
        st.write("**Sheet 2 (Bills)**")
        st.write(f"Last Row: {last_row_2}")
        if last_date_2:
            st.write(f"Latest Date: {last_date_2.strftime('%d/%m/%Y')}")
        st.write(f"Total Sale: ₹{total_amt_2:,.2f}")
        
        if abs(last_sheet_row - last_row_2) > 5:
            st.warning(f"🚨 Row count mismatch! Sheet 1 ({last_sheet_row}) vs Sheet 2 ({last_row_2})")

    # --- MAIN UI TABS ---
    tab_main, tab_comp = st.tabs(["📊 Main Report", "📈 Sales Comparison"])

    with tab_main:
        # Supplier filtering logic
        if 'Supplier Name' in valid_data.columns:
            all_suppliers = sorted(valid_data['Supplier Name'].unique())
            def select_all(): st.session_state.s_select = all_suppliers
            def clear_all(): st.session_state.s_select = []
            
            c1, c2 = st.sidebar.columns(2)
            c1.button("Select All", on_click=select_all)
            c2.button("Clear All", on_click=clear_all)
            
            selected = st.sidebar.multiselect("Choose Suppliers", options=all_suppliers, default=all_suppliers, key="s_select")
            filtered_df = valid_data[valid_data['Supplier Name'].isin(selected)]
        else:
            filtered_df = valid_data

        st.subheader("Key Performance Indicators")
        # Metrics Display (Standard Aggregates)
        ts = filtered_df['Sale Amt'].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sales", f"₹{ts:,.0f}")
        m2.metric("Lorry Hire", f"₹{filtered_df['Lorry Hire'].sum():,.0f}")
        m3.metric("Lorry Hire (Cash)", f"₹{filtered_df['Lorry Hire ( Cash)'].sum():,.0f}")
        
        st.markdown("---")
        st.subheader("Monthly Arrivals")
        monthly = filtered_df.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index().sort_values('Month_Sort')
        fig = px.bar(monthly, x='Month', y='Sale Amt', title="Sales Trend")
        st.plotly_chart(fig, use_container_width=True)

    with tab_comp:
        st.subheader("🔄 Monthly Total Sales Comparison")
        s1_m = raw_data.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index().rename(columns={'Sale Amt': 'As per Patti'})
        s2_m = raw_data_2.groupby(['Month_Sort', 'Month'])['Sale Amount'].sum().reset_index().rename(columns={'Sale Amount': 'As per Bills'})
        
        comp_df = pd.merge(s1_m, s2_m, on=['Month_Sort', 'Month'], how='outer').fillna(0).sort_values('Month_Sort')
        st.dataframe(comp_df[['Month', 'As per Patti', 'As per Bills']].style.format({'As per Patti': '₹{:,.0f}', 'As per Bills': '₹{:,.0f}'}), use_container_width=True)
        
        fig_comp = px.line(comp_df, x='Month', y=['As per Patti', 'As per Bills'], markers=True)
        st.plotly_chart(fig_comp, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
