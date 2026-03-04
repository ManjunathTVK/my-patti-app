import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Business Report Dashboard")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    h1 {
        font-size: 2.2rem !important;
    }
    button[data-baseweb="tab"] {
        margin-right: 2rem !important;
    }
    button[data-baseweb="tab"] > div {
        font-size: 20px !important;
    }
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

    date_col = get_fuzzy_col(df.columns, 'arrivaldate')
    if date_col:
        df.rename(columns={date_col: 'Arrival Date'}, inplace=True)
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/')
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], format='%d/%m/%Y', errors='coerce')

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

    patti_col = get_fuzzy_col(df.columns, 'pattino')
    if patti_col:
        df.rename(columns={patti_col: 'Patti No'}, inplace=True)
        df = df.sort_values('Patti No')
    
    if 'Arrival Date' in df.columns:
        df['Month'] = df['Arrival Date'].dt.strftime('%b-%Y') 
        df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
    else:
        df['Month'] = 'Unknown'
        
    return df, last_row_fetched

@st.cache_data(ttl=600)
def load_comparison_data(url):
    # 1. Load raw CSV
    df = pd.read_csv(url)
    
    # --- ROW TRACKING ---
    df['Sheet_Row_Number'] = df.index + 2
    last_row_fetched = int(df['Sheet_Row_Number'].max()) if not df.empty else 0
    
    df.columns = df.columns.str.strip()
    
    # 2. FLEXIBLE DATE CLEANING (Fixes the Feb 5th - 28th issue)
    latest_date = None
    if 'Date' in df.columns:
        # Convert to string and clean separators first
        df['Date'] = df['Date'].astype(str).str.strip()
        df['Date'] = df['Date'].str.replace('-', '/').str.replace('.', '/')
        df['Date'] = df['Date'].replace(['nan', 'None', '', ' '], np.nan)
        
        # dayfirst=True allows it to handle 04/02 (Feb 4) and 02/13 (Feb 13) in the same column
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Track latest date for the Health Check sidebar
        latest_date = df['Date'].max()
        
        # Drop only the rows where the date is truly unreadable
        df = df.dropna(subset=['Date'])
        
        # Generate Sorting Columns
        df['Month'] = df['Date'].dt.strftime('%b-%Y') 
        df['Month_Sort'] = df['Date'].dt.to_period('M')
        
    # 3. Clean Sale Amount
    total_amt = 0
    col_name = 'Sale Amount'
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        total_amt = df[col_name].sum()
        
    return df, last_row_fetched, latest_date, total_amt

try:
    # --- LOAD DATA ---
    raw_data, last_sheet_row = load_and_clean_data(SHEET_URL)
    raw_data_2, last_row_2, last_date_2, total_amt_2 = load_comparison_data(SHEET_URL_2)
    
    if 'Arrival Date' in raw_data.columns:
        valid_data = raw_data.dropna(subset=['Arrival Date']).copy()
    else:
        valid_data = raw_data.copy()
    
    # --- SIDEBAR HEALTH CHECK ---
    st.sidebar.header("Control Panel")
    with st.sidebar.expander("🛠 Data Health Check", expanded=True):
        st.write("**Sheet 1 (Patti)**")
        st.write(f"Last Row: {last_sheet_row}")
        st.write(f"Valid Rows: {len(valid_data)}")
        st.markdown("---")
        st.write("**Sheet 2 (Bills)**")
        st.write(f"Last Row: {last_row_2}")
        if last_date_2:
            st.write(f"Latest Date: {last_date_2.strftime('%d/%m/%Y')}")
        st.write(f"Total Sale: ₹{total_amt_2:,.2f}")

    if 'Supplier Name' in valid_data.columns:
        all_suppliers = sorted(valid_data['Supplier Name'].unique())
        def select_all(): st.session_state.supplier_selection = all_suppliers
        def clear_all(): st.session_state.supplier_selection = []
            
        st.sidebar.subheader("Filter Suppliers")
        c1, c2 = st.sidebar.columns(2)
        c1.button("Select All", on_click=select_all, use_container_width=True)
        c2.button("Clear All", on_click=clear_all, use_container_width=True)
        
        selected_suppliers = st.sidebar.multiselect(
            "Choose Suppliers", options=all_suppliers, default=all_suppliers, key="supplier_selection"
        )
        filtered_df = valid_data[valid_data['Supplier Name'].isin(selected_suppliers)] if selected_suppliers else valid_data.iloc[0:0]
    else:
        filtered_df = valid_data

    # --- TABS ---
    tab_main, tab_comp = st.tabs(["📊 Main Report", "📈 Sales Comparison"])

    with tab_main:
        st.subheader("Key Metrics")
        # KPI Calculations
        ts = filtered_df['Sale Amt'].sum() if 'Sale Amt' in filtered_df.columns else 0
        lh = filtered_df['Lorry Hire'].sum() if 'Lorry Hire' in filtered_df.columns else 0
        lc = filtered_df['Lorry Hire ( Cash)'].sum() if 'Lorry Hire ( Cash)' in filtered_df.columns else 0
        cl = filtered_df['Cleaning coolie'].sum() if 'Cleaning coolie' in filtered_df.columns else 0
        un = filtered_df['Un Coolie'].sum() if 'Un Coolie' in filtered_df.columns else 0
        we = filtered_df['Wei Coolie'].sum() if 'Wei Coolie' in filtered_df.columns else 0
        py = filtered_df['Payment amt'].sum() if 'Payment amt' in filtered_df.columns else 0
        bal = ts - (lh + lc + cl + un + we + py)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sales", f"₹{ts:,.0f}")
        m2.metric("Lorry Hire", f"₹{lh:,.0f}")
        m3.metric("Lorry Hire (Cash)", f"₹{lc:,.0f}")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Cleaning Coolie", f"₹{cl:,.0f}")
        m5.metric("Un Coolie", f"₹{un:,.0f}")
        m6.metric("Wei Coolie", f"₹{we:,.0f}")
        
        m7, m8, m9 = st.columns(3)
        m7.metric("Patty Amount", f"₹{filtered_df['Patty Amt'].sum():,.0f}")
        m8.metric("Total Payments", f"₹{py:,.0f}")
        m9.metric("Balance", f"₹{bal:,.0f}", delta_color="inverse")

        st.markdown("---")
        st.subheader("🏢 Supplierwise Patti Details")
        agg_cols = ['Sale Amt', 'Lorry Hire', 'Lorry Hire ( Cash)', 'Cleaning coolie', 'Un Coolie', 'Wei Coolie', 'Patty Amt', 'Payment amt']
        supplier_stats = filtered_df.groupby('Supplier Name')[agg_cols].sum().reset_index()
        st.dataframe(supplier_stats.style.format({c: '₹{:,.0f}' for c in agg_cols}), use_container_width=True)

    with tab_comp:
        st.subheader("🔄 Monthly Total Sales Comparison")
        s1_m = raw_data.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index().rename(columns={'Sale Amt': 'As per Patti'})
        s2_m = raw_data_2.groupby(['Month_Sort', 'Month'])['Sale Amount'].sum().reset_index().rename(columns={'Sale Amount': 'As per Bills'})
        
        comp_df = pd.merge(s1_m, s2_m, on=['Month_Sort', 'Month'], how='outer').fillna(0).sort_values('Month_Sort')
        
        st.dataframe(comp_df[['Month', 'As per Patti', 'As per Bills']].style.format({'As per Patti': '₹{:,.0f}', 'As per Bills': '₹{:,.0f}'}), use_container_width=True)
        
        fig = px.line(comp_df, x='Month', y=['As per Patti', 'As per Bills'], markers=True, title="Comparison Trend")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")

