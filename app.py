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
    /* Tabs Styling */
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
    
    # Helper to find column fuzzily
    def get_fuzzy_col(df_cols, target):
        # target: simple string e.g. "lorryhirecash"
        # returns correctly capitalized column name from df or None
        map_cols = {c.lower().replace(' ', '').replace('(', '').replace(')', '').replace('_', ''): c for c in df_cols}
        normalized_target = target.lower().replace(' ', '').replace('(', '').replace(')', '').replace('_', '')
        return map_cols.get(normalized_target)

    # 2. INTENSE DATE CLEANING
    date_col = get_fuzzy_col(df.columns, 'arrivaldate')
    if date_col:
        # Standardize name for easier access later
        df.rename(columns={date_col: 'Arrival Date'}, inplace=True)
        
        df['Arrival Date'] = df['Arrival Date'].astype(str).str.strip()
        df['Arrival Date'] = df['Arrival Date'].str.replace('-', '/').str.replace('.', '/')
        df['Arrival Date'] = df['Arrival Date'].replace(['nan', 'None', '', ' '], np.nan)
        # Force parsing DD/MM/YYYY
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'], format='%d/%m/%Y', errors='coerce')

    # 3. FORWARD FILL TEXT
    # Map desired names to actual columns
    col_mapping = {}
    
    supp_col = get_fuzzy_col(df.columns, 'suppliername')
    if supp_col:
        df[supp_col] = df[supp_col].astype(str).str.strip().replace(['nan', 'None', '', ' '], np.nan).ffill()
        col_mapping[supp_col] = 'Supplier Name'

    # 4. NUMERIC CLEANING
    # Targets: Sale Amt, Patty Amt, Payment Amt, Cleaning, Un, Wei, Lorry Hire, Lorry Hire Cash
    targets_numeric = {
        'saleamt': 'Sale Amt',
        'pattyamt': 'Patty Amt',
        'paymentamt': 'Payment amt',
        'cleaningcoolie': 'Cleaning coolie',
        'uncoolie': 'Un Coolie',
        'weicoolie': 'Wei Coolie',
        'lorryhirebank': 'Lorry Hire', # User clarified this is "Lorry Hire Bank" in sheet
        'lorryhire': 'Lorry Hire',     # Fallback if both exist or user reverts
        'lorryhirecash': 'Lorry Hire ( Cash)'
    }
    
    for key, std_name in targets_numeric.items():
        actual_col = get_fuzzy_col(df.columns, key)
        if actual_col:
            # Clean numeric
            df[actual_col] = df[actual_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0)
            col_mapping[actual_col] = std_name
            
    # Rename all found columns to standard names
    df.rename(columns=col_mapping, inplace=True)

    # 5. GENERATE MONTHLY FEATURES
    # We keep a copy of the 'raw' count before dropping NaNs for the health check
    # MOVED: Filtering for valid dates happens in the main app logic now, to keep cancelled pattis for audit.
    # df = df.dropna(subset=['Arrival Date']) 
    
    # df = df.sort_values('Arrival Date')
    patti_col = get_fuzzy_col(df.columns, 'pattino')
    if patti_col:
        df.rename(columns={patti_col: 'Patti No'}, inplace=True)
        df = df.sort_values('Patti No')
    
    if 'Arrival Date' in df.columns:
        df['Month'] = df['Arrival Date'].dt.strftime('%b-%Y') 
        df['Month_Sort'] = df['Arrival Date'].dt.to_period('M')
    else:
        # Fallback if arrival date missing completely from sheet
        df['Month'] = 'Unknown'
        
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
        df['Month'] = df['Date'].dt.strftime('%b-%Y') 
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
    
    # --- PREPARE VALID DATA FOR REPORTING (Exclude Cancelled/No Date) ---
    if 'Arrival Date' in raw_data.columns:
        valid_data = raw_data.dropna(subset=['Arrival Date']).copy()
    else:
        valid_data = raw_data.copy()
    
    # --- SIDEBAR & DEBUGGER ---
    st.sidebar.header("Control Panel")
    
    with st.sidebar.expander("🛠 Data Health Check", expanded=True):
        st.write(f"**Last Row in Google Sheet:** {last_sheet_row}")
        st.write(f"Total Rows Fetched: {len(raw_data)}")
        st.write(f"Valid Sales Rows: {len(valid_data)}")
        if 'Supplier Name' in valid_data.columns:
            st.write(f"Unique Suppliers: {valid_data['Supplier Name'].nunique()}")
        
        st.markdown("---")
        st.caption("Detected Columns (After Standardization):")
        st.code(list(raw_data.columns))
        
        if not valid_data.empty and 'Arrival Date' in valid_data.columns:
            st.write("Date Range detected:")
            st.info(f"{valid_data['Arrival Date'].min().date()} to {valid_data['Arrival Date'].max().date()}")

    if 'Supplier Name' in valid_data.columns:
        all_suppliers = sorted(valid_data['Supplier Name'].unique())
        
        # Callbacks for Select/Clear All
        def select_all():
            st.session_state.supplier_selection = all_suppliers
        
        def clear_all():
            st.session_state.supplier_selection = []
            
        st.sidebar.subheader("Filter Suppliers")
        c1, c2 = st.sidebar.columns(2)
        c1.button("Select All", on_click=select_all, use_container_width=True)
        c2.button("Clear All", on_click=clear_all, use_container_width=True)
        
        selected_suppliers = st.sidebar.multiselect(
            "Choose Suppliers", 
            options=all_suppliers, 
            default=all_suppliers, 
            key="supplier_selection"
        )
        
        # Filter based on VALID data for charts/tables
        if not selected_suppliers:
            st.warning("No suppliers selected!")
            filtered_df = valid_data[valid_data['Supplier Name'].isin([])] # Empty
        else:
            filtered_df = valid_data[valid_data['Supplier Name'].isin(selected_suppliers)]
    else:
        filtered_df = valid_data

    # --- TABS FOR ANALYSIS ---
    tab_main, tab_comp = st.tabs(["📊 Main Report", "📈 Sales Comparison"])

    with tab_main:
        # --- DYNAMIC HEADER ---
        if not selected_suppliers:
            header_text = "Key Metrics (No Selection)"
        elif len(selected_suppliers) == len(all_suppliers):
            header_text = "Key Metrics (All Suppliers)"
        elif len(selected_suppliers) <= 3:
            header_text = f"Key Metrics ({', '.join(selected_suppliers)})"
        else:
            header_text = f"Key Metrics ({len(selected_suppliers)} Suppliers Selected)"
            
        st.markdown(f"<h3 style='font-size: 18px; margin-bottom: 10px;'>{header_text}</h3>", unsafe_allow_html=True)

        # --- TOP KPI METRICS ---
        
        # Calculate Aggregates (COLUMNS ARE NOW STANDARDIZED)
        total_sales = filtered_df['Sale Amt'].sum() if 'Sale Amt' in filtered_df.columns else 0
        lorry_hire = filtered_df['Lorry Hire'].sum() if 'Lorry Hire' in filtered_df.columns else 0
        lorry_cash = filtered_df['Lorry Hire ( Cash)'].sum() if 'Lorry Hire ( Cash)' in filtered_df.columns else 0
        cleaning = filtered_df['Cleaning coolie'].sum() if 'Cleaning coolie' in filtered_df.columns else 0
        un_coolie = filtered_df['Un Coolie'].sum() if 'Un Coolie' in filtered_df.columns else 0
        wei_coolie = filtered_df['Wei Coolie'].sum() if 'Wei Coolie' in filtered_df.columns else 0
        patty_amt = filtered_df['Patty Amt'].sum() if 'Patty Amt' in filtered_df.columns else 0
        payment_amt = filtered_df['Payment amt'].sum() if 'Payment amt' in filtered_df.columns else 0
        
        # Formula: Balance = Sale Amt - (Lorry Hire + Lorry Hire(Cash) + Cleaning + Un + Wei + Payment)
        expenses = lorry_hire + lorry_cash + cleaning + un_coolie + wei_coolie
        balance = total_sales - (expenses + payment_amt)

        # Build 3x3 Grid
        
        # ROW 1
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Total Sales", f"₹{total_sales:,.0f}")
        r1c2.metric("Lorry Hire", f"₹{lorry_hire:,.0f}")
        r1c3.metric("Lorry Hire (Cash)", f"₹{lorry_cash:,.0f}")
        
        st.markdown("---") # Separator between rows for clarity or just rely on spacing? using spacing.
        
        # ROW 2
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Cleaning Coolie", f"₹{cleaning:,.0f}")
        r2c2.metric("Un Coolie", f"₹{un_coolie:,.0f}")
        r2c3.metric("Wei Coolie", f"₹{wei_coolie:,.0f}")
        
        # ROW 3
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric("Patty Amount", f"₹{patty_amt:,.0f}")
        r3c2.metric("Total Payments", f"₹{payment_amt:,.0f}")
        r3c3.metric("Balance", f"₹{balance:,.0f}", delta_color="inverse") # Red if negative implies deficit? Or just highlight.

        st.markdown("---")

        # --- SUPPLIER PERFORMANCE TABLE ---
        st.subheader("🏢 Supplierwise Patti Details")
        
        # Define Columns to Aggregate
        agg_cols = [
            'Sale Amt', 'Lorry Hire', 'Lorry Hire ( Cash)', 
            'Cleaning coolie', 'Un Coolie', 'Wei Coolie', 
            'Patty Amt', 'Payment amt'
        ]
        
        # Ensure these cols exist in filtered_df for aggregation
        safe_df = filtered_df.copy()
        for c in agg_cols:
            if c not in safe_df.columns:
                safe_df[c] = 0
                
        # Aggregate by Supplier
        supplier_stats = safe_df.groupby('Supplier Name')[agg_cols].sum().reset_index()
        
        # Calculate Balance per Supplier
        # Balance = Sale - (Expenses + Payments)
        supplier_expenses = (
            supplier_stats['Lorry Hire'] + 
            supplier_stats['Lorry Hire ( Cash)'] + 
            supplier_stats['Cleaning coolie'] + 
            supplier_stats['Un Coolie'] + 
            supplier_stats['Wei Coolie']
        )
        supplier_stats['Balance'] = supplier_stats['Sale Amt'] - (supplier_expenses + supplier_stats['Payment amt'])
        
        # Reorder Columns for better readability
        final_cols = ['Supplier Name'] + agg_cols + ['Balance']
        supplier_stats = supplier_stats[final_cols]
        
        # Sort by Sale Amt descending
        supplier_stats = supplier_stats.sort_values('Sale Amt', ascending=False)
        
        # Display Table
        st.dataframe(
            supplier_stats.style.format({c: '₹{:,.0f}' for c in agg_cols + ['Balance']})
            .background_gradient(subset=['Sale Amt'], cmap="Greens")
            .background_gradient(subset=['Balance'], cmap="Reds", vmin=None, vmax=0), # Highlight negative balance?
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        # Download Button
        csv_supp = supplier_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Supplier Summary (CSV)",
            data=csv_supp,
            file_name="supplier_performance_summary.csv",
            mime="text/csv"
        )
        
        st.markdown("---")

        # --- MONTHLY SALES REPORT ---
        st.subheader("📅 Monthly Arrival")
        
        monthly_summary = filtered_df.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        monthly_summary = monthly_summary.sort_values('Month_Sort')

        # Calculate Total for Table Display
        total_sales = monthly_summary['Sale Amt'].sum()
        total_row = pd.DataFrame([{'Month': 'TOTAL', 'Sale Amt': total_sales}])
        monthly_display = pd.concat([monthly_summary, total_row], ignore_index=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(
                monthly_display[['Month', 'Sale Amt']].style.format({'Sale Amt': '₹{:,.0f}'})
                .apply(lambda x: ['font-weight: bold' if x.name == len(monthly_display)-1 else '' for i in x], axis=1),
                hide_index=True, use_container_width=True
            )

        with c2:
            fig = px.bar(monthly_summary, x='Month', y='Sale Amt', 
                          title="Arrival Trend (Based on Arrival Date)")
            fig.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

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
        st.info("Comparing Total Sales as per Patti and Sales as per Bills")
        
        # Load Sheet 2 Data
        raw_data_2 = load_comparison_data(SHEET_URL_2)
        
        # 1. Aggregate SHEET 1 (Total, Unfiltered)
        s1_monthly = raw_data.groupby(['Month_Sort', 'Month'])['Sale Amt'].sum().reset_index()
        s1_monthly.rename(columns={'Sale Amt': 'As per Patti'}, inplace=True)
        
        # 2. Aggregate SHEET 2 (Using 'Sale Amount')
        s2_monthly = raw_data_2.groupby(['Month_Sort', 'Month'])['Sale Amount'].sum().reset_index()
        s2_monthly.rename(columns={'Sale Amount': 'As per Bills'}, inplace=True)
        
        # 3. Merge
        comparison_df = pd.merge(s1_monthly, s2_monthly, on=['Month_Sort', 'Month'], how='outer').fillna(0)
        comparison_df = comparison_df.sort_values('Month_Sort')
        
        # --- ADD TOTAL ROW ---
        total_s1 = comparison_df['As per Patti'].sum()
        total_s2 = comparison_df['As per Bills'].sum()
        
        # Create a DataFrame for the Total row
        total_row = pd.DataFrame([{
            'Month': 'TOTAL', 
            'Month_Sort': '',  # Empty or distinct to avoid sorting issues 
            'As per Patti': total_s1, 
            'As per Bills': total_s2
        }])
        
        # Concatenate: Original Data + Total Row
        comparison_table = pd.concat([comparison_df, total_row], ignore_index=True)
        
        # 4. Visualize
        col_A, col_B = st.columns([1, 2])
        
        with col_A:
            st.dataframe(
                comparison_table[['Month', 'As per Patti', 'As per Bills']].style.format(
                    {'As per Patti': '₹{:,.0f}', 'As per Bills': '₹{:,.0f}'}
                ).apply(lambda x: ['font-weight: bold' if x.name == len(comparison_table)-1 else '' for i in x], axis=1),
                hide_index=True, use_container_width=True
            )
            
        with col_B:
            # Reshape for Plotly
            comp_plot = comparison_df.melt(
                id_vars=['Month', 'Month_Sort'], 
                value_vars=['As per Patti', 'As per Bills'],
                var_name='Source', value_name='Sales'
            )
            fig_comp = px.line(comp_plot, x='Month', y='Sales', color='Source', markers=True, 
                               title="Sales Comparison Trend", color_discrete_sequence=px.colors.qualitative.Set1)
            fig_comp.update_traces(line=dict(width=4))
            fig_comp.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # --- CUMULATIVE COMPARISON ---
        st.subheader("📈 Cumulative Sales Growth")
        
        # Calculate Cumulative Sum
        comparison_df['Cumulative (As per Patti)'] = comparison_df['As per Patti'].cumsum()
        comparison_df['Cumulative (As per Bills)'] = comparison_df['As per Bills'].cumsum()
        comparison_df['Difference'] = comparison_df['Cumulative (As per Patti)'] - comparison_df['Cumulative (As per Bills)']
        
        st.caption("Cumulative Progress Table")
        st.dataframe(
            comparison_df[['Month', 'Cumulative (As per Patti)', 'Cumulative (As per Bills)', 'Difference']].style.format(
                {'Cumulative (As per Patti)': '₹{:,.0f}', 'Cumulative (As per Bills)': '₹{:,.0f}', 'Difference': '₹{:,.0f}'}
            ).apply(lambda x: ['color: red' if v < 0 else '' for v in x], subset=['Difference']),
            hide_index=True, use_container_width=True
        )

        # Reshape for Plotly
        cum_plot = comparison_df.melt(
            id_vars=['Month', 'Month_Sort'], 
            value_vars=['Cumulative (As per Patti)', 'Cumulative (As per Bills)'],
            var_name='Source', value_name='Cumulative Sales'
        )
        fig_cum = px.line(cum_plot, x='Month', y='Cumulative Sales', color='Source', markers=True, 
                          title="Cumulative Growth Trail", color_discrete_sequence=px.colors.qualitative.Set1)
        fig_cum.update_traces(line=dict(width=4))
        fig_cum.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_cum, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # --- AUDIT & HEALTH CHECK ---
        st.subheader("📋 Data Audit & Missing Pattis")
        
        # --- 1. PREPARE RAW DATA FOR AUDIT ---
        audit_raw = raw_data.copy()
        if 'Patti No' in audit_raw.columns:
            audit_raw['Patti No'] = pd.to_numeric(audit_raw['Patti No'], errors='coerce')
            audit_raw = audit_raw.dropna(subset=['Patti No']).sort_values('Patti No')
            audit_raw['Patti No'] = audit_raw['Patti No'].astype(int)

        # --- 2. GLOBAL METRICS ---
        
        # A. Last Patti No & Supplier
        if not audit_raw.empty:
            last_patti_idx = audit_raw['Patti No'].idxmax()
            last_patti_val = audit_raw.loc[last_patti_idx, 'Patti No']
            last_patti_supp = audit_raw.loc[last_patti_idx, 'Supplier Name']
            last_patti_display = f"{last_patti_val} ({last_patti_supp})"
            
            min_patti = audit_raw['Patti No'].min()
            max_patti = audit_raw['Patti No'].max()
        else:
            last_patti_display = "No Data"
            min_patti = 0
            max_patti = 0

        # B. Last Bill Date
        last_bill_date = raw_data_2['Date'].max()
        if pd.notnull(last_bill_date):
            last_bill_display = pd.to_datetime(last_bill_date).strftime('%d/%m/%Y')
        else:
            last_bill_display = "-"

        # --- DISPLAY METRICS ---
        c1, c2 = st.columns(2)
        c1.metric("Last Patti No Reported", last_patti_display)
        c2.metric("Date of Last Bill", last_bill_display)

        # --- 3. MISSING PATTI SEQUENCE (Logic: Missing = Not Present OR Sale Amt is 0) ---
        if not audit_raw.empty and (max_patti > min_patti):
            # A. Define "Valid" Pattis (Those with Sale Amt > 0)
            # Ensure 'Sale Amt' is numeric
            if 'Sale Amt' in audit_raw.columns:
                audit_raw['Sale Amt'] = pd.to_numeric(audit_raw['Sale Amt'], errors='coerce').fillna(0)
            
            valid_pattis = audit_raw[audit_raw['Sale Amt'] != 0]['Patti No'].unique()
            valid_set = set(valid_pattis)
            
            # B. Define Full Range
            full_range = set(range(min_patti, max_patti + 1))
            
            # C. Identify "Ineffective/Missing" (Gaps + Zero Value)
            missing_ids = sorted(list(full_range - valid_set))
            
            if missing_ids:
                st.warning(f"Found {len(missing_ids)} missing or cancelled Patti Numbers (where Sale Amt is 0 or row is missing).")
                
                # D. Build Tables for Display
                missing_records = []
                
                # Create a lookup for Supplier Names from the raw data (to catch "Cancelled" rows)
                # If multiple rows share a Patti No, we take the one with a Value if possible, or failing that, the first one.
                # Here we just want the Supplier Name if the row exists.
                patti_supplier_map = audit_raw.dropna(subset=['Supplier Name']).set_index('Patti No')['Supplier Name'].to_dict()
                
                for p_no in missing_ids:
                    # Check if this ID exists in raw data (meaning it's a Zero Value / Cancelled Row)
                    if p_no in patti_supplier_map:
                        supp_name = patti_supplier_map[p_no]
                    else:
                        supp_name = "" # Truly missing gap
                    
                    missing_records.append({'Patti No': p_no, 'Supplier Name': supp_name})
                
                display_missing = pd.DataFrame(missing_records)
                
                # Show preview (Use a slightly larger height for visibility)
                st.dataframe(display_missing, hide_index=True, use_container_width=True, height=300)
                
                # Download Button
                csv = display_missing.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Missing/Cancelled Patti List (CSV)",
                    data=csv,
                    file_name="missing_cancelled_patti_nos.csv",
                    mime="text/csv"
                )
            else:
                st.success("No missing or zero-value Patti numbers detected in the range.")
        else:
            st.info("Not enough data to calculate missing sequences.")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if your Arrival Date column uses the format DD/MM/YYYY.")
