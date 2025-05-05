import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image # Import Image
from shapely.geometry import MultiPoint # For Convex Hull
import numpy as np # For calculations
from io import BytesIO # To read uploaded file content
# Import DBSCAN for clustering
from sklearn.cluster import DBSCAN
# Import for Base64 encoding the image
import base64
import os # To check if image file exists

st.set_page_config(layout="wide")

st.title("Mobile Network Performance Analyzer")

# --- Sidebar ---
# Add vertical space at the top
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Center logo using columns
logo_sidebar_cols = st.sidebar.columns([1, 2, 1])
with logo_sidebar_cols[1]:
    try:
        logo = Image.open("FLYCOMM@2x-768x453.png")
        st.image(logo, use_column_width=True)
    except FileNotFoundError:
        st.warning("Logo image not found.")

# Add space below logo
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv", "txt"])

# List of reports
report_list = [
    "Data Overview",
    "Network Coverage Quality Map",
    "Network Performance Benchmarking",
    "Anomalies",
]

# Report Selection in Sidebar
if 'selected_report' not in st.session_state:
    st.session_state.selected_report = report_list[0]

st.sidebar.header("Reports")
st.session_state.selected_report = st.sidebar.radio(
    "Choose a report:",
    report_list,
    index=report_list.index(st.session_state.selected_report),
    key='report_radio'
)

# Initialize session state for filters
filter_defaults = {
    'date_start': pd.to_datetime('2024-01-01'),
    'date_end': pd.to_datetime('today').normalize(), # Normalize to remove time part
    'selected_operator': "All Operators",
    'selected_techs': ["All Techs"],
    'selected_modems': ["All Modems"],
    'show_flysense': False,
    'selected_single_modem': None
}
for key, default_value in filter_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Main Area ---
if uploaded_file is not None:
    # Read and process data only once using caching
    @st.cache_data
    def load_and_process_data(uploaded_file_content):
        try:
            df_loaded = pd.read_csv(BytesIO(uploaded_file_content))
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return pd.DataFrame() # Return empty DataFrame on read error

        # --- Standardize Columns ---
        if 'modem_id' in df_loaded.columns and 'download' in df_loaded.columns and 'upload' in df_loaded.columns:
            rename_map = {'download': 'downloadMbps', 'upload': 'uploadMbps'}
            df_loaded.rename(columns=rename_map, inplace=True)

        # --- Ensure Date/Time columns ---
        if 'timestamp' in df_loaded.columns and 'date' not in df_loaded.columns:
            try:
                datetime_col = pd.to_datetime(df_loaded['timestamp'], errors='coerce')
                if not datetime_col.isnull().all():
                    df_loaded['date'] = datetime_col.dt.strftime('%m/%d/%Y')
                    df_loaded['time'] = datetime_col.dt.strftime('%H:%M:%S')
            except Exception:
                pass # Ignore timestamp parsing errors silently within cache
        
        # Ensure essential columns exist (add N/A or None)
        essential_cols = ['tech', 'modem_id', 'lat', 'lng', 'date', 'Operator']
        for col in essential_cols:
            if col not in df_loaded.columns:
                df_loaded[col] = None # Use None for easier handling later

        # --- Merge PLMN Data (Simplified - requires PLMN file) ---
        try:
            plmn_df = pd.read_csv("israel_plmn-2607.csv")
            if 'mcc' in df_loaded.columns and 'mnc' in df_loaded.columns and \
               'MCC' in plmn_df.columns and 'MNC' in plmn_df.columns and 'Operator' in plmn_df.columns:
                
                # Ensure consistent types before merge
                for col in ['mcc', 'mnc']: df_loaded[col] = df_loaded[col].astype(str)
                for col in ['MCC', 'MNC']: plmn_df[col] = plmn_df[col].astype(str)
                
                # Perform merge, overwrite Operator only if original is None/NA
                df_merged = pd.merge(df_loaded, plmn_df[['MCC', 'MNC', 'Operator']].rename(columns={'Operator': 'PLMN_Operator'}),
                               left_on=['mcc', 'mnc'], right_on=['MCC', 'MNC'], how='left')
                
                # If 'Operator' column didn't exist or was all None, use PLMN names
                if 'Operator' not in df_loaded.columns or df_loaded['Operator'].isnull().all():
                     df_loaded['Operator'] = df_merged['PLMN_Operator']
                else: # Otherwise, fill only missing values in original Operator column
                     df_loaded['Operator'].fillna(df_merged['PLMN_Operator'], inplace=True)
                
                # Fallback for remaining NAs in Operator
                if 'mcc' in df_loaded.columns and 'mnc' in df_loaded.columns:
                     op_fallback = df_loaded['mcc'].astype(str) + '-' + df_loaded['mnc'].astype(str)
                     df_loaded['Operator'].fillna(op_fallback, inplace=True)
            
        except FileNotFoundError:
             pass # Ignore if PLMN file not found
        except Exception:
             pass # Ignore other PLMN errors
             
        # Final fallback for Operator if still missing
        if 'Operator' not in df_loaded.columns or df_loaded['Operator'].isnull().all():
             df_loaded['Operator'] = 'Unknown'
        else:
             df_loaded['Operator'].fillna('Unknown', inplace=True)

        return df_loaded

    df_raw = load_and_process_data(uploaded_file.getvalue())

    if df_raw.empty:
        st.error("Failed to load or process data.")
    else:
        # Add explicit try block here
        try:
            # --- Populate Filter Options from loaded data ---
            operator_options = ["All Operators"]
            tech_options = ["All Techs"]
            modem_options = ["All Modems"]
            min_date_data = st.session_state.date_start # Keep session state default
            max_date_data = st.session_state.date_end   # Keep session state default

            try:
                if 'Operator' in df_raw.columns:
                    unique_ops = df_raw['Operator'].dropna().astype(str).unique()
                    if len(unique_ops) > 0:
                        operator_options.extend(sorted(unique_ops))
                if 'tech' in df_raw.columns:
                    unique_techs = df_raw['tech'].dropna().astype(str).unique()
                    if len(unique_techs) > 0:
                        tech_options.extend(sorted(unique_techs))
                if 'modem_id' in df_raw.columns:
                    unique_modems = df_raw['modem_id'].dropna().astype(str).unique()
                    if len(unique_modems) > 0:
                        modem_options.extend(sorted(unique_modems))
                if 'date' in df_raw.columns:
                     parsed_dates = pd.to_datetime(df_raw['date'], errors='coerce', format='%m/%d/%Y')
                     if not parsed_dates.isnull().all():
                         min_date_data = parsed_dates.min()
                         max_date_data = parsed_dates.max()
            except Exception as e:
                st.warning(f"Could not fully populate filter options from data: {e}")

            # --- Render Top Filter Bar --- 
            filter_cols = st.columns([1, 1, 1, 1, 1, 1])
            with filter_cols[0]:
                # Use min/max dates from data if available and valid
                min_d = min_date_data if pd.notna(min_date_data) else None
                max_d = max_date_data if pd.notna(max_date_data) else None
                # Ensure value is within min/max bounds
                date_start_value = st.session_state.date_start
                if min_d and date_start_value < min_d: date_start_value = min_d
                if max_d and date_start_value > max_d: date_start_value = max_d
                st.session_state.date_start = st.date_input('Start date', value=date_start_value, min_value=min_d, max_value=max_d, key='date_input_start')
            with filter_cols[1]:
                date_end_value = st.session_state.date_end
                if min_d and date_end_value < min_d: date_end_value = min_d
                if max_d and date_end_value > max_d: date_end_value = max_d
                st.session_state.date_end = st.date_input('End date', value=date_end_value, min_value=min_d, max_value=max_d, key='date_input_end')
            with filter_cols[2]:
                sel_op = st.session_state.selected_operator
                current_op_index = 0
                if sel_op in operator_options:
                     current_op_index = operator_options.index(sel_op)
                st.session_state.selected_operator = st.selectbox("Operator:", options=operator_options, index=current_op_index, key='top_op_select')
            with filter_cols[3]:
                sel_techs = st.session_state.selected_techs
                if not isinstance(sel_techs, list) or not all(t in tech_options for t in sel_techs):
                     sel_techs = ["All Techs"]
                st.session_state.selected_techs = st.multiselect("Tech:", options=tech_options, default=sel_techs, key='tech_multi')
                if "All Techs" in st.session_state.selected_techs and len(st.session_state.selected_techs) > 1: st.session_state.selected_techs = ["All Techs"]
                if not st.session_state.selected_techs: st.session_state.selected_techs = ["All Techs"]
            with filter_cols[4]:
                show_flysense = st.session_state.show_flysense
                if show_flysense:
                    # REMOVED Single modem selection - always show all in Flysense view
                    st.write("Modem: All (Flysense)") # Placeholder to fill the column space
                    st.session_state.selected_single_modem = None # Ensure single modem selection is cleared
                else:
                    valid_modem_options_multi = [m for m in modem_options if pd.notna(m)]
                    sel_modems = st.session_state.selected_modems
                    if not isinstance(sel_modems, list) or not all(m in valid_modem_options_multi for m in sel_modems):
                        sel_modems = ["All Modems"]
                    st.session_state.selected_modems = st.multiselect("Modem ID:", options=valid_modem_options_multi, default=sel_modems, key='modem_select_multi')
                    if "All Modems" in st.session_state.selected_modems and len(st.session_state.selected_modems) > 1: st.session_state.selected_modems = ["All Modems"]
                    if not st.session_state.selected_modems: st.session_state.selected_modems = ["All Modems"]
            with filter_cols[5]:
                st.session_state.show_flysense = st.toggle("Show Flysense View", value=st.session_state.show_flysense, key='flysense_toggle')

            # Ensure correct datetime objects for filtering
            st.session_state.date_start = pd.to_datetime(st.session_state.date_start)
            st.session_state.date_end = pd.to_datetime(st.session_state.date_end)

            # --- Apply Global Filters ---
            df_filtered_global = df_raw.copy()
            try:
                # Date Filter
                if 'date' in df_filtered_global.columns:
                    try:
                        df_filtered_global['date_parsed'] = pd.to_datetime(df_filtered_global['date'], errors='coerce', format='%m/%d/%Y')
                        if not df_filtered_global['date_parsed'].isnull().all():
                            df_filtered_global = df_filtered_global.dropna(subset=['date_parsed'])
                            df_filtered_global = df_filtered_global[
                                (df_filtered_global['date_parsed'] >= st.session_state.date_start) & 
                                (df_filtered_global['date_parsed'] <= st.session_state.date_end)
                            ]
                        else:
                            st.warning("Could not parse 'date' column for filtering.")
                    except ValueError:
                         # Try fallback parsing
                         try:
                              df_filtered_global['date_parsed'] = pd.to_datetime(df_filtered_global['date'], errors='coerce')
                              if not df_filtered_global['date_parsed'].isnull().all():
                                  df_filtered_global = df_filtered_global.dropna(subset=['date_parsed'])
                                  df_filtered_global = df_filtered_global[
                                      (df_filtered_global['date_parsed'] >= st.session_state.date_start) & 
                                      (df_filtered_global['date_parsed'] <= st.session_state.date_end)
                                  ]
                              else:
                                  st.warning("Could not parse 'date' column for filtering (fallback failed).")
                         except Exception as e_parse:
                              st.warning(f"Error parsing date column: {e_parse}")
                
                # Operator Filter
                if st.session_state.selected_operator != "All Operators" and 'Operator' in df_filtered_global.columns:
                     df_filtered_global = df_filtered_global[df_filtered_global['Operator'].astype(str) == st.session_state.selected_operator]
                
                # Tech Filter
                if st.session_state.selected_techs != ["All Techs"] and 'tech' in df_filtered_global.columns:
                    df_filtered_global = df_filtered_global[df_filtered_global['tech'].astype(str).isin(st.session_state.selected_techs)]
                
                # Modem Filter (Conditional)
                if not st.session_state.show_flysense and st.session_state.selected_modems != ["All Modems"] and 'modem_id' in df_filtered_global.columns:
                     df_filtered_global = df_filtered_global[df_filtered_global['modem_id'].astype(str).isin(st.session_state.selected_modems)]

                st.write(f"Showing data for: **{st.session_state.selected_operator}** from **{st.session_state.date_start.strftime('%Y-%m-%d')}** to **{st.session_state.date_end.strftime('%Y-%m-%d')}** ({len(df_filtered_global)} samples)")

            except Exception as e:
                st.error(f"Error applying global filters: {e}")
                df_filtered_global = df_raw.copy() # Fallback to unfiltered on error

            # --- Report Display Area --- 
            if df_filtered_global.empty:
                 st.warning("No data available for the selected filters.")
            else:
                 # Display selected report using the globally filtered data
                 # (Make sure all report sections use df_filtered_global and handle potential missing columns gracefully)
                 if st.session_state.selected_report == "Data Overview":
                     st.header("Data Overview")
                     # --- Metric Cards ---
                     st.subheader("Average Performance Metrics")
                     data_for_cards = df_filtered_global
                     metrics_to_show = {
                         'RSRP (dBm)': 'rsrp', 'RSRQ (dB)': 'rsrq', 'SNR (dB)': 'snr',
                         'Latency (ms)': 'latency', 'Jitter (ms)': 'jitter',
                         'Download (Mbps)': 'downloadMbps', 'Upload (Mbps)': 'uploadMbps'
                     }
                     cols = st.columns(4)
                     col_idx = 0
                     for label, col_name in metrics_to_show.items():
                         if col_name in data_for_cards.columns:
                             numeric_series = pd.to_numeric(data_for_cards[col_name], errors='coerce')
                             avg_val = numeric_series.mean()
                             if pd.notna(avg_val):
                                 with cols[col_idx % 4]:
                                     st.metric(label=label, value=f"{avg_val:.2f}")
                                 col_idx += 1
                     if col_idx == 0:
                         st.info("No average metrics could be calculated.")

                     # --- New: Time Series Chart by Band and Parameters ---
                     st.subheader("Time Series by Band and Parameters")
                     # Check for required columns
                     required_cols = ['band', 'date', 'time']
                     time_series_params = ['rsrp', 'rsrq', 'snr', 'enb', 'latency', 'jitter', 'downloadMbps', 'uploadMbps']
                     available_params = [p for p in time_series_params if p in df_filtered_global.columns]
                     if not all(col in df_filtered_global.columns for col in required_cols):
                         st.warning("Missing required columns for time series chart (need 'band', 'date', 'time').")
                     elif not available_params:
                         st.warning("No available parameters to plot.")
                     else:
                         # Parse datetime
                         df_filtered_global['datetime'] = pd.to_datetime(df_filtered_global['date'].astype(str) + ' ' + df_filtered_global['time'].astype(str), errors='coerce')
                         # Drop rows with missing datetime or band
                         ts_df = df_filtered_global.dropna(subset=['datetime', 'band'])
                         # Get available bands
                         bands = sorted(ts_df['band'].dropna().astype(str).unique())
                         if not bands:
                             st.warning("No bands available in data.")
                         else:
                             selected_band = st.selectbox("Select Band:", bands)
                             band_df = ts_df[ts_df['band'].astype(str) == selected_band]
                             if band_df.empty:
                                 st.warning("No data for selected band.")
                             else:
                                 param_choices = st.multiselect("Select parameters to plot:", available_params, default=[p for p in ['rsrp', 'downloadMbps', 'uploadMbps'] if p in available_params])
                                 if not param_choices:
                                     st.info("Select at least one parameter to plot.")
                                 else:
                                     # Convert selected columns to numeric if needed
                                     for p in param_choices:
                                         band_df[p] = pd.to_numeric(band_df[p], errors='coerce')
                                     band_df = band_df.sort_values('datetime')
                                     # Plotly line chart with spline
                                     fig = go.Figure()
                                     for p in param_choices:
                                         fig.add_trace(go.Scatter(
                                             x=band_df['datetime'],
                                             y=band_df[p],
                                             mode='lines+markers',
                                             name=p,
                                             line_shape='spline',
                                             connectgaps=True
                                         ))
                                     fig.update_layout(
                                         xaxis_title='Time',
                                         yaxis_title='Value',
                                         title=f"Time Series for Band {selected_band}",
                                         legend_title="Parameter",
                                         hovermode='x unified',
                                         margin={'l':0,'r':0,'t':40,'b':0}
                                     )
                                     st.plotly_chart(fig, use_container_width=True)

                 elif st.session_state.selected_report == "Network Coverage Quality Map":
                      st.header("Network Coverage Quality Map")
                      if 'lat' not in df_filtered_global.columns or 'lng' not in df_filtered_global.columns:
                          st.warning("Required columns ('lat', 'lng') not found.")
                      else:
                          if st.session_state.show_flysense:
                               st.subheader(f"Flysense View: All Sensors")
                               flysense_data = df_filtered_global.dropna(subset=['lat', 'lng', 'modem_id'])
                               if flysense_data.empty:
                                   st.warning("No valid data points with lat/lng/modem_id for Flysense view.")
                               else:
                                    # Calculate Centroids (still useful for icon placement later)
                                    centroids = flysense_data.groupby('modem_id')[['lat', 'lng']].mean().reset_index()
                                    st.info(f"Calculated {len(centroids)} sensor centroids.") # User feedback
                                    
                                    # Calculate overall center (based on all points for better map centering)
                                    overall_center_lat = flysense_data['lat'].mean()
                                    overall_center_lon = flysense_data['lng'].mean()
                                    
                                    # --- Clustering based on CENTROIDS --- 
                                    # Remove the commented out data limiting block
                                    
                                    if centroids.empty or len(centroids) < 2:
                                        st.warning("Not enough centroids (need at least 2) for clustering.")
                                        labels = np.zeros(len(centroids), dtype=int)
                                        centroids['cluster'] = labels # Add cluster column anyway
                                        n_clusters = 0 if centroids.empty else 1
                                    else:
                                        # Prepare CENTROID coords for DBSCAN
                                        coords_for_dbscan = centroids[['lng', 'lat']].values
                                        print(f"[DEBUG] Running DBSCAN on {len(coords_for_dbscan)} centroids...") # DEBUG
    
                                        # Use DBSCAN to find clusters of centroids
                                        dbscan_eps = 0.05 # Adjust based on expected centroid spacing
                                        dbscan_min_samples = 2 # Need at least 2 centroids to form a spatial cluster
                                        try:
                                            st.info(f"Running clustering (DBSCAN) on {len(coords_for_dbscan)} centroids...")
                                            print(f"[DEBUG] Starting DBSCAN fit on centroids (eps={dbscan_eps}, min_samples={dbscan_min_samples})...") # DEBUG
                                            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_for_dbscan)
                                            print("[DEBUG] DBSCAN fit completed.") # DEBUG
                                            labels = db.labels_
                                            
                                            # Add labels to the centroids DataFrame
                                            centroids['cluster'] = labels
                                            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                            st.success(f"Identified {n_clusters} centroid cluster(s).")
                                        except ImportError:
                                             st.error("sklearn library not found. Please install it: pip install scikit-learn")
                                             labels = np.zeros(len(centroids), dtype=int)
                                             centroids['cluster'] = labels
                                             n_clusters = 1
                                        except Exception as e_dbscan:
                                             st.error(f"Error during centroid clustering: {e_dbscan}")
                                             print(f"[DEBUG] Error during DBSCAN on centroids: {e_dbscan}") # DEBUG
                                             labels = np.zeros(len(centroids), dtype=int)
                                             centroids['cluster'] = labels
                                             n_clusters = 1
                                        
                                    # --- Load and Encode Custom Icon --- 
                                    icon_data_uri = None
                                    icon_path = "Flysense.png"
                                    try:
                                        if os.path.exists(icon_path):
                                            with open(icon_path, "rb") as image_file:
                                                encoded_string = base64.b64encode(image_file.read()).decode()
                                            icon_data_uri = f"data:image/png;base64,{encoded_string}"
                                        # No else: don't show warning if not found, just skip icon
                                    except Exception as e_icon:
                                         st.error(f"Error loading or encoding icon: {e_icon}")

                                    # Create Figure
                                    fig_fly = go.Figure()

                                    # --- Add Buffered Hull Polygon for each CENTROID cluster --- 
                                    if n_clusters > 0:
                                        unique_labels = set(centroids['cluster'])
                                        for k in unique_labels:
                                            if k == -1:
                                                continue
                                            cluster_mask = (centroids['cluster'] == k)
                                            cluster_centroids_df = centroids.loc[cluster_mask]
                                            # Get all modem_ids in this cluster
                                            cluster_modems = set(cluster_centroids_df['modem_id'])
                                            # Get all measurement points for these modems
                                            cluster_measurements_df = flysense_data[flysense_data['modem_id'].isin(cluster_modems)]
                                            cluster_points = cluster_measurements_df[['lng', 'lat']].values
                                            if len(cluster_points) >= 1:
                                                try:
                                                    mp = MultiPoint(cluster_points)
                                                    shape_to_buffer = mp
                                                    if len(cluster_points) >= 3:
                                                        hull = mp.convex_hull
                                                        if hull.geom_type in ['Polygon', 'LineString']:
                                                            shape_to_buffer = hull
                                                    elif len(cluster_points) == 2:
                                                        shape_to_buffer = mp.convex_hull
                                                    buffer_distance = 0.01
                                                    buffered_shape = shape_to_buffer.buffer(buffer_distance)
                                                    if buffered_shape.geom_type == 'Polygon':
                                                        hull_lon, hull_lat = buffered_shape.exterior.xy
                                                        hull_lon = np.append(hull_lon, hull_lon[0])
                                                        hull_lat = np.append(hull_lat, hull_lat[0])
                                                        fig_fly.add_trace(go.Scattermapbox(
                                                            mode = "lines", fill = "toself",
                                                            lon = hull_lon,
                                                            lat = hull_lat,
                                                            fillcolor = 'rgba(255,0,0,0.2)',
                                                            line = dict(width = 2, color = 'rgba(255,0,0,0.6)'),
                                                            name = f'Cluster {k} Area',
                                                            hoverinfo='name'
                                                        ))
                                                except Exception as e_hull:
                                                    st.warning(f"Could not calculate or buffer hull for centroid cluster {k}: {e_hull}")

                                    # Add Measurement Points (increase size)
                                    fig_fly.add_trace(go.Scattermapbox(
                                        mode = "markers",
                                        lon = flysense_data['lng'],
                                        lat = flysense_data['lat'],
                                        marker = go.scattermapbox.Marker(
                                             size=12, # Increased size for visibility
                                             color='rgba(0, 0, 255, 0.5)', # Semi-transparent blue
                                             opacity=0.5
                                        ),
                                        name='Measurements',
                                        hovertext=flysense_data['modem_id'] # Show modem ID on hover
                                    ))

                                    # --- Prepare Mapbox Layers (Including Custom Icons) --- 
                                    mapbox_layers = []
                                    if icon_data_uri and not centroids.empty:
                                         icon_source = {
                                             "type": "FeatureCollection",
                                             "features": [
                                                 {
                                                     "type": "Feature",
                                                     "geometry": {
                                                         "type": "Point",
                                                         "coordinates": [lon, lat]
                                                     },
                                                     "properties": {
                                                         "modem_id": modem_id
                                                     }
                                                 } for modem_id, lat, lon in zip(centroids['modem_id'], centroids['lat'], centroids['lng'])
                                             ]
                                         }
                                         mapbox_layers.append(
                                              {
                                                   "sourcetype": "geojson",
                                                   "source": icon_source,
                                                   "type": "symbol",
                                                   "symbol": {
                                                        "icon": "custom-icon",
                                                        "iconsize": 0.5
                                                        # "placement": "point" # Optional, can be added if needed
                                                   }
                                              }
                                         )

                                    # Update Layout
                                    layout_updates = {
                                         'mapbox': {
                                             'style': "open-street-map",
                                             'center': {'lat': overall_center_lat, 'lon': overall_center_lon},
                                             'zoom': 13, # Adjust zoom as needed
                                             'layers': mapbox_layers # Add the layers list here
                                         },
                                         'margin': {'l':0, 'r':0, 't':0, 'b':0},
                                         'showlegend': True,
                                         'legend': {
                                             'bgcolor': 'rgba(200,200,200,0.5)',
                                             'bordercolor': 'rgba(100,100,100,0.5)',
                                             'borderwidth': 2,
                                         },
                                         'hovermode': 'closest',
                                         'images': [
                                            {
                                                "name": "custom-icon",
                                                "source": icon_data_uri
                                            }
                                         ] if icon_data_uri else []
                                    }

                                    fig_fly.update_layout(**layout_updates)
                                    st.plotly_chart(fig_fly, use_container_width=True)
                          else:
                               # --- Standard View ---
                               map_metrics = ['rsrp', 'rsrq', 'snr', 'latency', 'jitter', 'downloadMbps', 'uploadMbps']
                               available_metrics = [m for m in map_metrics if m in df_filtered_global.columns]
                               if not available_metrics:
                                    st.warning(f"No standard map metrics found.")
                               else:
                                    selected_metric = st.selectbox("Select metric to display on map:", available_metrics)
                                    # Ensure selected metric is numeric
                                    df_filtered_global[selected_metric] = pd.to_numeric(df_filtered_global[selected_metric], errors='coerce')
                                    map_df = df_filtered_global.dropna(subset=['lat', 'lng', selected_metric])
                                    if not map_df.empty:
                                         color_scale = px.colors.sequential.Viridis_r if selected_metric in ['latency', 'jitter'] else px.colors.sequential.Viridis
                                         fig = px.scatter_mapbox(map_df, lat="lat", lon="lng", color=selected_metric,
                                                                size=[5] * len(map_df), # Increased fixed size for points
                                                                color_continuous_scale=color_scale,
                                                                mapbox_style="open-street-map",
                                                                hover_name=selected_metric,
                                                                title=f"{selected_metric.capitalize()} Coverage Map")
                                         fig.update_layout(
                                               mapbox_center_lat=map_df['lat'].mean(),
                                               mapbox_center_lon=map_df['lng'].mean(),
                                               margin={"r":0,"t":0,"l":0,"b":0}
                                          )
                                         st.plotly_chart(fig, use_container_width=True)
                                    else:
                                         st.warning(f"No valid data for map.")
              
                 # --- [Other Reports using df_filtered_global with similar robustness checks] ---
                 elif st.session_state.selected_report == "Network Performance Benchmarking":
                     st.header("Network Performance Benchmarking")
                     required_cols = ['latency', 'jitter', 'downloadMbps', 'uploadMbps', 'tech', 'Operator'] 
                     if not all(col in df_filtered_global.columns for col in required_cols):
                          st.warning(f"Missing required columns for benchmarking.")
                     else:
                          # Convert relevant columns to numeric
                          for col in ['latency', 'jitter', 'downloadMbps', 'uploadMbps']:
                              df_filtered_global[col] = pd.to_numeric(df_filtered_global[col], errors='coerce')
                          # Drop rows with missing tech or operator
                          bench_df = df_filtered_global.dropna(subset=['tech', 'Operator'])
                          # Group by tech and operator, calculate means
                          group_cols = ['tech', 'Operator']
                          agg_cols = ['latency', 'jitter', 'downloadMbps', 'uploadMbps']
                          summary = bench_df.groupby(group_cols)[agg_cols].mean().reset_index()
                          st.subheader("Average Performance by Technology and Operator")
                          st.dataframe(summary, use_container_width=True)
                          # Bar chart: average download/upload by technology
                          st.subheader("Average Download/Upload Speed by Technology")
                          tech_summary = bench_df.groupby('tech')[['downloadMbps', 'uploadMbps']].mean().reset_index()
                          tech_summary = tech_summary.sort_values('downloadMbps', ascending=False)
                          fig = px.bar(tech_summary, x='tech', y=['downloadMbps', 'uploadMbps'],
                                       barmode='group',
                                       labels={'value': 'Mbps', 'tech': 'Technology', 'variable': 'Metric'},
                                       title='Average Download/Upload Speed by Technology')
                          st.plotly_chart(fig, use_container_width=True)
                          # Show filtered data table (optional)
                          with st.expander("Show Raw Filtered Data"):
                              st.dataframe(bench_df, use_container_width=True)

                 elif st.session_state.selected_report == "Anomalies":
                     st.header("Anomalies Report")
                     tabs = st.tabs(["Performance", "RF Status", "Network"])
                     # --- Performance Tab ---
                     with tabs[0]:
                         st.subheader("Performance Anomalies")
                         perf_params = ["downloadMbps", "uploadMbps", "latency", "jitter"]
                         perf_labels = {"downloadMbps": "Download (Mbps)", "uploadMbps": "Upload (Mbps)", "latency": "Latency (ms)", "jitter": "Jitter (ms)"}
                         perf_defaults = {"downloadMbps": 10, "uploadMbps": 2, "latency": 100, "jitter": 30}
                         perf_directions = {"downloadMbps": "min", "uploadMbps": "min", "latency": "max", "jitter": "max"}
                         st.write("Set thresholds for anomalies:")
                         perf_thresholds = {}
                         cols = st.columns(len(perf_params))
                         for i, p in enumerate(perf_params):
                             if perf_directions[p] == "min":
                                 perf_thresholds[p] = cols[i].number_input(f"Min {perf_labels[p]}", value=perf_defaults[p], key=f"perf_thr_{p}")
                             else:
                                 perf_thresholds[p] = cols[i].number_input(f"Max {perf_labels[p]}", value=perf_defaults[p], key=f"perf_thr_{p}")
                         # User selects parameter to analyze
                         selected_param = st.selectbox("Select parameter to analyze:", perf_params, format_func=lambda x: perf_labels[x])
                         # Prepare data
                         perf_df = df_filtered_global[["datetime", "date", "time"] + perf_params].copy() if "datetime" in df_filtered_global.columns else df_filtered_global[["date", "time"] + perf_params].copy()
                         for p in perf_params:
                             if p in perf_df.columns:
                                 perf_df[p] = pd.to_numeric(perf_df[p], errors='coerce')
                         # Determine anomalies
                         if perf_directions[selected_param] == "min":
                             perf_df["is_anomaly"] = perf_df[selected_param] < perf_thresholds[selected_param]
                         else:
                             perf_df["is_anomaly"] = perf_df[selected_param] > perf_thresholds[selected_param]
                         # Line+markers chart
                         st.subheader(f"{perf_labels[selected_param]} Over Time (Anomalies Highlighted)")
                         if "datetime" in perf_df.columns:
                             x_vals = perf_df["datetime"]
                         else:
                             x_vals = perf_df["date"].astype(str) + " " + perf_df["time"].astype(str)
                         fig = go.Figure()
                         # Normal points
                         fig.add_trace(go.Scatter(
                             x=x_vals[~perf_df["is_anomaly"]],
                             y=perf_df.loc[~perf_df["is_anomaly"], selected_param],
                             mode="lines+markers",
                             name="Normal",
                             line=dict(color="green"),
                             marker=dict(color="green", size=8, symbol="circle"),
                             showlegend=True
                         ))
                         # Anomaly points
                         fig.add_trace(go.Scatter(
                             x=x_vals[perf_df["is_anomaly"]],
                             y=perf_df.loc[perf_df["is_anomaly"], selected_param],
                             mode="markers",
                             name="Anomaly",
                             marker=dict(color="red", size=10, symbol="diamond"),
                             showlegend=True
                         ))
                         fig.update_layout(
                             xaxis_title="Time",
                             yaxis_title=perf_labels[selected_param],
                             legend_title="Status",
                             hovermode='x unified',
                             margin={'l':0,'r':0,'t':40,'b':0}
                         )
                         st.plotly_chart(fig, use_container_width=True)
                         # Pie chart
                         st.subheader("Anomaly Distribution")
                         anomaly_count = perf_df["is_anomaly"].sum()
                         normal_count = len(perf_df) - anomaly_count
                         pie_fig = go.Figure(data=[
                             go.Pie(labels=["Normal", "Anomaly"], values=[normal_count, anomaly_count],
                                    marker_colors=["green", "red"], hole=0.4)
                         ])
                         pie_fig.update_layout(margin={'l':0,'r':0,'t':40,'b':0})
                         st.plotly_chart(pie_fig, use_container_width=True)
                         # Table of anomalies only
                         st.subheader("Anomaly Table")
                         anomaly_table = perf_df[perf_df["is_anomaly"]]
                         if not anomaly_table.empty:
                             st.dataframe(anomaly_table, use_container_width=True, hide_index=True)
                         else:
                             st.info("No anomalies detected for the selected parameter.")
                     # --- RF Status Tab ---
                     with tabs[1]:
                         st.subheader("RF Status Anomalies")
                         rf_params = ["rsrp", "rsrq", "snr"]
                         rf_labels = {"rsrp": "RSRP (dBm)", "rsrq": "RSRQ (dB)", "snr": "SNR (dB)"}
                         rf_defaults = {"rsrp": -110, "rsrq": -15, "snr": 3}
                         rf_directions = {"rsrp": "min", "rsrq": "min", "snr": "min"}
                         st.write("Set thresholds for anomalies:")
                         rf_thresholds = {}
                         cols = st.columns(len(rf_params))
                         for i, p in enumerate(rf_params):
                             rf_thresholds[p] = cols[i].number_input(f"Min {rf_labels[p]}", value=rf_defaults[p], key=f"rf_thr_{p}")
                         # User selects parameter to analyze
                         selected_rf_param = st.selectbox("Select parameter to analyze:", rf_params, format_func=lambda x: rf_labels[x])
                         # Prepare data
                         rf_df = df_filtered_global[["datetime", "date", "time"] + rf_params].copy() if "datetime" in df_filtered_global.columns else df_filtered_global[["date", "time"] + rf_params].copy()
                         for p in rf_params:
                             if p in rf_df.columns:
                                 rf_df[p] = pd.to_numeric(rf_df[p], errors='coerce')
                         # Determine anomalies
                         rf_df["is_anomaly"] = rf_df[selected_rf_param] < rf_thresholds[selected_rf_param]
                         # Line+markers chart
                         st.subheader(f"{rf_labels[selected_rf_param]} Over Time (Anomalies Highlighted)")
                         if "datetime" in rf_df.columns:
                             x_vals = rf_df["datetime"]
                         else:
                             x_vals = rf_df["date"].astype(str) + " " + rf_df["time"].astype(str)
                         fig = go.Figure()
                         # Normal points
                         fig.add_trace(go.Scatter(
                             x=x_vals[~rf_df["is_anomaly"]],
                             y=rf_df.loc[~rf_df["is_anomaly"], selected_rf_param],
                             mode="lines+markers",
                             name="Normal",
                             line=dict(color="green"),
                             marker=dict(color="green", size=8, symbol="circle"),
                             showlegend=True
                         ))
                         # Anomaly points
                         fig.add_trace(go.Scatter(
                             x=x_vals[rf_df["is_anomaly"]],
                             y=rf_df.loc[rf_df["is_anomaly"], selected_rf_param],
                             mode="markers",
                             name="Anomaly",
                             marker=dict(color="red", size=10, symbol="diamond"),
                             showlegend=True
                         ))
                         fig.update_layout(
                             xaxis_title="Time",
                             yaxis_title=rf_labels[selected_rf_param],
                             legend_title="Status",
                             hovermode='x unified',
                             margin={'l':0,'r':0,'t':40,'b':0}
                         )
                         st.plotly_chart(fig, use_container_width=True)
                         # Pie chart
                         st.subheader("Anomaly Distribution")
                         anomaly_count = rf_df["is_anomaly"].sum()
                         normal_count = len(rf_df) - anomaly_count
                         pie_fig = go.Figure(data=[
                             go.Pie(labels=["Normal", "Anomaly"], values=[normal_count, anomaly_count],
                                    marker_colors=["green", "red"], hole=0.4)
                         ])
                         pie_fig.update_layout(margin={'l':0,'r':0,'t':40,'b':0})
                         st.plotly_chart(pie_fig, use_container_width=True)
                         # Table of anomalies only
                         st.subheader("Anomaly Table")
                         anomaly_table = rf_df[rf_df["is_anomaly"]]
                         if not anomaly_table.empty:
                             st.dataframe(anomaly_table, use_container_width=True, hide_index=True)
                         else:
                             st.info("No anomalies detected for the selected parameter.")
                     # --- Network Tab ---
                     with tabs[2]:
                         st.subheader("Network Band/Technology Distribution")
                         # Pie chart: all bands, colored by technology, size by sample count
                         if 'band' in df_filtered_global.columns and 'tech' in df_filtered_global.columns:
                             band_tech_df = df_filtered_global.dropna(subset=['band', 'tech'])
                             band_tech_df['band'] = band_tech_df['band'].astype(str)
                             band_tech_df['tech'] = band_tech_df['tech'].astype(str)
                             band_counts = band_tech_df.groupby(['band', 'tech']).size().reset_index(name='count')
                             band_counts['label'] = band_counts['band'] + ' (' + band_counts['tech'] + ')'
                             fig = go.Figure(data=[
                                 go.Pie(labels=band_counts['label'], values=band_counts['count'],
                                        marker=dict(colors=px.colors.qualitative.Plotly),
                                        hole=0.4)
                             ])
                             fig.update_layout(
                                 title="Sample Distribution by Band and Technology",
                                 margin={'l':0,'r':0,'t':40,'b':0}
                             )
                             st.plotly_chart(fig, use_container_width=True)
                         else:
                             st.info("No band/technology data available for pie chart.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # Add more detailed error logging if needed
            # import traceback
            # st.error(traceback.format_exc())
else:
    st.info("Please upload a CSV file using the sidebar to begin analysis.")