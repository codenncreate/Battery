import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
# from mpl_toolkits.mplot3d import Axes3D # Not used in the provided snippet
import time
import base64
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import joblib
# import xgboost as xgb # Not directly used, but joblib loads an xgb model
import shap
import numpy as np
# import zipfile # Not used
# import os # Not used
# import io # Not used

st.set_page_config(layout="wide", page_icon="⚡", page_title="Electica")

# Load the config file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create an authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render the login widget
try:
    authenticator.login()
except Exception as e:
    st.error(e)

# --- Helper function for SOH comparison (cached) ---
@st.cache_data
def calculate_soh_comparison(_df_input_actual_soh, _model_input):
    # _df_input_actual_soh is the dataframe with the true 'SOH' values
    # _model_input is the trained model
    soh_list = []
    expected_features = ['cycle', 'voltage_measured', 'current_measured', 'temperature_measured', 'time']
    
    if 'battery' not in _df_input_actual_soh.columns:
        st.error("Critical error: 'battery' column missing in the uploaded data for SOH comparison.")
        return pd.DataFrame() # Return empty dataframe
        
    unique_batteries = _df_input_actual_soh['battery'].unique()

    for battery in unique_batteries:
        battery_data = _df_input_actual_soh[_df_input_actual_soh['battery'] == battery]
        if battery_data.empty or 'cycle' not in battery_data.columns:
            continue
        
        max_cycle_val = battery_data['cycle'].max()
        if pd.isna(max_cycle_val):
            continue
        
        val = int(max_cycle_val)

        for cycle in range(1, val + 1):
            cycle_data = battery_data[battery_data['cycle'] == cycle]
            if cycle_data.empty:
                continue
            
            actual_avg_soh = cycle_data['SOH'].mean() if 'SOH' in cycle_data else float('nan')

            # Prepare features for prediction from cycle_data
            missing_cols = [col for col in expected_features if col not in cycle_data.columns]
            if missing_cols:
                st.warning(f"Missing columns for prediction: {missing_cols} in cycle {cycle} for battery {battery}. Skipping this entry.")
                continue

            X_cycle = cycle_data[expected_features]
            
            predicted_avg_soh = float('nan') # Default
            try:
                if not X_cycle.empty:
                    predictions = _model_input.predict(X_cycle)
                    predicted_avg_soh = predictions.mean()
            except Exception as e:
                st.warning(f"Error during prediction for battery {battery}, cycle {cycle}: {e}. Skipping this entry.")

            soh_list.append({
                'Battery': battery, 'Cycle': cycle,
                'Actual SOH': actual_avg_soh, 'Predicted SOH': predicted_avg_soh
            })

    soh_df_output = pd.DataFrame(soh_list)
    if not soh_df_output.empty:
        soh_df_output.dropna(subset=['Predicted SOH', 'Actual SOH'], how='any', inplace=True)
        soh_df_output['Actual SOH'] = soh_df_output['Actual SOH'].round(2)
        soh_df_output['Predicted SOH'] = soh_df_output['Predicted SOH'].round(2)
        soh_df_output['Predicted SOH'] = soh_df_output['Predicted SOH'].clip(upper=1)
    return soh_df_output

# Authenticate users
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{st.session_state.get("name")}*!')

    image_path = "logo3.png"
    url = "https://www.electica.in/"
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        markdown_content = f'<a href="{url}" target="_blank"><img src="data:image/png;base64,{encoded_image}" width="300"></a>'
        st.markdown(markdown_content, unsafe_allow_html=True)
    except FileNotFoundError:
        st.sidebar.warning(f"Logo file '{image_path}' not found. Displaying text link instead.")
        st.markdown(f'<a href="{url}" target="_blank">Electica</a>', unsafe_allow_html=True)


    st.title("Battery Analytics Dashboard")

    if 'show_markdown' not in st.session_state:
        st.session_state.show_markdown = True

    if st.session_state.show_markdown:
        markdown_text = """
        ##### Welcome!

        Unlock comprehensive insights into your battery's performance and health. 
        - **Data Exploration**: Upload battery information to see performance patterns through interactive charts.
        - **Performance Analysis**: Assess the predictive model's accuracy and overall effectiveness.
        - **Predictive Insights**: Leverage Model Forecasts to find key degradation factors for better battery management.
        """
        st.markdown(markdown_text, unsafe_allow_html=True)

    st.sidebar.title("Upload Data for Analysis")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

    # Initialize session state variables
    if 'insights_generated' not in st.session_state:
        st.session_state.insights_generated = False
    if 'data_frame' not in st.session_state:
        st.session_state.data_frame = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "Model Performance" # Default

    if uploaded_file is not None:
        if st.sidebar.button("Generate Insights"):
            st.session_state.show_markdown = False
            st.session_state.insights_generated = True

            @st.cache_data # Cache data loading
            def load_data(file):
                try:
                    data = pd.read_csv(file)
                    return data
                except Exception as e:
                    st.error(f"Error loading CSV file: {e}")
                    return None

            df_loaded = load_data(uploaded_file)
            
            if df_loaded is not None:
                st.session_state.data_frame = df_loaded.copy()
                try:
                    st.session_state.model = joblib.load('battery_soh_model2.pkl')
                    st.sidebar.success("Data loaded successfully!")

                    # Pre-calculate SOH comparison results here
                    st.session_state.results_df = calculate_soh_comparison(
                        st.session_state.data_frame.copy(), 
                        st.session_state.model
                    )

                except FileNotFoundError:
                    st.error("Model file 'battery_soh_model2.pkl' not found. Please ensure it's in the correct path.")
                    st.session_state.insights_generated = False # Reset flag if model fails
                except Exception as e:
                    st.error(f"Error loading the model: {e}")
                    st.session_state.insights_generated = False # Reset flag

                 # Data Summary (only if df is loaded)
                df_summary = st.session_state.data_frame
                st.sidebar.header("Summary Statistics")
                st.sidebar.write("Total Rows:", df_summary.shape[0])
                st.sidebar.write("Total Columns:", df_summary.shape[1])

                categorical_cols = df_summary.select_dtypes(include=['object', 'category']).columns
                st.sidebar.write("#### Categorical Columns:", len(categorical_cols))
                for i, col in enumerate(categorical_cols):
                    st.sidebar.write(f"{i+1}. {col}")
                # st.sidebar.write("Unique Categories in Categorical Columns:") # Can be verbose
                # for col in categorical_cols:
                #     st.sidebar.markdown(f"{col}: {df_summary[col].unique()[:5]}...") # Show first 5

                numerical_cols = df_summary.select_dtypes(include=['float64', 'int64']).columns
                st.sidebar.write("#### Numerical Columns:", len(numerical_cols))
                for i, col in enumerate(numerical_cols):
                    st.sidebar.markdown(f"{i+1}. {col}")
                if not numerical_cols.empty:
                    st.sidebar.write("Numerical Summary:")
                    st.sidebar.write(df_summary[numerical_cols].describe())
            else:
                st.session_state.insights_generated = False # Reset if data loading failed


    if st.session_state.insights_generated and st.session_state.data_frame is not None and st.session_state.model is not None:
        df = st.session_state.data_frame # Use data from session state
        model = st.session_state.model   # Use model from session state
        temp_df_viz = df.copy() # For visualizations that might alter df

        with st.expander("View Data"):
            st.dataframe(df, use_container_width=True)

        st.write("## Data Analysis")
        
        required_viz_cols = ['cycle', 'SOH', 'battery', 'voltage_measured', 'current_measured', 'temperature_measured', 'time']
        if not all(col in temp_df_viz.columns for col in required_viz_cols):
            st.error(f"One or more required columns for visualization are missing from the data: {required_viz_cols}")
        else:
            with st.spinner('Generating initial plots...'):
                progress_bar_viz = st.progress(0)
                progress_steps_viz = 6
                
                col1, col2, col3 = st.columns(3)
                
                batteries = temp_df_viz['battery'].unique()
                dark_mode_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                battery_color_map = {battery: dark_mode_colors[i % len(dark_mode_colors)] for i, battery in enumerate(batteries)}

                with col1:
                    fig = px.line(temp_df_viz, x='cycle', y='SOH', color='battery',
                                  color_discrete_map=battery_color_map,
                                  title='State of Health (SOH) Over Cycles',
                                  category_orders={"battery": batteries})
                    fig.update_layout(legend_title_text="Battery")
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar_viz.progress(1 / progress_steps_viz)
                    time.sleep(0.5)

                with col2:
                    corr_cols = ['voltage_measured', 'current_measured', 'temperature_measured', 'SOH']
                    corr_matrix = temp_df_viz[corr_cols].corr(numeric_only=True)
                    fig = px.imshow(corr_matrix, text_auto=".2f", title='Correlation Heatmap', color_continuous_scale='Inferno')
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar_viz.progress(2 / progress_steps_viz)
                    time.sleep(0.5)

                with col3:
                    hist_data = [temp_df_viz[temp_df_viz['battery'] == battery]['SOH'] for battery in batteries]
                    group_labels = batteries
                    fig = ff.create_distplot(hist_data, group_labels, colors=[battery_color_map.get(battery) for battery in batteries],
                                             show_hist=False, show_rug=False, curve_type='normal') # kde or normal
                    for i in range(len(fig.data)):
                        if group_labels[i] in battery_color_map: # Check if key exists
                           fig.data[i].line.color = battery_color_map[group_labels[i]]
                           # fig.data[i].fillcolor = battery_color_map[group_labels[i]] # fillcolor not standard for create_distplot traces like this for normal curve
                    fig.update_layout(title_text='SOH Distribution Curve', xaxis_title_text='SOH', yaxis_title_text='Density', legend_title_text='Battery')
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar_viz.progress(3 / progress_steps_viz)
                    time.sleep(0.5)

                def plot_trend_over_time(column_name, data, target_col):
                    var = column_name
                    num_bins = 10
                    # Filter for valid cycle values before binning
                    x = data[(data['cycle'] >= 0) & pd.notna(data['cycle']) & (data['cycle'] <= data['cycle'].max())].copy() # Use .copy()
                    if x.empty or x['cycle'].max() == 0:
                        st.warning(f"Not enough cycle data to plot {var} trend.")
                        return
                    max_cycle = int(x['cycle'].max())
                    step_size = max(1, max_cycle // num_bins)
                    bin_edges = range(0, max_cycle + step_size, step_size)
                    if len(bin_edges) < 2 : bin_edges = [0, max_cycle] # handle case with few cycles
                    
                    x.loc[:, 'cycle_bin'] = pd.cut(x['cycle'], bins=bin_edges, right=False, include_lowest=True)
                    aggregated_data = x.groupby(['cycle_bin', 'time'], as_index=False, observed=True)[var].mean()
                    fig_trend = px.line(aggregated_data, x='time', y=var, color='cycle_bin', title=f'{var} Trend over Time')
                    target_col.plotly_chart(fig_trend, use_container_width=True)

                with col1:
                    plot_trend_over_time('temperature_measured', temp_df_viz, col1)
                    progress_bar_viz.progress(4 / progress_steps_viz)
                    time.sleep(0.5)
                with col2:
                    plot_trend_over_time('current_measured', temp_df_viz, col2)
                    progress_bar_viz.progress(5 / progress_steps_viz)
                    time.sleep(0.5)
                with col3:
                    plot_trend_over_time('voltage_measured', temp_df_viz, col3)
                    progress_bar_viz.progress(6 / progress_steps_viz)
                    time.sleep(0.5)
                st.spinner() # Clear spinner

        # --- Analysis Type Selection ---
        st.session_state.analysis_type = st.radio(
            "Select Analysis Type:",
            ("Model Performance", "Model Predictions"),
            horizontal=True,
            key='analysis_type_radio', # Explicit key
            index=["Model Performance", "Model Predictions"].index(st.session_state.analysis_type) # Persist selection
        )
        
        # --- Model Performance Section ---
        if st.session_state.analysis_type == 'Model Performance':
            st.write("## Model Performance")
            results_perf_df = st.session_state.results_df # Use pre-calculated results

            if results_perf_df is None or results_perf_df.empty:
                st.warning("Model performance data could not be generated. Please check data and model.")
            else:
                st.write("#### Overall")
                with st.spinner('Generating overall performance plots...'):
                    progress_bar_mp = st.progress(0)
                    mp_steps = 3
                    col4, col5, col6 = st.columns(3)
                    xy = results_perf_df[['Actual SOH', 'Predicted SOH']].min().min()
                    with col4:
                        fig = px.scatter(results_perf_df, x='Actual SOH', y='Predicted SOH', title='Actual vs Predicted SOH', opacity=0.6)
                        fig.add_trace(go.Scatter(x=[xy,1], y=[xy,1], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                        fig.update_layout(legend_title_text="Legend") # Changed from Battery
                        st.plotly_chart(fig, use_container_width=True)
                        progress_bar_mp.progress(1/mp_steps); time.sleep(0.5)
                    
                    with col5: # SHAP Plot
                        X_shap = df[['cycle','voltage_measured', 'current_measured', 'temperature_measured', 'time']].copy()
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_shap)
                        shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
                        shap_importance = shap_df.abs().mean().sort_values(ascending=False)
                        fig_shap = go.Figure()
                        fig_shap.add_trace(go.Bar(x=shap_importance.values[::-1], y=shap_importance.index[::-1], orientation='h', marker=dict(color='orange'), name='Mean |SHAP|'))
                        fig_shap.update_layout(title='SHAP Feature Importance', xaxis_title='Mean |SHAP Value|', yaxis_title='Feature', template='plotly_dark')
                        st.plotly_chart(fig_shap, use_container_width=True)
                        progress_bar_mp.progress(2/mp_steps); time.sleep(0.5)

                    with col6: # Residuals Plot
                        residuals = results_perf_df['Actual SOH'] - results_perf_df['Predicted SOH']
                        fig_res = go.Figure()
                        fig_res.add_trace(go.Scatter(x=results_perf_df['Cycle'], y=residuals, mode='markers', name='Residuals', marker=dict(color='purple', size=10, opacity=0.6)))
                        if not results_perf_df.empty:
                            fig_res.add_trace(go.Scatter(x=[min(results_perf_df['Cycle']), max(results_perf_df['Cycle'])], y=[0, 0], mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
                        fig_res.update_layout(title="Residuals vs Cycle", xaxis_title="Cycle", yaxis_title="Residual (Actual - Predicted)", showlegend=True, template="plotly_dark")
                        st.plotly_chart(fig_res, use_container_width=True)
                        progress_bar_mp.progress(3/mp_steps); time.sleep(0.5)
                    st.spinner()

                st.write("#### Battery-wise")
                unique_batteries_perf = results_perf_df['Battery'].unique()
                if len(unique_batteries_perf) > 0:
                    with st.spinner('Generating battery-wise performance plots...'):
                        progress_bar_bp = st.progress(0)
                        bp_steps = len(unique_batteries_perf) * 3
                        plot_counter = 0

                        for battery_id in unique_batteries_perf:
                            st.subheader(f"Battery: {battery_id}")
                            battery_specific_df = results_perf_df[results_perf_df['Battery'] == battery_id]
                            if battery_specific_df.empty: continue

                            xy = battery_specific_df[['Actual SOH', 'Predicted SOH']].min().min()
                            col7, col8, col9 = st.columns(3)
                            with col7:
                                fig = px.scatter(battery_specific_df, x='Actual SOH', y='Predicted SOH', title='Actual vs Predicted SOH', opacity=0.6)
                                fig.add_trace(go.Scatter(x=[xy, 1], y=[xy, 1], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                                st.plotly_chart(fig, use_container_width=True)
                                plot_counter+=1; progress_bar_bp.progress(plot_counter/bp_steps); time.sleep(0.2)
                            with col8:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=battery_specific_df['Cycle'], y=battery_specific_df['Actual SOH'], mode='markers', name='Actual SOH', marker=dict(color='blue', size=10, opacity=0.7)))
                                fig.add_trace(go.Scatter(x=battery_specific_df['Cycle'], y=battery_specific_df['Predicted SOH'], mode='markers', name='Predicted SOH', marker=dict(color='orange', size=10, opacity=0.7, symbol='x')))
                                fig.update_layout(title="Actual vs Predicted SOH Over Cycles", xaxis_title="Cycle", yaxis_title="SOH", showlegend=True, template="plotly_dark", yaxis=dict(range=[0,1]))
                                st.plotly_chart(fig, use_container_width=True)
                                plot_counter+=1; progress_bar_bp.progress(plot_counter/bp_steps); time.sleep(0.2)
                            with col9:
                                residuals_battery = battery_specific_df['Actual SOH'] - battery_specific_df['Predicted SOH']
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=battery_specific_df['Cycle'], y=residuals_battery, mode='markers', name='Residuals', marker=dict(color='purple', size=10, opacity=0.6)))
                                fig.add_trace(go.Scatter(x=[min(battery_specific_df['Cycle']), max(battery_specific_df['Cycle'])], y=[0, 0], mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
                                fig.update_layout(title="Residuals vs Cycle", xaxis_title="Cycle", yaxis_title="Residual", showlegend=True, template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                                plot_counter+=1; progress_bar_bp.progress(plot_counter/bp_steps); time.sleep(0.2)
                        st.spinner()
                else:
                    st.info("No battery-specific performance data to display.")


        # --- Model Predictions Section ---
        elif st.session_state.analysis_type == 'Model Predictions':
            st.write("## Model Predictions")
            
            # Predictions are made on the original df
            df_pred = df.copy() # Work on a copy
            features_to_predict = ['cycle', 'voltage_measured', 'current_measured', 'temperature_measured', 'time']
            
            if not all(col in df_pred.columns for col in features_to_predict):
                st.error(f"One or more required columns for prediction are missing: {features_to_predict}")
            else:
                soh_predicted_values = model.predict(df_pred[features_to_predict])
                df_pred['SOH_predicted'] = soh_predicted_values
                df_pred['SOH_predicted'] = df_pred['SOH_predicted'].round(2).clip(upper=1)

                unique_batteries_pred = df_pred['battery'].unique()
                if len(unique_batteries_pred) > 0:
                    with st.spinner('Generating prediction plots...'):
                        progress_bar_pred = st.progress(0)
                        pred_steps = len(unique_batteries_pred)
                        
                        for idx, battery_id_pred in enumerate(unique_batteries_pred, start=1):
                            st.subheader(f"Battery: {battery_id_pred}")
                            battery_df_for_pred_plot = df_pred[df_pred['battery'] == battery_id_pred]
                            
                            if battery_df_for_pred_plot.empty: continue

                            avg_soh_per_cycle = battery_df_for_pred_plot.groupby('cycle')[['SOH_predicted']].mean().reset_index()
                            
                            # Display in a single column for this section
                            fig_pred_plot = go.Figure()
                            fig_pred_plot.add_trace(go.Scatter(x=avg_soh_per_cycle['cycle'], y=avg_soh_per_cycle['SOH_predicted'], mode='markers',
                                                        name='Predicted SOH', marker=dict(color='orange', size=10, opacity=0.7, symbol='x')))
                            fig_pred_plot.update_layout(title="Predicted SOH Over Cycles", xaxis_title="Cycle", yaxis_title="Predicted SOH",
                                                showlegend=True, yaxis=dict(range=[0,1]), template="plotly_dark")
                            if not avg_soh_per_cycle.empty: # Add x-axis range if data exists
                                fig_pred_plot.update_layout(xaxis=dict(range=[0, max(avg_soh_per_cycle['cycle'])]))
                            
                            st.plotly_chart(fig_pred_plot, use_container_width=True)
                            progress_bar_pred.progress(idx / pred_steps)
                            time.sleep(0.5) # Simulate time
                        st.spinner()
                        
                        # df_pred.to_csv("predictions.csv", index=False)
                        # st.success("The predictions file has been downloaded successfully.")
                else:
                    st.info("No batteries found in the data to make predictions for.")

    elif uploaded_file is None: # No file uploaded yet
        st.sidebar.info("Please upload a CSV file to get started.")
        if not st.session_state.insights_generated : # Only show if never generated
            st.sidebar.markdown("""
            ### CSV File Requirements
            The CSV file must contain the following columns:
            - **cycle**: The cycle number of the battery.
            - **voltage_measured**: The measured voltage of the battery.
            - **current_measured**: The measured current of the battery.
            - **temperature_measured**: The measured temperature of the battery.
            - **time**: The timestamp of the measurement.
            - **SOH**: The State of Health of the battery (for model performance).
            - **battery**: The identifier for the battery.
            """)
    # Handle cases where insights_generated is True but df or model is None (e.g. loading error)
    elif st.session_state.insights_generated and (st.session_state.data_frame is None or st.session_state.model is None):
        st.error("There was an issue loading the data or model. Please try uploading the file again or check the model file.")


elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')

# Update the config file (Be cautious with writing to config.yaml on every run in a deployed app)
# This is generally for updating credentials if you have a registration/password change feature.
try:
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
except Exception as e:
    st.warning(f"Could not save config file: {e}")