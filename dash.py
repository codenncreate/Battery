import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from mpl_toolkits.mplot3d import Axes3D
import time

# st.set_page_config(layout="wide", page_icon="🔋")
st.set_page_config(layout="wide", page_icon="⚡")

# Function to load data with caching
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Add an image as an icon
st.image("https://www.electica.in/assets/logo-31df17ef.png", width=200, use_column_width=False)

# Set the title of the app
st.title("Battery Analytics Dashboard")

st.sidebar.title("Load Data for Analysis")

# Sidebar for file upload and data summary
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Button to trigger data processing
if uploaded_file is not None:
    if st.sidebar.button("Generate Insights"):
    
        # Simulate loading time
        df = load_data(uploaded_file)
        temp = df.copy()  # Create a copy of the dataframe for plotting

        # Display success message in the sidebar
        st.sidebar.success("Data uploaded successfully!")

        # Data Summary in the sidebar
        st.sidebar.header("Summary Statistics")
        st.sidebar.write("Total Columns:", df.shape[1])

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        st.sidebar.write("#### Categorical Features:", len(categorical_cols))
        for i, col in enumerate(categorical_cols):
            st.sidebar.write(f"{i+1}. {col}")

        st.sidebar.write("Unique Categories in Categorical Columns:")
        for col in categorical_cols:
            st.sidebar.markdown(f"{col}: {df[col].unique()}")

        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        st.sidebar.write("#### Numerical Columns:", len(numerical_cols))
        for i, col in enumerate(numerical_cols):
            st.sidebar.markdown(f"{i+1}. {col}")

        st.sidebar.write("Numerical Summary:")
        st.sidebar.write(df[numerical_cols].describe())

            # Main window content
        with st.expander("View Data"):
            st.dataframe(df, use_container_width=True)

        st.write("## Data Analysis")

        with st.spinner('Generating plots...'):

            # Progress bar
            progress_bar = st.progress(0)
            progress_steps = 6  # Number of plots to generate

            
            # Create three columns for plots
            col1, col2, col3 = st.columns(3)

            # Define the color combination and battery sequence
            batteries = df['battery'].unique()
            dark_mode_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            battery_color_map = {battery: dark_mode_colors[i % len(dark_mode_colors)] for i, battery in enumerate(batteries)}

            # Plot 1: SOH vs Cycle for Each Battery
            with col1:
                fig = px.line(df, x='cycle', y='SOH', color='battery',
                              color_discrete_map=battery_color_map,
                              title='State of Health (SOH) Over Cycles',
                              category_orders={"battery": batteries})
                fig.update_layout(legend_title_text="Battery")
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(1 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Plot 2: Correlation Heatmap
            with col2:
                corr_matrix = temp[['voltage_measured', 'current_measured', 'temperature_measured', 'SOH']].corr(numeric_only=True)
                fig = px.imshow(corr_matrix, text_auto=".2f", title='Correlation Heatmap', color_continuous_scale='Inferno')
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(2 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Plot 3: Distribution Curve for SOH for Each Battery
            with col3:
                hist_data = [df[df['battery'] == battery]['SOH'] for battery in batteries]
                group_labels = batteries

                fig = ff.create_distplot(hist_data, group_labels, colors=[battery_color_map[battery] for battery in batteries],
                                        show_hist=False, show_rug=False, curve_type='normal')

                # Update layout and add opacity to the filled area
                for i in range(len(fig.data)):
                    fig.data[i].line.color = battery_color_map[group_labels[i]]
                    fig.data[i].fillcolor = battery_color_map[group_labels[i]]

                fig.update_layout(
                    title_text='SOH Distribution Curve',
                    xaxis_title_text='SOH',
                    yaxis_title_text='Density',
                    legend_title_text='Battery'
                )
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(3 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Plot 4: Lineplot for temperature_measured
            with col1:
                var = 'temperature_measured'
                bin_size = 20
                x = temp[(temp['cycle'] >= 0) & (temp['cycle'] <= temp['cycle'].max())]
                x['cycle_bin'] = pd.cut(x['cycle'], bins=range(0, x['cycle'].max() + bin_size, bin_size), right=False)
                aggregated_data = x.groupby(['cycle_bin', 'time'], as_index=False, observed=True)[var].mean()
                fig = px.line(aggregated_data, x='time', y=var, color='cycle_bin', title=f'{var} Trend over Time')
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(4 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Plot 5: Lineplot for current_measured
            with col2:
                var = 'current_measured'
                bin_size = 20
                x = temp[(temp['cycle'] >= 0) & (temp['cycle'] <= temp['cycle'].max())]
                x['cycle_bin'] = pd.cut(x['cycle'], bins=range(0, x['cycle'].max() + bin_size, bin_size), right=False)
                aggregated_data = x.groupby(['cycle_bin', 'time'], as_index=False, observed=True)[var].mean()
                fig = px.line(aggregated_data, x='time', y=var, color='cycle_bin', title=f'{var} Trend over Time')
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(5 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Plot 6: Lineplot for voltage_measured
            with col3:
                var = 'voltage_measured'
                bin_size = 20
                x = temp[(temp['cycle'] >= 0) & (temp['cycle'] <= temp['cycle'].max())]
                x['cycle_bin'] = pd.cut(x['cycle'], bins=range(0, x['cycle'].max() + bin_size, bin_size), right=False)
                aggregated_data = x.groupby(['cycle_bin', 'time'], as_index=False, observed=True)[var].mean()
                fig = px.line(aggregated_data, x='time', y=var, color='cycle_bin', title=f'{var} Trend over Time')
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(6 / progress_steps)
                time.sleep(1)  # Simulate time taken to generate the plot

            # Remove the spinner once all plots are generated
            st.spinner()

else:
    st.sidebar.write("Please upload a CSV file to get started.")