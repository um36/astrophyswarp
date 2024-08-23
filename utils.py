#This file includes all the definitions of the functions needed for the streamlit 
#web application including docstrings for each.
   
#importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to define the sine function
def sine_function(phi, A, C, D):
    """
    Computes the value of a sine function with amplitude, phase shift and vertical shift. 
    We don't include a frewuency as this would alter the period of the wave and not be consistent with the phi range of 360 degrees.
    Parameters:
    phi = between 0 and 360: The input angle(s) in degrees.
    A = float: The amplitude of the sine wave.
    C = float: The phase shift of the sine wave in degrees.
    D = float: The vertical shift of the sine wave.
    Returns = float or ndarray: The computed sine value(s) after applying the amplitude, phase shift, and vertical shift.
    """
    return A * np.sin(np.deg2rad(phi + C)) + D

# Function to adjust amplitude and phase shift values
def adjust_amplitude_phase(df, amp_col, phase_col):
    """
    Adjusts the amplitude and phase shift values in a DataFrame by adjusting negative amplitude values to be positive. If an amplitude is negative, 
    the corresponding phase shift is increased by 180 degrees. Ensuring that all phase shift values are within the range [0, 360) degrees.
    Parameters:
    df = pandas.DataFrame: The DataFrame containing the amplitude and phase shift columns to be adjusted.
    amp_col = str: The name of the column in the DataFrame that contains amplitude values (either height or velocity).
    phase_col = str: The name of the column in the DataFrame that contains phase shift values (either height or velocity).
    Returns = pandas.DataFrame: The modified DataFrame with adjusted amplitude and phase shift values.
    """
    # Adjust negative amplitudes
    df[phase_col] = np.where(df[amp_col] < 0, df[phase_col] + 180, df[phase_col])
    df[amp_col] = np.abs(df[amp_col])
    # Adjust phase shifts to be within [0, 360)
    df[phase_col] = df[phase_col] % 360
    return df

# Function to plot graph for model across all years (phase shift and amplitude)
def plot_graph(df, title, ycol1, ycol2, ylabel1, ylabel2, phase_shift=False):
    """
    Plots a graph with height and velocity data for selected radii.
    This function generates a plot with two y-axes, where one axis represents height (Kpc) and the 
    other represents velocity (Km/s). It allows plotting these values against time for different 
    radii. The function includes options for displaying the phase shift as well which allows the user to 
    choose between amplitude or phase shift to plot.
    Parameters:
    df = pandas.DataFrame: The DataFrame containing the data to be plotted. The DataFrame should include columns
         for radius ('R'), time ('t'), and the height and velocity values.
    title = str: The title of the plot.
    ycol1 = str: The name of the column in the DataFrame that contains the height data.
    ycol2 = str: The name of the column in the DataFrame that contains the velocity data.
    ylabel1 = str: The label for the y-axis corresponding to the height data.
    ylabel2 = str: The label for the y-axis corresponding to the velocity data.
    phase_shift = bool, optional: Whether to include phase shift in the plot (default is False).
    Returns = matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = None  # Initialize ax2 as None, will be used later if needed
    
    # Define color gradients for the different radii
    height_colors = plt.cm.cool(np.linspace(0.3, 0.7, len(selected_radii)))
    velocity_colors = plt.cm.gist_heat(np.linspace(0.3, 0.7, len(selected_radii)))
    
    # Loop through each selected radius to plot height and velocity
    for idx, radius in enumerate(selected_radii):
        if show_height:  # Check if height data should be plotted
            # Filter the DataFrame for the current radius
            df_selected_r_h = df[df['R'] == radius]
            # Plot the height data on ax1 (primary y-axis)
            ax1.plot(df_selected_r_h['t'], df_selected_r_h[ycol1], 
                     label=f'Height R={radius}', linestyle='-', marker='o', color=height_colors[idx])
        
        if show_velocity:  # Check if velocity data should be plotted
            if ax2 is None:
                ax2 = ax1.twinx()  # Create a secondary y-axis for velocity data if not already created
            # Filter the DataFrame for the current radius
            df_selected_r_v = df[df['R'] == radius]
            # Plot the velocity data on ax2 (secondary y-axis)
            ax2.plot(df_selected_r_v['t'], df_selected_r_v[ycol2], 
                     label=f'Velocity R={radius}', linestyle='--', marker='x', color=velocity_colors[idx])
    
    # Set labels and formatting for the primary y-axis (height)
    ax1.set_ylabel(ylabel1, color='blue', fontsize=14)
    ax1.set_xlim(left=0)  # Set the x-axis limit to start from 0
    ax1.set_xlabel('Time (Gyr)')  # Set the label for the x-axis
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)  # Format y-axis ticks
    
    # If a secondary y-axis was created, set its labels and formatting
    if ax2:
        ax2.set_ylabel(ylabel2, color='red', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    
    # Set the title of the plot
    plt.title(title, fontsize=16)
    
    # Handle the legends for both axes
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), bbox_transform=ax1.transAxes)
    if ax2:
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9), bbox_transform=ax1.transAxes)
    
    # Adjust layout to fit everything within the figure area
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)
    
    # Return the figure object
    return fig
     
# Function to adjust phase shifts based on a custom interval
def adjust_phase_shifts(df, phase_col, start_point):
    """
    Adjusts the phase shifts in a DataFrame to be within a custom interval.
    This function shifts the phase values such that they are adjusted relative to a 
    specified starting point. The phase values are constrained within the range of 
    [start_point, start_point + 360).
    Parameters:
    df = pandas.DataFrame: The DataFrame containing the phase shift data to be adjusted.
    phase_col = str: The name of the column in the DataFrame that contains the phase shift values.
    start_point = float: The starting point for the phase adjustment. The phase values will be adjusted 
    relative to this point and wrapped around within a 360-degree interval.
    Returns = pandas.DataFrame: The modified DataFrame with adjusted phase shift values.
    """
    # Adjust the phase values based on the start_point and wrap around 360 degrees
    df[phase_col] = (df[phase_col] - start_point) % 360 + start_point
    return df  # Return the modified DataFrame
           
#function to calculate phase difference between height and velocity
def calculate_phase_difference(combined_params_df, selected_R):
    """
    Calculates the phase difference between height and velocity for a given value of R for all years.
    This function filters the combined parameters DataFrame based on the selected value of R, 
    sorts the data by time 't', and computes the phase difference between the 'C_height' 
    and 'C_velocity' columns. The phase difference is adjusted to ensure it falls within 
    the range of -180 to 180 degrees.
    Parameters:
    combined_params_df = pd.DataFrame: DataFrame containing the combined parameters with columns
    'R', 't', 'C_height', and 'C_velocity'.
    selected_R = float: The value of R to filter the DataFrame by.
    Returns = pd.DataFrame: A DataFrame with the filtered and sorted data, including the calculated
    'Phase_Difference' column.
    """
   # Adjust phase difference to ensure it is within the range of -180 to 180 degrees
    def adjust_phase_differencehv(diff):
        """
        Adjusts the phase difference to be within the range of -180 to 180 degrees. THis is mainly linked to the below function.
        Parameters:
        diff = float: The original raw phase difference value.
        Returns = float: The adjusted phase difference value.
        """
        if diff > 180:
            return diff - 360
        elif diff < -180:
            return diff + 360
        else:
            return diff
    # Filter combined parameters DataFrame by selected R value
    filtered_params = combined_params_df[combined_params_df['R'] == selected_R].copy()
    # Ensure the DataFrame is sorted by time 't'
    filtered_params = filtered_params.sort_values(by='t')
    # Calculate the phase difference between 'C_height' and 'C_velocity'
    filtered_params['Phase_Difference'] = filtered_params['C_height'] - filtered_params['C_velocity']

    # Apply the adjustment function to the 'Phase_Difference' column
    filtered_params['Phase_Difference'] = filtered_params['Phase_Difference'].apply(adjust_phase_differencehv)
    return filtered_params
    
# Function to plot phase difference
def plot_phase_difference(merged_df, selected_R_pha):
    """
    Plots the phase difference between height and velocity over time.
    This function creates a line plot showing how the phase difference between height and
    velocity varies with time. It assumes that the input DataFrame has columns 't' for time
    and 'Phase_Difference' for the calculated phase difference. The plot includes labels,
    a title, and grid lines.
    Parameters:
    merged_df = pd.DataFrame: DataFrame containing the columns 't' (time) and 'Phase_Difference'.
    selected_R_pha = float: The value of R used to generate the plot title. This is selected by the user.
    """
    # Create a new figure with specified size
    plt.figure(figsize=(10, 5))
    # Plot 'Phase_Difference' against time 't'
    plt.plot(merged_df['t'], merged_df['Phase_Difference'], marker='o', linestyle='-', color='b')
    
    # Set labels for the x and y axes
    plt.xlabel('Time (Gyr)')
    plt.ylabel('Phase Difference (degrees)')
    # Set the title of the plot
    plt.title(f'Phase Difference Between Height and Velocity Over Time (Radius: {selected_R_pha})')
    # Set the x-axis limit to start from 0
    plt.xlim(left=0)
    # Add grid lines for better readability
    plt.grid(True)
    
    # Display the plot using Streamlit
    st.pyplot(plt)
        
# Calculate phase differences for consecutive R values
def calculate_differences(df, selected_r_values, selected_metric):
    """
    This function computes the difference of a specified metric (e.g., 'height' or 'velocity')
    between consecutive values of R. It assumes that the input DataFrame contains columns 
    for R values, time, and the metric of interest. For each pair of consecutive R values, it
    merges the data based on time and calculates the difference in the metric values.
    Parameters:
    df = pd.DataFrame: DataFrame containing columns 'R', 't' (time), and the metric of interest (e.g., 'C_height', 'C_velocity').
    selected_r_values = list of floats: List of R values for which differences are to be calculated range between 5.5 and 15.5 and must be consecutive.
    metric = str: The metric to calculate differences for, e.g., 'height' or 'velocity'.
    Returns = pd.DataFrame: A DataFrame containing time and the calculated differences for each pair of 
    consecutive R values.
    """
    differences = []

    # Ensure selected_metric is valid
    if selected_metric not in ['height', 'velocity']:
        raise ValueError("Selected metric must be either 'height' or 'velocity'.")

    # Debugging: Print selected R values
    print(f"Selected R values: {selected_r_values}")

    for i in range(len(selected_r_values) - 1):
        r1 = selected_r_values[i]
        r2 = selected_r_values[i + 1]

        # Filter the DataFrame for the current and next R values
        df_r1 = df[df['R'] == r1]
        df_r2 = df[df['R'] == r2]

        # Debugging: Print DataFrames
        print(f"DataFrame for R={r1}:\n", df_r1.head())
        print(f"DataFrame for R={r2}:\n", df_r2.head())

        if df_r1.empty or df_r2.empty:
            print(f"Warning: No data for R={r1} or R={r2}.")
            continue

        # Prepare column names for the difference calculation
        diff_column = f'{selected_metric}_diff_{r1}_{r2}'
        col_name = f'C_{selected_metric}'

        if col_name not in df_r1.columns or col_name not in df_r2.columns:
            print(f"Error: Metric column '{col_name}' is missing in DataFrame.")
            continue

        df_r1 = df_r1[['t', col_name]].rename(columns={col_name: 'value'})
        df_r2 = df_r2[['t', col_name]].rename(columns={col_name: 'value'})

        merged = pd.merge(df_r1, df_r2, on='t', suffixes=('_r1', '_r2'))

        # Calculate the difference
        merged[diff_column] = merged['value_r1'] - merged['value_r2']
        differences.append(merged[['t', diff_column']])

    if differences:
        return pd.concat(differences, ignore_index=True)
    else:
        print("No differences calculated.")
        return pd.DataFrame()
   
# Adjust phase differences to fit within a specified interval
def adjust_phase_interval(diff_df, start_interval, end_interval):
    """
    Adjusts phase difference values to fall within a specified interval range.
    This function modifies the phase difference values between consecutive radii in a DataFrame to ensure that they
    fall within a given interval range by wrapping values around if necessary. Values outside
    the range are set to `None` and rows containing such values are removed from the DataFrame.
    Parameters:
    diff_df = pd.DataFrame: DataFrame containing phase difference columns. Only columns with
    'diff' in their name are adjusted.
    start_interval = float: The lower bound of the desired interval (in degrees).
    end_interval = float: The upper bound of the desired interval (in degrees).
    Returns = pd.DataFrame: The adjusted DataFrame with phase differences wrapped within the specified
    interval, and rows with out-of-bounds values removed.
    """
    def adjust_value(value):
       """
       Adjusts a single phase difference value to ensure it falls within the specified interval.
       Mainly used for the below function.
       Parameters:
       value (float): The phase difference value to be adjusted. This is selected by the user.
       Returns = float or None: The adjusted value if within the interval, otherwise None.
       """
       # Wrap the value around if it's less than the start interval
       while value < start_interval:
          value += 360
       # Wrap the value around if it's greater than the end interval
       while value > end_interval:
          value -= 360
       # Check if the adjusted value is within the specified interval
       return value if start_interval <= value <= end_interval else None

    # Apply the adjustment function to each column with 'diff' in its name
    for col in diff_df.columns:
        if 'diff' in col:
            diff_df[col] = diff_df[col].apply(adjust_value)
    
    # Drop rows where any 'diff' column has a None value (out-of-bounds)
    return diff_df.dropna()

# Interactive plot function for phase between radii
def plot_differences(selected_r_values, selected_metric, adjusted=False):
    """
    Plots the differences for a specified metric between consecutive R values.
    This function creates a line plot showing the differences in a specified metric (e.g., 
    'height' or 'velocity') between consecutive R values. The plot will reflect either the 
    raw or adjusted differences based on the `adjusted` parameter.
    Parameters:
    selected_r_values = list of float: List of R values for which differences are calculated.
    selected_metric = str: The metric to plot differences for (e.g., 'height' or 'velocity').
    adjusted = bool: If True, plot the adjusted differences; otherwise, plot the raw differences.
    Returns = None: Displays the plot using Streamlit.
    """
    selected_r_values = sorted(selected_r_values)
    # Ensure consecutive selection
    if not all(selected_r_values[i] + 1 == selected_r_values[i + 1] for i in range(len(selected_r_values) - 1)):
        st.write("Please select consecutive R values.")
        return
