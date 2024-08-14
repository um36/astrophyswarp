import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import os
import imageio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from ipywidgets import interact, widgets


st.markdown("""
    <style>
    .main {
        background-color: #e6e6fa;  /* Light purple overall background */
        color: #4B0082;  /* Indigo text color for contrast */
    }
    .sidebar .sidebar-content {
        background-color: #f0e6f6;  /* Lighter shade for the sidebar */
    }
    .stButton>button {
        background-color: #ff69b4;  /* Pink button color */
        color: white;
        border-radius: 5px;
        border: 2px solid #ff1493;  /* Hot pink outline */
    }
    .stSelectbox>div>div>input, .stMultiSelect>div>div>input {
        background-color: #d6eaff;  /* Light blue background for select box input */
        color: #4B0082;  /* Indigo text color */
        border: 2px solid #4B0082;  /* Indigo border */
        border-radius: 5px;
    }
    .stSelectbox>div>div>button, .stMultiSelect>div>div>button {
        color: #4B0082;  /* Indigo text color */
    }
    .stSlider>div>div>div>input {
        background-color: #4169e1;  /* Royal blue for slider background */
    }
    .stCheckbox>div:first-child>div>div {
        background-color: #4169e1;  /* Royal blue for checkbox background */
        border-radius: 5px;
    }
    .stCheckbox>div:first-child>div>div>div {
        color: white;  /* White checkmark color */
    }
    .stMarkdown>div {
        color: #4B0082;  /* Indigo text for markdown */
    }
    .stSubheader, .stHeader, .stTitle, .stText {
        color: #4B0082;  /* Indigo text for headers and general text */
    }
    </style>
""", unsafe_allow_html=True)

# Define emojis
rocket_emoji = "🚀"
galaxy_emoji = "🌌"
space_emoji = "🪐"
star_emoji = "✨"
st.title(f"{galaxy_emoji}Understanding the Milky Ways Warp!{galaxy_emoji}")
full_df= pd.read_csv('all_data.tab')
full_df['phi']=full_df['phi']*10
#put tabs in for general graphs, curve fit, heatmaps
two_df= pd.read_csv('total_data_df.csv')

# Function to define the sine function
def sine_function(phi, A, C, D):
    return A * np.sin(np.deg2rad(phi + C)) + D

# Function to adjust amplitude and phase shift values
def adjust_amplitude_phase(df, amp_col, phase_col):
    # Adjust negative amplitudes
    df[phase_col] = np.where(df[amp_col] < 0, df[phase_col] + 180, df[phase_col])
    df[amp_col] = np.abs(df[amp_col])
    
    # Adjust phase shifts to be within [0, 360)
    df[phase_col] = df[phase_col] % 360
    return df

# Create four tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Documentation","Initial analysis","Heat Map", "Curve Fit","Model Over time","Phase difference","Custom Graph","Animation"])

with tab1:
    st.subheader(f"{star_emoji}Introduction{star_emoji}")
    with st.expander("This project focuses on using computer simulations of galaxy collisions to investigate whether the warp created in them has the same behaviour as we see in the Milky Way."):
        st.write("The Milky Way is a disc galaxy, but we can now see that the outer part of the disc is warped. Because the galaxy is so big, and changes so slowly, we cannot directly see how this warp is changing over time. Theoretical models lead us to expect that the warp is a long-lasting, stable structure in the galaxy, and is only moving very slowly. However, we can now measure the velocities of large numbers of stars in the warp, and these are not consistent with a warp having a stable shape. The warp appears to precess around the galaxy as fast as the stars themselves are moving! We need to understand why this is happening. One possibility is that the warp is not long lasting but is instead the response to a recent interaction with another Galaxy.")
    st.subheader(f"{star_emoji}How to Guide{star_emoji}")
    with st.expander("This section contains a brief description about the different features of this streamlit app and how to navigate across the various tabs to best make use of the resource."):
        st.markdown(
        """
        **Overview:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The tabs above all briefly contain an explanation with the context behind why that specific piece of analysis was carried out as well as how it links to our overall results within the project. These go in order from the initial plots showing a general picture to more in-depth comparisons as we progress.
        
        **User Input Features:**  
        &nbsp;&nbsp;&nbsp;&nbsp;These are referring to the select boxes that appear under each title. They are there for the user to be able to have the option to view the data in a more personalised way and is more convenient for filtering and sorting the data that we want to be presented. For example, selecting an R value of 8.5 will filter the dataset so that only information and rows with an 8.5 value in the radius column are displayed on the chosen graph.
        
        **Other Interactive Features:**  
        &nbsp;&nbsp;&nbsp;&nbsp;As well as the standard select boxes, this tool consists of various other interactive features too. Items such as sliders help enable the user to engage with the data better and understand how graphs change over time with the year slider. There are also sections of the tool where the user can input numbers to specify a set interval, so we can customise the best-suited axis for certain graphs to optimise the smoothness of the data. As a result, we can gain a better insight into what is happening with the warp.
        
        **Saving Features:**  
        &nbsp;&nbsp;&nbsp;&nbsp;Moreover, in addition to the adjusting features within certain graphs, there’s the ability to save the graph that’s created using the save graph button. This will save the graph as an image, which can then be shared amongst fellow researchers. Plus, each saved graph can be viewed in the custom graphs tab in more detail within the Streamlit app.
        
        Below you can find the frequently asked questions regarding the data, the results, and the code.
        """,
        unsafe_allow_html=True
    )
    st.subheader(f"{star_emoji}FAQs{star_emoji}")
    with st.expander("What work has been previously completed on this?"):
        st.write("Here is the link to relevant research papers!")
        st.markdown(
        """
        [The tangled warp of the Milky Way](https://ui.adsabs.harvard.edu/abs/2024A%26A...688A..38J/abstract)
 
        [The disturbed outer Milky Way disc](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4988M/abstract)
 
        [The Milky Way's rowdy neighbours](https://ui.adsabs.harvard.edu/abs/2024arXiv240712095S/abstract)

        [Place to find other papers](https://ui.adsabs.harvard.edu/)
        """
    )
    with st.expander("What Data has been used and how was it obtained?"):
        st.markdown(
        """
        The previous code I was working with revolved around creating a simulation of a Milky Way like galaxy interacting with a dwarf galaxy passing through. After this was made snapshots were taken from the simulation data at each point in time giving information about velocity and height of the particles in the galaxy for varying sections of the galaxy and how it was affected by the collision.
 
        
        (This data is from the Disturbed outer milky way disc paper and can be found in the previous work section.)
        """,
        unsafe_allow_html=True
    )
    with st.expander("Can you provide a data dictionary?"):
        st.markdown(
        """
        **Overview:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The data dictionary below contains a bried description of the columns within our dataset including: t, phi, R, N, Zmean and vZ_mean.
        
        **t:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The Giga years is billions of years, 0 is the start of simulation and the dwarf galaxy passes through after 0.1 Gyr.
        
        **phi:**  
        &nbsp;&nbsp;&nbsp;&nbsp;the position around the galaxy (values between 0 and 350 degrees)
        
        **R:**  
        &nbsp;&nbsp;&nbsp;&nbsp;the radius of the particle or the horizontal distance between the particle and the centre of the galaxy (values between 5.5 and 15.5 Kpc). For example 5.5 means between 5 and 6.
        
         **N:**  
        &nbsp;&nbsp;&nbsp;&nbsp;the number of particles in the bin (the group of particles that fall within a specific R and phi value)

         **Zmean:**  
        &nbsp;&nbsp;&nbsp;&nbsp;the average vertical height between the galactic disk and the particle in Kpc for each step.

         **VZ_mean:**  
        &nbsp;&nbsp;&nbsp;&nbsp;the vertical velocity of the particle in Km/s.

        """,
        unsafe_allow_html=True
    )
    with st.expander("How have the Sine curves been fitted?"):
        st.markdown(
        """
        The sine curves have been fitted by filtering down to a specific R value and year and by looping over all of these we plot phi values on the x axis (between 0 and 350 degrees) and then either height or velocity on the y axis. Once these data points are plotted then we use a package called curve fit to find the optimum parameters for our sine function. 

        A sine function has parameters A,B,C,D where the function is Asin(Bx+C)+D. However for this project I chose to only use A,C and D values without changing the frequency. The reason for this is because for a specific year and R value, the phi values range from 0 to 360, so if they frequency/period of the sine wave changed for each year, this wouldnt match up and be consistent overtime.
        
        Furthermore, I added an uncertainty factor when calculating the fit. This takes into account that the data might not be 100% accurate and was created by using a value 1/sqrt(N) where N is the number of particles in the bin.

        Overall, this is then what has been added to the original data frame as extra columns for both height and velocity as A_height, C_height, D_height, A_velocity, C_velocity, and D_velocity columns. This is where A is the Amplitude of the cuve, C is the phase shift and D is the vertical shift of the curve.
        """,
        unsafe_allow_html=True)
    with st.expander("How was the streamlit app made?"):
        st.write("This streamlit app was created by Urvi Modha and was made by referring to various documentation that is available here:")
        st.markdown(
        """
        [Streamlit Documentation](https://docs.streamlit.io)
 
        [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
 
        [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
        """
    )
with tab2:
    st.subheader(f"{star_emoji}Initial analysis overview{star_emoji}")
    st.write('This graph displays the variation of the selected column (Zmean or vZ_mean) over time for different radius values (R) at a specific phi value. You can choose a phi value from the dropdown menu, select which column you want to plot on the y-axis, and select multiple radius values to compare their trends on the same graph. Plotting this graph gives an idea of a general picture for one specific region in the galactic disc.')
    # Create columns
    col1, col2, col3 = st.columns(3)
    # Select phi value
    phi_values = (full_df['phi'].unique())
    # Select column to plot on y-axis
    column_options = ['Zmean', 'vZ_mean']
    # Select multiple r values
    r_values = full_df['R'].unique()
    with col1:
        selected_phi = st.selectbox('Select phi', phi_values)
    with col2:
        selected_column = st.selectbox('Select column to plot on y-axis', column_options)
    with col3:
        selected_r_values = st.multiselect('Select radius values to plot', r_values)

    if selected_phi is not None and selected_column and selected_r_values:
        # Filter the full_df dataframe based on selections
        filtered_df = full_df[(full_df['phi'] == selected_phi) & (full_df['R'].isin(selected_r_values))]

    # Plotting
        fig, ax = plt.subplots()

        for r_value in selected_r_values:
            subset = filtered_df[filtered_df['R'] == r_value]
            ax.plot(subset['t'], subset[selected_column], label=f'radius = {r_value}')

        ax.set_xlabel('time')
        ax.set_ylabel(selected_column)
        ax.legend()
        ax.set_title(f'{selected_column} over time for $\phi$ = {selected_phi}')

        st.pyplot(fig)

    st.subheader(f"{star_emoji}Initial analysis for specific point in time{star_emoji}")
    st.write('This graph displays the variation of the selected column (Zmean or vZ_mean) across all phi positions for a specific year and different radius values (R). You can choose a year from the dropdown menu, select which column you want to plot on the y-axis, and select multiple radius values to compare their patterns across positions.')
    # user input options
    # Create columns
    column1, column2, column3 = st.columns(3)
    # Select year value
    year_value = (full_df['t'].unique())
    with column1:
        selected_year = st.selectbox('select year', year_value)
    with column2:
        selected_column_t = st.selectbox('Select column to plot at a year', column_options)
    with column3:
        selected_r_values_t = st.multiselect('Select radius values to plot at a year', r_values)

    if selected_year is not None and selected_column_t and selected_r_values_t:
        # Filter the full_df dataframe based on selections
        filtered_df_t = full_df[(full_df['t'] == selected_year) & (full_df['R'].isin(selected_r_values_t))]

    # Plotting
        fig_two, ax = plt.subplots()

        for r_value_t in selected_r_values_t:
            subset_t = filtered_df_t[filtered_df_t['R'] == r_value_t]
            ax.plot(subset_t['phi'], subset_t[selected_column_t], label=f'radius = {r_value_t}')

        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(selected_column_t)
        ax.legend()
        ax.set_title(f'{selected_column_t} over all positions for year = {selected_year}')

        st.pyplot(fig_two)
    st.write('The above graph highlights a lot of fluctuation at the start, before 0.1. After this point theres a concentrated height and velocity that shifts with time.')

with tab3:
    st.subheader(f"{star_emoji}Heat Map Analysis{star_emoji}")
    st.write('This heatmap visualizes the distribution of the selected column (Zmean or vZ_mean) across different phi positions and radius values (R) for a specific year. You can select a year and a column to analyze, and the heatmap will display the variation in values across the 2D plane defined by phi and R.')
    colu1,colu2 = st.columns(2)
    year_value_h = (full_df['t'].unique())
    column_options_h = ['Zmean', 'vZ_mean']
    with colu1:
        selected_year_h = st.selectbox('select year for heatmap', year_value_h)
    with colu2:
        selected_column_h = st.selectbox('Select column to plot for heatmap', column_options_h)
    filtered_df_t_h = full_df[(full_df['t'] == selected_year_h)]

    # Prepare data for heatmap
    heatmap_data = filtered_df_t_h.pivot(index='R', columns='phi', values=selected_column_h)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', norm=Normalize(vmin=-np.max(np.abs(heatmap_data)), vmax=np.max(np.abs(heatmap_data))), ax=ax)

    ax.set_xlabel('$phi$ (Degrees)')
    ax.set_ylabel('Radius (kpc)')
    ax.set_title(f'{selected_column_h} Heatmap at Time = {selected_year_h:.2f}')

    # Show the heatmap in Streamlit
    st.pyplot(fig)
    st.subheader(f"{star_emoji}Heat Map Circular{star_emoji}")
    st.write('This circular distribution plot shows the spatial distribution of the selected column (Zmean or vZ_mean) across a 2D plane for a specific year. The plot translates phi and radius values (R) into Cartesian coordinates (X and Y), allowing you to visualize how the values are distributed in space.')

    # Plot Circular Distribution
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert polar to Cartesian
    filtered_df_t_h['x'] = filtered_df_t_h['R'] * np.cos(np.radians(filtered_df_t_h['phi']))
    filtered_df_t_h['y'] = filtered_df_t_h['R'] * np.sin(np.radians(filtered_df_t_h['phi']))

    # Adjust circle size
    max_radius = filtered_df_t_h['R'].max()
    circle_size = 2000 / max_radius  # Adjust 2000 as necessary to get the desired effect

    # Scatter plot for height
    sc = ax.scatter(filtered_df_t_h['x'], filtered_df_t_h['y'], c=filtered_df_t_h[selected_column_h], cmap='coolwarm', norm=Normalize(vmin=-np.max(np.abs(filtered_df_t_h[selected_column_h])), vmax=np.max(np.abs(filtered_df_t_h[selected_column_h]))), s=circle_size)
    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_title(f'{selected_column_h} Distribution at Time = {selected_year_h:.2f}')
    plt.colorbar(sc, ax=ax, label='Height')

    # Show the scatter plot in Streamlit
    st.pyplot(fig)

    st.write("Initially, the heat map shows a wide range of variability with no clear pattern in both height and velocity. After year 0.1, there is a noticeable shift, with height/velocity values becoming more positive and more negative, peaking at 0.6 in the phi range between 170 and 270 degrees (for height), and between 190 and 240 (for velocity). This increase is more pronounced at larger radii, while smaller radii near the center show height values approaching zero with less pronounced variation. Over time, the values shift across different phi regions, and the overall smoothness of the data decreases, indicating evolving patterns and potential changes in underlying processes.")

with tab4:
    st.subheader(f"{star_emoji} Curve fit analysis for $\phi$ {star_emoji}")
    st.write('This section provides a curve fit analysis for the selected column (height or velocity) at a specific year and radius value (R). You can choose a year, column, and radius value to analyse, and then compare the data against either a manual or automatic sine wave fit. Use the checkboxes to toggle between the manual fit, where you can adjust the amplitude, phase shift, and vertical shift, and the automatic fit, which uses optimised parameters to fit the data. The graph will display the selected data alongside the chosen fitting curve, allowing you to explore how well the model represents the underlying data.')
    # User input options
    colum1, colum2, colum3 = st.columns(3)
    year_value = two_df['t'].unique()
    time_min = year_value.min()
    time_max = year_value.max()
    with colum1:
        selected_year_c = st.selectbox('Select year for fit', year_value)
    with colum2:
        selected_column_t_c = st.selectbox('Select column to plot for fit', ['height', 'velocity'])
    with colum3:
        selected_r_values_t_c = st.selectbox('Select radius value to plot for fit', two_df['R'].unique())

    if selected_year_c is not None and selected_column_t_c and selected_r_values_t_c:
        # Filter the two_df dataframe based on selections
        filtered_df_t_c = two_df[(two_df['t'] == selected_year_c) & (two_df['R'] == selected_r_values_t_c)]

        if not filtered_df_t_c.empty:
            if selected_column_t_c == 'height':
                A_opt = filtered_df_t_c['A_height'].values[0]
                C_opt = filtered_df_t_c['C_height'].values[0]
                D_opt = filtered_df_t_c['D_height'].values[0]
                y_data = filtered_df_t_c['Zmean']
            elif selected_column_t_c == 'velocity':
                A_opt = filtered_df_t_c['A_velocity'].values[0]
                C_opt = filtered_df_t_c['C_velocity'].values[0]
                D_opt = filtered_df_t_c['D_velocity'].values[0]
                y_data = filtered_df_t_c['vZ_mean']

            x_data = filtered_df_t_c['phi']

            # Checkboxes for showing manual and automatic fitting
            show_manual_fit = st.checkbox("Show Manual Fit")
            show_auto_fit = st.checkbox("Show Automatic Fit")

            if show_manual_fit:
                # Sliders for manual adjustment of amplitude and phase shift
                amplitude = st.slider("Amplitude (A)", min_value=0.0, max_value=10.0, value=5.0, step=0.0001)
                phase_shift = st.slider("Phase Shift (C)", min_value=-180.0, max_value=180.0, value=0.0)
                vertical_shift = st.slider("Vertical Shift (D)", min_value=0.0, max_value=10.0, value=5.0, step=0.0001)
                # Create a sine wave based on user inputs
                y_manual = amplitude * np.sin(np.radians(x_data + phase_shift)) + vertical_shift
                # Plot manual fitting
                fig, ax = plt.subplots()
                ax.plot(filtered_df_t_c['phi'], y_data, label='Data', color='blue')
                ax.plot(x_data, y_manual, label='Manual Fit', color='orange', linestyle='--')
                ax.set_xlabel(r'$\phi$ (degrees)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)

            if show_auto_fit:
                # Create a sine wave with the optimized parameters
                y_auto = sine_function(x_data, A_opt, C_opt, D_opt)

                # Plot automatic fitting
                fig, ax = plt.subplots()
                ax.plot(x_data, y_data, label='Data', color='blue')
                ax.plot(x_data, y_auto, label='Automatic Fit', color='green', linestyle='--')
                ax.set_xlabel(r'$\phi$ (degrees)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)

                # Display the optimized parameters
                st.write(f"Optimized Amplitude (A): {A_opt:.2f}")
                st.write(f"Optimized Phase Shift (C): {C_opt:.2f} degrees")
                st.write(f"Optimized Vertical Shift (D): {D_opt:.2f}")
        else:
            st.write("No data found for the selected year and radius value.")
    st.write('In conclusion, the automatic sine wave fit aligns closely with the observed data (with uncertainty is smoother than without), confirming that the variables height and velocity follow a periodic pattern with the provided parameters A, C and D. Manual adjustments show how deviations in amplitude, phase shift, and vertical shift can impact the fit, indicating areas where the model may require further refinement. However on the whole, the optimum parameters are slighty better overall.')
with tab5:

    # Adjusting amplitude and phase shift values for height and velocity
    two_pos_df = adjust_amplitude_phase(two_df.copy(), 'A_height', 'C_height')
    two_pos_df = adjust_amplitude_phase(two_pos_df, 'A_velocity', 'C_velocity')

    # Function to plot the graph
    def plot_graph(df, title, ycol1, ycol2, ylabel1, ylabel2, phase_shift=False):
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax2 = None

        height_colors = plt.cm.cool(np.linspace(0.3, 0.7, len(selected_radii)))
        velocity_colors = plt.cm.gist_heat(np.linspace(0.3, 0.7, len(selected_radii)))

        for idx, radius in enumerate(selected_radii):
            if show_height:
                df_selected_r_h = df[df['R'] == radius]
                ax1.plot(df_selected_r_h['t'], df_selected_r_h[ycol1], 
                        label=f'Height R={radius}', linestyle='-', marker='o', color=height_colors[idx])

            if show_velocity:
                if ax2 is None:
                    ax2 = ax1.twinx()
                df_selected_r_v = df[df['R'] == radius]
                ax2.plot(df_selected_r_v['t'], df_selected_r_v[ycol2], 
                        label=f'Velocity R={radius}', linestyle='--', marker='x', color=velocity_colors[idx])

        ax1.set_ylabel(ylabel1, color='blue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)
        if ax2:
            ax2.set_ylabel(ylabel2, color='red', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
        plt.title(title, fontsize=16)

        # Handle legends
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), bbox_transform=ax1.transAxes)
        if ax2:
            ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9), bbox_transform=ax1.transAxes)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        st.pyplot(fig)
        return fig

    # Streamlit layout
    st.subheader(f"{star_emoji} Model across all years {star_emoji}")
    st.write('This section allows you to analyse the amplitude (A) or phase shift (C) of height and velocity data across all years for selected radius values (R). You can choose whether to display the amplitude or phase shift model, and select which radius values and data types (height and/or velocity) to include in the plot. With the resulting graph you can compare the amplitude or phase shift variations for both height and velocity overtime.')
    # User input options
    opt1, opt2, opt3, opt4 = st.columns(4)
    with opt1:     
        plot_type = st.selectbox("Choose the type of plot", ["Amplitude (A)", "Phase Shift (C)"])
    with opt2:
        selected_radii = st.multiselect("Select R values:", options=two_df['R'].unique())
    with opt3:
        show_velocity = st.checkbox("Show Velocity Data")
    with opt4:
        show_height = st.checkbox("Show Height Data")

    if not show_velocity and not show_height:
        st.write("Please select at least one data type (Velocity or Height) to plot.")
    else:
        if plot_type == "Amplitude (A)":
            st.subheader("Amplitude (A) Model across all years")
            st.write('The amplitude in the data represents the maximum deviation of particles from the galactic plane (for height) or the maximum velocity in the vertical direction (for velocity) at a given radius and time.')
            fig = plot_graph(two_pos_df, 'Amplitude (A) over Time', 'A_height', 'A_velocity', 'Amplitude (A)', 'Amplitude (A)')
            with st.expander('Amplitude height results'):
                st.markdown(
        """
        **What amplitude represents:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The amplitude of the galactic warp, represented by the height data , shows how pronounced the warp is. A higher amplitude means a greater deviation of stars and gas from the galactic plane. This helps us track how the warps strength changes over time, especially after events like a dwarf galaxy passing by.
        
        **Variation with Radius:**  
        &nbsp;&nbsp;&nbsp;&nbsp; The warps amplitude typically varies with distance from the galactic centre. Generally, the warp is more pronounced in the outer regions of the galaxy because they are less tightly bound and more affected by disturbances. By examining this variation, we can understand how different parts of the galaxy respond to the same disturbance. The pattern for overtime is consistent across radii as the general shape follows through with varying magnitudes as you get to the outer regions.
        
        **Evolution Over Time:**  
        &nbsp;&nbsp;&nbsp;&nbsp;Monitoring how amplitude changes at various radii over time can reveal if the warp is growing, decaying, or remaining stable. For instance, after a significant perturbation, the amplitude might initially increase and then gradually decrease if the warp is relaxing. Alternatively, a continuous increase might indicate a persistent or escalating warp. From the above graph after the dwarf galaxy interacts with the Milky Way like galaxy, an increase in amplitude at different radii is seen, which indicates that the warp is being affected by the gravitational disturbance. In this interaction we can see that the warps amplitude is fluctuating! Sometimes either decreasing, suggesting a return to stability, but also sometimes increasing, indicating that the warp is becoming more pronounced at certain points in time. On the whole this accentuates the fact that the warp is more complicated and more factors are at play. 
        
        """,
        unsafe_allow_html=True
    )
            with st.expander('Amplitude velocity results'):
                st.markdown(
        """
        **What amplitude represents:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The amplitude of vertical velocity reflects how fast the particles are moving to keep up with the warp. Higher velocity amplitudes indicate more vigorous motion within the warp, which helps us understand the dynamics and how it affected the warps evolution.
        
        **Variation with Radius:**  
        &nbsp;&nbsp;&nbsp;&nbsp; Similarly the amplitude for velocity shows similar patterns to height for different radii as the two are correlated this is expected.
        
        **Evolution Over Time:**  
        &nbsp;&nbsp;&nbsp;&nbsp;This initial spike after 0.1 Gyr indicates that the warp is not only being induced but that the particles within the galaxy are moving more to keep up with the warp in response to the gravitational interaction. After the initial perturbation, the velocity amplitude might either decrease, suggesting the galaxy is returning to a more stable state, or continue to grow, indicating that the warp's dynamic motion is escalating. Theres a combination of the two for outer radii suggesting these things at varied points in time.')
        
        """,
        unsafe_allow_html=True
    )
        elif plot_type == "Phase Shift (C)":
            st.subheader("Original Phase Shift (C) Model across all years")
            st.write('The phase shift of the galactic warps height data indicates the timing and distribution of vertical displacements from the galactic plane. The phase shift of the galactic warp’s velocity data reflects the timing and distribution of vertical velocities within the galactic disk. The combined data represents the curvature of particles within the warp at a given radius and time.')
            fig = plot_graph(two_pos_df, 'Phase Shift (C) over Time', 'C_height', 'C_velocity', 'Phase Shift (C)', 'Phase Shift (C)', phase_shift=True)

            st.subheader(f"{star_emoji}Adjust Phase Shift Interval{star_emoji}")
            st.write('After adjusting the phase shift interval, this graph shows the updated phase shift (C) for height and velocity data over time. By customising the start point of the 360-degree interval, we can fine-tune the phase shift to produce a smoother graph that better aligns with the data points. This adjustment helps in choosing the optimal interval that reveals the most consistent patterns in the phase shift model for the selected radius values.')
            start_point = st.number_input("Start Point for 360-degree Interval", value=0, step=1)

            # Function to adjust phase shifts based on a custom interval
            def adjust_phase_shifts(df, phase_col, start_point):
                df[phase_col] = (df[phase_col] - start_point) % 360 + start_point
                return df

            update_graph = st.checkbox('See phase shift updated Graph')
            if update_graph:
                adjusted_height_df = adjust_phase_shifts(two_pos_df.copy(), 'C_height', start_point)
                adjusted_velocity_df = adjust_phase_shifts(two_pos_df.copy(), 'C_velocity', start_point)
                st.subheader("Adjusted Phase Shift (C) Model")
                fig = plot_graph(pd.concat([adjusted_height_df, adjusted_velocity_df]), 'Adjusted Phase Shift (C) over Time', 'C_height', 'C_velocity', 'Phase Shift (C)', 'Phase Shift (C)', phase_shift=True)

            # Store the final figure in session state
            st.session_state['fig_final'] = fig
            with st.expander('Phase shift height and velocity results'):
                st.markdown(
        """
        **What phase shift represents:**  
        &nbsp;&nbsp;&nbsp;&nbsp;A phase shift shows how the peaks and troughs of the warp’s height are aligned with angular positions in the galaxy, as well as reveals how the speed of upward and downward motion aligns with angular positions.  Consistent phase shifts across different radii suggest uniform behaviour, while varying phase shifts indicate how the warp’s vertical displacement and velocity evolves or propagates over time. This helps us understand how the warp’s structure and alignment change, particularly in response to external perturbations like the passage of a dwarf galaxy.
        
        **Variation with Radius:**  
        &nbsp;&nbsp;&nbsp;&nbsp;The phase shift of height varies with distance from the galactic centre, showing a general pattern where the phase shift tends to take longer to sync at larger radii. The phase shifts of height at velocity at different radii show very similar trends however they slightly lag behind each other. This observation could be because the peak and curve of the warp has the potential to be different for each radius and the warp is also moving where the particles are moving to keep up and therefore the particles following it around take longer to get to the same point.
        
        **Evolution Over Time:**  
        &nbsp;&nbsp;&nbsp;&nbsp;Initially, phase shifts for both height and velocity fluctuate broadly (0-360 degrees) as the system adjusts to disturbances. Over time, these shifts stabilize to a narrower range apart from at outer radii. This suggests that the warp’s vertical displacement stabilizes after initial perturbations and that perhaps outer regions of the galaxy exhibit more pronounced and delayed responses compared to the inner regions.
        
        """,
        unsafe_allow_html=True
    )
        # Save graph as
        filename = st.text_input("Enter the filename without extension:", "graph")
        if st.button('Save Phase Shift Graph'):
            if 'fig_final' in st.session_state:
                if not os.path.exists('saved_graphs'):
                    os.makedirs('saved_graphs')
                file_path = os.path.join('saved_graphs', f'{filename}.png')
                st.session_state['fig_final'].savefig(file_path)
                st.success(f'Graph saved as {file_path}')
            else:
                st.error("No graph to save. Please create or update a graph first.")
with tab6:
    def calculate_phase_difference(combined_params_df, selected_R):
        # Filter combined parameters DataFrame by selected R value
        filtered_params = combined_params_df[combined_params_df['R'] == selected_R].copy()
        
        # Ensure the DataFrame is sorted by 't'
        filtered_params = filtered_params.sort_values(by='t')
        
        # Calculate the phase difference in 'C' values between height and velocity
        filtered_params['Phase_Difference'] = filtered_params['C_height'] - filtered_params['C_velocity']
        
        # Adjust for differences greater than 180 degrees or less than -180 degrees
        def adjust_phase_differencehv(diff):
            if diff > 180:
                return diff - 360
            elif diff < -180:
                return diff + 360
            else:
                return diff

        filtered_params['Phase_Difference'] = filtered_params['Phase_Difference'].apply(adjust_phase_differencehv)
        
        return filtered_params

    # Function to plot phase difference
    def plot_phase_difference(merged_df):
        plt.figure(figsize=(10, 5))
        plt.plot(merged_df['t'], merged_df['Phase_Difference'], marker='o', linestyle='-', color='b')
        plt.xlabel('Year')
        plt.ylabel('Phase Difference (degrees)')
        plt.title(f'Phase Difference Between Height and Velocity Over Time (Radius: {selected_R_pha})')
        plt.grid(True)
        st.pyplot(plt)

    st.subheader(f"{star_emoji} Difference in Phase shift H and V{star_emoji}")
    st.write('This analysis explores the phase shift difference between height and velocity for a selected radius (R) over time. The phase shift difference is calculated by subtracting the velocity phase shift from the height phase shift. Any differences outside the range of ±180 degrees are adjusted to maintain consistency. This visualisation helps in understanding how the phase relationships between height and velocity evolve over time for a specific radius, highlighting periods of synchronisation or divergence.')
    selected_R_pha = st.selectbox('Select Radius (R):', two_df['R'].unique())

    # Calculate phase difference
    merged_df = calculate_phase_difference(two_pos_df, selected_R_pha)
    # Plot phase difference
    plot_phase_difference(merged_df)
    st.write('With the above plot we are looking at points with the value of 90 degrees to be where velocity and height data are in sync. The reason for this is because the phase difference between velocity and displacement in simple harmonic motion is a quarter-cycle, or 90 degrees. Due to velocity being the derivative of displacement, and the derivative of a sine or cosine function is shifted by a quarter-cycle. Overall, from the graph all radii show at the beginning timeframe large phase shift differences as the velocity and height are still affected by the interaction with the dwarf galaxy and this disturbance causes them to be out of sync. In contrast some radii experience times where height and velocity difference is fluctuating around 90 or -90 degrees, however this time period varies for each radii without a set pattern.')

    st.subheader(f"{star_emoji}Phase Difference Between Radii{star_emoji}")
    st.write("This section focuses on the phase shift differences across consecutive radius values for either height or velocity. By analysing these differences, we can observe how phase shifts vary across different regions. This visualisation makes it easier to spot patterns or unusual changes in the system's behavior.")
    # Adjusting amplitude and phase shift values for height and velocity
    two_pos_df = adjust_amplitude_phase(two_df, 'A_height', 'C_height')
    two_pos_df = adjust_amplitude_phase(two_pos_df, 'A_velocity', 'C_velocity')

    # Get unique radii values from the DataFrame
    r_values = sorted(two_pos_df['R'].unique())

    # Calculate differences for consecutive R values
    def calculate_differences(df, selected_r_values, metric):
        differences = []
        for i in range(len(selected_r_values) - 1):
            r1 = selected_r_values[i]
            r2 = selected_r_values[i + 1]
            
            df_r1 = df[df['R'] == r1]
            df_r2 = df[df['R'] == r2]
            
            if df_r1.empty or df_r2.empty:
                continue
            
            diff_column = f'{metric}_diff_{r1}_{r2}'
            df_r1 = df_r1[['t', f'C_{metric}']].rename(columns={f'C_{metric}': 'value'})
            df_r2 = df_r2[['t', f'C_{metric}']].rename(columns={f'C_{metric}': 'value'})
            
            merged = pd.merge(df_r1, df_r2, on='t', suffixes=('_r1', '_r2'))
            merged[diff_column] = merged[f'value_r1'] - merged[f'value_r2']
            
            differences.append(merged[['t', diff_column]])

        return pd.concat(differences, ignore_index=True)

    # Adjust phase differences to fit within a specified interval
    def adjust_phase_interval(diff_df, start_interval, end_interval):
        def adjust_value(value):
            while value < start_interval:
                value += 360
            while value > end_interval:
                value -= 360
            return value if start_interval <= value <= end_interval else None
        
        for col in diff_df.columns:
            if 'diff' in col:  # Only apply to difference columns
                diff_df[col] = diff_df[col].apply(adjust_value)
        
        return diff_df.dropna()  # Remove rows where phase differences are out of bounds

    # Interactive plot function
    def plot_differences(selected_r_values, selected_metric, adjusted=False):
        selected_r_values = sorted(selected_r_values)
        
        # Ensure consecutive selection
        if not all(selected_r_values[i] + 1 == selected_r_values[i + 1] for i in range(len(selected_r_values) - 1)):
            st.write("Please select consecutive R values.")
            return
        
        # Calculate differences
        diff_df = calculate_differences(two_pos_df, selected_r_values, selected_metric)
        
        # Apply interval adjustment if required
        if adjusted:
            diff_df = adjust_phase_interval(diff_df, start_interval, end_interval)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i in range(len(selected_r_values) - 1):
            r1 = selected_r_values[i]
            r2 = selected_r_values[i + 1]
            column_name = f'{selected_metric}_diff_{r1}_{r2}'
            
            if column_name in diff_df.columns:
                ax.plot(diff_df['t'], diff_df[column_name], marker='o', label=f'{selected_metric.capitalize()} diff {r1}-{r2}')
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{selected_metric.capitalize()} Difference')
        ax.set_title(f'{selected_metric.capitalize()} Difference Over Time')
        ax.legend()
        st.pyplot(fig)  # Display plot in Streamlit
        st.write('For the graph above the phase difference between (5.5 and 6.5 etc) the inner radii have a difference close to zero. On the other hand, as you get further out this differnce has a slightly bigger range around zero and this point where it fluctuates around zero is significantly less for bigger radii.') 
               
    # Widgets for user interaction
    newercol1, newercol2 = st.columns(2)
    with newercol1:
        r_selector = st.multiselect(
            'Select R Values:',
            options=r_values  # Allow selection from 5.5 to 15.5
        )
    with newercol2:
        metric_selector = st.radio(
            'Select Metric:',
            options=['height', 'velocity'],
            index=0,
            horizontal=True  # Display options horizontally
        )

    # Check if any R values are selected
    if not r_selector:
        st.write("No R values selected, so data cannot be plotted.")
    else:
        # Plot the initial phase differences
        plot_differences(r_selector, metric_selector)
        
        # User input for interval adjustment
        st.write(f"{star_emoji}Adjust Phase Shift Interval{star_emoji}")
        st.write('In this section, users can adjust the phase shift interval to optimize the display of phase differences between radii. By customizing the start and end intervals, you can refine the graph to reveal smoother transitions and more coherent patterns in phase differences. This adjustment allows for better visualization of data points, ensuring that the observed trends are within a meaningful and interpretable range.')
        newestcol1, newestcol2= st.columns(2)
        with newestcol1:
            start_interval = st.number_input("Start Interval (degrees):", value=-50.0, step=1.0)
        with newestcol2:
            end_interval = st.number_input("End Interval (degrees):", value=50.0, step=1.0)
        update_graph = st.checkbox('See Updated Graph')
        if update_graph:
            plot_differences(r_selector, metric_selector, adjusted=True)

        # Save graph as
        

        filename = st.text_input("Enter the filename (without extension):", "graph")
        
        if st.button('Save Graph'):
            if update_graph:
                # Save the adjusted graph
                diff_df = calculate_differences(two_pos_df, r_selector, metric_selector)
                adjusted_df = adjust_phase_interval(diff_df, start_interval, end_interval)
                
                fig, ax = plt.subplots(figsize=(14, 8))
                for i in range(len(r_selector) - 1):
                    r1 = r_selector[i]
                    r2 = r_selector[i + 1]
                    column_name = f'{metric_selector}_diff_{r1}_{r2}'
                    
                    if column_name in adjusted_df.columns:
                        ax.plot(adjusted_df['t'], adjusted_df[column_name], marker='o', label=f'{metric_selector.capitalize()} diff {r1}-{r2}')
                
                ax.set_xlabel('Year')
                ax.set_ylabel(f'{metric_selector.capitalize()} Difference')
                ax.set_title(f'{metric_selector.capitalize()} Difference Over Time')
                ax.legend()
                file_path = os.path.join('saved_graphs', f'{filename}.png')
                plt.savefig(file_path)
                st.success(f"Graph saved as {filename}.png")
            else:
                st.error("No updated graph to save. Please update the graph first.")

with tab7:
    st.subheader(f"{star_emoji}Custom Graphs and results{star_emoji}")
    st.write('In this section, you can view and analyse custom graphs that you have previously saved. The saved graphs are displayed in expandable sections, each showing the graph image along with a text area where you can add your analysis or notes. This feature allows you to review your saved visualizations and document your observations or insights directly alongside the graphs.')
    # Assume you have saved graphs in a folder called 'saved_graphs'
    saved_graphs_folder = 'saved_graphs'

    # List all files in the saved_graphs folder
    if os.path.exists(saved_graphs_folder):
        files = os.listdir(saved_graphs_folder)
        
        if files:
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(saved_graphs_folder, file)
                    graph_name = os.path.splitext(file)[0]

                    with st.expander(f"Graph: {graph_name}"):
                        # Display the image
                        st.image(file_path, caption=graph_name)

                        # Text area for analysis
                        analysis = st.text_area(f"Analysis for {graph_name}", key=file)
                        st.write(analysis)
        else:
            st.info("No graphs saved yet.")
    else:
        st.warning(f"The folder '{saved_graphs_folder}' does not exist.")

with tab8:
    st.subheader(f"{star_emoji}Animation across all Years{star_emoji}")
    st.write("This section allows you to explore how a selected variable changes over time for a specific radius. By choosing a radius and a variable such as Zmean or vZ_mean, you can view an animation showing the evolution of the data across different years. The animation displays the variable's distribution and fitted curves for each year, helping you observe trends and variations over time. Use the slider to navigate through the years and analyze how the variable's behavior shifts.")
    # Adjusting amplitude and phase shift values for height and velocity
    two_pos_df = adjust_amplitude_phase(two_df, 'A_height', 'C_height')
    two_pos_df = adjust_amplitude_phase(two_df, 'A_velocity', 'C_velocity')

    # User options
    newcol1,newcol2 = st.columns(2)
    with newcol1:
        selected_R_ani = st.selectbox('Select Radius R:', two_df['R'].unique())
    with newcol2:
        variable = st.selectbox('Select column for animation:', ['Zmean', 'vZ_mean'])

    # Determine which parameters to use based on user selection
    if variable == 'Zmean':
        param_cols = {'A': 'A_height', 'C': 'C_height', 'D': 'D_height'}
    else:
        param_cols = {'A': 'A_velocity', 'C': 'C_velocity', 'D': 'D_velocity'}

    # Filter data based on selected R
    filtered_data_ani = two_pos_df[two_pos_df['R'] == selected_R_ani]
    # Get unique years for animation
    years = filtered_data_ani['t'].unique()
    # Initialize Plotly figure
    fig = make_subplots(rows=1, cols=1)

    # Initialize lists for y-values
    all_y_values = []

    # Add traces for each year
    for year in filtered_data_ani['t'].unique():
        frame_data = filtered_data_ani[filtered_data_ani['t'] == year]
        clean_data = frame_data.dropna(subset=['phi', variable])
        clean_data = clean_data[np.isfinite(clean_data[variable])]
        
        if clean_data.empty:
            continue

        all_y_values.extend(clean_data[variable].tolist())
        
        A_fitted = clean_data[param_cols['A']].values[0]
        C_fitted = clean_data[param_cols['C']].values[0]
        D_fitted = clean_data[param_cols['D']].values[0]

        phi_range = np.linspace(0, 360, 100)
        fitted_curve = sine_function(phi_range, A_fitted, C_fitted, D_fitted)
        all_y_values.extend(fitted_curve)
        
        fig.add_trace(
            go.Scatter(x=clean_data['phi'], y=clean_data[variable], mode='lines', name=f'Year {year}', visible=False)
        )
        
        fig.add_trace(
            go.Scatter(x=phi_range, y=fitted_curve, mode='lines', line=dict(dash='dash', color='red'), name=f'Fitted Year {year}', visible=False)
        )

    # Set the first trace visible
    if len(fig.data) > 0:
        fig.data[0].visible = True
        if len(fig.data) > 1:
            fig.data[1].visible = True

    # Create sliders and buttons for the animation
    steps = []
    for i in range(0, len(fig.data), 2):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": f"{variable} over phi (Year: {years[i//2]})"}],
        )
        step["args"][0]["visible"][i] = True
        if i + 1 < len(fig.data):
            step["args"][0]["visible"][i + 1] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    # Set consistent y-axis range based on all y-values
    global_min = min(all_y_values)
    global_max = max(all_y_values)

    fig.update_layout(
        sliders=sliders,
        title=f"{variable} over phi for Radius {selected_R_ani}",
        xaxis_title='Phi (degrees)',
        yaxis_title=variable,
        yaxis=dict(range=[global_min, global_max]),  # Set the y-axis range
        annotations=[
            dict(
                x=0.5,
                y=-0.2,
                xref='paper',
                yref='paper',
                text="",
                showarrow=False
            )
        ]
    )

    st.plotly_chart(fig)
    st.write('The analysis of the data for radii from 5.5 to 15.5 kpc from the galactic center reveals a progression in the behavior of ripples as the radius increases. For smaller radii (5.5 kpc), the ripples after the object passes are small in height and gradually shift from 230 to 300 degrees, with only minor changes in size. As the radius increases to 10.5 kpc and beyond, these ripples become more pronounced, growing in height and taking longer to stabilize. Particularly at 13.5 kpc, the ripples exhibit significant negative heights, peaking at -13.967 and taking longer to settle, a pattern that persists and intensifies at 14.5 and 15.5 kpc, where the maximum height reaches -16.178. These observations suggest that the warp in the galactic disk becomes more pronounced at larger radii, with increasing ripple height and longer periods before stabilization, indicating a stronger and more complex warp effect as one moves further from the center. This trend suggests that the outer regions of the galaxy experience more impact due to the interaction.')
