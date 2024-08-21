import numpy as np
# Define the sine function to fit, keeping the period consistent
def sine_function(x, A, C):
    return A * np.sin(np.radians(x + C))

# Create velocity_df and height_df
    velocity_df = full_df[['t', 'R', 'phi', 'vZ_mean']].copy()
    height_df = full_df[['t', 'R', 'phi', 'Zmean']].copy()

    # Columns to store amplitude (A) and phase shift (C)
    velocity_df['A'] = np.nan
    velocity_df['C'] = np.nan
    height_df['A'] = np.nan
    height_df['C'] = np.nan

    # Fit sine curve and extract parameters for each year and radius
    years = velocity_df['t'].unique()
    radii = velocity_df['R'].unique()

    for year in years:
        for radius in radii:
            # Filter data for the current year and radius
            df_velocity = velocity_df[(velocity_df['t'] == year) & (velocity_df['R'] == radius)]
            df_height = height_df[(height_df['t'] == year) & (height_df['R'] == radius)]
            
            # Ensure we have data to fit
            if len(df_velocity) >= 30:  # Assuming at least 30 points for a good fit
                phi_velocity = df_velocity['phi'].values
                v = df_velocity['vZ_mean'].values
                
                # Fit the sine curve for velocity
                try:
                    # Fit the sine function to the data
                    params, params_covariance = curve_fit(sine_function,df_velocity['phi'] , df_velocity['vZ_mean'], p0=[10, 0])
                    # Extract optimized parameters
                    A_opt, C_opt = params
                    velocity_df.loc[(velocity_df['t'] == year) & (velocity_df['R'] == radius), 'A'] = A_opt
                    velocity_df.loc[(velocity_df['t'] == year) & (velocity_df['R'] == radius), 'C'] = C_opt
                except RuntimeError:
                    print(f"Could not fit sine curve for velocity: t = {year}, r = {radius}")
            
            if len(df_height) >= 30:  # Assuming at least 36 points for a good fit
                phi_height = df_height['phi'].values
                z = df_height['Zmean'].values
                
                # Fit the sine curve for height
                try:
                    params_h, params_covariance = curve_fit(sine_function,df_height['phi'] , df_height['Zmean'], p0=[10, 0])
                    # Extract optimized parameters
                    A_opt_h, C_opt_h = params_h
                    height_df.loc[(height_df['t'] == year) & (height_df['R'] == radius), 'A'] = A_opt_h
                    height_df.loc[(height_df['t'] == year) & (height_df['R'] == radius), 'C'] = C_opt_h
                except RuntimeError:
                    print(f"Could not fit sine curve for height: t = {year}, r = {radius}")
        #The df should now have the sine optimum points in both and can plot based on user selection
        #selecting the R, either height or velocity, A or C to plot
     
    # Function to adjust phase shift values based on the sign of A
    def adjust_phase_shift(df):
        # Add 180 to C where A is negative
        df.loc[df['A'] < 0, 'C'] += 180
        #add 360 when c is negative
        df['C'] = np.where(df['C'] < 0, df['C'] + 360, df['C'])
     

    # Apply adjustment to both dataframes
    adjust_phase_shift(height_df)
    adjust_phase_shift(velocity_df)

    # Ensure amplitude values are all positive in height_df and velocity_df
    height_df['A'] = height_df['A'].abs()
    velocity_df['A'] = velocity_df['A'].abs()

    # Saving DataFrames to CSV files
    velocity_df.to_csv('velocity_data.csv', index=False)
    height_df.to_csv('height_data.csv', index=False)
    # Read from CSV
    velocity_df = pd.read_csv('velocity_data.csv')
    height_df = pd.read_csv('height_data.csv')
########after reading it 
    # Adjust the phase shift C in height_df for degrees
    height_df['C'] = (height_df['C'] + 180) % 360 - 180
    # Adjust the phase shift C in velocity_df for degrees
    velocity_df['C'] = (velocity_df['C'] + 180) % 360 - 180

    def adjust_phase_shifts(df):
        adjusted_c = [df['C'].iloc[0]]  # Start with the first value
        
        for i in range(1, len(df)):
            current_c = df['C'].iloc[i]
            previous_c = adjusted_c[-1]

            # Calculate possible adjusted versions of current_c
            options = [
                current_c,
                current_c + 360,
                current_c - 360,
                current_c + 720,
                current_c - 720
            ]
            
            # Select the option that results in the smallest difference with previous_c
            best_c = min(options, key=lambda x: abs(x - previous_c))
            adjusted_c.append(best_c)
        
        df['C'] = adjusted_c
        return df

    # Apply the function to both dataframes
    height_df = adjust_phase_shifts(height_df)
    velocity_df = adjust_phase_shifts(velocity_df)

##################################lmfit model code
    # Define the sine function to fit, keeping the period consistent
def sine_function(phi, A, C, D):
    return A * np.sin(np.radians(phi) + np.radians(C)) + D

# Assuming full_df is already defined
velocity_df = full_df[['t', 'R', 'phi', 'vZ_mean']].copy()
height_df = full_df[['t', 'R', 'phi', 'Zmean']].copy()

# Columns to store amplitude (A) and phase shift (C)
velocity_df['A'] = np.nan
velocity_df['C'] = np.nan
velocity_df['D'] = np.nan  # Include offset D
height_df['A'] = np.nan
height_df['C'] = np.nan
height_df['D'] = np.nan  # Include offset D

# Fit sine curve and extract parameters for each year and radius
years = velocity_df['t'].unique()
radii = velocity_df['R'].unique()

for year in years:
    for radius in radii:
        # Filter data for the current year and radius
        df_velocity = velocity_df[(velocity_df['t'] == year) & (velocity_df['R'] == radius)]
        df_height = height_df[(height_df['t'] == year) & (height_df['R'] == radius)]
        
        # Ensure we have data to fit
        if len(df_velocity) >= 30:
            phi_velocity = df_velocity['phi'].values
            v = df_velocity['vZ_mean'].values
            
            # Fit the sine curve for velocity
            try:
                model = Model(sine_function)
                params = model.make_params(A=10, C=0, D=np.mean(v))
                result = model.fit(v, params, phi=phi_velocity)
                
                # Extract optimized parameters
                A_opt = result.params['A'].value
                C_opt = result.params['C'].value
                D_opt = result.params['D'].value
                velocity_df.loc[(velocity_df['t'] == year) & (velocity_df['R'] == radius), 'A'] = A_opt
                velocity_df.loc[(velocity_df['t'] == year) & (velocity_df['R'] == radius), 'C'] = C_opt
                velocity_df.loc[(velocity_df['t'] == year) & (velocity_df['R'] == radius), 'D'] = D_opt
            except Exception as e:
                print(f"Could not fit sine curve for velocity: t = {year}, r = {radius}, error: {e}")
        
        if len(df_height) >= 30:
            phi_height = df_height['phi'].values
            z = df_height['Zmean'].values
            
            # Fit the sine curve for height
            try:
                model = Model(sine_function)
                params = model.make_params(A=10, C=0, D=np.mean(z))
                result = model.fit(z, params, phi=phi_height)
                
                # Extract optimized parameters
                A_opt_h = result.params['A'].value
                C_opt_h = result.params['C'].value
                D_opt_h = result.params['D'].value
                height_df.loc[(height_df['t'] == year) & (height_df['R'] == radius), 'A'] = A_opt_h
                height_df.loc[(height_df['t'] == year) & (height_df['R'] == radius), 'C'] = C_opt_h
                height_df.loc[(height_df['t'] == year) & (height_df['R'] == radius), 'D'] = D_opt_h
            except Exception as e:
                print(f"Could not fit sine curve for height: t = {year}, r = {radius}, error: {e}")

# Function to adjust phase shift values based on the sign of A
def adjust_phase_shift(df):
    # Add 180 to C where A is negative
    df.loc[df['A'] < 0, 'C'] += 180
    df['A'] = df['A'].abs()
    # Normalize C to the range [0, 360)
    df['C'] = np.mod(df['C'], 360)

# Apply adjustment to both dataframes
adjust_phase_shift(height_df)
adjust_phase_shift(velocity_df)

# Ensure smooth phase transition
def adjust_phase_shifts(df):
    adjusted_c = [df['C'].iloc[0]]  # Start with the first value
    
    for i in range(1, len(df)):
        current_c = df['C'].iloc[i]
        previous_c = adjusted_c[-1]

        # Calculate possible adjusted versions of current_c
        options = [
            current_c,
            current_c + 360,
            current_c - 360
        ]
        
        # Select the option that results in the smallest difference with previous_c
        best_c = min(options, key=lambda x: abs(x - previous_c))
        adjusted_c.append(best_c)
    
    df['C'] = adjusted_c
    return df

# Apply the function to both dataframes
height_df = adjust_phase_shifts(height_df)
velocity_df = adjust_phase_shifts(velocity_df)


################################################################
#function for sine curve and uncertanty 
    # Function to define the sine function
        def sine_function(phi, A, C, D):
            return A * np.sin(np.deg2rad(phi + C)) + D

        # Function to fit sine curve and return parameters
        def fit_sine_curve(df, variable):
            results = []
            years = df['t'].unique()
            radii = df['R'].unique()

            for year in years:
                for radius in radii:
                    frame_data = df[(df['t'] == year) & (df['R'] == radius)]
                    clean_data = frame_data.dropna(subset=['phi', variable])
                    clean_data = clean_data[np.isfinite(clean_data[variable])]

                    if clean_data.empty:
                        continue

                    initial_A = clean_data['A'].iloc[0] if 'A' in clean_data.columns else 1
                    initial_C = clean_data['C'].iloc[0] if 'C' in clean_data.columns else 0
                    initial_D = clean_data[variable].mean()
                    # Calculate uncertainties as 1/sqrt(N)
                    #N = clean_data['N']  
                    #uncertainties = 1 / np.sqrt(N)

                    try:
                        #popt, _ = curve_fit(sine_function, clean_data['phi'], clean_data[variable], p0=[initial_A, initial_C, initial_D], sigma=uncertainties, absolute_sigma=True)
                        popt, _ = curve_fit(sine_function, clean_data['phi'], clean_data[variable], p0=[initial_A, initial_C, initial_D])
                    except RuntimeError:
                        continue

                    A_fitted, C_fitted, D_fitted = popt
                    results.append({
                        't': year,
                        'R': radius,
                        'A': A_fitted,
                        'C': C_fitted,
                        'D': D_fitted
                    })

            return pd.DataFrame(results)

        # Fit sine curves for height and velocity for all years and radii
        height_params = fit_sine_curve(full_df, 'Zmean')
        velocity_params = fit_sine_curve(full_df, 'vZ_mean')

        # Merge the height and velocity parameters into a single DataFrame
        combined_params_df = pd.merge(height_params, velocity_params, on=['t', 'R'], suffixes=('_height', '_velocity'))

        # Create a new DataFrame that includes the original data along with the fitted parameters
        total_data_without_uncertainty_df = full_df.merge(combined_params_df, on=['t', 'R'], how='left')

        # Display the new DataFrame
        st.write(total_data_without_uncertainty_df.head(20))
        # Save the new DataFrame to a CSV file
        # csv_file_path = 'total_data_without_uncertainty_df.csv'
        # total_data_without_uncertainty_df.to_csv(csv_file_path, index=False)
