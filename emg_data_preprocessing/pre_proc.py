
# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################################################

# Create function to read and reorder csv file
def csv_file(file_path):
    
    # Read the csv file into a dataframe
    df = pd.read_csv(file_path, delimiter=',')

    # Set the column names
    df.columns = ['Trapezium', 'Stimulation', 'Deltoid', 'Triceps','Biceps', 'ECU', 'ECR', 'FCR', 'APL', 'FCU', 'Right Bicep', 'OM', 'FDI', 'Right Deltoid', 'OP', 'ED']

     # Add a new column 'time' where its value is equal to the number of rows 
    df['time'] = range(1, len(df) + 1)
    
    # Set the column order with Predictions column at the end
    column_order = ['time', 'Trapezium', 'Deltoid', 'Triceps','Biceps', 'ECU', 'ECR', 'FCR', 'APL', 'FCU', 'Right Bicep', 'OM', 'FDI', 'Right Deltoid', 'OP', 'ED', 'Stimulation']
    # Reorder the columns
    df = df[column_order]
    
    return df
    
# Plot time vs prediction signal
def plot_pred(df):
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['Stimulation'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Stimulation')
    ax.set_title('Stimulation vs Time')
    plt.show()
    
# Get column stats 
def get_column_stats(df, col_name):
    col_min = df[col_name].min()
    col_max = df[col_name].max()
    col_mean = df[col_name].mean()
    
    return (col_min, col_max, col_mean)

# Apply moving average smoothing to a column
def smooth_column(df, column_name, window_size):
    # Create a new column with the smoothed values
    df[column_name] = df[column_name].rolling(window_size, center=True, min_periods=1).mean()
    return df

# Apply smoothing
def threshold(x, mean):
    
    # Set the threshold
    if x < mean:
        return 0
    else:
        return 1

#####################################################################################################################
def run():
    # Call txt file
    df = csv_file('/Users/ariasarch/Desktop/Book1.csv')
    
    # Plot the stim
    plot_pred(df)
    
    # # Apply moving average smoothing to the 'Stimulation' column
    # df = smooth_column(df, 'Stimulation', 10)
    
    # Get stats
    col_min, col_max, col_mean = get_column_stats(df, 'Stimulation')
    
    # Plot the stim
    plot_pred(df)
    
    # Apply smoothing
    df['Stimulation'] = df['Stimulation'].apply(threshold, mean = df['Stimulation'].mean())
    
    # Plot the new stim
    plot_pred(df)
    
    # Drop the time column
    df = df.drop('time', axis=1)
    
    # # Set the column order with Predictions column at the end
    # column_order = ['ECR', 'Biceps', 'ED', 'ECU', 'Stimulation']
    
    # # Reorder the columns
    # df = df[column_order]
    
    # Save modified DataFrame as a new CSV file
    df.to_csv('/Users/ariasarch/Desktop/preprocessed_emg_data_round2.csv', index=False)