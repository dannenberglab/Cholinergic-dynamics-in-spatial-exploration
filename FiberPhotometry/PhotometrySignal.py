import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signalsci
import tdt
from filterpy.kalman import KalmanFilter
from scipy.optimize import minimize
from scipy.signal import resample
from scipy.stats import pearsonr
import pandas as pd
from scipy import stats
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from matplotlib.patches import Polygon
##################### Old Version
def calculate_dftof_oldversion(filepath, sr_analysis):
    # calculate df to f
    # Parameters
    # sampling rate after downsampling; chosen to be 30 to approximately match video sampling rate
    # (sampling rate of behavior)
    # Initialize data struct
    data = {
        'time': [],
        'signal': [],
        'TTL': [],
        'controlSignal': []
    }
    # Get list of .dat files in the specified directory
    file_list = [file for file in os.listdir(filepath) if file.endswith('.dat')]
    # import fiber photometry data from data.dat files to a data dictionary including: time, signal, TTL, controlsignal
    for file_name in file_list:
        temp = np.loadtxt(str(filepath) + "/" + str(file_name), skiprows=10)
        data['time'] = np.concatenate((data['time'], temp[:, 0]))
        data['signal'] = np.concatenate((data['signal'], temp[:, 1]))
        data['TTL'] = np.concatenate((data['TTL'], temp[:, 2]))
        data['controlSignal'] = np.concatenate((data['controlSignal'], temp[:, 3]))
        # sort according to time; that corrects for temporal sequence errors
        # caused by loading the data in the wrong sequence (e.g. data_10 is
        # loaded before data_1)
        sort_vec = np.argsort(data['time'])
        data['time'] = data['time'][sort_vec]
        data['signal'] = data['signal'][sort_vec]
        data['TTL'] = data['TTL'][sort_vec]
        data['controlSignal'] = data['controlSignal'][sort_vec]

    # get sampling rate
    sr = 1 / (data['time'][1] - data['time'][0])
    # Extract start point of video recording
    peaks, locs = signalsci.find_peaks(data['TTL'], height=2)
    startloc = int(peaks[0])
    # startTime = data['time'][startLoc]

    # Convert to arrays if necessary
    signal = np.array(data['signal'][startloc:])
    controlsignal = np.array(data['controlSignal'][startloc:])

    # Resample signals to 30 Hz
    signal = resample(signal, int(len(signal) * sr_analysis / sr) + 1)
    controlsignal = resample(controlsignal, int(len(controlsignal) * sr_analysis / sr) + 1)

    # Replace first and last 0.5 seconds of data with future/past values
    signal[:15] = signal[15]
    signal[-15:] = signal[(len(signal) - 1)]
    controlsignal[:15] = controlsignal[15]
    controlsignal[-15:] = controlsignal[(len(controlsignal) - 1)]

    # time axis
    time = np.ones(len(signal))
    for i in range(1, len(time) + 1):
        time[i - 1] = i * (1 / sr_analysis) - (1 / sr_analysis)

    ## data processing
    ## adjust for different change in bleaching
    diffbleach = signal - controlsignal
    # Fit a polynomial of degree 2 to the time and diffBleach data
    coefficients = np.polyfit(time, diffbleach, 2)
    # Calculate the result of the fitted model using time
    diffbleach_model_result = np.polyval(coefficients, time)
    # Corrected controlSignal
    correctedControlSignal = controlsignal + diffbleach_model_result

    # Define the objective function
    def objective_func(x):
        a = x[0]
        b = x[1]
        return np.sum((signal - (correctedControlSignal * a + b)) ** 2)

    # Set initial values
    x0 = [1, 0]
    # Solve the optimization problem
    sol = minimize(objective_func, x0)
    # Get the optimized values
    a = sol.x[0]
    b = sol.x[1]
    # Calculate the scaled controlSignal
    scaledControlSignal = correctedControlSignal * a + b

    ## compute change in fluorescence normalized to scaled control signal
    ## (deltaF/F)
    dfof = (signal - scaledControlSignal) / scaledControlSignal

    # dFoF = dFoF[onset_num_frame + 1:onset_num_frame + 1 + exp_window]
    # mainsignal = signal[onset_num_frame + 1:onset_num_frame + 1 + exp_window]
    # controlSignal = controlSignal[onset_num_frame + 1:onset_num_frame + 1 + exp_window]
    # scaledControlSignal = scaledControlSignal[onset_num_frame + 1:onset_num_frame + 1 + exp_window]

    plt.plot(time, signal, label='resampled GCaMP signal')
    plt.plot(time, controlsignal, label='resampled isosbestic control signal')
    plt.plot(time, scaledControlSignal, label='scaledControlSignal')
    plt.xlabel('Time (units)')
    plt.legend()
    plt.show()

    plt.plot(time, dfof)
    plt.xlabel('Time (s)')
    plt.ylabel('\u0394'+'F/F')
    plt.show()

    return dfof, signal, controlsignal, scaledControlSignal

##################### TDT system
def load_TDT_data(file_path, sampling_rate,start_of_recording,end_of_recording):
    data = tdt.read_block(file_path, t1=start_of_recording, t2=end_of_recording)
    # loads a .mat file containing fiber photometry data recorded with the TDT system
    fs = data.streams._405A.fs  # sampling rate
    print(fs)
    x465 = data.streams._465A.data # GCaMP signal (465 nm)
    x405 = data.streams._405A.data  # isosbestic control signal (405 nm)

    # Resample signals to fsp of the video
    resampled_x465 = resample(x465, int(len(x465)*sampling_rate/fs)+1)
    resampled_x405 = resample(x405, int(len(x405)*sampling_rate/fs)+1)

    # compute the time-line from the sampling rate
    time=[]
    for i in np.arange(0, len(resampled_x405), 1):
        time.append(i / sampling_rate)

    # smoothed_1s_x405 = pd.Series(x405).rolling(window=int(fs)).mean()
    # smoothed_1s_x465 = pd.Series(x465).rolling(window=int(fs)).mean()
    # # Replace first and last 1 seconds of data with future/past values
    # one_s=int(fs)
    # smoothed_1s_x405[:one_s] = smoothed_1s_x405[one_s]
    # smoothed_1s_x405[-one_s:] = smoothed_1s_x405[(len(smoothed_1s_x405)-1)]
    # smoothed_1s_x465[:one_s] = smoothed_1s_x465[one_s]
    # smoothed_1s_x465[-one_s:] = smoothed_1s_x465[(len(smoothed_1s_x465)-1)]

    # plt.plot(time, resampled_x405, 'red', label='resampled isosbestic control signal')
    # plt.plot(time, resampled_x465, 'blue', label='resampled GCaMP signal')
    # plt.legend()
    # plt.show()

    signal_tuple = (resampled_x465, resampled_x405, sampling_rate, time)
    return signal_tuple

def calculate_dftof_TDT(file_path, sampling_rate,start_of_recording,end_of_recording):

    x465, x405, fs, time = load_TDT_data(file_path, sampling_rate,start_of_recording,end_of_recording)
    #adjust for different change in bleaching
    diffBleach = x465 - x405
    # Fit a polynomial of degree 2 to the time and diffBleach data
    coefficients = np.polyfit(time, diffBleach, 2)
    # Calculate the result of the fitted model using time
    diffBleach_model_result = np.polyval(coefficients, time)
    # Corrected controlSignal
    correctedControlSignal = x405 + diffBleach_model_result

    #compute the parameters for optimal scaling of the corrected control
    # signal to correct for differences in fluorescence intensity/LED power
    # between GCaMP signal and isosbestic control signal
    def objective_func(x):
        a = x[0]
        b = x[1]
        return np.sum((x465 - (correctedControlSignal * a + b)) ** 2)

    # Set initial values
    x0 = [1, 0]
    # Solve the optimization problem
    sol =minimize(objective_func, x0)
    # Get the optimized values
    a = sol.x[0]
    b = sol.x[1]
    # Calculate the scaled controlSignal
    scaledControlSignal = correctedControlSignal * a + b

    dFoF = (x465 - scaledControlSignal) / scaledControlSignal

    plt.plot(time, x465, label='resampled GCaMP signal')
    plt.plot(time, scaledControlSignal, label='scaledControlSignal')
    plt.xlabel('Time (units)')
    plt.legend()
    plt.show()

    plt.plot(time, dFoF)
    plt.xlabel('Time (s)')
    plt.ylabel('\u0394'+'F/F')
    plt.legend()
    plt.show()

    return dFoF, scaledControlSignal

########## calculate speed
#####################
###################
################## remove this part later
#### extract a column from csv file
def calculate_speed(x_values, y_values, time_interval):
    speed_values = []

    for i in range(1, len(x_values)):
        x_diff = x_values[i] - x_values[i - 1]
        y_diff = y_values[i] - y_values[i - 1]
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
        speed = distance / time_interval
        speed_values.append(speed)

    return speed_values
def normalize_signal(signal):
    min_sig = min(signal)
    signal_1 = [sig - min_sig for sig in signal]
    max_sig = max(signal_1)
    normalized_signal = [sig / max_sig for sig in signal_1]
    return normalized_signal


def extract_column(csv_file, column_index):
    column_vector = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:
                column_vector.append((row[column_index]))

    return column_vector
def smooth_signal(signal, window_size,sr_analysis):
    smoothed_signal = []
    window_size=(window_size*sr_analysis)
    half_window = int(window_size // 2)

    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        average = sum(signal[start:end]) / (end - start)
        smoothed_signal.append(average)

    return np.array(smoothed_signal)

#### multiply vector with a float number
def multiply_vector(vector, factor):
    result = []
    for cell in vector:
        result.append((cell) * factor)
    return result
##########
def apply_kalman_filter(x, y, timestep):
    num_points = len(x)

    # Create the Kalman filter
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Define the state transition matrix (dynamic model)
    dt = timestep
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Define the measurement matrix (measurement model)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Define the covariance matrices
    kf.P *= 1000  # Initial state covariance
    kf.R = np.diag([0.1, 0.1])  # Measurement noise covariance
    kf.Q = np.eye(4)  # Process noise covariance

    # Initialize the state vector
    kf.x = np.array([x[0], y[0], 0, 0])

    speeds = [0] * num_points

    for i in range(1, num_points):
        # Predict the state and covariance
        kf.predict()

        # Update the state and covariance using the current measurement
        kf.update(np.array([[x[i], y[i]]]))

        # Extract the estimated position and speed
        estimated_state = kf.x
        speed = math.sqrt(estimated_state[2]**2 + estimated_state[3]**2)
        speeds[i] = speed

    return speeds

def calculate_two_kalman_speed(x, y, dt):
    # Initialize the forward Kalman filter
    forward_filter = KalmanFilter(dim_x=2, dim_z=2)
    forward_filter.F = np.array([[1., dt],
                                 [0., 1.]])  # State transition matrix
    forward_filter.H = np.array([[1., 0.],
                                 [0., 1.]])  # Measurement function
    forward_filter.Q = np.eye(2) * 0.1  # Process noise covariance
    forward_filter.R = np.eye(2) * 0.1  # Measurement noise covariance

    # Initialize the backward Kalman filter
    backward_filter = KalmanFilter(dim_x=2, dim_z=2)
    backward_filter.F = np.array([[1., -dt],
                                  [0., 1.]])  # State transition matrix
    backward_filter.H = np.array([[1., 0.],
                                  [0., 1.]])  # Measurement function
    backward_filter.Q = np.eye(2) * 0.1  # Process noise covariance
    backward_filter.R = np.eye(2) * 0.1  # Measurement noise covariance

    # Run the forward filter
    forward_filter.x = np.array([x[0], y[0]])  # Initial state
    forward_speeds = []
    for i in range(1, len(x)):
        forward_filter.predict()
        forward_filter.update(np.array([x[i], y[i]]))
        forward_speeds.append(forward_filter.x[1])

    # Run the backward filter
    backward_filter.x = np.array([x[-1], y[-1]])  # Initial state
    backward_speeds = []
    for i in range(len(x) - 2, -1, -1):
        backward_filter.predict()
        backward_filter.update(np.array([x[i], y[i]]))
        backward_speeds.append(backward_filter.x[1])

    # Reverse the backward speeds to match the chronological order
    backward_speeds.reverse()

    # Combine the forward and backward speeds with weighted average
    forward_weight = 0.5  # Weight for forward velocities
    backward_weight = 0.5  # Weight for backward velocities

    # Calculate the weighted average of forward and backward speeds
    speeds = []
    for forward_speed, backward_speed in zip(forward_speeds, backward_speeds):
        combined_speed = (forward_weight * forward_speed) + (backward_weight * backward_speed)
        speeds.append(combined_speed)

    return speeds
########################
def speed_analysis(csv_file, sr_analysis, ind1):
    ### read deeplabcut csv file

    nose_x = extract_column(csv_file, ind1)
    nose_y = extract_column(csv_file, ind1+1)
    nose_likelihood = extract_column(csv_file, 3)

    nose_x = list(map(float, nose_x[3:]))
    nose_y = list(map(float, nose_y[3:]))
    print(nose_x[0],nose_y[0])
    nose_likelihood = list(map(float, nose_likelihood[3:]))

    x = smooth_signal(nose_x, 0.5, sr_analysis)
    y = smooth_signal(nose_y, 0.5, sr_analysis)
    likelihood = nose_likelihood
    ### define scale for coordination
    x_pixels_per_meter = 875  # number of pixels per 100cm in x-dimension
    y_pixels_per_meter = 875  # number of pixels per 100cm in y-dimension
    # pixel_aspect_ratio = y_pixels_per_meter/x_pixels_per_meter
    scaleFactor = 100 / y_pixels_per_meter  # y coordinates because x coordinates will be corrected
    x = np.array(x)
    y = np.array(y)
    likelihood = np.array(likelihood)

    for t in range(1, len(x)):
        if np.isnan(x[t]):
            x[t] = x[t - 1]
        if np.isnan(y[t]):
            y[t] = y[t - 1]

    ######### calculate the speed

    speeds = calculate_speed(x, y, (1 / sr_analysis))

    speeds = multiply_vector(speeds, scaleFactor)
    speeds = np.array(speeds)
    for t in range(1, len(speeds)):
        if speeds[t] > 200:
            speeds[t] = np.mean(speeds[t - 10:t - 1])
        if np.isnan(speeds[t]) or np.isinf(speeds[t]):
            speeds[t] = speeds[t - 1]
    ######################################
    kalman_speeds = apply_kalman_filter(x, y, (1 / sr_analysis))
    kalman_speeds = multiply_vector(kalman_speeds, scaleFactor)
    kalman_speeds = np.array(kalman_speeds)
    for t in range(1, len(kalman_speeds)):
        if kalman_speeds[t] > 200:
            kalman_speeds[t] = np.mean(kalman_speeds[t - 10:t - 1])
        if np.isnan(kalman_speeds[t]) or np.isinf(kalman_speeds[t]):
            kalman_speeds[t] = kalman_speeds[t - 1]

    kalman_speeds = kalman_speeds[0:len(kalman_speeds) - 1]

    ####################################

    twokalman_speeds = calculate_two_kalman_speed(x, y, (1 / sr_analysis))
    twokalman_speeds = multiply_vector(twokalman_speeds, scaleFactor)
    twokalman_speeds = np.array(twokalman_speeds)

    for t in range(1, len(twokalman_speeds)):
        if twokalman_speeds[t] > 200:
            twokalman_speeds[t] = np.mean(twokalman_speeds[t - 10:t - 1])
        if np.isnan(twokalman_speeds[t]) or np.isinf(twokalman_speeds[t]):
            twokalman_speeds[t] = twokalman_speeds[t - 1]

    ####################################

    time_vec = np.ones(len(speeds))
    for i in range(1, len(time_vec) + 1):
        time_vec[i - 1] = i * (1 / sr_analysis) - (1 / sr_analysis)
    return time_vec, speeds, kalman_speeds, twokalman_speeds

#######
def speed_analysis_new():
    ### read deeplabcut csv file
    ind1=4
    sr_analysis=10
    multi_x = float(40 / 1100)  # pixel to cm
    multi_y= float(40 / 1100)
    # path="E:/lab/Cholinergic Prj/Data/30/"
    # Full_GCaMP7sChAT_611915_210426_Rec1_15min_Learning
    # Full_GCaMP7sChAT_611915_210426_Rec2_15min_ObjectExploration_Recall
    # Full_GCaMP7sChAT_611915_210430_Rec1_15min_ObjectExploration_Learning
    # Full_GCaMP7sChAT_611915_210430_Rec2_15min_ObjectExploration_Recall
    # Full_GCaMP7sChAT_611916_210503_Rec1_Light_15min_objectLocationMemory_Learning
    # Full_GCaMP7sChAT_611916_210503_Rec2_Light_15min_objectLocationMemory_Recall
    # Full_GCaMP7sChAT_611916_210508_Rec1_Light_objectLocationMemory_learning
    # Full_GCaMP7sChAT_611916_210508_Rec2_Light_objectLocationMemory_recall
    # Full_GCaMP7sChAT_611926_210531_Rec1_Light_15min_ObjectLocationMemory_Learning
    # Full_GCaMP7sChAT_611926_210531_Rec2_Light_15min_ObjectLocationMemory_Recall
    # Full_GCaMP7sChAT_611926_210604_Rec1_Light_15min_ObjectLocationMemory_Learning
    # Full_GCaMP7sChAT_611926_210604_Rec2_Light_15min_ObjectLocationMemory_Recall

    # Full_FrodoMouse_230505_134344_cropped_6928_26585_edited_learning
    # Full_FrodoMouse_230505_134344_cropped_98786_120890_edited_recall
    # Full_LuminousMouse_230405_153254_cropped_6299_24880_edited_learning
    # Full_LuminousMouse_230405_153254_cropped_97817_116618_edited_recall
    # Full_LuminousMouse_230411_110626_cropped_9383_28356_edited_learning
    # Full_LuminousMouse_230411_110626_cropped_100954_120670_edited_recall

    # name_exp="Full_GCaMP7sChAT_611926_210604_Rec1_Light_15min_ObjectLocationMemory_Learning"
    # csv_file=path + name_exp + "/" + name_exp + ".csv"
    csv_file="E:/lab/Cholinergic Prj/Data/Patrick/final_pat_test/Patrick_RecallDLC_resnet50_pat_dlcJul29shuffle1_100000_filtered.csv"
    body_x = extract_column(csv_file, ind1)
    body_y = extract_column(csv_file, ind1+1)
    body_likelihood = extract_column(csv_file, ind1+2)

    body_x = list(map(float, body_x[3:]))
    body_y = list(map(float, body_y[3:]))
    body_likelihood = list(map(float, body_likelihood[3:]))

    # video_file=path + name_exp + "/" + name_exp +".mp4"
    video_file="E:/lab/Cholinergic Prj/Data/Patrick/final_pat_test/Patrick_Recall.mp4"
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    print(len(frame))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title("First Frame")
    plt.show()

    # ######### calculate the speed

    x = np.array(body_x) * multi_x
    y = np.array(body_y) * multi_y
    body_likelihood = np.array(body_likelihood, dtype=float)
    for j in range(1, len(x)):
        if body_likelihood[j] < 0.6:
            x[j] = np.nan
            y[j] = np.nan

    ### interpolate NaNs
    df = pd.DataFrame({'x': x, 'y': y})
    # Interpolate linearly over NaN values
    df_interpolated = df.interpolate(method='linear')
    x = df_interpolated['x'].values
    y = df_interpolated['y'].values

    speeds = calculate_speed(x, y, (1 / sr_analysis))
    for t in range(1, len(speeds)):
        if speeds[t] > 150:
            speeds[t] = np.nan
    df_s = pd.DataFrame({'speed': speeds})
    speeds = df_s.interpolate(method='linear')['speed'].values

    # ######################################
    kalman_speeds = apply_kalman_filter(x, y, (1 / sr_analysis))
    kalman_speeds = np.array(kalman_speeds)
    # for t in range(1, len(kalman_speeds)):
    #     if kalman_speeds[t] > 200:
    #         kalman_speeds[t] = np.mean(kalman_speeds[t - 10:t - 1])
    kalman_speeds = kalman_speeds[0:len(kalman_speeds) - 1]

    #################### two kalman
    dt=1/sr_analysis
    transition_matrix = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

    # We only observe position: [x, y]
    observation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    # Process noise covariance
    process_covariance = np.eye(4) * 1 #Smaller values = smoother, less responsive filter. Larger values = more responsive but noisier estimates.

    # Measurement noise covariance
    # Adjust this based on known measurement noise characteristics
    observation_covariance = np.eye(2) * 0.001 #Smaller values mean more accurate measurements, so the filter puts more weight on the incoming data.

    # Initial guess for the state mean: we know initial x,y, assume vx, vy = 0
    initial_state_mean = [x[0], 0.0, y[0], 0.0]

    # Initial covariance: somewhat uncertain initial velocity
    initial_state_covariance = np.eye(4) * 1 #If we are pretty sure about the initial position (because it’s measured), we give it a low uncertainty. If we have no clue about the initial velocity, we give it a higher uncertainty.

    # Create KalmanFilter instance
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=process_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )

    # Observations: stack x and y into Nx2
    observations = np.column_stack([x, y])

    # Run smoothing (two-pass: forward + backward)
    smoothed_state_means, smoothed_state_covariances = kf.smooth(observations)

    # Extract velocities from smoothed states
    vx = smoothed_state_means[:, 1]
    vy = smoothed_state_means[:, 3]

    # Compute speed
    speed = np.sqrt(vx ** 2 + vy ** 2)
    #############

    # plt.plot(speeds[5500:5800])
    # plt.plot(speed[5500:5800])
    # plt.show()
    # plt.plot(speeds[5500:5800])
    # plt.plot(kalman_speeds[5500:5800])
    # plt.show()
    # #
    # plt.plot(speeds)
    # plt.show()
    # plt.plot(speed)
    # plt.show()
    # plt.plot(kalman_speeds)
    # plt.show()
    # sr_analysis=30
    # speeds=resample(speeds, int(len(speeds) * sr_analysis / 10) + 1)
    # speed=resample(kalman_speeds, int(len(speed) * sr_analysis / 10) + 1)[0:len(speeds)]
    # kalman_speeds=resample(kalman_speeds, int(len(kalman_speeds) * sr_analysis / 10) + 1)
    # print(len(speeds),len(speed),len(kalman_speeds))
    df = pd.DataFrame({
        'pure_speed': speeds,
        'twoKal_speed': speed,
        'kal_speed': kalman_speeds
    })
    # Save them all into one CSV with headers and no index
    # name_exp = "Patrick_Recall"
    # df.to_csv(path + name_exp + 'combined_speeds.csv', index=False)
    df.to_csv("E:/lab/Cholinergic Prj/Data/Patrick" + name_exp + 'combined_speeds.csv', index=False)

    # # # # ## plot
    path_df="C:/Users/ffarokhi/Desktop/BlancaData/only_patrick/"
    path_df = path_df + name_exp + "/" + name_exp + "_data.csv"
    df = pd.read_csv(path_df)
    dfof=np.array(df['dfof'])
    dfof=smooth_signal(dfof, 1, sr_analysis)
    print(len(dfof),len(speeds))
    main_speed=np.array(speed)
    # main_speed=smooth_signal(main_speed, 1, sr_analysis)
    log_speed = np.log2(main_speed + 0.01)
    if np.isnan(log_speed).any() or np.isinf(log_speed).any():
        df_log = pd.DataFrame({'log_speed': log_speed})
        log_speed = df_log.interpolate(method='linear')['log_speed'].values
    time_vec = np.ones(len(speeds))
    for i in range(1, len(time_vec) + 1):
        time_vec[i - 1] = i * (1 / sr_analysis) - (1 / sr_analysis)
    # Rlog, f1 = show_speed_dfof(time_vec, dfof, 1, log_speed, sr_analysis)
    Rlog, f1 = show_speed_dfof(time_vec[1:], dfof, 1, log_speed[1:], sr_analysis)
    # Rlog, f1 = show_speed_dfof(time_vec[15:], dfof[:len(dfof)-15], 1, log_speed[15:], sr_analysis)
    plt.show()

def show_logspeed_dfof(time_vec,dfof,smooth_time_window,input_speed,sr_analysis):
    ## find min as offset for log

    smooth_speed=smooth_signal(input_speed, smooth_time_window,sr_analysis)
    unique_speed = np.unique(smooth_speed)
    sorted_speed = np.sort(unique_speed)
    offset = sorted_speed[1]
    log2_smoothspeed = np.log2(smooth_speed + offset)

    # unique_speed = np.unique(input_speed)
    # sorted_speed = np.sort(unique_speed)
    # offset = sorted_speed[1]
    # log2_speed= np.log2(input_speed + offset)
    #
    # print(log2_speed)
    ### smooth signals
    smoothed_dfof=smooth_signal(dfof, smooth_time_window,sr_analysis)

    f1, (s1, s2, s3) = plt.subplots(3, 1, figsize=(8, 10))

    s1.plot(time_vec, log2_smoothspeed, linewidth=1, color='k')
    s1.plot(time_vec, smooth_signal(log2_smoothspeed, smooth_time_window,sr_analysis), linewidth=2, color=[0, 0.4470, 0.7410])
    s1.set_title(f'{smooth_time_window}-sec smoothed Log running speed')
    s1.set_ylabel('Log Running speed (cm/s)')
    s1.set_xticklabels([])
    s1.tick_params(direction='out')

    s2.plot(time_vec, dfof, linewidth=1, color='k')
    s2.plot(time_vec, smoothed_dfof, linewidth=2, color=[0.8500, 0.3250, 0.0980])
    s2.set_title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    s2.set_ylabel('dF/F (a.u.)')
    s2.set_xticklabels([])
    s2.tick_params(direction='out')

    s3.plot(time_vec, normalize_signal(log2_smoothspeed), linewidth=2, color=[0, 0.4470, 0.7410])
    s3.plot(time_vec, normalize_signal(smoothed_dfof), linewidth=2, color=[0.8500, 0.3250, 0.0980])
    s3.set_title('Superimposition')
    s3.set_xlabel('Time (s)')
    s3.set_ylabel('Normalized data (a.u.)')
    s3.tick_params(direction='out')

    ####### corr log
    # Compute Pearson's R
    Rlog, _ = pearsonr(log2_smoothspeed, smoothed_dfof)
    dim = [.75, .2, .1, .1]
    str = f'Rlog = {Rlog:.2f}'
    s3.text(dim[0], dim[1], str, transform=s3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)

    return Rlog,f1

def show_speed_dfof(time_vec, dfof, speed, smooth_time_window, sr_analysis, log):

    ## smoothing signals
    window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
    kernel = np.ones(window_samples) / window_samples
    smoothed_speed = np.convolve(speed, kernel, mode='same')
    smoothed_dfof = np.convolve(dfof, kernel, mode='same')


   ## measuring the log of speed
    if log: # log=1/0
        log_speed = np.log2(smoothed_speed + 0.01)
        if np.isnan(log_speed).any() or np.isinf(log_speed).any():
            ## interpolate NanN
            df_log = pd.DataFrame({'log_speed': log_speed})
            log_speed = df_log.interpolate(method='linear')['log_speed'].values
        smoothed_speed=log_speed

    ## plotting
    f1, (s1, s2, s3) = plt.subplots(3, 1, figsize=(8, 10))

    s1.plot(time_vec, smoothed_speed, linewidth=1, color='#666869')
    s1.set_title(f'{smooth_time_window}-sec smoothed running speed')
    s1.set_ylabel('Running speed (cm/s)')
    s1.set_xticklabels([])
    s1.tick_params(direction='out')

    s2.plot(time_vec, smoothed_dfof, linewidth=1 ,color='#38843F') # ,color=[0.8500, 0.3250, 0.0980]
    s2.set_title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    s2.set_ylabel(r'$\Delta$'+'F/F')
    s2.set_xticklabels([])
    s2.tick_params(direction='out')

    s3.plot(time_vec, stats.zscore(smoothed_speed), linewidth=1, color='#666869')
    s3.plot(time_vec, stats.zscore(smoothed_dfof), linewidth=1.1,color='#38843F')
    s3.set_title('Superimposition')
    s3.set_xlabel('Time (s)')
    s3.set_ylabel('Z-score')
    s3.tick_params(direction='out')

    ####### corr log
    # Compute Pearson's R
    R, _ = pearsonr(smoothed_speed, smoothed_dfof)
    dim = [.87, .1, .1, .1]
    str = f'R = {R:.2f}'
    s3.text(dim[0], dim[1], str, transform=s3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)

    return R,f1



def exponential_func(t, a, b, c):
    return a * np.exp(b * t) + c

def fit_exp(sig, fps_video):
    # Fit the exponential model to the data

    time_values = np.linspace(0, len(sig) - 1, len(sig))
    p0 = (1, -1 / (fps_video * 30), 1)  # Initial guesses for a, b, c
    # min_b = -1 / (fps_video)
    max_b=-1 / (fps_video*5*60)
    popt, pcov = curve_fit(exponential_func, time_values, sig, p0=p0, bounds=([0, -np.inf, -np.inf], [np.inf, max_b, np.inf]), maxfev=10000)

    # Get the optimized parameters
    a_opt, b_opt, c_opt = popt

    # Generate the fitted curve
    fitted_signal = exponential_func(time_values, *popt)

    # Print the optimized parameters
    print(f"Fitted parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")
    tau = (-1 / b_opt)/ fps_video
    print(f"Time constant (tau): {tau}")

    return fitted_signal,tau

def show_speed_dfof_decays_signle(time_vec, signals, smooth_time_window, sr_analysis):
    dfof = stats.zscore(signals[0])
    corrected_dfof = stats.zscore(signals[1])
    predicted_dfof = stats.zscore(signals[2])
    input_speed = (signals[3])

    # smoothing signals
    window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
    kernel = np.ones(window_samples) / window_samples

    smooth_speed = np.convolve(input_speed, kernel, mode='same')
    fitted_speed, tau_speed = fit_exp(input_speed, sr_analysis)

    smoothed_dfof = np.convolve(dfof, kernel, mode='same')
    fitted_dfof, tau_dfof = fit_exp(dfof, sr_analysis)

    smoothed_corrected_dfof = np.convolve(corrected_dfof, kernel, mode='same')
    fitted_corrected_dfof, tau_corrected_dfof = fit_exp(corrected_dfof, sr_analysis)

    smoothed_predicted_dfof = np.convolve(predicted_dfof, kernel, mode='same')
    fitted_predicted_dfof, tau_predicted_dfof = fit_exp(predicted_dfof, sr_analysis)


    # z_speed = stats.zscore(input_speed)
    # z_dfof = stats.zscore(dfof)
    # zfitted_speed, ztau_speed = fit_exp(z_speed, sr_analysis)
    # zfitted_dfof, ztau_dfof = fit_exp(z_dfof, sr_analysis)

    f1, (s1, s2, s3) = plt.subplots(3, 1, figsize=(8, 10))

    # s1.plot(time_vec, input_speed, linewidth=1, color='k')
    s1.plot(time_vec, smooth_speed, linewidth=1, color='#666869')
    s1.plot(time_vec, fitted_speed, color='#666869', linewidth=3)
    s1.set_title(f'{smooth_time_window}-sec smoothed (speed)')
    s1.set_ylabel('(speed) (cm/s)')
    s1.set_xticklabels([])
    s1.text(350, 4, f'(speed) τ = {tau_speed:.2f}' + " (s)", fontsize=12, color='black')
    s1.tick_params(direction='out')

    s2.plot(time_vec, smoothed_dfof, linewidth=1, color='#109D49')
    s2.plot(time_vec, fitted_dfof, color='green', linewidth=3)
    s2.text(350, 2, f'(zscore dfof) τ = {tau_dfof:.2f}' + " (s)", fontsize=12, color='green')

    s2.plot(time_vec, smoothed_predicted_dfof, linewidth=1, color='#515254')
    s2.plot(time_vec, fitted_predicted_dfof, color='black', linewidth=3)
    s2.text(350, 1.2, f'predict(logspeed) τ = {tau_predicted_dfof:.2f}' + " (s)", fontsize=12, color='black')

    s2.set_title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    s2.set_ylabel(r'$\Delta$' + 'F/F')
    s2.set_xticklabels([])
    s2.tick_params(direction='out')

    # s2.set_ylim([-0.05, 0.073])
    # s2.set_yticks([-0.04, -0.02, 0.0, 0.02, 0.04,0.06])

    s3.plot(time_vec, smoothed_corrected_dfof, linewidth=1, color='#90CB81')
    s3.plot(time_vec, fitted_corrected_dfof, linewidth=3, color='#6ABD45')
    s3.text(350, 2, f'(zscoredfof-(a*logspeed+b)) τ = {tau_corrected_dfof:.2f}' + " (s)", fontsize=12,
            color='#6ABD45')
    s3.set_title('Superimposition')
    s3.set_xlabel('Time (s)')
    s3.set_ylabel('Z-score')
    s3.tick_params(direction='out')

    # Rasterize the figure before saving
    for obj in f1.get_children():
        obj.set_rasterized(True)

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)
    # plt.show()

    return tau_speed,tau_dfof,tau_corrected_dfof,tau_predicted_dfof
def show_speed_dfof_decays(time_vec, signals, sems, smooth_time_window, sr_analysis):

    dfof=signals[0]
    corrected_dfof=signals[1]
    predicted_dfof=signals[2]
    input_speed=signals[3]

    dfof_sem=sems[0]
    corrected_dfof_sem=sems[1]
    predicted_dfof_sem=sems[2]
    speed_sem=sems[3]

    # smoothing signals
    window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
    kernel = np.ones(window_samples) / window_samples

    smooth_speed = np.convolve(input_speed, kernel, mode='same')
    speed_sem = np.convolve(speed_sem, kernel, mode='same')
    fitted_speed, tau_speed = fit_exp(input_speed, sr_analysis)

    smoothed_dfof = np.convolve(dfof, kernel, mode='same')
    dfof_sem = np.convolve(dfof_sem, kernel, mode='same')
    fitted_dfof, tau_dfof = fit_exp(dfof, sr_analysis)

    smoothed_corrected_dfof = np.convolve(corrected_dfof, kernel, mode='same')
    corrected_dfof_sem = np.convolve(corrected_dfof_sem, kernel, mode='same')
    fitted_corrected_dfof, tau_corrected_dfof = fit_exp(corrected_dfof, sr_analysis)

    smoothed_predicted_dfof = np.convolve(predicted_dfof, kernel, mode='same')
    predicted_dfof_sem = np.convolve(predicted_dfof_sem, kernel, mode='same')
    fitted_predicted_dfof, tau_predicted_dfof = fit_exp(predicted_dfof, sr_analysis)


    f1, (s1, s2, s3) = plt.subplots(3, 1, figsize=(8, 10))

    # s1.plot(time_vec, input_speed, linewidth=1, color='k')
    s1.fill_between(time_vec, smooth_speed - speed_sem, smooth_speed + speed_sem, color='lightgray')
    s1.plot(time_vec, smooth_speed, linewidth=1, color='#666869')
    s1.plot(time_vec, fitted_speed, color='#666869', linewidth=3)
    s1.set_title(f'{smooth_time_window}-sec smoothed speed')
    s1.set_ylabel('(speed) (cm/s)')
    s1.set_xticklabels([])
    s1.text(350, 4, f'avg(speed) τ = {tau_speed:.2f}' + " (s)", fontsize=12, color='black')
    s1.tick_params(direction='out')

    s2.fill_between(time_vec, smoothed_dfof - dfof_sem, smoothed_dfof + dfof_sem, color='#C4E5D8')
    s2.plot(time_vec, smoothed_dfof, linewidth=1 ,color='#109D49')
    s2.plot(time_vec,fitted_dfof, color='green', linewidth=3)
    s2.text(350, 2, f'avg(zscore dfof) τ = {tau_dfof:.2f}' + " (s)", fontsize=12, color='green')

    s2.fill_between(time_vec, smoothed_predicted_dfof - predicted_dfof_sem,
                    smoothed_predicted_dfof + predicted_dfof_sem, color='#8F9194')
    s2.plot(time_vec, smoothed_predicted_dfof, linewidth=1, color='#515254')
    s2.plot(time_vec, fitted_predicted_dfof, color='black', linewidth=3)
    s2.text(350, 1.2, f'predict(logspeed) τ = {tau_predicted_dfof:.2f}' + " (s)", fontsize=12, color='black')

    s2.set_title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    s2.set_ylabel(r'$\Delta$'+'F/F')
    s2.set_xticklabels([])
    s2.tick_params(direction='out')

    s3.fill_between(time_vec, smoothed_corrected_dfof - corrected_dfof_sem,
                    smoothed_corrected_dfof + corrected_dfof_sem, color='#DBECCB')
    s3.plot(time_vec, smoothed_corrected_dfof, linewidth=1, color='#90CB81')
    s3.plot(time_vec, fitted_corrected_dfof, linewidth=3, color='#6ABD45')
    s3.text(350, 2, f'avg(zscoredfof-(a*logspeed+b)) τ = {tau_corrected_dfof:.2f}' + " (s)", fontsize=12, color='#6ABD45')
    s3.set_title('Superimposition')
    s3.set_xlabel('Time (s)')
    s3.set_ylabel('Z-score')
    s3.tick_params(direction='out')

    ####### corr log
    # Compute Pearson's R
    # Rlog, _ = pearsonr(input_speed, dfof)
    # dim = [.87, .1, .1, .1]
    # str = f'R = {Rlog:.2f}'
    # s3.text(dim[0], dim[1], str, transform=s3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)
    # plt.show()

    #f=plt.figure(figsize=(10, 5))
    # plt.plot(time_vec,z_speed,label='zscore(1s_smoothed_dfof)',color="gray")
    # plt.plot(time_vec,z_dfof,label='zscore(1s_smoothed_speed)',color="green")
    # plt.plot(time_vec, zfitted_speed, linewidth=4, color='black')
    # plt.plot(time_vec, zfitted_dfof, linewidth=4, color='blue')
    # plt.xlabel('t')
    # plt.text(100, 3.5, f'τ = {ztau_dfof:.2f}' + " (s)", fontsize=16, color='blue')
    # plt.text(300, 3.5, f'τ = {ztau_speed:.2f}' + " (s)", fontsize=16, color='black')
    # plt.legend()
    # plt.ylim([-4, 4])
    # plt.title("tau for zscore (dfof/speed) - not smoothed ")
    return 0

def show_speed_dfof_decays_seperate(time_vec, signals, sems, smooth_time_window, sr_analysis,path):

    dfof=signals[0]
    corrected_dfof=signals[1]
    predicted_dfof=signals[2]
    input_speed=signals[3]

    dfof_sem=sems[0]
    corrected_dfof_sem=sems[1]
    predicted_dfof_sem=sems[2]
    speed_sem=sems[3]

    # smoothing signals
    window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
    kernel = np.ones(window_samples) / window_samples

    smooth_speed = np.convolve(input_speed, kernel, mode='same')
    speed_sem = np.convolve(speed_sem, kernel, mode='same')
    fitted_speed, tau_speed = fit_exp(input_speed, sr_analysis)

    smoothed_dfof = np.convolve(dfof, kernel, mode='same')
    dfof_sem = np.convolve(dfof_sem, kernel, mode='same')
    fitted_dfof, tau_dfof = fit_exp(dfof, sr_analysis)

    smoothed_corrected_dfof = np.convolve(corrected_dfof, kernel, mode='same')
    corrected_dfof_sem = np.convolve(corrected_dfof_sem, kernel, mode='same')
    fitted_corrected_dfof, tau_corrected_dfof = fit_exp(corrected_dfof, sr_analysis)

    smoothed_predicted_dfof = np.convolve(predicted_dfof, kernel, mode='same')
    predicted_dfof_sem = np.convolve(predicted_dfof_sem, kernel, mode='same')
    fitted_predicted_dfof, tau_predicted_dfof = fit_exp(predicted_dfof, sr_analysis)

    # Plotting the first subplot separately
    f1 = plt.figure(figsize=(8, 4))
    plt.fill_between(time_vec, smooth_speed - speed_sem, smooth_speed + speed_sem, color='lightgray')
    # plt.plot(time_vec, smooth_speed, linewidth=1, color='#666869')
    # plt.plot(time_vec, fitted_speed, color='#666869', linewidth=3)
    # plt.title(f'{smooth_time_window}-sec smoothed speed')
    # plt.ylabel('(speed) (cm/s)')
    # plt.text(350, 4, f'avg(speed) τ = {tau_speed:.2f}' + " (s)", fontsize=12, color='black')
    # plt.tick_params(direction='out')
    # plt.tight_layout()
    plt.savefig(path + "decays1_avg.svg", format="svg")
    plt.show()

    # Plotting the second subplot separately
    f2 = plt.figure(figsize=(8, 4))
    plt.fill_between(time_vec, smoothed_dfof - dfof_sem, smoothed_dfof + dfof_sem, color='#C4E5D8')
    # plt.plot(time_vec, smoothed_dfof, linewidth=1, color='#109D49')
    # plt.plot(time_vec, fitted_dfof, color='green', linewidth=3)
    # plt.text(350, 2, f'avg(zscore dfof) τ = {tau_dfof:.2f}' + " (s)", fontsize=12, color='green')

    plt.fill_between(time_vec, smoothed_predicted_dfof - predicted_dfof_sem,
                     smoothed_predicted_dfof + predicted_dfof_sem, color='#8F9194')
    # plt.plot(time_vec, smoothed_predicted_dfof, linewidth=1, color='#515254')
    # plt.plot(time_vec, fitted_predicted_dfof, color='black', linewidth=3)
    # plt.text(350, 1.2, f'predict(logspeed) τ = {tau_predicted_dfof:.2f}' + " (s)", fontsize=12, color='black')
    #
    # plt.title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    # plt.ylabel(r'$\Delta$' + 'F/F')
    # plt.tick_params(direction='out')
    # plt.tight_layout()
    plt.savefig(path + "decays2_avg.svg", format="svg")
    plt.show()

    # Plotting the third subplot separately
    f3 = plt.figure(figsize=(8, 4))
    plt.fill_between(time_vec, smoothed_corrected_dfof - corrected_dfof_sem,
                     smoothed_corrected_dfof + corrected_dfof_sem, color='#DBECCB')
    # plt.plot(time_vec, smoothed_corrected_dfof, linewidth=1, color='#90CB81')
    # plt.plot(time_vec, fitted_corrected_dfof, linewidth=3, color='#6ABD45')
    # plt.text(350, 2, f'avg(zscoredfof-(a*logspeed+b)) τ = {tau_corrected_dfof:.2f}' + " (s)", fontsize=12,
    #          color='#6ABD45')

    # plt.title('Superimposition')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Z-score')
    plt.tick_params(direction='out')
    plt.tight_layout()
    plt.savefig(path + "decays3_avg.svg", format="svg")
    plt.show()


def show_speed_dfof_decays_temp(time_vec, signals, sems, smooth_time_window, sr_analysis):

    dfof=signals[0]
    corrected_dfof=signals[1]
    predicted_dfof=signals[2]
    input_speed=signals[3]

    dfof_sem=sems[0]
    corrected_dfof_sem=sems[1]
    predicted_dfof_sem=sems[2]
    speed_sem=sems[3]




    smooth_speed = smooth_signal(input_speed, smooth_time_window, sr_analysis)
    speed_sem = smooth_signal(speed_sem, smooth_time_window, sr_analysis)
    fitted_speed, tau_speed = fit_exp(input_speed, sr_analysis)

    smoothed_dfof = smooth_signal(dfof, smooth_time_window, sr_analysis)
    dfof_sem = smooth_signal(dfof_sem, smooth_time_window, sr_analysis)
    fitted_dfof, tau_dfof = fit_exp(dfof, sr_analysis)

    smoothed_corrected_dfof = smooth_signal(corrected_dfof, smooth_time_window, sr_analysis)
    corrected_dfof_sem = smooth_signal(corrected_dfof_sem, smooth_time_window, sr_analysis)
    fitted_corrected_dfof, tau_corrected_dfof = fit_exp(corrected_dfof, sr_analysis)

    smoothed_predicted_dfof = smooth_signal(predicted_dfof, smooth_time_window, sr_analysis)
    predicted_dfof_sem = smooth_signal(predicted_dfof_sem, smooth_time_window, sr_analysis)
    fitted_predicted_dfof, tau_predicted_dfof = fit_exp(predicted_dfof, sr_analysis)


    z_speed = stats.zscore(input_speed)
    z_dfof = stats.zscore(dfof)
    zfitted_speed, ztau_speed = fit_exp(z_speed, sr_analysis)
    zfitted_dfof, ztau_dfof = fit_exp(z_dfof, sr_analysis)


    plt.figure(figsize=(9, 5))
    plt.fill_between(time_vec, smoothed_corrected_dfof - corrected_dfof_sem,
                    smoothed_corrected_dfof + corrected_dfof_sem, color='#DBECCB')
    plt.plot(time_vec, smoothed_corrected_dfof, linewidth=4, color='#3D8B43')
    plt.plot(time_vec, fitted_corrected_dfof, linewidth=8, color='#38843F')
    plt.text(350, 2, f'avg(zscoredfof-(a*logspeed+b)) τ = {tau_corrected_dfof:.2f}' + " (s)", fontsize=12, color='#6ABD45')
    plt.xlabel('Time (s)')
    plt.ylabel('Z-score')
    plt.tick_params(direction='out')


    return 0
def corr_speed_dfof(dfof,smooth_time_window,input_speed,sr_analysis):
    ## find min as offset for log

    if smooth_time_window==0:
        smooth_speed = input_speed
        smoothed_dfof = dfof
        print("wrong smooth window size")
    else:
        smooth_speed=input_speed
        smoothed_dfof=smooth_signal(dfof, smooth_time_window,sr_analysis)

    ####### corr log
    # Compute Pearson's R
    Rlog, _ = pearsonr(smooth_speed, smoothed_dfof)

    return Rlog



def corr_speed_dfof_band(time_vec, dfof, input_speed, smooth_time_window_big, smooth_time_window_s,  sr_analysis):
    ## find min as offset for log

    if (smooth_time_window_big==0 or smooth_time_window_s==0):
        print("not valid smoothing window")
    else:
        smooth_speed=input_speed #smooth_signal(input_speed, 1,sr_analysis)
        smoothed_dfof=smooth_signal(dfof, smooth_time_window_s,sr_analysis)-smooth_signal(dfof, smooth_time_window_big,sr_analysis)
        # Compute Pearson's R
        Rlog, _ = pearsonr(smooth_speed, smoothed_dfof)

    return Rlog

def behaviors(csv_file, details):
    with open(details, 'r') as file:
        # Read the first line
        line = file.readline()
        # Remove any trailing newline character
        line = line.rstrip('\n')

    onset = int(line.split(",")[0])
    print("The onset time:", onset)
    novel_obj_loc = line.split(",")[1]
    print("The novel location up/down: ", novel_obj_loc)

    data = pd.read_csv(csv_file).values
    # Assign names to columns
    column_names = ['row_num','background','Walking','Rearing_supported_by_wall','Rearing_unsupported_by_wall','Sniffing_at_wall','Grooming','Sniffing_at_ground','stationary_head_movements','Sniffing_at_corners','interact_novel_loc','Stretched_sniffing_novel_loc','interact_familiar_loc','Stretched_sniffing_familiar_loc']
    behaviors_df = pd.DataFrame(data, columns=column_names)

    ### Walking and Grooming
    walking = np.array(behaviors_df['Walking'])[1:]
    grooming = np.array(behaviors_df['Grooming'])[1:]

    ### exploration stationary or non-stationary object
    interact_novel_loc_times = np.array(behaviors_df['interact_novel_loc'])
    Stretched_sniffing_novel_times = np.array(behaviors_df['Stretched_sniffing_novel_loc'])
    interact_familiar_times = np.array(behaviors_df['interact_familiar_loc'])
    Stretched_sniffing_familiar_times = np.array(behaviors_df['Stretched_sniffing_familiar_loc'])
    if novel_obj_loc == 'u':
        exp_non_statobj = (Stretched_sniffing_novel_times | interact_novel_loc_times)[1:]
        exp_statobj = (interact_familiar_times | Stretched_sniffing_familiar_times)[1:]
    else:  # down
        exp_statobj = (Stretched_sniffing_novel_times | interact_novel_loc_times)[1:]
        exp_non_statobj = (interact_familiar_times | Stretched_sniffing_familiar_times)[1:]

    ### rearing
    all_rearings = (np.array(behaviors_df['Rearing_supported_by_wall']) | np.array(behaviors_df['Rearing_unsupported_by_wall']))[1:]
    for i in range(0, len(all_rearings)):
        if exp_statobj[i] == 1 or exp_non_statobj[i] == 1:
            all_rearings[i] = 0

    return exp_non_statobj,exp_statobj,walking,all_rearings,grooming

















