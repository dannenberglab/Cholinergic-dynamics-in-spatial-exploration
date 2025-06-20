import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import tdt
from scipy.optimize import minimize
from scipy.signal import resample
from scipy.stats import pearsonr
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


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

    # plt.plot(time, resampled_x405, 'red', label='resampled isosbestic control signal')
    # plt.plot(time, resampled_x465, 'blue', label='resampled GCaMP signal')
    # plt.legend()
    # plt.show()

    signal_tuple = (resampled_x465, resampled_x405, sampling_rate, time)
    return signal_tuple

def calculate_dftof_TDT(file_path, sampling_rate,start_of_recording,end_of_recording):

    x465, x405, fs, time = load_TDT_data(file_path, sampling_rate,start_of_recording,end_of_recording)
    # adjust for different change in bleaching
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

    # set initial values
    x0 = [1, 0]
    # Solve the optimization problem
    sol =minimize(objective_func, x0)
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
    plt.show()

    return dFoF, scaledControlSignal

##### Movement speed
def calculate_speed(x_values, y_values, time_interval):
    speed_values = []

    for i in range(1, len(x_values)):
        x_diff = x_values[i] - x_values[i - 1]
        y_diff = y_values[i] - y_values[i - 1]
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
        speed = distance / time_interval
        speed_values.append(speed)

    return speed_values
#### extract a column from csv file
def extract_column(csv_file, column_index):
    column_vector = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:
                column_vector.append((row[column_index]))

    return column_vector
def speed_analysis_new():
    ### read deeplabcut csv file
    ind1=4 # column in the csv file of deeplabcut
    sr_analysis=30 # sampling rate of the video
    multi_x = float(40 / 1100)  # pixel to cm --> needs to be updated if videos are from diffrent distance of the arena
    multi_y= float(40 / 1100)

    csv_file="E:/lab/Cholinergic Prj/Data/Full_GCaMP7sChAT_611926_210604_Rec1_Light_15min_ObjectLocationMemory_Learning.csv"
    body_x = extract_column(csv_file, ind1)
    body_y = extract_column(csv_file, ind1+1)
    body_likelihood = extract_column(csv_file, ind1+2)

    body_x = list(map(float, body_x[3:]))
    body_y = list(map(float, body_y[3:]))
    body_likelihood = list(map(float, body_likelihood[3:]))

    ## for visualization of the areana in the video
    # # video_file=path + name_exp + "/" + name_exp +".mp4"
    # video_file="E:/lab/Cholinergic Prj/Full_GCaMP7sChAT_611926_210604_Rec1_Light_15min_ObjectLocationMemory_Learning.mp4"
    # cap = cv2.VideoCapture(video_file)
    # ret, frame = cap.read()
    # print(len(frame))
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(frame_rgb)
    # plt.title("First Frame")
    # plt.show()

    # ######### calculate the speed
    x = np.array(body_x) * multi_x
    y = np.array(body_y) * multi_y
    body_likelihood = np.array(body_likelihood, dtype=float)
    for j in range(1, len(x)):
        if body_likelihood[j] < 0.9:
            x[j] = np.nan
            y[j] = np.nan

    ### interpolate NaNs
    df = pd.DataFrame({'x': x, 'y': y})
    # interpolate linearly over NaN values
    df_interpolated = df.interpolate(method='linear')
    x = df_interpolated['x'].values
    y = df_interpolated['y'].values

    speeds = calculate_speed(x, y, (1 / sr_analysis))
    for t in range(1, len(speeds)):
        if speeds[t] > 150:
            speeds[t] = np.nan
    df_s = pd.DataFrame({'speed': speeds})
    speeds = df_s.interpolate(method='linear')['speed'].values

    # saving the movement speed signal into a csv file
    df = pd.DataFrame({
        'pure_speed': speeds,
    })
    name_exp="Full_GCaMP7sChAT_611926_210604_Rec1_Light_15min_ObjectLocationMemory_Learning"
    df.to_csv("E:/lab/Cholinergic Prj/" + name_exp + '_speedsignal.csv', index=False)

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

def corr_speed_dfof(dfof,smooth_time_window,input_speed,sr_analysis):
    ## find min as offset for log

    if smooth_time_window==0:
        smooth_speed = input_speed
        smoothed_dfof = dfof
        print("wrong smooth window size")
    else:
        smooth_speed=input_speed
        window_samples = int(smooth_time_window * sr_analysis)
        kernel = np.ones(window_samples) / window_samples
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')

    ####### corr log
    # Compute Pearson's R
    Rlog, _ = pearsonr(smooth_speed, smoothed_dfof)

    return Rlog

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

def show_speed_dfof_decays_signle(time_vec, signals, smooth_time_window, sr_analysis, exp_name):
    dfof = signals[0]
    corrected_dfof = signals[1]
    predicted_dfof = signals[2]
    input_speed = signals[3]

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

    # # Rasterize the figure before saving
    # for obj in f1.get_children():
    #     obj.set_rasterized(True)

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)
    # plt.savefig('E:/lab/Cholinergic Prj/final files/results/' + str(exp_name) + "novelty.svg", format="svg")
    # plt.show()

    return tau_speed,tau_dfof,tau_corrected_dfof,tau_predicted_dfof









































#### multiply vector with a float number
def multiply_vector(vector, factor):
    result = []
    for cell in vector:
        result.append((cell) * factor)
    return result
##########


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
    plt.plot(time_vec, smooth_speed, linewidth=1, color='#666869')
    plt.plot(time_vec, fitted_speed, color='#666869', linewidth=3)
    plt.title(f'{smooth_time_window}-sec smoothed speed')
    plt.ylabel('(speed) (cm/s)')
    plt.text(350, 4, f'avg(speed) τ = {tau_speed:.2f}' + " (s)", fontsize=12, color='black')
    plt.tick_params(direction='out')
    plt.savefig(path + "decays1_avg.svg", format="svg")
    plt.show()

    # Plotting the second subplot separately
    f2 = plt.figure(figsize=(8, 4))
    plt.fill_between(time_vec, smoothed_dfof - dfof_sem, smoothed_dfof + dfof_sem, color='#C4E5D8')
    plt.plot(time_vec, smoothed_dfof, linewidth=1, color='#109D49')
    plt.plot(time_vec, fitted_dfof, color='green', linewidth=3)
    plt.text(350, 2, f'avg(zscore dfof) τ = {tau_dfof:.2f}' + " (s)", fontsize=12, color='green')

    plt.fill_between(time_vec, smoothed_predicted_dfof - predicted_dfof_sem,
                     smoothed_predicted_dfof + predicted_dfof_sem, color='#8F9194')
    plt.plot(time_vec, smoothed_predicted_dfof, linewidth=1, color='#515254')
    plt.plot(time_vec, fitted_predicted_dfof, color='black', linewidth=3)
    plt.text(350, 1.2, f'predict(logspeed) τ = {tau_predicted_dfof:.2f}' + " (s)", fontsize=12, color='black')

    plt.title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    plt.ylabel(r'$\Delta$' + 'F/F')
    plt.tick_params(direction='out')
    plt.savefig(path + "decays2_avg.svg", format="svg")
    plt.show()

    # Plotting the third subplot separately
    f3 = plt.figure(figsize=(8, 4))
    plt.fill_between(time_vec, smoothed_corrected_dfof - corrected_dfof_sem,
                     smoothed_corrected_dfof + corrected_dfof_sem, color='#DBECCB')
    plt.plot(time_vec, smoothed_corrected_dfof, linewidth=1, color='#90CB81')
    plt.plot(time_vec, fitted_corrected_dfof, linewidth=3, color='#6ABD45')
    plt.text(350, 2, f'avg(zscoredfof-(a*logspeed+b)) τ = {tau_corrected_dfof:.2f}' + " (s)", fontsize=12,
             color='#6ABD45')

    plt.title('Superimposition')
    plt.xlabel('Time (s)')
    plt.ylabel('Z-score')
    plt.tick_params(direction='out')
    plt.savefig(path + "decays3_avg.svg", format="svg")
    plt.show()



