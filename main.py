import random
from FiberPhotometry.PhotometrySignal import *
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.interpolate import interp1d
import statsmodels.formula.api as smf
from scipy.stats import t
from scipy.stats import zscore
from datetime import datetime
import os
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


## this function "scatter_plot_dfof_speed" is used in another function called plot_speed_dfof) for figure 2
def scatter_plot_dfof_speed(path, expriment_name, dfof, speed, smooth_time_window, sr_analysis, log):
    ## smoothing signals
    if smooth_time_window > 0:
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_speed = np.convolve(speed, kernel, mode='same')
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')
    else:
        smoothed_speed = np.array(speed)
        smoothed_dfof = np.array(dfof)

    ## measuring the log of speed
    if log:  # log=1/0
        log_speed = np.log2(smoothed_speed + 0.01)
        if np.isnan(log_speed).any() or np.isinf(log_speed).any():
            ## interpolate NanN
            df_log = pd.DataFrame({'log_speed': log_speed})
            log_speed = df_log.interpolate(method='linear')['log_speed'].values
        smoothed_speed = log_speed

    # scatter plot
    scatter_dfof = []
    scatter_speed = []

    for j in range(0, len(dfof), 5):
        scatter_dfof.append(smoothed_dfof[j])
        scatter_speed.append(smoothed_speed[j])

    scatter_speed = np.array(scatter_speed)
    scatter_dfof = np.array(scatter_dfof)

    def logarithmic_function(x, a, b):
        return a * np.log(x) + b

    def linear_function(x, a, b):
        return a * x + b

    if log == 0:
        valid_indices = np.where((scatter_speed > 0) & (scatter_speed < 60))
        scatter_speed = scatter_speed[valid_indices]
        scatter_dfof = scatter_dfof[valid_indices]

    # Perform curve fitting
    if log:
        popt, pcov = curve_fit(linear_function if log else logarithmic_function,
                               scatter_speed, scatter_dfof)
        a, b = popt
        x_curve = np.linspace(min(scatter_speed), max(scatter_speed), 100)
        y_curve = linear_function(x_curve, a, b)
        plt.xlabel('Log2( Running Speed)')
    else:
        popt, pcov = curve_fit(logarithmic_function, scatter_speed, scatter_dfof)
        a, b = popt
        x_curve = np.linspace(min(scatter_speed), max(scatter_speed), 100)
        y_curve = logarithmic_function(x_curve, a, b)
        plt.xlabel('Running Speed')

    # Plot scatter points and fitted curve
    plt.scatter(scatter_speed, scatter_dfof, marker=".", linewidths=0.1, color="#666869")
    plt.plot(x_curve, y_curve, color='r', label='Fitted curve: {:.2f} * ln(x) + {:.2f}'.format(a, b))
    plt.legend()
    plt.ylabel(r'$\Delta$' + 'F/F')
    plt.savefig(path + "scatter_speed_" + expriment_name + ".svg", format="svg")
    plt.show()


## this function "avg_scatter_plot_dfof_speed" is used in another function called plot_speed_dfof) for figure 2
def avg_scatter_plot_dfof_speed(path, expriment_name, dfof, speed, smooth_time_window, sr_analysis, log, bin_size):
    ## smoothing signals
    if smooth_time_window > 0:
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_speed = np.convolve(speed, kernel, mode='same')
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')
    else:
        smoothed_speed = np.array(speed)
        smoothed_dfof = np.array(dfof)

    ## measuring the log of speed
    if log:  # log=1/0
        log_speed = np.log2(smoothed_speed + 0.01)
        if np.isnan(log_speed).any() or np.isinf(log_speed).any():
            ## interpolate NanN
            df_log = pd.DataFrame({'log_speed': log_speed})
            log_speed = df_log.interpolate(method='linear')['log_speed'].values
        smoothed_speed = log_speed

    if log:
        valid_indices = np.where((smoothed_speed < 5))  # (smoothed_speed > 0) &
    else:
        valid_indices = np.where((smoothed_speed < 30))  # (smoothed_speed > 0) &

    bin_speed = smoothed_speed[valid_indices]
    bin_dfof = zscore(smoothed_dfof)[valid_indices]

    bin_edges = np.arange(-0.5, 4.5, bin_size)  # 1 cm/s bins
    # assign each speed value to a bin
    bin_indices = np.digitize(bin_speed, bin_edges)
    # Calculate mean scatter dfof values for each bin
    mean_dfof = []
    for x in range(0, len(bin_edges)):
        # if there are any data points in the bin
        if np.any(bin_indices == x):
            mean_dfof.append(bin_dfof[bin_indices == x].mean())
        else:
            mean_dfof.append(np.nan)

    # plt.plot(bin_edges, mean_dfof, marker='o')
    # if log:
    #     plt.xlabel('log2 running Speed (cm/s)')
    # else:
    #     plt.xlabel('Running Speed (cm/s)')
    # plt.ylabel('Mean dfof')
    # plt.title('Mean dfof vs Speed')
    # plt.grid(True)
    # # plt.savefig(path + "mean_logspeed_binned" + expriment_name + ".svg", format="svg")
    # plt.show()
    return mean_dfof


## this function is for generating plot in figure 2 of the paper
def plot_speed_dfof(path, directory_path, log):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    list_exp = []
    R_list = []
    mean_dfof_exp = []
    mean_dfof_exp_sample = []
    mean_dfof_exp_test = []
    merged_data = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        onset = int(line.split(",")[0])
        onset = onset * sr_analysis

        ## metadata for each session
        with open(filepath_exp + "\mouse_id.txt", 'r') as file:
            mouse_ID = file.readline().strip().replace(" ", "")
            n_trial = file.readline().strip().replace(" ", "")
            # task_t = file.readline().strip().replace(" ", "")
            # last character of each
            mouse_ID = int(mouse_ID[-1] if mouse_ID else '')
            n_trial = int(n_trial[-1] if n_trial else '')
            # task_t = int(task_t[-1] if task_t else '')

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 900 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 900 * sr_analysis

        list_exp.append(expriment_name)
        ##############################
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        time = data[0][onset:analysis_win]
        dfof = data[1][onset:analysis_win]
        speed = data[2][onset:analysis_win]

        # ## Part 1: correlations --> uncomment this part to plot figure 2 A and D samples --> plot correlations
        # R, f1 = show_speed_dfof(time, dfof, speed, 1, sr_analysis, log)
        # plt.savefig(path+"dfof_speed_"+expriment_name+".svg",format="svg")
        # plt.show()
        # R_list.append(R)
        # df = pd.DataFrame({
        #     'Name_Exp': list_exp,
        #     'R': R_list
        # })
        # df.to_csv(path+'Rlog.csv', index=False)

        # ## Part 2 -->uncomment this part to plot figure 2 B and E samples --> scatter plots
        # scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, log)

        # ## Part 3 -->uncomment this part to plot figure 2 C and F  --> avg scatter plots
        bin_size = 0.2  ## if you change the function to use movement speed instead of log of speed, you need to change bin size as well
        mean_dfof = avg_scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, 1, bin_size)
        mean_dfof_exp.append(mean_dfof[0:30])
        if task == 'Learning' or task == 'learning':
            mean_dfof_exp_sample.append(mean_dfof[0:30])
        else:
            mean_dfof_exp_test.append(mean_dfof[0:30])
        print(len(mean_dfof))

        ## the following part creates dataframes for each session to be used in LMM for figure 2F
        # Determine if the task is 'sample' or 'test'
        smooth_time_window = 0.5
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_speed = np.convolve(speed, kernel, mode='same')
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')
        log_speed = np.log2(smoothed_speed + 0.01)
        if np.isnan(log_speed).any() or np.isinf(log_speed).any():
            ## interpolate NanN
            df_log = pd.DataFrame({'log_speed': log_speed})
            log_speed = df_log.interpolate(method='linear')['log_speed'].values

        task_SampleOrTest = 0 if task_type == 0 else 1
        # Create a temporary DataFrame for the current experiment
        temp_df = pd.DataFrame({
            'dfof': stats.zscore(smoothed_dfof),
            'speed': log_speed,
            'task_type': task_SampleOrTest,
            'session_id': i,
            'mouseid': mouse_ID
        })
        # Append the temporary DataFrame to the list
        merged_data.append(temp_df)
        #
        i += 1  ## do Not comment this

    # ## Part 3 -->uncomment this part to plot figure 2 C and F  --> avg scatter plots
    ## the LLM for compring Sample Vs Test in figure 2 F
    df = pd.concat(merged_data, ignore_index=True)
    # LMM for figure 2F
    model = smf.mixedlm(
        "dfof ~ speed * task_type",  # fixed effects: speed, interaction
        data=df,
        groups=df["session_id"]  # random effects by session id
    )
    result = model.fit()
    print(result.summary())

    # plots for figure 2
    mean_dfof_across_experiments = np.nanmean(mean_dfof_exp, axis=0)  # Average across experiments
    sem_dfof_across_experiments = np.nanstd(mean_dfof_exp, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    x_values = np.arange(-0.5, 4.5, bin_size)
    plt.errorbar(x_values, mean_dfof_across_experiments, yerr=sem_dfof_across_experiments,
                 fmt='o-', capsize=3, label='Mean ± SEM')
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Mean dfof')
    plt.title('Mean dfof vs Speed')
    plt.legend()
    plt.savefig(path + "mean_logspeed_binned" + ".svg", format="svg")
    plt.show()

    mean_dfof_across_experiments_sample = np.nanmean(mean_dfof_exp_sample, axis=0)  # Average across experiments
    sem_dfof_across_experiments_sample = np.nanstd(mean_dfof_exp_sample, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    mean_dfof_across_experiments_test = np.nanmean(mean_dfof_exp_test, axis=0)  # Average across experiments
    sem_dfof_across_experiments_test = np.nanstd(mean_dfof_exp_test, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    plt.errorbar(x_values, mean_dfof_across_experiments_sample, yerr=sem_dfof_across_experiments_sample,
                 fmt='o-', capsize=3, label='Sample sessions', color="#048786")
    plt.errorbar(x_values, mean_dfof_across_experiments_test, yerr=sem_dfof_across_experiments_test,
                 fmt='o-', capsize=3, label='Test sessions', color="#871719")
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Mean dfof')
    plt.title('Mean dfof vs Speed')
    plt.legend()
    plt.savefig(path + "mean_logspeed_binned_seperate" + ".svg", format="svg")
    plt.show()


## this function is for generating plot in figure 2 G of the paper
def find_best_smoothing_corr(path, directory_path):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    list_exp = []
    all_R_corr = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        onset = int(line.split(",")[0])
        onset = onset * sr_analysis

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 900 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 900 * sr_analysis

        list_exp.append(expriment_name)
        ##############################
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        time = data[0][onset:analysis_win]
        dfof = data[1][onset:analysis_win]
        speed = data[2][onset:analysis_win]

        ## smoothing signals
        smooth_time_window = 1
        log = 1
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_speed = np.convolve(speed, kernel, mode='same')
        # smoothed_dfof = np.convolve(dfof, kernel, mode='same')

        ## measuring the log of speed
        if log:  # log=1/0
            log_speed = np.log2(smoothed_speed + 0.01)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                ## interpolate NanN
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values
            smoothed_speed = log_speed

        rcorr = []
        x = []
        range1 = [0.25, 0.5]
        range2 = np.arange(1, 32, 1)  # [0.25, 0.5,0.75,1,2]  # Adjust the step as needed
        range3 = np.arange(32, 257, 32)
        merged_range = np.concatenate((range1, range2, range3))
        for p in merged_range:
            Rlog = corr_speed_dfof(dfof, p, smoothed_speed, sr_analysis)
            print(p, Rlog)
            rcorr.append(Rlog)
            x.append(p)

        plt.plot(x, rcorr)
        plt.scatter(x, rcorr)
        plt.xlabel("window size(s) for smoothing")
        plt.ylabel("R")
        plt.savefig(path + "R_30" + expriment_name + ".svg", format="svg")
        plt.xscale("log")
        plt.savefig(path + "R_30_log" + expriment_name + ".svg", format="svg")
        plt.show()

        all_R_corr.append(rcorr)
        i = i + 1
    print(len(merged_range))
    df_R_corr = pd.DataFrame(all_R_corr)
    # df_R_corr.to_csv(path + "Rcorr_logspeed_dfofsmooth.csv", index=False)

    mean_across_experiments = np.mean(all_R_corr, axis=0)  # Average across experiments
    sem_across_experiments = np.std(all_R_corr, axis=0) / np.sqrt(len(all_R_corr))  # SEM

    plt.xscale("log")
    plt.plot(x, mean_across_experiments, label='Mean')
    # plt.scatter(x, mean_across_experiments)  # optional styling
    # plt.errorbar(x, mean_across_experiments, yerr=sem_across_experiments,
    #              fmt='o-', capsize=3, label='Mean ± SEM', color='black')
    plt.fill_between(
        x,
        mean_across_experiments - sem_across_experiments,
        mean_across_experiments + sem_across_experiments,
        color='gray', alpha=0.5
    )

    plt.xlabel("window size (s) for smoothing")
    plt.ylabel("R")
    plt.legend()
    plt.savefig(path + "mean_best_corr" + ".svg", format="svg")
    plt.show()

## stats for enviromental novelty :
def stat_cluster_test(observed_cluster_mass, observed_cluster_size, savePath, csv_file):
    # largestMass_cluster_obs, sizeOf_largestMass_cluster_obs, index_largeMass_obs,
    # massOf_largestSize_cluster_obs,largestSize_cluster_obs, index_largeSize_obs

    df = pd.read_csv(csv_file)

    dis = np.abs(df["largestMass_cluster_obs"].values)
    plt.hist(dis)  # bins=len(cluster_mass)
    plt.xlabel("largestMass_cluster")
    plt.ylabel("number of clusters")
    plt.savefig(savePath + "nullDis_Mass.svg", format="svg")
    plt.show()

    p_value_m = (np.sum(dis >= abs(observed_cluster_mass)) + 1) / (len(dis) + 1)
    print(f"Monte Carlo p-value mass: {p_value_m}")

    dis = np.abs(df["largestSize_cluster_obs"].values)
    plt.hist(dis)
    plt.xlabel("largestSize_cluster")
    plt.ylabel("number of clusters")
    plt.savefig(savePath + "nullDis_size.svg", format="svg")
    plt.show()
    p_value_m = (np.sum(dis >= abs(observed_cluster_size)) + 1) / (len(dis) + 1)
    print(f"Monte Carlo p-value size: {p_value_m}")

def compute_t_threshold(n_samples_1, n_samples_2, p_value):
    if n_samples_1 == n_samples_2:  ## a paired or one-sample test
        df = n_samples_1 - 1  # Degrees of freedom
    else:  # an independent two-sample t-test
        df = n_samples_1 + n_samples_2 - 2
    #  loc (mean) ,  scale (std_dev) to adjust the t-distribution
    t_critical = t.ppf(1 - p_value / 2, df)  # Two-tailed test  1-p_value/2,   one tail 1-p_value

    return t_critical


def compare_Plots(signal_2D_1, signal1_name, signal_2D_2, signal2_name, p_value, t_test, sr, plot=0):
    # t_test = {‘two-sided’, ‘less’, ‘greater’}
    lent = signal_2D_1.shape[1]
    t_val = np.zeros(lent)
    t_critical_both = compute_t_threshold(signal_2D_1.shape[0], signal_2D_2.shape[0], p_value)  # /signal_2D_1.shape[0]

    # Perform two-sample t-test and mark significant differences
    for i in range(lent):
        t_stat, p_value = stats.ttest_ind(signal_2D_1[:, i], signal_2D_2[:, i], alternative=t_test, equal_var=False)
        t_val[i] = t_stat

    # Create a figure for the bar plot
    if plot:
        # Plot the t-values and critical t-lines
        x_positions = np.arange(lent) * (1 / sr)
        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
        ax_bar.plot(x_positions, t_val, label="t-statistic values")
        ax_bar.axhline(t_critical_both, color='red', linestyle='--', label=str("Critical t-value"))
        # ax_bar.axhline(t_critical_2, color='blue', linestyle='--', label=str("Critical t-value for " + signal2_name))
        ax_bar.axhline(-t_critical_both, color='red', linestyle='--')  # , label=str("Critical t-value for " + "both")
        # ax_bar.axhline(-t_critical_2, color='blue', linestyle='--', label=str("Critical t-value for " + signal2_name))
        plt.legend()
        plt.xlabel("time(s)")
        plt.ylabel("t_value")
        # print(x_positions[contact_index])
        # plt.text(x_positions[contact_index],2.3,"contact point:" + str(x_positions[contact_index]), fontsize=12)
        # ax_bar.axvline(x_positions[contact_index], color='red', linestyle='--')
        # plt.savefig("E:\\lab\\Cholinergic Prj\\final files\\temp\\t_val.svg", format="svg")
        plt.show()
    #
    # cluster_mass_fun(t_val, t_critical_both)
    # print("t_critical", t_critical_both)

    return t_val, t_critical_both


def cluster_mass_fun(t_val, t_critical_1, plotOption_histOfClusters=0):
    cluster_mass = []
    cluster_size = []
    cluster_st_ed = []
    mass_temp = 0
    flag = 0
    i = 0
    size = 0
    t_val_abs = np.abs(t_val)
    while i < len(t_val) - 1:
        if t_val_abs[i] >= t_critical_1:
            flag = 1
            size = i
            while flag:
                mass_temp += t_val[i]
                i += 1
                if t_val_abs[i] < t_critical_1:
                    flag = 0
                if i >= len(t_val) - 1:
                    break
            cluster_mass.append(mass_temp)
            cluster_size.append(i - size + 1)
            cluster_st_ed.append([size, i])
            mass_temp = 0
        i += 1
    if plotOption_histOfClusters:
        plt.hist(cluster_mass)  # bins=len(cluster_mass)
        plt.xlabel("cluster mass (summed t values)")
        plt.ylabel("number of clusters")
        plt.text(0, 1, "total #clusters" + str(len(cluster_mass)), fontsize=12)
        # plt.savefig("E:\\lab\\Cholinergic Prj\\final files\\temp\\hist_main.svg", format="svg")
        df = pd.DataFrame({
            'cluster_mass': cluster_mass,
            'cluster_size': cluster_size,
            'cluster_start': [start for start, end in cluster_st_ed],
            'cluster_end': [end for start, end in cluster_st_ed],
            'len': [end - start + 1 for start, end in cluster_st_ed],
            'percentage': [(end - start + 1) / len(t_val) for start, end in cluster_st_ed],
            'p_mass': '',
            'p_size': ''
        })
        # df.to_csv("E:\\lab\\Cholinergic Prj\\final files\\temp\\hist_main.csv", index=False)
        plt.show()

    ### only big the cluster
    if len(cluster_mass):
        max_index = np.argmax(np.abs(cluster_mass))
        big_cluster = cluster_mass[max_index]
        size_cluster = cluster_size[max_index]
        index_largeMass = cluster_st_ed[max_index]

        max_index = np.argmax(cluster_size)
        bigmass_cluster = cluster_mass[max_index]
        bigsize_cluster = cluster_size[max_index]
        index_largeSize = cluster_st_ed[max_index]
        return big_cluster, size_cluster, index_largeMass, bigmass_cluster, bigsize_cluster, index_largeSize  # , cluster_mass
    else:
        return 0, 0, 0, 0, 0, 0


## use this function "novel_env" to plot figure 2 H and I
def novel_env(path, directory_path):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    speed_list = []
    dfof_list = []
    list_exp = []
    corrected_dfof_list = []
    corrected_dfof_fit_list = []
    coeffs = []
    predicted_dfof = []
    win_plt = int(14 * 60 * sr_analysis)
    tau_speed_list = []
    tau_dfof_list = []
    tau_corrected_dfof_list = []
    tau_predicted_dfof_list = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        onset = int(line.split(",")[0])
        onset = onset * sr_analysis

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 900 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 900 * sr_analysis
        list_exp.append(expriment_name)
        ############################## calulations
        if task_type == 1 or task_type == 0:
            analysis_win = analysis_win - onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            time = data[0][onset:analysis_win]
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            smooth_time_window = 1
            log = 1
            window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
            kernel = np.ones(window_samples) / window_samples
            speed = np.convolve(speed, kernel, mode='same')
            dfof = np.convolve(dfof, kernel, mode='same')
            zdfof = stats.zscore(dfof)

            ## measuring the log of speed
            if log:  # log=1/0
                log_speed = np.log2(speed + 0.001)
                if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                    ## interpolate NanN
                    df_log = pd.DataFrame({'log_speed': log_speed})
                    log_speed = df_log.interpolate(method='linear')['log_speed'].values

            log_speed = log_speed.reshape(-1, 1)
            # perform linear regression to calculate speed predicted cholirnegic activity
            model = LinearRegression()
            model.fit(log_speed, zdfof)
            pzdfof = model.predict(log_speed)
            corrected_dfof = zdfof - pzdfof

            corrected_dfof_list.append(corrected_dfof[:win_plt])
            predicted_dfof.append(pzdfof[:win_plt])
            speed_list.append(speed[:win_plt])
            dfof_list.append(zdfof[:win_plt])

            # uncomment this if you want to plot each single session
            # signals = [zdfof[:win_plt], corrected_dfof[:win_plt], pzdfof[:win_plt], speed[:win_plt]]
            # tau_speed,tau_dfof,tau_corrected_dfof,tau_predicted_dfof = show_speed_dfof_decays_signle(time[:win_plt], signals, smooth_time_window, sr_analysis ,str(expriment_name))
            # plt.savefig(path + str(expriment_name) + "novelty.svg", format="svg")
            # plt.show()
            # tau_speed_list.append(tau_speed)
            # tau_dfof_list.append(tau_dfof)
            # tau_corrected_dfof_list.append(tau_corrected_dfof)
            # tau_predicted_dfof_list.append(tau_predicted_dfof)
        i += 1

    # # uncomment this part to save taus in a csv file
    # data = {
    #     'exp_name': list_exp,
    #     'tau_speed': tau_speed_list,
    #     'tau_dfof': tau_dfof_list,
    #     'tau_corrected_dfof': tau_corrected_dfof_list,
    #     'tau_predicted_dfof': tau_predicted_dfof_list
    # }
    # df = pd.DataFrame(data)
    # df.to_csv(path+'combined_results.csv', index=False)

    speed_list = np.array(speed_list)
    avg_speed = np.mean(speed_list, axis=0)
    std_speed = np.std(speed_list, axis=0)  # Standard deviation
    n = speed_list.shape[0]  # Number of observations
    speed_sem = std_speed / np.sqrt(n)  # Standard error of the mean

    dfof_list = np.array(dfof_list)
    avg_dfof = np.mean(dfof_list, axis=0)
    std_dfof = np.std(dfof_list, axis=0)  # Standard deviation
    n = dfof_list.shape[0]  # Number of observations
    dfof_sem = std_dfof / np.sqrt(n)  # Standard error of the mean

    predicted_dfof = np.array(predicted_dfof)
    predicted_dfof_avg = np.mean(predicted_dfof, axis=0)
    std_predicted_dfof = np.std(predicted_dfof, axis=0)  # Standard deviation
    n = predicted_dfof.shape[0]  # Number of observations
    predicted_dfof_sem = std_predicted_dfof / np.sqrt(n)  # Standard error of the mean

    corrected_dfof_list = np.array(corrected_dfof_list)
    corrected_dfof = np.mean(corrected_dfof_list, axis=0)
    std_corrected_dfof = np.std(corrected_dfof_list, axis=0)  # Standard deviation
    n = speed_list.shape[0]  # Number of observations
    corrected_dfof_sem = std_corrected_dfof / np.sqrt(n)  # Standard error of the mean

    signals = [avg_dfof, corrected_dfof, predicted_dfof_avg, avg_speed]
    sems = [dfof_sem, corrected_dfof_sem, predicted_dfof_sem, speed_sem]
    Rlog = show_speed_dfof_decays(time[:win_plt], signals, sems, 1, sr_analysis)
    # plt.savefig(path + "novelty_sem.svg", format="svg")
    plt.show()

    # running nonparametric cluster - based permutation test
    p_value = 0.05
    t_val, t_critic = compare_Plots(dfof_list, "dfof", predicted_dfof, "predicted_dfof", p_value, "two-sided",
                                    sr_analysis, 1)
    (largestMass_cluster_obs, sizeOf_largestMass_cluster_obs, index_largeMass_obs, massOf_largestSize_cluster_obs,
     largestSize_cluster_obs, index_largeSize_obs) = cluster_mass_fun(t_val, t_critic, 1)
    print("t_critical------------>", t_critic)
    ## report the largest cluster mass and size
    print("largestMass_cluster_obs", largestMass_cluster_obs)
    print("sizeOf_largestMass_cluster_obs", sizeOf_largestMass_cluster_obs, "--> ", sizeOf_largestMass_cluster_obs/sr_analysis , "seconds")
    print("index_largeMass_obs", index_largeMass_obs)
    print("largestSize_cluster_obs", largestSize_cluster_obs)
    print("massOf_largestSize_cluster_obs", massOf_largestSize_cluster_obs)
    print("index_largeSize_obs", index_largeSize_obs)

    ## Creating the permutation distribution representing the null hypothesis ##
    num = dfof_list.shape[0]  # num of observations
    both_gp = np.concatenate((dfof_list, dfof_list))
    rnd_shuffled = both_gp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    csv_file = path + "Clusters_" + "dfof" + "_VS_" + "predicted_dfof" + f"_{timestamp}.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write(
                "largestMass_cluster_obs,sizeOf_largestMass_cluster_obs,massOf_largestSize_cluster_obs,largestSize_cluster_obs\n")

    for i in range(500): # num_iters
        print("iter--> ", i)
        np.random.shuffle(rnd_shuffled)
        t_val, t_critic = compare_Plots(rnd_shuffled[0:num, :], "sig1",
                                        rnd_shuffled[num:, :], "sig2", 0.05, "two-sided", sr_analysis,
                                        1)
        biggestMass_cluster, sizeOf_biggestMass_cluster, index_largeMass, massOf_biggestSize_cluster, biggestSize_cluster, index_largeSize = cluster_mass_fun(
            t_val, t_critic, 1)

        # save results to the CSV file
        DF = pd.DataFrame({'biggestMass_cluster': [biggestMass_cluster],
                           'sizeOf_biggestMass_cluster': [sizeOf_biggestMass_cluster],
                           'massOf_biggestSize_cluster': [massOf_biggestSize_cluster],
                           'biggestSize_cluster': [biggestSize_cluster]})
        DF.to_csv(csv_file, mode='a', header=False, index=False)


## this function generated csv file including data required for generating figure 3 of the paper
def time_analysis_washout(path, directory_path):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    print(folder_names)
    total_non_stat = []
    total_stat = []
    total_ratio = []
    list_task = []
    all_ratio = pd.DataFrame()

    list_expriment_name = []
    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            # Remove any trailing newline character
            line = line.rstrip('\n')

        onset = int(line.split(",")[0])
        onset = onset * sr_analysis
        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 60 * 15 * sr_analysis  ## !!!! for summary change 15 min to 12 min and comment the washout part
            # 720
        else:  ## Recall 3 min   !!!! for summary change 15 min to 3 min and comment the washout part
            task_type = 1
            analysis_win = 60 * 15 * sr_analysis
        ##
        list_task.append(task)
        list_expriment_name.append(expriment_name)
        ############################## calulations
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        exp_non_statobj = data[3][onset:analysis_win]
        exp_statobj = data[4][onset:analysis_win]

        non_stat_v = []
        stat_v = []
        diff = []
        ratio = []
        for w in range(0, 13, 1):
            non_stat_v.append(sum(exp_non_statobj[(w) * sr_analysis * 60:(w + 3) * sr_analysis * 60]) / sr_analysis)
            stat_v.append(sum(exp_statobj[(w) * sr_analysis * 60:(w + 3) * sr_analysis * 60]) / sr_analysis)
            diff.append(non_stat_v[w] - stat_v[w])
            ratio.append((non_stat_v[w] - stat_v[w]) / (non_stat_v[w] + stat_v[w]))
        all_ratio[f'{task_type}_{i}'] = pd.Series(ratio)
        # !!summary part:
        total_non_stat.append(sum(exp_non_statobj))
        total_stat.append(sum(exp_statobj))
        total_ratio.append((total_non_stat[i] - total_stat[i]) / (total_non_stat[i] + total_stat[i]))
        i = i + 1
    # Save the DataFrame to a CSV file
    output_filename = path + "wash_time_ratio.csv"
    all_ratio.to_csv(output_filename, index=False)

    # !!summary part: Save total_non_stat, total_stat, and total_ratio to a new CSV file
    summary_data = pd.DataFrame({
        'list_expriment_name': list_expriment_name,
        'Task': list_task,
        'Total_Non_Stat': total_non_stat,
        'Total_Stat': total_stat,
        'Total_Ratio': total_ratio
    })
    summary_filename = path + "summary_time.csv"
    summary_data.to_csv(summary_filename, index=False)

## use this function to
# plot figure 4 A and B
def behavioral_interactions(path, directory_path):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    num_exp = 0
    average_interaction_matrix = np.zeros((5, 5), dtype=float)
    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        onset = int(line.split(",")[0])
        onset = onset * sr_analysis

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 900 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 900 * sr_analysis

        ############################## calulations
        if task_type == 0:  # or task_type == 0:  ## change this to 0 for plotting Sample sessions and 1 for Test sessions
            analysis_win = analysis_win - onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            exp_non_statobj = np.array(data[3][onset:analysis_win]).astype(bool)
            exp_statobj = np.array(data[4][onset:analysis_win]).astype(bool)
            walking = np.array(data[5][onset:analysis_win]).astype(bool)
            rearings = np.array(data[6][onset:analysis_win]).astype(bool)
            groomings = np.array(data[7][onset:analysis_win]).astype(bool)

            ### List of time series arrays
            time_series = [exp_non_statobj, exp_statobj, walking, rearings, groomings]
            ### initialize the interaction matrix
            interaction_matrix = np.zeros((len(time_series), len(time_series)), dtype=int)
            behavior_names = ['Exp_Non_Statobj', 'Exp_Statobj', 'Walking', 'Rearings', 'Groomings']

            # calculating of the interactions
            for y in range(len(time_series)):
                for j in range(len(time_series)):
                    interaction_matrix[y, j] = np.sum(time_series[y] & time_series[j])
            # Save individual matrix to CSV
            df_matrix = pd.DataFrame(interaction_matrix, columns=behavior_names, index=behavior_names)
            save_name = f"{expriment_name}_interaction_matrix.csv"
            df_matrix.to_csv(os.path.join(path, save_name))

            # Add the interaction matrix to the accumulator
            average_interaction_matrix += interaction_matrix
            print("Interaction Matrix:")
            print(interaction_matrix)
            # # Plotting
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(interaction_matrix, annot=True, fmt='d', xticklabels=behavior_names, yticklabels=behavior_names,
            #             cmap='viridis', cbar=True)
            # plt.title('Behavioral Interaction Matrix')
            # plt.show()
            num_exp += 1
        i += 1
    # divide by the number of experiments to get the average interaction matrix
    print("i", i)
    print("exp_num", num_exp)
    average_interaction_matrix /= num_exp
    average_interaction_matrix /= 30
    # Round to the nearest integer and convert to int
    average_interaction_matrix = np.round(average_interaction_matrix).astype(int)
    print("numer of exp: ", i)
    # report the average
    print("Average Interaction Matrix:")
    print(average_interaction_matrix)

    # plot the full matrix
    behavior_names = ['Exp_Non_Statobj', 'Exp_Statobj', 'Walking', 'Rearings', 'Groomings']
    plt.figure(figsize=(10, 8))
    sns.heatmap(average_interaction_matrix, annot=True, fmt='d', xticklabels=behavior_names, yticklabels=behavior_names,
                cmap='viridis', annot_kws={"fontsize": 20}, cbar=True)
    plt.title('Average Behavioral Interaction Matrix')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig(path + "matrix_inter_full_test.svg", format="svg")
    plt.show()

    # plot the matrix scaled on non-diagonal ones
    # a mask for the diagonal
    mask = np.eye(average_interaction_matrix.shape[0], dtype=bool)
    # diagonal values = NaN so they appear white
    masked_matrix = average_interaction_matrix.astype(float)
    masked_matrix[mask] = np.nan
    masked_matrix = np.round(masked_matrix).astype(int)
    # plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(masked_matrix, annot=True, fmt='d', xticklabels=behavior_names, yticklabels=behavior_names,
                cmap='viridis', cbar=True, mask=mask, annot_kws={"fontsize": 20},
                cbar_kws={'label': 'Interaction Value'})
    plt.title('Average Behavioral Interaction Matrix')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig(path +"matrix_inter_nondiagonals_test.svg", format="svg")  # Save as SVG
    plt.show()

    # extract the diagonal elements
    diagonal_elements = np.diag(average_interaction_matrix)
    # labels for the pie chart (same as behavior names)
    behavior_names = ['Exp Non Statobj', 'Exp Statobj', 'Walking', 'Rearings', 'Groomings']
    # plotting the pie chart
    plt.figure(figsize=(20, 20))
    plt.pie(diagonal_elements, labels=behavior_names, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 60})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(path + "piechart_sample.svg", format="svg")
    plt.show()

def Create_dataset_for_LLM(path, directory_path):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    all_data = []
    list_exp = []
    coefficients_list = []
    all_sessions = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        onset = int(line.split(",")[0])
        onset = onset * sr_analysis

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 15 * 60 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 15 * 60 * sr_analysis

        ## metadata for each session
        with open(filepath_exp + "/mouse_id.txt", 'r') as file:
            mouse_ID = file.readline().strip().replace(" ", "")
            n_trial = file.readline().strip().replace(" ", "")
            # task_t = file.readline().strip().replace(" ", "")
            # last character of each
            mouse_ID = int(mouse_ID[-1] if mouse_ID else '')
            n_trial = int(n_trial[-1] if n_trial else '')
            # task_t = int(task_t[-1] if task_t else '')

        ############################## calulations
        if task_type == 1 or task_type == 0:
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            if len(data[1]) - onset > analysis_win:
                analysis_win = analysis_win
            else:
                analysis_win = len(data[1])
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            time_vector = np.arange(len(dfof)) / sr_analysis
            #####
            log = 1
            window_samples = int(0.5 * sr_analysis)  # number of samples in the 0.5s window
            kernel = np.ones(window_samples) / window_samples

            dfof = np.convolve(dfof, kernel, mode='same')
            dfof = zscore(dfof)
            speed = np.convolve(speed, kernel, mode='same')

            ## calculate log speed
            if log:  # log=1/0
                log_speed = np.log2(speed + 0.001)
                if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                    ## interpolate NanN
                    df_log = pd.DataFrame({'log_speed': log_speed})
                    log_speed = df_log.interpolate(method='linear')['log_speed'].values

            session_df = pd.DataFrame({
                'session_id': i,
                'mouse_id': mouse_ID,
                'trial': n_trial,
                'task': task_type,
                'time': time_vector,
                'dfof': dfof,
                'speed': log_speed,
                'exp_non_statobj': data[3][onset:analysis_win],
                'exp_statobj': data[4][onset:analysis_win],
                'locomotion': data[5][onset:analysis_win],
                'rearing': data[6][onset:analysis_win],
                'grooming': data[7][onset:analysis_win]
            })

            all_sessions.append(session_df)

        i = i + 1
    full_data = pd.concat(all_sessions, ignore_index=True)
    full_data["mouse_trial"] = full_data["mouse_id"].astype(str) + "_T" + full_data["trial"].astype(str)
    full_data.to_csv(path + "all20sessions_allbehaviors.csv")


# plotting figure 5 C
def cholinergicLevel_diffBehaviors(path):
    # Load the data
    df_long = pd.read_csv(path + "all20sessions_allbehaviors.csv")
    # define behaviors
    behaviors = ['locomotion', 'grooming', 'rearing', 'exp_statobj', 'exp_non_statobj']
    all_dfof = []
    all_behavior_names = []
    all_task_labels = []

    for task_val in [0, 1]:  # 0 = Sample, 1 = Test
        df_filtered = df_long[df_long['task'] == task_val]
        for behavior in behaviors:
            idx_active = df_filtered[behavior] > 0
            dfof_values = df_filtered.loc[idx_active, 'dfof']

            all_dfof.extend(dfof_values)
            all_behavior_names.extend([behavior] * len(dfof_values))
            all_task_labels.extend([f"Task {task_val}"] * len(dfof_values))

    plot_df = pd.DataFrame({
        'Behavior': all_behavior_names,
        'dfof': all_dfof,
        'Task': all_task_labels
    })

    # Plot
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Behavior', y='dfof', hue='Task', data=plot_df, width=0.6, showfliers=False)
    plt.xlabel("Behavior")
    plt.ylabel("z-score (ΔF/F)")
    plt.title("ΔF/F Across Behaviors by Task")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.ylim([-4.5, 3.5])
    plt.savefig(path + "boxplot_zscore_sample&test.svg", format="svg")
    plt.show()



## plot figure 4D-C
def bar_plot_timewashedLLM(path):
    # LMM_windowed.csv is the output of time_wash_LLM Matlab script
    df = pd.read_csv(path + "LMM_windowed.csv")

    exp_non_statobj_sample = df['exp_non_statobj_sample']
    exp_statobj_sample = df['exp_statobj_sample']
    exp_non_statobj_test = df['exp_non_statobj_test']
    exp_statobj_test = df['exp_statobj_test']
    exp_non_statobj_sample_se = df['exp_non_statobj_sample_se']
    exp_statobj_sample_se = df['exp_statobj_sample_se']
    exp_non_statobj_test_se = df['exp_non_statobj_test_se']
    exp_statobj_test_se = df['exp_statobj_test_se']

    diff_sample = df['exp_non_statobj_sample'] - df['exp_statobj_sample']
    diff_test = df['exp_non_statobj_test'] - df['exp_statobj_test']

    plt.figure(figsize=(10, 7))
    # plt.fill_between(np.arange(len(data_mean)), data_mean - data_sem, data_mean + data_sem, color='lightgray')
    plt.errorbar(np.arange(len(exp_non_statobj_sample)), exp_non_statobj_sample, yerr=exp_non_statobj_sample_se,
                 fmt='-o', color='#8E584D',
                 ecolor='#C7ACA3', capsize=5, label="exp_non_statobj")
    # plt.fill_between(np.arange(len(data_mean2)), data_mean2 - data_sem2, data_mean2 + data_sem2, color='lightgray')
    plt.errorbar(np.arange(len(exp_statobj_sample)), exp_statobj_sample, yerr=exp_statobj_sample_se, fmt='-o',
                 color='#F58020',
                 ecolor='#FDC89B', capsize=5, label="exp_statobj")
    plt.legend()
    plt.ylim([-0.42, 0.82])
    path = 'E:/lab/Cholinergic Prj/final files/GLMs/'
    plt.savefig(path + "OBJs_Sample.svg", format="svg")
    plt.show()

    plt.figure(figsize=(10, 7))
    # plt.fill_between(np.arange(len(data_mean)), data_mean - data_sem, data_mean + data_sem, color='lightgray')
    plt.errorbar(np.arange(len(exp_non_statobj_test)), exp_non_statobj_test, yerr=exp_non_statobj_test_se, fmt='-o',
                 color='#8E584D',
                 ecolor='#C7ACA3', capsize=5, label="exp_non_statobj")
    # plt.fill_between(np.arange(len(data_mean2)), data_mean2 - data_sem2, data_mean2 + data_sem2, color='lightgray')
    plt.errorbar(np.arange(len(exp_statobj_test)), exp_statobj_test, yerr=exp_statobj_test_se, fmt='-o',
                 color='#F58020',
                 ecolor='#FDC89B', capsize=5, label="exp_statobj")
    plt.legend()
    plt.savefig(path + "OBJs_TEST.svg", format="svg")
    plt.ylim([-0.42, 0.82])
    plt.show()

    # Compute difference
    diff_test = exp_non_statobj_test - exp_statobj_test
    sem_diff_test = np.sqrt(exp_non_statobj_test_se ** 2 + exp_statobj_test_se ** 2)

    diff_sample = exp_non_statobj_sample - exp_statobj_sample
    sem_diff_sample = np.sqrt(exp_non_statobj_sample_se ** 2 + exp_statobj_sample_se ** 2)

    plt.figure(figsize=(10, 7))
    # plt.fill_between(np.arange(len(data_mean)), data_mean - data_sem, data_mean + data_sem, color='lightgray')
    plt.errorbar(np.arange(len(diff_test)), diff_test, yerr=sem_diff_test, fmt='-o', color='#048786',
                 ecolor='#D7E1E2', capsize=5, label="Test")
    plt.errorbar(np.arange(len(diff_sample)), diff_sample, yerr=sem_diff_sample, fmt='-o', color='#871719',
                 ecolor='#DDC0B5', capsize=5, label="Sample")
    plt.legend()
    # plt.savefig(path+"AvgSEM_Diff.svg",format="svg")
    plt.show()

    # Compute differences and pooled SEM
    diff_sample = df['exp_non_statobj_sample'] - df['exp_statobj_sample']
    sem_diff_sample = np.sqrt(df['exp_non_statobj_sample_se'] ** 2 + df['exp_statobj_sample_se'] ** 2)

    diff_test = df['exp_non_statobj_test'] - df['exp_statobj_test']
    sem_diff_test = np.sqrt(df['exp_non_statobj_test_se'] ** 2 + df['exp_statobj_test_se'] ** 2)


    # Create DataFrame of the differences
    df_diff = pd.DataFrame({
        'estimate': pd.concat([diff_sample, diff_test], ignore_index=True),
        'task': ['Sample'] * len(df) + ['Test'] * len(df),
        'time_window': list(range(len(df))) * 2
    })

    df_diff['task'] = pd.Categorical(df_diff['task'], categories=['Sample', 'Test'])
    # OLS model: difference in object exploration between task phases
    model = smf.ols("estimate ~ task * time_window", data=df_diff)
    result = model.fit()
    print(result.summary())

## this function is used in "strech_time_Behaviors" function
def find_startANDendpoints(behavior_sig):
    starts = []
    ends = []
    for sweep in range(0, len(behavior_sig)):
        if np.array_equal([0, 1], np.array(behavior_sig[sweep: sweep + 2])):
            starts.append(sweep + 1)
        if np.array_equal([1, 0], np.array(behavior_sig[sweep: sweep + 2])):
            ends.append(sweep)
    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:len(starts) - 1]
    # print("starts", starts)
    # print("ends", ends)
    return starts, ends


## this function is used in "strech_time_Behaviors" function
def merge_startANDend_points(starts, ends, min_distance):
    st_ed = []
    se_state = [starts[0], ends[0]]
    for se in range(0, len(starts) - 1):
        if starts[se + 1] - ends[se] <= min_distance:
            se_state[1] = ends[se + 1]
        else:
            st_ed.append(se_state)
            se_state = [starts[se + 1], ends[se + 1]]  # ?????
    st_ed.append(se_state)
    print("starts_ends ", st_ed)
    print("number of bouts", len(st_ed))
    return st_ed


## this function is used in "strech_time_Behaviors" function
def remove_small_win(st_ed, min_win_size):
    print("current st_ed", st_ed)
    new_st_ed = []
    for i in range(0, len(st_ed)):
        print(st_ed[i], st_ed[i][1] - st_ed[i][0])
        if st_ed[i][1] - st_ed[i][0] > min_win_size:
            new_st_ed.append(st_ed[i])
    return new_st_ed


## this function is used in "strech_time_Behaviors" function
def remove_long_win(st_ed, min_win_size):
    new_st_ed = []
    for i in range(0, len(st_ed)):
        print(st_ed[i], st_ed[i][1] - st_ed[i][0])
        if st_ed[i][1] - st_ed[i][0] < min_win_size:
            new_st_ed.append(st_ed[i])
    return new_st_ed

def look_final_behavior(all_main, all_sts, all_eds):
    all_main = np.array(all_main)
    all_sts = np.array(all_sts)
    all_eds = np.array(all_eds)

    all_main_sum = np.sum(all_main, axis=0)

    # ## start
    all_sts_sum = np.sum(all_sts, axis=0)

    # ## end
    all_eds_sum = np.sum(all_eds, axis=0)
    ### merge
    merged = np.concatenate((all_sts_sum, all_main_sum, all_eds_sum))

    return merged
def strech_time_Behaviors_only_4_statistics(path, directory_path, name_behavior, whichTask, mouseid, trialid, num_iters_clusterTest):
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0

    streched_baseline_dfof = []
    streched_baseline_pdfof = []
    streched_win = []
    start_offs = []
    end_offs = []

    speed_streched_win = []
    speed_start_offs = []
    speed_end_offs = []

    pdfof_streched_win = []
    pdfof_start_offs = []
    pdfof_end_offs = []

    streched_win_behave0 = []
    start_offs_behave0 = []
    end_offs_behave0 = []

    streched_win_behave1 = []
    start_offs_behave1 = []
    end_offs_behave1 = []

    streched_win_behave2 = []
    start_offs_behave2 = []
    end_offs_behave2 = []

    streched_win_behave3 = []
    start_offs_behave3 = []
    end_offs_behave3 = []

    streched_win_behave4 = []
    start_offs_behave4 = []
    end_offs_behave4 = []

    streched_win_bk = []
    start_bk = []
    end_bk = []

    total_calculated_st = 0
    total_calculated_st_persession = []
    total_win_be = 0
    total_win_persession = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        # labels_file = filepath_exp + '/' + expriment_name + '_labels.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            # Remove any trailing newline character
            line = line.rstrip('\n')

        onset = int(line.split(",")[0])
        onset = onset * sr_analysis
        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 15 * 60 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 15 * 60 * sr_analysis
        ##
        ## metadata for each session
        with open(filepath_exp + "\mouse_id.txt", 'r') as file:
            mouse_ID = file.readline().strip().replace(" ", "")
            n_trial = file.readline().strip().replace(" ", "")
            # task_t = file.readline().strip().replace(" ", "")
            # last character of each
            mouse_ID = int(mouse_ID[-1] if mouse_ID else '')
            n_trial = int(n_trial[-1] if n_trial else '')
            # task_t = int(task_t[-1] if task_t else '')

        if mouseid==0:
            mouse_ID=0

        if trialid==0:
            n_trial=0

        ############################## calulations
        if whichTask == "both":
            task_list = [0, 1]
        elif whichTask == "sample" or whichTask == "Sample":
            task_list = [0]
        elif whichTask == "Test" or whichTask == "test":
            task_list = [1]
        else:
            print("Error Task type")
        if (task_type in task_list) and (mouse_ID == mouseid) and (n_trial == trialid):
            analysis_win = analysis_win - onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            time = data[0][onset:analysis_win]
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            exp_non_statobj = data[3][onset:analysis_win]
            exp_statobj = data[4][onset:analysis_win]
            walking = data[5][onset:analysis_win]
            rearings = data[6][onset:analysis_win]
            groomings = data[7][onset:analysis_win]

            behavior_walking = np.array(walking)
            behavior_grooming = np.array(groomings)
            behavior_rearings = np.array(rearings)
            behavior_exp_statobj = np.array(exp_statobj)
            behavior_exp_non_statobj = np.array(exp_non_statobj)

            result_or = np.logical_not(np.logical_or(behavior_exp_non_statobj, behavior_exp_statobj))
            rearing_noObj = np.logical_and(result_or, behavior_rearings).astype(int)

            behavior_exp_non_statobj_norearing = np.logical_and(behavior_exp_non_statobj,
                                                                np.logical_not(behavior_rearings)).astype(int)
            behavior_exp_statobj_norearing = np.logical_and(behavior_exp_statobj,
                                                            np.logical_not(behavior_rearings)).astype(int)

            # behavior_rearings=rearing_noObj
            list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,
                              behavior_exp_non_statobj]
            behavior_bk = (np.logical_or.reduce([behavior_walking,
                                                 behavior_grooming,
                                                 behavior_rearings,
                                                 behavior_exp_statobj,
                                                 behavior_exp_non_statobj]))

            ##  logical NOT on the OR result
            behavior_bk = np.logical_not(behavior_bk)
            list_behaviors.append(behavior_bk)

            if name_behavior == "behavior_grooming":
                behavior_sig = behavior_grooming
            elif name_behavior == "behavior_walking":
                behavior_sig = behavior_walking
            elif name_behavior == "behavior_rearings":
                behavior_sig = rearing_noObj
            elif name_behavior == "behavior_exp_statobj":
                behavior_sig = behavior_exp_statobj
            elif name_behavior == "behavior_exp_non_statobj":
                behavior_sig = behavior_exp_non_statobj
            else:
                print("error behavior name ")
                break

            min_distance = int(0 * sr_analysis)
            offset = int(5 * sr_analysis)
            min_win_size = int(2 * sr_analysis)
            # max_win_size= int(10*sr_analysis)
            min_dis_toShow = int(4 * sr_analysis)

            ##### predicting the dfof using speed
            window_samples = int(0.5 * sr_analysis)  # number of samples in the 0.5s window
            kernel = np.ones(window_samples) / window_samples

            dfof = np.array(dfof)
            dfof = np.convolve(dfof, kernel, mode='same')  ## with half second.
            dfof = zscore(dfof)

            speed = np.array(speed)
            speed = np.convolve(speed, kernel, mode='same')  ## with half second.

            ## calculate log speed
            log_speed = np.log2(speed + 0.001)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ## model to measure the predicted dfof
            log_speed = log_speed.reshape(-1, 1)
            # Perform linear regression
            model = LinearRegression()
            model.fit(log_speed, dfof)
            # Get the coefficient and intercept
            coefficient = model.coef_[0]
            intercept = model.intercept_
            predicted_dfof = model.predict(log_speed)

            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)

            #### merge start and end points
            st_ed = merge_startANDend_points(starts, ends, min_distance)
            total_win_be += len(st_ed)
            print("len st_ends", len(st_ed))
            total_win_persession.append(len(st_ed))

            ## find the max window size
            # max_window = 0
            # for t in range(0,len(st_ed)):
            #     win_d=st_ed[t][1] - st_ed[t][0] + 1
            #     if win_d > max_window:
            #         max_window=win_d
            # # print("max_window",max_window)
            max_window = 5000

            ## functions
            def start_end_ind(st_ed, y, offset, dfof, behavior_sig, min_dis_toShow):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                flag = 0
                signal = []
                behavior_main = []
                temp_speed = []
                p_dfof = []

                if st_ed[y][1] - st_ed[y][0] > min_win_size:
                    if y < len(st_ed) - 1:
                        if end_idx < len(dfof) and start_idx > 0 and (
                                st_ed[y][0] - st_ed[y - 1][1]) >= min_dis_toShow and (
                                st_ed[y + 1][0] - st_ed[y][1]) >= min_dis_toShow:
                            signal = (dfof[start_idx:end_idx])
                            behavior_main = behavior_sig[start_idx:end_idx]
                            temp_speed = speed[start_idx:end_idx]
                            p_dfof = predicted_dfof[start_idx:end_idx]
                            flag = 1
                    else:
                        if end_idx < len(dfof) and (st_ed[y][0] - st_ed[y - 1][1]) >= min_dis_toShow:
                            signal = dfof[start_idx: end_idx]
                            temp_speed = speed[start_idx: end_idx]
                            p_dfof = predicted_dfof[start_idx: end_idx]
                            behavior_main = behavior_sig[start_idx: end_idx]
                            flag = 1
                # if 0:
                #     plt.plot(p_dfof, color="red")
                #     plt.plot(signal, color="green")
                #
                #     # plt.plot(temp_speed,color="black")
                #     plt.show()
                return signal, flag, behavior_main, temp_speed, p_dfof

            def extract_st_ends_behaviors(behavior_main, offset, len_signal):
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start == 1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start == 0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len_signal - offset: len_signal]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)
                return behavior_win_start, behavior_win_end

            def start_end_ind_strechedmain(st_ed, offset, sig, len_signal):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1

                # Handle negative indices by inserting NaN for first and last windows
                if start_idx < 0:
                    nan_fill = np.full(abs(start_idx), np.nan)  # Fill NaN for negative range
                    signal = np.concatenate([nan_fill, sig[:end_idx]])
                else:
                    signal = sig[start_idx:end_idx]

                # If the end index exceeds the length of dfof, pad with NaN
                if end_idx > len(sig):
                    nan_fill = np.full(end_idx - len(sig), np.nan)
                    signal = np.concatenate([signal, nan_fill])

                # split the signal to st end and main
                start_off = signal[:offset]
                end_off = signal[len_signal - offset:len_signal]
                main_part = signal[offset:len_signal - offset]

                # Create an array of main part signal values (time points)
                original_x = np.arange(len(main_part))
                # Define the new x values (time points) after stretching
                new_x = np.linspace(0, len(main_part) - 1, max_window)
                # Create a function for linear interpolation
                interp_func = interp1d(original_x, main_part, kind='linear')
                # Interpolate the signal at new x values
                new_signal = interp_func(new_x)
                return new_signal, start_off, end_off, signal

            ## Main code
            tempflag = 0
            num_sections = 10
            #  keep track of used index ranges
            used_ranges = []
            used_ranges_dfof = []
            for y in range(0, len(st_ed)):  # len(st_ed)

                signal, flag, behavior_main, temp_speed, p_dfof = start_end_ind(st_ed, y, offset, dfof, behavior_sig,
                                                                                min_dis_toShow)
                total_calculated_st += flag

                if flag == 1:
                    tempflag += 1
                    # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                    len_signal = len(signal)
                    streched_main_signal0, start_off0, end_off0, bewalking = start_end_ind_strechedmain(st_ed, offset,
                                                                                                        list_behaviors[
                                                                                                            0],
                                                                                                        len_signal)
                    streched_main_signal1, start_off1, end_off1, begrooming = start_end_ind_strechedmain(st_ed, offset,
                                                                                                         list_behaviors[
                                                                                                             1],
                                                                                                         len_signal)
                    streched_main_signal2, start_off2, end_off2, berearings = start_end_ind_strechedmain(st_ed, offset,
                                                                                                         list_behaviors[
                                                                                                             2],
                                                                                                         len_signal)
                    streched_main_signal3, start_off3, end_off3, beexp_statobj = start_end_ind_strechedmain(st_ed,
                                                                                                            offset,
                                                                                                            list_behaviors[
                                                                                                                3],
                                                                                                            len_signal)
                    streched_main_signal4, start_off4, end_off4, beexp_non_statobj = start_end_ind_strechedmain(st_ed,
                                                                                                                offset,
                                                                                                                list_behaviors[
                                                                                                                    4],
                                                                                                                len_signal)
                    streched_main_bk, start_offbk, end_offbk, bebk = start_end_ind_strechedmain(st_ed, offset,
                                                                                                list_behaviors[5],
                                                                                                len_signal)

                    streched_win_bk.append(np.nanmean(streched_main_bk.reshape(-1, 20), axis=1))
                    start_bk.append(start_offbk)
                    end_bk.append(end_offbk)

                    streched_win_behave0.append(np.nanmean(streched_main_signal0.reshape(-1, 20), axis=1))
                    start_offs_behave0.append(start_off0)
                    end_offs_behave0.append(end_off0)

                    streched_win_behave1.append(np.nanmean(streched_main_signal1.reshape(-1, 20), axis=1))
                    start_offs_behave1.append(start_off1)
                    end_offs_behave1.append(end_off1)

                    streched_win_behave2.append(np.nanmean(streched_main_signal2.reshape(-1, 20), axis=1))
                    start_offs_behave2.append(start_off2)
                    end_offs_behave2.append(end_off2)

                    streched_win_behave3.append(np.nanmean(streched_main_signal3.reshape(-1, 20), axis=1))
                    start_offs_behave3.append(start_off3)
                    end_offs_behave3.append(end_off3)

                    streched_win_behave4.append(np.nanmean(streched_main_signal4.reshape(-1, 20), axis=1))
                    start_offs_behave4.append(start_off4)
                    end_offs_behave4.append(end_off4)

                    ### Main signal ###
                    ### start and end of behavior for NaN detection
                    behavior_win_start, behavior_win_end = extract_st_ends_behaviors(behavior_main, offset, len_signal)

                    # split the siganl to start and end and main part
                    start_off = signal[:offset]
                    end_off = signal[len_signal - offset:len_signal]
                    main_part = signal[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    original_x = np.arange(len(main_part))
                    # Define the new x values (time points) after stretching
                    new_x = np.linspace(0, len(main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(original_x, main_part, kind='linear')
                    # Interpolate the signal at new x values
                    new_signal = interp_func(new_x)

                    ## save for avg
                    streched_win.append(np.nanmean(new_signal.reshape(-1, 20), axis=1))
                    start_offs.append(behavior_win_start * start_off)
                    end_offs.append(behavior_win_end * end_off)

                    ### Baseline for dfof ###
                    section_length = len(main_part)

                    for _ in range(num_sections):
                        max_attempts = 1000
                        for _ in range(max_attempts):
                            start_index = random.randint(0, len(dfof) - section_length)
                            end_index = start_index + section_length

                            # Check for overlap with any previously selected section
                            overlap = any(not (end_index <= used_start or start_index >= used_end)
                                          for used_start, used_end in used_ranges_dfof)

                            if not overlap:
                                used_ranges_dfof.append((start_index, end_index))
                                cut_section = dfof[start_index:end_index]
                                original_x_base = np.arange(len(cut_section))
                                new_x_base = np.linspace(0, len(cut_section) - 1, max_window)
                                interp_func = interp1d(original_x_base, cut_section, kind='linear')
                                new_base = interp_func(new_x_base)
                                streched_baseline_dfof.append(np.nanmean(new_base.reshape(-1, 20), axis=1))
                                break
                        else:
                            print("Warning: Could not find a non-overlapping section after max attempts.")

                    ### End Main signal ###

                    ###  Speed Signal ###
                    speed_start_off = temp_speed[:offset]
                    speed_end_off = temp_speed[len_signal - offset:len_signal]
                    speed_main_part = temp_speed[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    speed_original_x = np.arange(len(speed_main_part))
                    # Define the new x values (time points) after stretching
                    speed_new_x = np.linspace(0, len(speed_main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(speed_original_x, speed_main_part, kind='linear')
                    # Interpolate the signal at new x values
                    speed_new_signal = interp_func(speed_new_x)

                    ## save for avg
                    speed_streched_win.append(np.nanmean(speed_new_signal.reshape(-1, 20), axis=1))
                    speed_start_offs.append(behavior_win_start * speed_start_off)
                    speed_end_offs.append(behavior_win_end * speed_end_off)

                    ### End Speed Signal ###

                    ###  Predicted dfof Signal ###
                    pdfof_start_off = p_dfof[:offset]
                    pdfof_end_off = p_dfof[len_signal - offset:len_signal]
                    pdfof_main_part = p_dfof[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    pdfof_original_x = np.arange(len(pdfof_main_part))
                    # Define the new x values (time points) after stretching
                    pdfof_new_x = np.linspace(0, len(pdfof_main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(pdfof_original_x, pdfof_main_part, kind='linear')
                    # Interpolate the signal at new x values
                    pdfof_new_signal = interp_func(pdfof_new_x)

                    # plt.plot(pdfof_main_part,color="red")
                    # plt.plot(main_part,color="blue")
                    pdfof_streched_win.append(np.nanmean(pdfof_new_signal.reshape(-1, 20), axis=1))
                    pdfof_start_offs.append(behavior_win_start * pdfof_start_off)
                    pdfof_end_offs.append(behavior_win_end * pdfof_end_off)

                    ## baseline for pdfof
                    section_length = len(pdfof_main_part)

                    for __ in range(num_sections):
                        max_attempts = 1000
                        for __ in range(max_attempts):
                            start_index = random.randint(0, len(predicted_dfof) - section_length)
                            end_index = start_index + section_length

                            # Check for overlap with any previously selected section
                            overlap = any(not (end_index <= used_start or start_index >= used_end)
                                          for used_start, used_end in used_ranges)

                            if not overlap:
                                used_ranges.append((start_index, end_index))
                                cut_section = predicted_dfof[start_index:end_index]
                                original_x_base = np.arange(len(cut_section))
                                new_x_base = np.linspace(0, len(cut_section) - 1, max_window)
                                interp_func = interp1d(original_x_base, cut_section, kind='linear')
                                new_base = interp_func(new_x_base)
                                streched_baseline_pdfof.append(np.nanmean(new_base.reshape(-1, 20), axis=1))
                                break
                        else:
                            print("Warning: Could not find a non-overlapping section after max attempts.")

                    ###  End Predicted dfof Signal ###

            ## go to next expriment file
            total_calculated_st_persession.append(tempflag)
            # plt.show()
        i = i + 1

    print("total_bouts", total_win_be)
    print(total_win_persession)
    print("total_calculated_st", total_calculated_st)
    print(total_calculated_st_persession)

    ###  Presenting arrays of dfof, pdfof, speed and behaviors in all sessions for before, during and after behavior ###
    all_main = np.array(streched_win)
    all_sts = np.array(start_offs)
    all_eds = np.array(end_offs)

    all_base_dfof = np.array(streched_baseline_dfof)
    all_base_pdfof = np.array(streched_baseline_pdfof)

    speed_all_main = np.array(speed_streched_win)
    speed_all_sts = np.array(speed_start_offs)
    speed_all_eds = np.array(speed_end_offs)

    pdfof_all_main = np.array(pdfof_streched_win)
    pdfof_all_sts = np.array(pdfof_start_offs)
    pdfof_all_eds = np.array(pdfof_end_offs)

    final_sig_behavior0 = look_final_behavior(streched_win_behave0, start_offs_behave0, end_offs_behave0)
    final_sig_behavior1 = look_final_behavior(streched_win_behave1, start_offs_behave1, end_offs_behave1)
    final_sig_behavior2 = look_final_behavior(streched_win_behave2, start_offs_behave2, end_offs_behave2)
    final_sig_behavior3 = look_final_behavior(streched_win_behave3, start_offs_behave3, end_offs_behave3)
    final_sig_behavior4 = look_final_behavior(streched_win_behave4, start_offs_behave4, end_offs_behave4)
    final_sig_bk = look_final_behavior(streched_win_bk, start_bk, end_bk)
    ### End Presenting arrays of dfof, pdfof, speed and behaviors in all sessions for before, during and after behavior ###

    ### Plot behavior percentage of all sessions ###
    plt.plot(final_sig_behavior0, color='#1f77b4', label='Locomotion')  # Blue
    plt.plot(final_sig_behavior1, color='#2ca02c', label='Grooming')  # Green
    plt.plot(final_sig_behavior2, color='#9467bd', label='Rearing')  # Purple
    plt.plot(final_sig_behavior3, color='#ff7f0e', label='Exp StatObj')  # Orange
    plt.plot(final_sig_behavior4, color='#8c564b', label='Exp NonStatObj')  # Brown
    plt.plot(final_sig_bk, color='#7f7f7f', label='Background Signal')  # Gray
    plt.legend(fontsize=10)  # title="Behaviors"
    plt.title("Behaviors")
    plt.xlabel("Time")
    plt.ylabel("numbers of each behavior")
    # plt.grid(True)
    # plt.savefig(path + "behaviors_numbers" + name_behavior + ".svg", format="svg")
    plt.show()

    # Plot percentages
    total_counts = (
            final_sig_behavior0 +
            final_sig_behavior1 +
            final_sig_behavior2 +
            final_sig_behavior3 +
            final_sig_behavior4 +
            final_sig_bk
    )
    final_sig_behavior0_percent = (final_sig_behavior0 / total_counts) * 100
    final_sig_behavior1_percent = (final_sig_behavior1 / total_counts) * 100
    final_sig_behavior2_percent = (final_sig_behavior2 / total_counts) * 100
    final_sig_behavior3_percent = (final_sig_behavior3 / total_counts) * 100
    final_sig_behavior4_percent = (final_sig_behavior4 / total_counts) * 100
    final_sig_bk_percent = (final_sig_bk / total_counts) * 100

    plt.plot(final_sig_behavior0_percent, color='#1f77b4', label='Walking')  # Blue
    plt.plot(final_sig_behavior1_percent, color='#2ca02c', label='Grooming')  # Green
    plt.plot(final_sig_behavior2_percent, color='#9467bd', label='Rearing')  # Purple
    plt.plot(final_sig_behavior3_percent, color='#ff7f0e', label='Exp StatObj')  # Orange
    plt.plot(final_sig_behavior4_percent, color='#8c564b', label='Exp NonStatObj')  # Brown
    plt.plot(final_sig_bk_percent, color='#7f7f7f', label='Background Signal')  # Gray
    plt.legend(fontsize=10)
    # plt.title("Behaviors")
    # plt.xlabel("Time")
    plt.ylabel("Percentage of Each Behavior")
    plt.ylim(0, 110)
    plt.xlim(0, len(final_sig_behavior0_percent))
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(path + "behaviors_percentage_" + name_behavior + ".svg", format="svg")
    plt.show()
    ### End Plot behavior percentage of all sessions ###

    ##### Mean and SEM for dfof, pdfof, speed and behaviors across all sessions for before, during and after behavior #####
    ##### then merging them #####

    #### dfof
    # main

    all_main_avg = np.nanmean(all_main, axis=0)
    # all_main_avg = np.convolve(all_main_avg, kernel, mode='same')
    std_main = np.nanstd(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg = np.nanmean(all_sts, axis=0)
    # all_sts_avg = np.convolve(all_sts_avg, kernel, mode='same')
    std_st = np.nanstd(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg = np.nanmean(all_eds, axis=0)
    # all_eds_avg = np.convolve(all_eds_avg, kernel, mode='same')
    std_ed = np.nanstd(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean
    # merge dfof
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    ##### speed
    # main
    all_main_avg_speed = np.nanmean(speed_all_main, axis=0)
    # all_main_avg_speed = np.convolve(all_main_avg_speed, kernel, mode='same')
    std_main_speed = np.nanstd(speed_all_main, axis=0)  # Standard deviation
    n = speed_all_main.shape[0]  # Number of observations
    all_main_sem_speed = std_main_speed / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg_speed = np.nanmean(speed_all_sts, axis=0)
    # all_sts_avg_speed = np.convolve(all_sts_avg_speed, kernel, mode='same')
    std_st_speed = np.nanstd(speed_all_sts, axis=0)  # Standard deviation
    n = speed_all_sts.shape[0]  # Number of observations
    all_st_sem_speed = std_st_speed / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg_speed = np.nanmean(speed_all_eds, axis=0)
    # all_eds_avg_speed = np.convolve(all_eds_avg_speed, kernel, mode='same')
    std_ed_speed = np.nanstd(speed_all_eds, axis=0)  # Standard deviation
    n = speed_all_eds.shape[0]  # Number of observations
    all_ed_sem_speed = std_ed_speed / np.sqrt(n)  # Standard error of the mean
    # merge speed
    merged_speed = np.concatenate((all_sts_avg_speed, all_main_avg_speed, all_eds_avg_speed))
    merge_sem_speed = np.concatenate((all_st_sem_speed, all_main_sem_speed, all_ed_sem_speed))

    ##### predicted dfof
    # main
    all_main_avg_pdfof = np.nanmean(pdfof_all_main, axis=0)
    # all_main_avg_pdfof = np.convolve(all_main_avg_pdfof, kernel, mode='same')
    std_main_pdfof = np.nanstd(pdfof_all_main, axis=0)  # Standard deviation
    n = pdfof_all_main.shape[0]  # Number of observations
    all_main_sem_pdfof = std_main_pdfof / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg_pdfof = np.nanmean(pdfof_all_sts, axis=0)
    # all_sts_avg_pdfof = np.convolve(all_sts_avg_pdfof, kernel, mode='same')
    std_st_pdfof = np.nanstd(pdfof_all_sts, axis=0)  # Standard deviation
    n = pdfof_all_sts.shape[0]  # Number of observations
    all_st_sem_pdfof = std_st_pdfof / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg_pdfof = np.nanmean(pdfof_all_eds, axis=0)
    # all_eds_avg_pdfof = np.convolve(all_eds_avg_pdfof, kernel, mode='same')
    std_ed_pdfof = np.nanstd(pdfof_all_eds, axis=0)  # Standard deviation
    n = pdfof_all_eds.shape[0]  # Number of observations
    all_ed_sem_pdfof = std_ed_pdfof / np.sqrt(n)  # Standard error of the mean
    # merge pdfof
    merged_pdfof = np.concatenate((all_sts_avg_pdfof, all_main_avg_pdfof, all_eds_avg_pdfof))
    merge_sem_pdfof = np.concatenate((all_st_sem_pdfof, all_main_sem_pdfof, all_ed_sem_pdfof))

    #####  END Mean and SEM #####

    ######### Final Plots #########

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot speed signal on the left y-axis
    ax1.set_ylim(0, 14)
    ax1.plot(merged_speed, label='Average speed', linewidth=3, color='#58585A')
    ax1.fill_between(np.arange(len(merged_speed)), merged_speed - merge_sem_speed, merged_speed + merge_sem_speed,
                     color='#D3D3D3', alpha=0.5)
    ax1.set_ylabel('avg(Speed)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Add vertical lines for seperating before , during and after behavior
    ax1.axvline(x=offset, color='gray', linestyle='--')
    ax1.axvline(x=len(merged) - offset, color='gray', linestyle='--')
    ax1.set_xlim([0, len(merged)])

    # Create a twin axis for the second signals on the right y-axis , for dfof and pdfof
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.2, 1.2)
    ## dfof
    ax2.plot(merged, label='Average dfof', linewidth=3, color='#38843F')
    ax2.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#D2E9CD', alpha=0.5)
    ## pdfof
    ax2.plot(merged_pdfof, label='Average predicted dfof', linewidth=3, color='#F1605F')
    ax2.fill_between(np.arange(len(merged_pdfof)), merged_pdfof - merge_sem_pdfof, merged_pdfof + merge_sem_pdfof,
                     color='#F9CDE1', alpha=0.5)
    ax2.set_ylabel('avg(zscore(dfof))', color='#0F8140')
    ax2.tick_params(axis='y', labelcolor='#0F8140')
    ## plot and save
    fig.tight_layout()
    plt.legend()
    plt.title(name_behavior)
    plt.savefig(path + "main_" + name_behavior + ".svg", format="svg")
    plt.show()

    ######### End Final Plots #########

    ########################## Statistics ##########################
    ##### Nonparametric Cluster-Based Permutation Test to assess differences between the two signals temporally #####
    def Cluster_Based_Test(num_iters, signals_condition1, name_sig1, signals_condition2, name_sig2, sr_analysis,
                           plotOption_tvalues, plotOption_histOfClusters, csvPath, name_behavior):

        ## Calculating the t-value time series for the observed two conditions ##
        t_val_1, t_critic_1 = compare_Plots(signals_condition1, name_sig1, signals_condition2, name_sig2, 0.05,
                                            "two-sided",
                                            sr_analysis, 1)
        ## Finding clusters based on largest mass (biggestMass_cluster_obs) and largest size/length (biggestSize_cluster_obs) ##
        largestMass_cluster_obs, sizeOf_largestMass_cluster_obs, index_largeMass_obs, massOf_largestSize_cluster_obs, largestSize_cluster_obs, index_largeSize_obs = cluster_mass_fun(
            t_val_1, t_critic_1, 1)

        print("t_critical------------>", t_critic_1)
        ## report the largest cluster mass and size
        print("largestMass_cluster_obs", largestMass_cluster_obs)
        print("sizeOf_largestMass_cluster_obs", sizeOf_largestMass_cluster_obs)
        print("index_largeMass_obs", index_largeMass_obs)

        print("largestSize_cluster_obs", largestSize_cluster_obs)
        print("massOf_largestSize_cluster_obs", massOf_largestSize_cluster_obs)
        print("index_largeSize_obs", index_largeSize_obs)

        ## report the percentages of the comparision that is significant
        print("percentage of the largest cluster mass %",
              sizeOf_largestMass_cluster_obs / signals_condition2.shape[1] * 100)
        print("percentage of the largest cluster size %", largestSize_cluster_obs / signals_condition2.shape[1] * 100)

        num = signals_condition1.shape[0]  # num of observations

        ### Mean and SEM for signals across all sessions ###
        condition1_avg = np.nanmean(signals_condition1, axis=0)
        std_condition1 = np.nanstd(signals_condition1, axis=0)  # Standard deviation
        condition1_sem = std_condition1 / np.sqrt(num)  # Standard error of the mean

        condition2_avg = np.nanmean(signals_condition2, axis=0)
        std_condition2 = np.nanstd(signals_condition2, axis=0)  # Standard deviation
        condition2_sem = std_condition2 / np.sqrt(num)  # Standard error of the mean

        ## plot
        plt.plot(condition1_avg, label=name_sig1, linewidth=3, color='#38843F')
        plt.fill_between(np.arange(len(condition1_avg)), condition1_avg - condition1_sem,
                         condition1_avg + condition1_sem, color='#D2E9CD', alpha=0.5)
        plt.plot(condition2_avg, label=name_sig2, linewidth=3, color='red')
        plt.fill_between(np.arange(len(condition2_avg)), condition2_avg - condition2_sem,
                         condition2_avg + condition2_sem, color='#F9CDE1', alpha=0.5)
        plt.legend()
        plt.savefig(csvPath + "main_signals" + name_behavior + ".svg", format="svg")
        plt.title(name_behavior)
        plt.show()

        ## Creating the permutation distribution representing the null hypothesis ##
        ## plotOption=0/1 for plotting the t_values or histogram of clusters
        both_gp = np.concatenate((signals_condition1, signals_condition2))
        rnd_shuffled = both_gp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        csv_file = csvPath + "Clusters_" + str(name_behavior) + "_" + str(name_sig1) + "_VS_" + str(
            name_sig2) + f"_{timestamp}.csv"
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as f:
                f.write(
                    "largestMass_cluster_obs,sizeOf_largestMass_cluster_obs,massOf_largestSize_cluster_obs,largestSize_cluster_obs\n")

        for i in range(num_iters):
            print("iter--> ", i)
            np.random.shuffle(rnd_shuffled)

            t_val, t_critic = compare_Plots(rnd_shuffled[0:num, :], "sig1",
                                            rnd_shuffled[num:, :], "sig2", 0.05, "two-sided", sr_analysis,
                                            plotOption_tvalues)
            biggestMass_cluster, sizeOf_biggestMass_cluster, index_largeMass, massOf_biggestSize_cluster, biggestSize_cluster, index_largeSize = cluster_mass_fun(
                t_val, t_critic, plotOption_histOfClusters)

            # save results to the CSV file
            DF = pd.DataFrame({'biggestMass_cluster': [biggestMass_cluster],
                               'sizeOf_biggestMass_cluster': [sizeOf_biggestMass_cluster],
                               'massOf_biggestSize_cluster': [massOf_biggestSize_cluster],
                               'biggestSize_cluster': [biggestSize_cluster]})
            DF.to_csv(csv_file, mode='a', header=False, index=False)

        stat_cluster_test(largestMass_cluster_obs, largestSize_cluster_obs, csvPath, csv_file)
        return csv_file

    ### End of Cluster_Based_Test function ##

    ### Based on which two signals you want to compare, such as dfof and pdfof during, before, and after behaviors,
    # or dfof and baseline during the behavior, you can select different inputs for the Cluster_Based_Test function.
    ## input options:
    ## dfof vs basline during behavior : all_main vs all_base_dfof
    ## predicted dfof vs basline during behavior : pdfof_all_main vs all_base_pdfof
    ## dfof vs predicted dfof during behavior : all_main vs pdfof_all_main
    ## dfof vs predicted dfof before, during, and after behavior : whole_dfof_signal vs whole_dfof_pdfof
    ## dfof vs predicted dfof before and after behavior : BeforeAfter_dfof_signal vs BeforeAfter_dfof_pdfof

    ## uncomment to create your desired inputs
    whole_dfof_signal = np.hstack((all_sts, all_main, all_eds))
    whole_dfof_pdfof = np.hstack((pdfof_all_sts, pdfof_all_main, pdfof_all_eds))
    # BeforeAfter_dfof_signal = np.hstack((all_sts, all_eds))
    # BeforeAfter_dfof_pdfof = np.hstack((pdfof_all_sts, pdfof_all_eds))
    # whole_dfof_speed = np.hstack((speed_all_sts,speed_all_main, speed_all_eds))

    Cluster_Based_Test(num_iters_clusterTest, all_main, "Avg_" + "dfof", pdfof_all_main, "Avg_" + "pdfof", sr_analysis,
                       0, 0, path, name_behavior)

    ########################## End Statistics ##########################

### detect the rise and decay time for grooming and locomotion :
def strech_time_Behaviors_only_4_increaseTime(name_behavior, whichTask, StartOrEnd, activityThreshold,
                                              considered_seconds):
    path = 'E:/lab/Cholinergic Prj/final files/results/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'E:/lab/Cholinergic Prj/everything about the paper/BlancaData/all_30/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0

    streched_baseline_dfof = []
    streched_baseline_pdfof = []
    streched_win = []
    start_offs = []
    end_offs = []

    speed_streched_win = []
    speed_start_offs = []
    speed_end_offs = []

    pdfof_streched_win = []
    pdfof_start_offs = []
    pdfof_end_offs = []

    streched_win_behave0 = []
    start_offs_behave0 = []
    end_offs_behave0 = []

    streched_win_behave1 = []
    start_offs_behave1 = []
    end_offs_behave1 = []

    streched_win_behave2 = []
    start_offs_behave2 = []
    end_offs_behave2 = []

    streched_win_behave3 = []
    start_offs_behave3 = []
    end_offs_behave3 = []

    streched_win_behave4 = []
    start_offs_behave4 = []
    end_offs_behave4 = []

    streched_win_bk = []
    start_bk = []
    end_bk = []

    total_calculated_st = 0
    total_calculated_st_persession = []
    total_win_be = 0
    total_win_persession = []

    index_all_rise = []
    index_all_rise_pdfof = []

    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data = filepath_exp + '/' + expriment_name + '_data.csv'
        # labels_file = filepath_exp + '/' + expriment_name + '_labels.csv'
        #####################################################
        ## details
        with open(filepath_exp + "\details.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            # Remove any trailing newline character
            line = line.rstrip('\n')

        onset = int(line.split(",")[0])
        onset = onset * sr_analysis
        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 15 * 60 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 15 * 60 * sr_analysis
        ##
        ############################## calulations
        if whichTask == "both":
            task_list = [0, 1]
        elif whichTask == "sample" or whichTask == "Sample":
            task_list = [0]
        elif whichTask == "Test" or whichTask == "test":
            task_list = [1]
        else:
            print("Error Task type")
        if task_type in task_list:
            analysis_win = analysis_win - onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            time = data[0][onset:analysis_win]
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            exp_non_statobj = data[3][onset:analysis_win]
            exp_statobj = data[4][onset:analysis_win]
            walking = data[5][onset:analysis_win]
            rearings = data[6][onset:analysis_win]
            groomings = data[7][onset:analysis_win]

            behavior_walking = np.array(walking)
            behavior_grooming = np.array(groomings)
            behavior_rearings = np.array(rearings)
            behavior_exp_statobj = np.array(exp_statobj)
            behavior_exp_non_statobj = np.array(exp_non_statobj)

            result_or = np.logical_not(np.logical_or(behavior_exp_non_statobj, behavior_exp_statobj))
            rearing_noObj = np.logical_and(result_or, behavior_rearings).astype(int)

            behavior_exp_non_statobj_norearing = np.logical_and(behavior_exp_non_statobj,
                                                                np.logical_not(behavior_rearings)).astype(int)
            behavior_exp_statobj_norearing = np.logical_and(behavior_exp_statobj,
                                                            np.logical_not(behavior_rearings)).astype(int)

            # behavior_rearings=rearing_noObj
            list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,
                              behavior_exp_non_statobj]
            behavior_bk = (np.logical_or.reduce([behavior_walking,
                                                 behavior_grooming,
                                                 behavior_rearings,
                                                 behavior_exp_statobj,
                                                 behavior_exp_non_statobj]))

            ##  logical NOT on the OR result
            behavior_bk = np.logical_not(behavior_bk)
            list_behaviors.append(behavior_bk)

            if name_behavior == "behavior_grooming":
                behavior_sig = behavior_grooming
            elif name_behavior == "behavior_walking":
                behavior_sig = behavior_walking
            elif name_behavior == "behavior_rearings":
                behavior_sig = behavior_rearings
            elif name_behavior == "behavior_exp_statobj":
                behavior_sig = behavior_exp_statobj
            elif name_behavior == "behavior_exp_non_statobj":
                behavior_sig = behavior_exp_non_statobj
            else:
                print("error behavior name ")
                break

            min_distance = int(0 * sr_analysis)
            offset = int(5 * sr_analysis)
            min_win_size = int(2 * sr_analysis)
            # max_win_size= int(10*sr_analysis)
            min_dis_toShow = int(4 * sr_analysis)

            ##### predicting the dfof using speed
            window_samples = int(0.5 * sr_analysis)  # number of samples in the 0.5s window
            kernel = np.ones(window_samples) / window_samples

            dfof = np.array(dfof)
            dfof = np.convolve(dfof, kernel, mode='same')  ## with half second.
            dfof = zscore(dfof)

            speed = np.array(speed)
            speed = np.convolve(speed, kernel, mode='same')  ## with half second.

            ## calculate log speed
            log_speed = np.log2(speed + 0.001)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ## model to measure the predicted dfof
            log_speed = log_speed.reshape(-1, 1)
            # Perform linear regression
            model = LinearRegression()
            model.fit(log_speed, dfof)
            # Get the coefficient and intercept
            coefficient = model.coef_[0]
            intercept = model.intercept_
            predicted_dfof = model.predict(log_speed)

            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)

            #### merge start and end points
            st_ed = merge_startANDend_points(starts, ends, min_distance)
            total_win_be += len(st_ed)
            print("len st_ends", len(st_ed))
            total_win_persession.append(len(st_ed))

            ## find the max window size
            # max_window = 0
            # for t in range(0,len(st_ed)):
            #     win_d=st_ed[t][1] - st_ed[t][0] + 1
            #     if win_d > max_window:
            #         max_window=win_d
            # # print("max_window",max_window)
            max_window = 5000
            reshape_size = 20

            ## functions
            def start_end_ind(st_ed, y, offset, dfof, behavior_sig, min_dis_toShow):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                flag = 0
                signal = []
                behavior_main = []
                temp_speed = []
                p_dfof = []

                if st_ed[y][1] - st_ed[y][0] > min_win_size:
                    if y < len(st_ed) - 1:
                        if end_idx < len(dfof) and start_idx > 0 and (
                                st_ed[y][0] - st_ed[y - 1][1]) >= min_dis_toShow and (
                                st_ed[y + 1][0] - st_ed[y][1]) >= min_dis_toShow:
                            signal = dfof[start_idx:end_idx]
                            behavior_main = behavior_sig[start_idx:end_idx]
                            temp_speed = speed[start_idx:end_idx]
                            p_dfof = predicted_dfof[start_idx:end_idx]
                            flag = 1
                    else:
                        if end_idx < len(dfof) and (st_ed[y][0] - st_ed[y - 1][1]) >= min_dis_toShow:
                            signal = dfof[start_idx: end_idx]
                            temp_speed = speed[start_idx: end_idx]
                            p_dfof = predicted_dfof[start_idx: end_idx]
                            behavior_main = behavior_sig[start_idx: end_idx]
                            flag = 1
                # if 0:
                #     plt.plot(p_dfof, color="red")
                #     plt.plot(signal, color="green")
                #
                #     # plt.plot(temp_speed,color="black")
                #     plt.show()
                return signal, flag, behavior_main, temp_speed, p_dfof

            def extract_st_ends_behaviors(behavior_main, offset, len_signal):
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start == 1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start == 0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len_signal - offset: len_signal]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)
                return behavior_win_start, behavior_win_end

            def start_end_ind_strechedmain(st_ed, offset, sig, len_signal):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1

                # Handling negative indices by inserting NaN for first and last windows
                if start_idx < 0:
                    nan_fill = np.full(abs(start_idx), np.nan)  # Fill NaN for negative range
                    signal = np.concatenate([nan_fill, sig[:end_idx]])
                else:
                    signal = sig[start_idx:end_idx]

                # If the end index exceeds the length of dfof, pad with NaN
                if end_idx > len(sig):
                    nan_fill = np.full(end_idx - len(sig), np.nan)
                    signal = np.concatenate([signal, nan_fill])

                # split the signal to st end and main
                start_off = signal[:offset]
                end_off = signal[len_signal - offset:len_signal]
                main_part = signal[offset:len_signal - offset]

                # Creating an array of main part signal values (time points)
                original_x = np.arange(len(main_part))
                # new x values after stretching
                new_x = np.linspace(0, len(main_part) - 1, max_window)
                # a function for linear interpolation
                interp_func = interp1d(original_x, main_part, kind='linear')
                # Interpolate the signal at new x values
                new_signal = interp_func(new_x)
                return new_signal, start_off, end_off, signal

            def find_halfRise_indices(signal, halfRise):
                halfRise_indices = []
                if name_behavior == "behavior_grooming":
                    for i in range(0, len(signal) - 1):
                        if signal[i + 1] <= halfRise:  # and signal[i] <= halfRise:
                            halfRise_indices.append(i + 1)
                elif name_behavior == "behavior_walking":
                    for i in range(0, len(signal) - 1):
                        if signal[i + 1] >= halfRise:  # and signal[i] <= halfRise:
                            halfRise_indices.append(i + 1)
                else:
                    print("no behavior detected")
                return halfRise_indices  # [np.nanmedian(np.array(halfRise_indices))]

            ## Main code

            tempflag = 0
            win_during = int(1 * sr_analysis)

            for y in range(0, len(st_ed)):  # len(st_ed)

                signal, flag, behavior_main, temp_speed, p_dfof = start_end_ind(st_ed, y, offset, dfof, behavior_sig,
                                                                                min_dis_toShow)
                total_calculated_st += flag
                if flag == 1:
                    tempflag += 1
                    # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                    len_signal = len(signal)
                    streched_main_signal0, start_off0, end_off0, bewalking = start_end_ind_strechedmain(st_ed, offset,
                                                                                                        list_behaviors[
                                                                                                            0],
                                                                                                        len_signal)
                    streched_main_signal1, start_off1, end_off1, begrooming = start_end_ind_strechedmain(st_ed, offset,
                                                                                                         list_behaviors[
                                                                                                             1],
                                                                                                         len_signal)
                    streched_main_signal2, start_off2, end_off2, berearings = start_end_ind_strechedmain(st_ed, offset,
                                                                                                         list_behaviors[
                                                                                                             2],
                                                                                                         len_signal)
                    streched_main_signal3, start_off3, end_off3, beexp_statobj = start_end_ind_strechedmain(st_ed,
                                                                                                            offset,
                                                                                                            list_behaviors[
                                                                                                                3],
                                                                                                            len_signal)
                    streched_main_signal4, start_off4, end_off4, beexp_non_statobj = start_end_ind_strechedmain(st_ed,
                                                                                                                offset,
                                                                                                                list_behaviors[
                                                                                                                    4],
                                                                                                                len_signal)
                    streched_main_bk, start_offbk, end_offbk, bebk = start_end_ind_strechedmain(st_ed, offset,
                                                                                                list_behaviors[5],
                                                                                                len_signal)

                    streched_win_bk.append(np.nanmean(streched_main_bk.reshape(-1, reshape_size), axis=1))
                    start_bk.append(start_offbk)
                    end_bk.append(end_offbk)

                    streched_win_behave0.append(np.nanmean(streched_main_signal0.reshape(-1, reshape_size), axis=1))
                    start_offs_behave0.append(start_off0)
                    end_offs_behave0.append(end_off0)

                    streched_win_behave1.append(np.nanmean(streched_main_signal1.reshape(-1, reshape_size), axis=1))
                    start_offs_behave1.append(start_off1)
                    end_offs_behave1.append(end_off1)

                    streched_win_behave2.append(np.nanmean(streched_main_signal2.reshape(-1, reshape_size), axis=1))
                    start_offs_behave2.append(start_off2)
                    end_offs_behave2.append(end_off2)

                    streched_win_behave3.append(np.nanmean(streched_main_signal3.reshape(-1, reshape_size), axis=1))
                    start_offs_behave3.append(start_off3)
                    end_offs_behave3.append(end_off3)

                    streched_win_behave4.append(np.nanmean(streched_main_signal4.reshape(-1, reshape_size), axis=1))
                    start_offs_behave4.append(start_off4)
                    end_offs_behave4.append(end_off4)

                    ### Main signal ###
                    ### start and end of behavior for NaN detection
                    behavior_win_start, behavior_win_end = extract_st_ends_behaviors(behavior_main, offset, len_signal)

                    # split the siganl to start and end and main part
                    start_off = signal[:offset]
                    end_off = signal[len_signal - offset:len_signal]
                    main_part = signal[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    original_x = np.arange(len(main_part))
                    # Define the new x values (time points) after stretching
                    new_x = np.linspace(0, len(main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(original_x, main_part, kind='linear')
                    # Interpolate the signal at new x values
                    new_signal = interp_func(new_x)

                    ## save for avg
                    streched_win.append(np.nanmean(new_signal.reshape(-1, reshape_size), axis=1))
                    start_offs.append(behavior_win_start * start_off)
                    end_offs.append(behavior_win_end * end_off)

                    if StartOrEnd:
                        # rise time dfof
                        halfrise = (np.nanmean(start_offs[-1]) + np.nanmean(main_part)) * activityThreshold
                        halfRise_indices_st = find_halfRise_indices(
                            start_offs[-1][len(start_offs[-1]) - considered_seconds * win_during:len(start_offs[-1])],
                            halfrise)
                        halfRise_indices_main = find_halfRise_indices(main_part[:win_during], halfrise)
                        all_index = [-abs(x) for x in halfRise_indices_st] + halfRise_indices_main
                        index_all_rise.append(np.nanmedian(all_index))
                    else:
                        # for the end
                        halfrise = (np.nanmean(end_offs[-1]) + np.nanmean(main_part)) * activityThreshold
                        halfRise_indices_end = find_halfRise_indices(end_offs[-1][:considered_seconds * win_during],
                                                                     halfrise)
                        halfRise_indices_main = find_halfRise_indices(
                            main_part[len(main_part) - win_during:len(main_part)], halfrise)
                        all_index = [-abs(x) for x in halfRise_indices_main] + halfRise_indices_end
                        index_all_rise.append(np.nanmedian(all_index))

                    # print(all_index)
                    # start_signal=start_offs[-1]
                    # plt.plot(np.concatenate((start_signal, main_part, behavior_win_end * end_off)), color="red")
                    # plot_sig=np.concatenate((start_signal,main_part[:win_during]))
                    # plt.plot(plot_sig)
                    # for y in all_index:
                    #     plt.axhline(y=plot_sig[y], color='r', linestyle='--')  # Customize color and style if you want
                    # plt.axvline(x=len(start_signal), color='green', linestyle='--')
                    # plt.axvline(x=len(start_signal)+len(main_part), color='green', linestyle='--')
                    # plt.text(1, 1, "halfRise:"+str(halfrise), fontsize=12, color='blue')
                    # plt.show()

                    ### End Main signal ###

                    ###  Speed Signal ###
                    speed_start_off = temp_speed[:offset]
                    speed_end_off = temp_speed[len_signal - offset:len_signal]
                    speed_main_part = temp_speed[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    speed_original_x = np.arange(len(speed_main_part))
                    # Define the new x values (time points) after stretching
                    speed_new_x = np.linspace(0, len(speed_main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(speed_original_x, speed_main_part, kind='linear')
                    # Interpolate the signal at new x values
                    speed_new_signal = interp_func(speed_new_x)

                    ## save for avg
                    speed_streched_win.append(np.nanmean(speed_new_signal.reshape(-1, reshape_size), axis=1))
                    speed_start_offs.append(behavior_win_start * speed_start_off)
                    speed_end_offs.append(behavior_win_end * speed_end_off)

                    ### End Speed Signal ###

                    ###  Predicted dfof Signal ###
                    pdfof_start_off = p_dfof[:offset]
                    pdfof_end_off = p_dfof[len_signal - offset:len_signal]
                    pdfof_main_part = p_dfof[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    pdfof_original_x = np.arange(len(pdfof_main_part))
                    # Define the new x values (time points) after stretching
                    pdfof_new_x = np.linspace(0, len(pdfof_main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(pdfof_original_x, pdfof_main_part, kind='linear')
                    # Interpolate the signal at new x values
                    pdfof_new_signal = interp_func(pdfof_new_x)

                    pdfof_streched_win.append(np.nanmean(pdfof_new_signal.reshape(-1, reshape_size), axis=1))
                    pdfof_start_offs.append(behavior_win_start * pdfof_start_off)
                    pdfof_end_offs.append(behavior_win_end * pdfof_end_off)

                    if StartOrEnd:
                        ## rise time predicted dfof
                        halfrise = (np.nanmean(pdfof_start_offs[-1]) + np.nanmean(pdfof_main_part)) * activityThreshold
                        halfRise_indices_st = find_halfRise_indices(pdfof_start_offs[-1][len(
                            pdfof_start_offs[-1]) - considered_seconds * win_during:len(pdfof_start_offs[-1])],
                                                                    halfrise)
                        halfRise_indices_main = find_halfRise_indices(pdfof_main_part[:win_during], halfrise)
                        all_index = [-abs(x) for x in halfRise_indices_st] + halfRise_indices_main
                        index_all_rise_pdfof.append(np.nanmedian(all_index))
                    else:
                        # end
                        halfrise = (np.nanmean(pdfof_end_offs[-1]) + np.nanmean(pdfof_main_part)) * activityThreshold
                        halfRise_indices_end = find_halfRise_indices(
                            pdfof_end_offs[-1][:considered_seconds * win_during], halfrise)
                        halfRise_indices_main = find_halfRise_indices(
                            pdfof_main_part[len(pdfof_main_part) - win_during:len(pdfof_main_part)], halfrise)
                        all_index = [-abs(x) for x in halfRise_indices_main] + halfRise_indices_end
                        index_all_rise_pdfof.append(np.nanmedian(all_index))

                    # print(all_index)
                    # start_signal=pdfof_start_offs[-1]
                    # plt.plot(np.concatenate((start_signal, pdfof_main_part, behavior_win_end * pdfof_end_off)), color="red")
                    # plot_sig=np.concatenate((start_signal,pdfof_main_part[:win_during]))
                    # plt.plot(plot_sig)
                    # for y in all_index:
                    #     plt.axhline(y=plot_sig[y], color='r', linestyle='--')  # Customize color and style if you want
                    # plt.axvline(x=len(start_signal), color='green', linestyle='--')
                    # plt.axvline(x=len(start_signal)+len(pdfof_main_part), color='green', linestyle='--')
                    # plt.text(1, 1, "halfRise:"+str(halfrise), fontsize=12, color='blue')
                    # plt.show()

                    ###  End Predicted dfof Signal ###

            ## go to next expriment file
            total_calculated_st_persession.append(tempflag)

        i = i + 1

    print("all indexxx", index_all_rise)
    print("median rise time dfof", np.nanmedian(index_all_rise))
    # print("mode rise time dfof", stats.mode(index_all_rise)[0])
    plt.hist(index_all_rise)
    plt.show()

    print("all indexxx", index_all_rise_pdfof)
    print("median rise time pdfof", np.nanmedian(index_all_rise_pdfof))
    # print("mode rise time pdfof", stats.mode(index_all_rise_pdfof)[0])
    plt.hist(index_all_rise_pdfof)
    plt.show()

    max_len = max(len(index_all_rise), len(index_all_rise_pdfof))
    index_all_rise += [np.nan] * (max_len - len(index_all_rise))
    index_all_rise_pdfof += [np.nan] * (max_len - len(index_all_rise_pdfof))

    df = pd.DataFrame({
        'dfof': index_all_rise,
        'pdfof': index_all_rise_pdfof
    })
    df.to_csv('E:/lab/Cholinergic Prj/final files/results/time_' + str(activityThreshold) + name_behavior + str(
        StartOrEnd) + str(considered_seconds) + '.csv', index=False)

    print("total_bouts", total_win_be)
    print(total_win_persession)
    print("total_calculated_st", total_calculated_st)
    print(total_calculated_st_persession)

    ###  Presenting arrays of dfof, pdfof, speed and behaviors in all sessions for before, during and after behavior ###
    all_main = np.array(streched_win)
    all_sts = np.array(start_offs)
    all_eds = np.array(end_offs)

    all_base_dfof = np.array(streched_baseline_dfof)
    all_base_pdfof = np.array(streched_baseline_pdfof)

    speed_all_main = np.array(speed_streched_win)
    speed_all_sts = np.array(speed_start_offs)
    speed_all_eds = np.array(speed_end_offs)

    pdfof_all_main = np.array(pdfof_streched_win)
    pdfof_all_sts = np.array(pdfof_start_offs)
    pdfof_all_eds = np.array(pdfof_end_offs)

    final_sig_behavior0 = look_final_behavior(streched_win_behave0, start_offs_behave0, end_offs_behave0)
    final_sig_behavior1 = look_final_behavior(streched_win_behave1, start_offs_behave1, end_offs_behave1)
    final_sig_behavior2 = look_final_behavior(streched_win_behave2, start_offs_behave2, end_offs_behave2)
    final_sig_behavior3 = look_final_behavior(streched_win_behave3, start_offs_behave3, end_offs_behave3)
    final_sig_behavior4 = look_final_behavior(streched_win_behave4, start_offs_behave4, end_offs_behave4)
    final_sig_bk = look_final_behavior(streched_win_bk, start_bk, end_bk)
    ### End Presenting arrays of dfof, pdfof, speed and behaviors in all sessions for before, during and after behavior ###

    ### Plot behavior percentage of all sessions ###
    plt.plot(final_sig_behavior0, color='#1f77b4', label='Locomotion')  # Blue
    plt.plot(final_sig_behavior1, color='#2ca02c', label='Grooming')  # Green
    plt.plot(final_sig_behavior2, color='#9467bd', label='Rearing')  # Purple
    plt.plot(final_sig_behavior3, color='#ff7f0e', label='Exp StatObj')  # Orange
    plt.plot(final_sig_behavior4, color='#8c564b', label='Exp NonStatObj')  # Brown
    plt.plot(final_sig_bk, color='#7f7f7f', label='Background Signal')  # Gray
    plt.legend(fontsize=10)  # title="Behaviors"
    plt.title("Behaviors")
    plt.xlabel("Time")
    plt.ylabel("numbers of each behavior")
    # plt.grid(True)
    plt.savefig(path + "behaviors_numbers" + name_behavior + ".svg", format="svg")
    plt.show()

    # Plot percentages
    total_counts = (
            final_sig_behavior0 +
            final_sig_behavior1 +
            final_sig_behavior2 +
            final_sig_behavior3 +
            final_sig_behavior4 +
            final_sig_bk
    )

    final_sig_behavior0_percent = (final_sig_behavior0 / total_counts) * 100
    final_sig_behavior1_percent = (final_sig_behavior1 / total_counts) * 100
    final_sig_behavior2_percent = (final_sig_behavior2 / total_counts) * 100
    final_sig_behavior3_percent = (final_sig_behavior3 / total_counts) * 100
    final_sig_behavior4_percent = (final_sig_behavior4 / total_counts) * 100
    final_sig_bk_percent = (final_sig_bk / total_counts) * 100

    plt.plot(final_sig_behavior0_percent, color='#1f77b4', label='Walking')  # Blue
    plt.plot(final_sig_behavior1_percent, color='#2ca02c', label='Grooming')  # Green
    plt.plot(final_sig_behavior2_percent, color='#9467bd', label='Rearing')  # Purple
    plt.plot(final_sig_behavior3_percent, color='#ff7f0e', label='Exp StatObj')  # Orange
    plt.plot(final_sig_behavior4_percent, color='#8c564b', label='Exp NonStatObj')  # Brown
    plt.plot(final_sig_bk_percent, color='#7f7f7f', label='Background Signal')  # Gray
    plt.legend(fontsize=10)
    # plt.title("Behaviors")
    # plt.xlabel("Time")
    plt.ylabel("Percentage of Each Behavior")
    plt.ylim(0, 110)
    plt.xlim(0, len(final_sig_behavior0_percent))
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(path + "behaviors_percentage" + name_behavior + ".svg", format="svg")
    plt.show()
    ### End Plot behavior percentage of all sessions ###

    ##### Mean and SEM for dfof, pdfof, speed and behaviors across all sessions for before, during and after behavior #####
    ##### then merging them #####

    #### dfof
    # main

    all_main_avg = np.nanmean(all_main, axis=0)
    # all_main_avg = np.convolve(all_main_avg, kernel, mode='same')
    std_main = np.nanstd(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg = np.nanmean(all_sts, axis=0)
    # all_sts_avg = np.convolve(all_sts_avg, kernel, mode='same')
    std_st = np.nanstd(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg = np.nanmean(all_eds, axis=0)
    # all_eds_avg = np.convolve(all_eds_avg, kernel, mode='same')
    std_ed = np.nanstd(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean
    # merge dfof
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    ##### speed
    # main
    all_main_avg_speed = np.nanmean(speed_all_main, axis=0)
    # all_main_avg_speed = np.convolve(all_main_avg_speed, kernel, mode='same')
    std_main_speed = np.nanstd(speed_all_main, axis=0)  # Standard deviation
    n = speed_all_main.shape[0]  # Number of observations
    all_main_sem_speed = std_main_speed / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg_speed = np.nanmean(speed_all_sts, axis=0)
    # all_sts_avg_speed = np.convolve(all_sts_avg_speed, kernel, mode='same')
    std_st_speed = np.nanstd(speed_all_sts, axis=0)  # Standard deviation
    n = speed_all_sts.shape[0]  # Number of observations
    all_st_sem_speed = std_st_speed / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg_speed = np.nanmean(speed_all_eds, axis=0)
    # all_eds_avg_speed = np.convolve(all_eds_avg_speed, kernel, mode='same')
    std_ed_speed = np.nanstd(speed_all_eds, axis=0)  # Standard deviation
    n = speed_all_eds.shape[0]  # Number of observations
    all_ed_sem_speed = std_ed_speed / np.sqrt(n)  # Standard error of the mean
    # merge speed
    merged_speed = np.concatenate((all_sts_avg_speed, all_main_avg_speed, all_eds_avg_speed))
    merge_sem_speed = np.concatenate((all_st_sem_speed, all_main_sem_speed, all_ed_sem_speed))

    ##### predicted dfof
    # main
    all_main_avg_pdfof = np.nanmean(pdfof_all_main, axis=0)
    # all_main_avg_pdfof = np.convolve(all_main_avg_pdfof, kernel, mode='same')
    std_main_pdfof = np.nanstd(pdfof_all_main, axis=0)  # Standard deviation
    n = pdfof_all_main.shape[0]  # Number of observations
    all_main_sem_pdfof = std_main_pdfof / np.sqrt(n)  # Standard error of the mean
    # start
    all_sts_avg_pdfof = np.nanmean(pdfof_all_sts, axis=0)
    # all_sts_avg_pdfof = np.convolve(all_sts_avg_pdfof, kernel, mode='same')
    std_st_pdfof = np.nanstd(pdfof_all_sts, axis=0)  # Standard deviation
    n = pdfof_all_sts.shape[0]  # Number of observations
    all_st_sem_pdfof = std_st_pdfof / np.sqrt(n)  # Standard error of the mean
    # end
    all_eds_avg_pdfof = np.nanmean(pdfof_all_eds, axis=0)
    # all_eds_avg_pdfof = np.convolve(all_eds_avg_pdfof, kernel, mode='same')
    std_ed_pdfof = np.nanstd(pdfof_all_eds, axis=0)  # Standard deviation
    n = pdfof_all_eds.shape[0]  # Number of observations
    all_ed_sem_pdfof = std_ed_pdfof / np.sqrt(n)  # Standard error of the mean
    # merge pdfof
    merged_pdfof = np.concatenate((all_sts_avg_pdfof, all_main_avg_pdfof, all_eds_avg_pdfof))
    merge_sem_pdfof = np.concatenate((all_st_sem_pdfof, all_main_sem_pdfof, all_ed_sem_pdfof))

    #####  END Mean and SEM #####

    ######### Final Plots #########

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot speed signal on the left y-axis
    ax1.set_ylim(0, 16)
    ax1.plot(merged_speed, label='Average speed', linewidth=3, color='#58585A')
    ax1.fill_between(np.arange(len(merged_speed)), merged_speed - merge_sem_speed, merged_speed + merge_sem_speed,
                     color='#D3D3D3', alpha=0.5)
    ax1.set_ylabel('avg(Speed)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Add vertical lines for seperating before , during and after behavior
    ax1.axvline(x=offset, color='gray', linestyle='--')
    ax1.axvline(x=len(merged) - offset, color='gray', linestyle='--')
    ax1.set_xlim([0, len(merged)])

    # Create a twin axis for the second signals on the right y-axis , for dfof and pdfof
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.2, 1.2)
    ## dfof
    ax2.plot(merged, label='Average dfof', linewidth=3, color='#38843F')
    ax2.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#D2E9CD', alpha=0.5)
    ## pdfof
    ax2.plot(merged_pdfof, label='Average predicted dfof', linewidth=3, color='#F1605F')
    ax2.fill_between(np.arange(len(merged_pdfof)), merged_pdfof - merge_sem_pdfof, merged_pdfof + merge_sem_pdfof,
                     color='#F9CDE1', alpha=0.5)
    ax2.set_ylabel('avg(zscore(dfof))', color='#0F8140')
    ax2.tick_params(axis='y', labelcolor='#0F8140')
    ## plot and save
    fig.tight_layout()
    plt.legend()
    plt.savefig(path + "main_signals" + name_behavior + ".svg", format="svg")
    plt.title(name_behavior)
    plt.show()

    ######### End Final Plots #########


if __name__ == '__main__':
    print("Cholinergic Activity")
    path = 'E:/lab/Cholinergic Prj/final files/results/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'E:/lab/Cholinergic Prj/everything about the paper/BlancaData/all_30/'  ## path to the data
    # plot_speed_dfof(path, directory_path,1)
    # find_best_smoothing_corr(path, directory_path)
    # novel_env(path, directory_path)
    # time_analysis_washout(path, directory_path)
    # behavioral_interactions(path, directory_path)
    # Create_dataset_for_LLM(path, directory_path)
    # bar_plot_timewashedLLM(path)

    # name_behavior= behavior_walking,behavior_grooming,behavior_rearings,behavior_exp_statobj,behavior_exp_non_statobj
    # csv_file = strech_time_Behaviors_only_4_statistics(path, directory_path, "behavior_walking", "both", 0, 0, 3)
    # stat_cluster_test(2297.2, 567, path , csv_file)

    # StartOrEnd = 1 # 1 onset and 0 offset
    # activityThreshold = 0.98 # %
    # considered_seconds = 3 # seconds
    # strech_time_Behaviors_only_4_increaseTime("behavior_grooming", "both", StartOrEnd,
    #                                           activityThreshold, considered_seconds)
