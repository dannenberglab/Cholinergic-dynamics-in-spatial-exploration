import re
from FiberPhotometry.PhotometrySignal import *
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress, zscore
from scipy import stats
import seaborn as sns
from scipy.interpolate import interp1d
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import cv2
import statsmodels.formula.api as smf

## this function generated csv file including data required for generating figure 3 of the paper
def time_analysis_washout():
    path = 'C:/Users/ffarokhi/Desktop/test/' ## path for saving the result
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/' ## path to the data

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
    list_expriment_name=[]
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
            analysis_win = 60 * 15 * sr_analysis ## !!!! for summary change 15 min to 12 min and comment the washout part
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
        all_ratio[f'iter_{i}'] = pd.Series(ratio)
        ## !!summary part:
        # total_non_stat.append(sum(exp_non_statobj))
        # total_stat.append(sum(exp_statobj))
        # total_ratio.append((total_non_stat[i]-total_stat[i])/(total_non_stat[i]+total_stat[i]))
        i = i + 1
    # Save the DataFrame to a CSV file
    output_filename = path + "wash_time_new.csv"
    all_ratio.to_csv(output_filename, index=False)

    ## !!summary part: Save total_non_stat, total_stat, and total_ratio to a new CSV file
    # summary_data = pd.DataFrame({
    #     'list_expriment_name':list_expriment_name,
    #     'Task': list_task,
    #     'Total_Non_Stat': total_non_stat,
    #     'Total_Stat': total_stat,
    #     'Total_Ratio': total_ratio
    # })
    # summary_filename = path + "summary_time_new.csv"
    # summary_data.to_csv(summary_filename, index=False)

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

    # valid_indices = np.where((scatter_speed > 0))  # & (scatter_speed < 60)
    # scatter_speed = scatter_speed[valid_indices]
    # scatter_dfof = scatter_dfof[valid_indices]

    # Perform curve fitting
    if log:
        popt, pcov = curve_fit(linear_function, scatter_speed, scatter_dfof)
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
    # plt.savefig(path + "scatter_logspeed_" + expriment_name + ".svg", format="svg")
    plt.show()


## this function "avg_scatter_plot_dfof_speed" is used in another function called plot_speed_dfof) for figure 2
def avg_scatter_plot_dfof_speed(path, expriment_name, dfof, speed, smooth_time_window, sr_analysis, log,bin_size):
    ## smoothing signals
    if smooth_time_window>0:
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
        valid_indices = np.where((smoothed_speed<5)) #(smoothed_speed > 0) &
    else:
        valid_indices = np.where((smoothed_speed<30)) #(smoothed_speed > 0) &

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
def plot_speed_dfof():
    path = 'C:/Users/ffarokhi/Desktop/test/' ## path for saving the result
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/' ## path to the data
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))] # Folder containing different sessions of the experiment.
    i = 0
    list_exp = []
    R_list=[]
    mean_dfof_exp=[]
    mean_dfof_exp_sample=[]
    mean_dfof_exp_test=[]
    merged_data=[]

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

        with open(filepath_exp + "\mouse_id.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        mouseID=int(line)

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

        ### uncomment this part to plot figure 2 A and D samples --> plot correlations
        # R, f1 = show_speed_dfof(time, dfof, speed, 1, sr_analysis, 1)
        # plt.savefig(path+"dfof_logspeed4_"+expriment_name+".svg",format="svg")
        # plt.show()
        # R_list.append(R)
        # df = pd.DataFrame({
        #     'Name_Exp': list_exp,
        #     'R': R_list
        # })
        # df.to_csv(path+'Rlog.csv', index=False)

        ### uncomment this part to plot figure 2 B and E samples --> scatter plots
        # scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, 1)

        ### uncomment this part to plot figure 2 C and F samples --> avg scatter plots
        bin_size=0.2 ## if you change the function to use movement speed instead of log of speed, you need to change bin size as well
        mean_dfof=avg_scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, 1, bin_size)
        mean_dfof_exp.append(mean_dfof[0:30])
        if task == 'Learning' or task == 'learning':
            mean_dfof_exp_sample.append(mean_dfof[0:30])
        else:
            mean_dfof_exp_test.append(mean_dfof[0:30])
        print(len(mean_dfof))

        ## the following part creates dataframes for each session to be used in LMM for figure 2F
        # Determine if the task is 'sample' or 'test'
        smooth_time_window=0.5
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_speed = np.convolve(speed, kernel, mode='same')
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')
        log_speed = np.log2(smoothed_speed + 0.01)
        if np.isnan(log_speed).any() or np.isinf(log_speed).any():
            ## interpolate NanN
            df_log = pd.DataFrame({'log_speed': log_speed})
            log_speed = df_log.interpolate(method='linear')['log_speed'].values

        group = 0 if task_type == 0 else 1
        # Create a temporary DataFrame for the current experiment
        temp_df = pd.DataFrame({
            'dfof': stats.zscore(smoothed_dfof),
            'speed': log_speed,
            'group': group,
            'experiment':i,
            'mouseid':mouseID
        })
        # Append the temporary DataFrame to the list
        merged_data.append(temp_df)

        i += 1

    df = pd.concat(merged_data, ignore_index=True)
    # avg Plot
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

    #LMM
    model = smf.mixedlm(
        "dfof ~ speed * group",  # fixed effects: speed, group, interaction
        data=df,
        groups=df["mouseid"],  # random effects grouped by mouse id and expriment id
        vc_formula={"experiment": "0 + C(experiment)"}
    )
    result = model.fit()
    print(result.summary())

## this function is for generating plot in figure 2 G of the paper
def find_best_smoothing_corr():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/'
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
        smooth_time_window=1
        log=1
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

        x=[0.25,0.5,1,2,4,8,16,32,64,128,256]
        plt.plot(x, rcorr)
        plt.scatter(x, rcorr)
        plt.xlabel("window size(s) for smoothing")
        plt.ylabel("R")
        plt.savefig(path+"R_30"+expriment_name+".svg",format="svg")
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
    # plt.savefig(path + "mean_best_corr" + ".svg", format="svg")
    plt.show()

## use this function "novel_env" to plot figure 2 H and I
def novel_env():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/' # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals

    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))] # Folder containing different sessions of the experiment.
    i=0
    speed_list=[]
    dfof_list=[]
    list_exp=[]
    corrected_dfof_list=[]
    corrected_dfof_fit_list=[]
    coeffs=[]
    predicted_dfof=[]
    win_plt=int(14 * 60 * sr_analysis)
    tau_speed_list=[]
    tau_dfof_list=[]
    tau_corrected_dfof_list=[]
    tau_predicted_dfof_list=[]

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
            analysis_win=analysis_win-onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            time = data[0][onset:analysis_win]
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            smooth_time_window=1
            log=1
            window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
            kernel = np.ones(window_samples) / window_samples
            speed = np.convolve(speed, kernel, mode='same')
            dfof= np.convolve(dfof, kernel, mode='same')


            ## measuring the log of speed
            if log:  # log=1/0
                log_speed = np.log2(speed + 0.01)
                if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                ## interpolate NanN
                    df_log = pd.DataFrame({'log_speed': log_speed})
                    log_speed = df_log.interpolate(method='linear')['log_speed'].values

            log_speed = log_speed.reshape(-1, 1)
            # Perform linear regression
            model = LinearRegression()
            model.fit(log_speed, dfof)
            # Get the coefficient and intercept
            coefficient = model.coef_[0]
            intercept = model.intercept_
            pdfof = model.predict(log_speed)
            corrected_dfof = dfof - pdfof

            corrected_dfof_list.append(stats.zscore(corrected_dfof)[:win_plt])
            predicted_dfof.append(stats.zscore(pdfof)[:win_plt])
            speed_list.append(speed[:win_plt])
            dfof_list.append(stats.zscore(dfof)[:win_plt])

            ### uncomment this if you want to plot each single session
            # signals = [dfof[:win_plt], corrected_dfof[:win_plt], pdfof[:win_plt], speed[:win_plt]]
            # tau_speed,tau_dfof,tau_corrected_dfof,tau_predicted_dfof = show_speed_dfof_decays_signle(time[:win_plt], signals, smooth_time_window, sr_analysis)
            # # plt.savefig(path + "decays" + str(expriment_name) + ".svg", format="svg")
            # plt.show()
            # tau_speed_list.append(tau_speed)
            # tau_dfof_list.append(tau_dfof)
            # tau_corrected_dfof_list.append(tau_corrected_dfof)
            # tau_predicted_dfof_list.append(tau_predicted_dfof)
        i+=1

    ## uncomment this part to save taus in a csv file
    # data = {
    #     'exp_name': list_exp,
    #     'tau_speed': tau_speed_list,
    #     'tau_dfof': tau_dfof_list,
    #     'tau_corrected_dfof': tau_corrected_dfof_list,
    #     'tau_predicted_dfof': tau_predicted_dfof_list
    # }
    # df = pd.DataFrame(data)
    # df.to_csv(path+'combined_results.csv', index=False)

    speed_list=np.array(speed_list)
    avg_speed=np.mean(speed_list,axis=0)
    std_speed = np.std(speed_list, axis=0)  # Standard deviation
    n = speed_list.shape[0]  # Number of observations
    speed_sem = std_speed / np.sqrt(n)  # Standard error of the mean

    dfof_list = np.array(dfof_list)
    avg_dfof=np.mean(dfof_list,axis=0)
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

    signals=[avg_dfof,corrected_dfof,predicted_dfof_avg,avg_speed]
    sems=[dfof_sem,corrected_dfof_sem,predicted_dfof_sem,speed_sem]
    # Rlog = show_speed_dfof_decays_seperate(time[:win_plt], signals, sems, 1, sr_analysis, path)
    Rlog = show_speed_dfof_decays(time[:win_plt], signals, sems, 1, sr_analysis)
    plt.savefig(path + "decays_avg.svg", format="svg")
    plt.show()

## use this function to plot figure 4 A and B
def behavioral_interactions():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    num_exp=0
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
        if task_type == 1 : #or task_type == 0:  ## change this to 0 for plotting Sample sessions and 1 for Test sessions
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
            behavior_names = ['Exp Non Statobj', 'Exp Statobj', 'Walking', 'Rearings', 'Groomings']

            # calculating of the interactions
            for y in range(len(time_series)):
                for j in range(len(time_series)):
                    interaction_matrix[y, j] = np.sum(time_series[y] & time_series[j])
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
            num_exp+=1
        i+=1
    # Divide by the number of experiments to get the average interaction matrix
    print("i",i)
    print("exp_num",num_exp)
    average_interaction_matrix /= num_exp
    average_interaction_matrix /= 30
    # Round to the nearest integer and convert to int
    average_interaction_matrix = np.round(average_interaction_matrix).astype(int)
    print("numer of exp: ",i)
    # Print the average interaction matrix
    print("Average Interaction Matrix:")
    print(average_interaction_matrix)

    # plot the full matrix
    behavior_names = ['Exp Non Statobj', 'Exp Statobj', 'Walking', 'Rearings', 'Groomings']
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
                cmap='viridis', cbar=True, mask=mask, annot_kws={"fontsize": 20}, cbar_kws={'label': 'Interaction Value'})
    plt.title('Average Behavioral Interaction Matrix')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig(path +"matrix_inter_nondiagonals_test.svg", format="svg")  # Save as SVG
    plt.show()

    # Extract the diagonal elements
    diagonal_elements = np.diag(average_interaction_matrix)
    # Labels for the pie chart (same as behavior names)
    behavior_names = ['Exp Non Statobj', 'Exp Statobj', 'Walking', 'Rearings', 'Groomings']
    # Plotting the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(diagonal_elements, labels=behavior_names, autopct='%1.1f%%', startangle=140,textprops={'fontsize': 32})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(path + "piechart_sample.svg", format="svg")
    plt.show()

## This function generated required data .csv for creating figures 4 C , D, G, H
def GLM_func():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    all_data = []
    list_exp=[]
    coefficients_list=[]
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
        if task_type == 1 or task_type == 0:
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

            smooth_time_window = 0.5
            log = 1
            window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
            kernel = np.ones(window_samples) / window_samples
            speed = np.convolve(speed, kernel, mode='same')
            dfof = np.convolve(dfof, kernel, mode='same')

            ## measuring the log of speed
            if log:  # log=1/0
                log_speed = np.log2(speed + 0.01)
                if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                    ## interpolate NanN
                    df_log = pd.DataFrame({'log_speed': log_speed})
                    log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ########## GLM
            X1 = pd.DataFrame({
                'exp_non_statobj': exp_non_statobj,
                'exp_statobj': exp_statobj,
                'speed': log_speed,
                'rearing': rearings,
                'grooming': groomings  # Corrected typo from 'groomig' to 'grooming'
            })
            # Add constant for intercept
            X1 = sm.add_constant(X1)
            model1 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result1 = model1.fit()
            print(result1.summary())

            summary_text = result1.summary().as_text()
            # extract the Pseudo R-squared (CS) value
            match = re.search(r'Pseudo R-squ\. \(CS\):\s+([0-9.]+)', summary_text)
            pseudo_r2_cs = float(match.group(1))

            result_df = pd.DataFrame({
                'exp_name': expriment_name,
                'task': task_type,
                'Coefficient': result1.params,
                'Std_Error': result1.bse,
                'z_value': result1.tvalues,
                'p_value': result1.pvalues,
                'Conf_Lower': result1.conf_int()[0],
                'Conf_Upper': result1.conf_int()[1]
            })

            model_stats = pd.DataFrame({
                'task': [task_type],
                'exp_name': [expriment_name],
                'Log-Likelihood': [result1.llf],
                'Deviance': [result1.deviance],
                'Pearson Chi2': [result1.pearson_chi2],
                'Pseudo R-Squared (CS)': pseudo_r2_cs
            })

            coeffs = result1.params
            coeffs_dict = coeffs.to_dict()
            coeffs_dict['exp'] = expriment_name  # add run number for reference
            coefficients_list.append(coeffs_dict)

            filename = path + "/GLM_whole_coef.csv" ## includes : Coefficients
            coefficients_df = pd.DataFrame(coefficients_list)
            coefficients_df.to_csv(filename, index=False)

            filename = path + "/GLM_whole.csv" ## includes : Coefficient, Std_Error, z_value, p_value, Conf_Lower, Conf_Upper
            if not os.path.isfile(filename):
                # If the file does not exist, write the DataFrame with headers
                result_df.to_csv(filename, index=False, mode='w')
            else:
                # If the file exists, append without headers
                result_df.to_csv(filename, index=False, mode='a', header=False)

            filename = path + "/GLM_whole_stats.csv" ## statistics of GLM
            if not os.path.isfile(filename):
                # If the file does not exist, write the DataFrame with headers
                model_stats.to_csv(filename, index=False, mode='w')
            else:
                # If the file exists, append without headers
                model_stats.to_csv(filename, index=False, mode='a', header=False)

        i = i+1


## use this function to generate figure 4 E and F
def washout_win():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/'

    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    nonstat_stat_obj_list_sample = []
    nonstat_stat_obj_list_test = []
    num_exp=0
    merged_data=[]
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

        with open(filepath_exp + "\mouse_id.txt", 'r') as file:
            # Read the first line
            line = file.readline()
            line = line.rstrip('\n')
        mouseID=int(line)

        print("The onset time:", onset)
        print("task:", task)
        if task == 'Learning' or task == 'learning':
            task_type = 0
            analysis_win = 900 * sr_analysis
        else:  ## Recall
            task_type = 1
            analysis_win = 900 * sr_analysis

        ############################## calulations
        if task_type == 1 or task_type == 0:
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

            smooth_time_window = 0.5
            log = 1
            window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
            kernel = np.ones(window_samples) / window_samples
            speed = np.convolve(speed, kernel, mode='same')
            dfof = np.convolve(dfof, kernel, mode='same')

            ## measuring the log of speed
            if log:  # log=1/0
                log_speed = np.log2(speed + 0.01)
                if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                    ## interpolate NanN
                    df_log = pd.DataFrame({'log_speed': log_speed})
                    log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ## GLM
            w = 0
            time_win = 0
            win_len=3*60*sr_analysis
            all_results = pd.DataFrame()
            while time_win <= analysis_win-win_len:
                ## GLM
                X1 = pd.DataFrame()
                X1['exp_non_statobj'] = exp_non_statobj[time_win:time_win + win_len]
                X1['exp_statobj'] = exp_statobj[time_win:time_win + win_len]
                X1['speed'] = log_speed[time_win:time_win + win_len]
                X1['rearing'] = rearings[time_win:time_win + win_len]
                X1['groomig'] = groomings[time_win:time_win + win_len]
                fof = dfof[time_win:time_win + win_len]
                if sum(exp_non_statobj[time_win:time_win + win_len]) >= 15 and sum(exp_statobj[time_win:time_win + win_len]) >= 15:
                    X1 = sm.add_constant(X1)
                    model2 = sm.GLM(fof, X1, family=sm.families.Gaussian())
                    result2 = model2.fit()
                    result_df2 = pd.DataFrame(result2.summary().tables[1].data)
                    first_two_columns = result_df2.iloc[:, :2]
                    transposed_data = first_two_columns.T
                    transposed_data.insert(0, 'task', None)
                    transposed_data.at[1, 'task'] = str(task)
                    all_results = pd.concat([all_results, transposed_data.iloc[1:,0:]], ignore_index=True)
                else:
                    columns = ["const", "exp_non_statobj", "exp_statobj", "speed", "rearing", "groomig"]
                    data = [np.nan] * len(columns)
                    nan_df = pd.DataFrame([data], columns=columns)
                    nan_df.insert(0, 'task', None)
                    nan_df.at[1, 'task'] = str(task)
                    all_results = pd.concat([all_results, nan_df.iloc[1:, 0:]], ignore_index=True)
                    print("no")
                time_win = time_win + 1 * 60 * sr_analysis
                w = w + 1

            nonstatObj_list=all_results.iloc[0:12,3:4].to_numpy().astype(float)
            statObj_list=all_results.iloc[0:12,4:5].to_numpy().astype(float)
            if task_type == 1:
                nonstat_stat_obj_list_test.append(nonstatObj_list-statObj_list)
            else:
                nonstat_stat_obj_list_sample.append(nonstatObj_list-statObj_list)
            num_exp+=1

        i = i + 1

    nonstat_stat_obj_list_sample = np.array(nonstat_stat_obj_list_sample).squeeze()
    nonstat_stat_obj_list_test = np.array(nonstat_stat_obj_list_test).squeeze()
    diff_test_sample= nonstat_stat_obj_list_test-nonstat_stat_obj_list_sample

    # Compute mean and SEM
    mean_diff_sample = np.nanmean(nonstat_stat_obj_list_sample, axis=0)
    sem_diff_sample = np.nanstd(nonstat_stat_obj_list_sample, axis=0) / np.sqrt(len(nonstat_stat_obj_list_sample))
    mean_diff_test = np.nanmean(nonstat_stat_obj_list_test, axis=0)
    sem_diff_test = np.nanstd(nonstat_stat_obj_list_test, axis=0) / np.sqrt(len(nonstat_stat_obj_list_test))
    mean_diff_test_sample = np.nanmean(diff_test_sample, axis=0)
    sem_diff_test_sample = np.nanstd(diff_test_sample, axis=0) / np.sqrt(len(diff_test_sample))

    plt.figure(figsize=(15, 7))
    # plt.fill_between(np.arange(len(data_mean)), data_mean - data_sem, data_mean + data_sem, color='lightgray')
    plt.errorbar(np.arange(len(mean_diff_sample)), mean_diff_sample, yerr=sem_diff_sample, fmt='-o', color='#048786',
                 ecolor='#D7E1E2', capsize=5, label="Learning")
    # plt.fill_between(np.arange(len(data_mean2)), data_mean2 - data_sem2, data_mean2 + data_sem2, color='lightgray')
    plt.errorbar(np.arange(len(mean_diff_test)), mean_diff_test, yerr=sem_diff_test, fmt='-o', color='#871719',
                 ecolor='#DDC0B5', capsize=5, label="Recall")
    plt.legend()
    # plt.savefig(path+"AvgSEM_Diff.svg",format="svg")
    plt.show()

    plt.figure(figsize=(15, 7))
    plt.errorbar(np.arange(len(mean_diff_test_sample)), mean_diff_test_sample, yerr=sem_diff_test_sample, fmt='-o', color='red',
                 ecolor='#FBD1CF', capsize=5, label="Recall")
    plt.legend()
    # plt.savefig(path+"AvgSEM_Diff_test_sample.svg",format="svg")
    plt.show()

    # Perform linear regression
    x = np.arange(1, 13)
    x_reshaped = x.reshape(-1, 1)

    model_test = LinearRegression()
    model_test.fit(x_reshaped, mean_diff_test)
    regression_line_test = model_test.predict(x_reshaped)

    model_sample = LinearRegression()
    model_sample.fit(x_reshaped, mean_diff_sample)
    regression_line_sample = model_sample.predict(x_reshaped)

    # Extract regression formulas
    coef_test, intercept_test = model_test.coef_[0], model_test.intercept_
    coef_sample, intercept_sample = model_sample.coef_[0], model_sample.intercept_

    formula_test = f"y = {coef_test:.5f}x + {intercept_test:.5f}"
    formula_sample = f"y = {coef_sample:.5f}x + {intercept_sample:.5f}"

    # Plot the scatter plot and regression line
    plt.figure(figsize=(15, 7))
    plt.scatter(x, mean_diff_test, color='#871719', label='Mean Diff Test (Signal)')
    plt.plot(x, regression_line_test, color='#871719', label=f'Regression Line ({formula_test})')
    plt.scatter(x, mean_diff_sample, color='#048786', label='Mean Diff Sample (Signal)')
    plt.plot(x, regression_line_sample, color='#048786', label=f'Regression Line ({formula_sample})')
    plt.legend(fontsize=12)
    plt.title('Scatter Plot with Regression Lines', fontsize=14)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    # plt.grid(True)
    # plt.savefig(path+"AvgSEM_Diff_regressions.svg",format="svg")
    plt.show()


    # Perform linear regression for diff test - sample
    x = np.arange(1, 13)
    x_reshaped = x.reshape(-1, 1)
    model_diff = LinearRegression()
    model_diff.fit(x_reshaped, mean_diff_test_sample)
    regression_line_diff = model_diff.predict(x_reshaped)
    # Extract regression formulas
    coef_diff, intercept_diff = model_diff.coef_[0], model_diff.intercept_
    formula_diff = f"y = {coef_diff:.5f}x + {intercept_diff:.5f}"
    # Plot the scatter plot and regression line
    plt.figure(figsize=(15, 7))
    plt.scatter(x, mean_diff_test_sample, color='#871719', label='Mean Diff (Signal)')
    plt.plot(x, regression_line_diff, color='#871719', label=f'Regression Line ({formula_diff})')
    plt.legend(fontsize=12)
    plt.title('Scatter Plot with Regression Lines', fontsize=14)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)
    plt.savefig(path+"Diff_regressions.svg",format="svg")
    plt.show()

    temp = pd.DataFrame({
        'diff': mean_diff_test_sample,
        'win': np.arange(1, 13, 1)
    })

    ols_model = smf.ols("diff ~ win ", data=temp)
    ols_result = ols_model.fit()
    print(ols_result.summary())

## this function is used in "strech_time_Behaviors_only_4_with_speed_acrosswins" function
def find_startANDendpoints(behavior_sig):
    starts = []
    ends = []
    for sweep in range(0, len(behavior_sig)):
        if np.array_equal([0, 1], np.array(behavior_sig[sweep: sweep + 2])):
            starts.append(sweep+1)
        if np.array_equal([1, 0], np.array(behavior_sig[sweep: sweep + 2])):
            ends.append(sweep)
    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:len(starts) - 1]
    # print("starts", starts)
    # print("ends", ends)
    return starts, ends

## this function is used in "strech_time_Behaviors_only_4_with_speed_acrosswins" function
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

## this function is used in "strech_time_Behaviors_only_4_with_speed_acrosswins" function
def remove_small_win(st_ed, min_win_size):
    print("current st_ed", st_ed)
    new_st_ed=[]
    for i in range(0, len(st_ed)):
        print(st_ed[i], st_ed[i][1] - st_ed[i][0])
        if st_ed[i][1] - st_ed[i][0] > min_win_size:
            new_st_ed.append(st_ed[i])
    return new_st_ed

## this function is used in "strech_time_Behaviors_only_4_with_speed_acrosswins" function
def remove_long_win(st_ed, min_win_size):
    new_st_ed=[]
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
## this function is used to plot cholinergic dynamics at onset during and offset of each behavior in figure 5
def strech_time_Behaviors_only_4_with_speed_acrosswins():
    path = 'C:/Users/ffarokhi/Desktop/test/'  # for saving the results
    # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    directory_path = 'C:/Users/ffarokhi/Documents/GitHub/Cholinergic-dynamics-in-spatial-exploration/all_data/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0

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

    total_calculated_st=0
    total_calculated_st_persession = []
    total_win_be=0
    total_win_persession=[]

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
            analysis_win = 15*60 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 15*60 * sr_analysis
        ##
        ############################## calulations
        if task_type == 1 or task_type == 0:
            analysis_win=analysis_win-onset
            data = pd.read_csv(file_data).values
            data = pd.DataFrame(data)
            time = data[0][onset:analysis_win]
            dfof = data[1][onset:analysis_win]
            speed = data[2][onset:analysis_win]
            exp_non_statobj = data[3][onset:analysis_win]
            exp_statobj = data[4][onset:analysis_win]
            walking=data[5][onset:analysis_win]
            rearings = data[6][onset:analysis_win]
            groomings = data[7][onset:analysis_win]

            window_samples = int(0.5 * sr_analysis)  # number of samples in the 0.5s window
            kernel = np.ones(window_samples) / window_samples
            speed = np.convolve(speed, kernel, mode='same')
            dfof = np.convolve(dfof, kernel, mode='same')

            ## calculate log speed
            log_speed = np.log2(speed + 0.01)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ########## GLM
            # X1 = pd.DataFrame()
            # X1['exp_non_statobj'] = exp_non_statobj
            # X1['exp_statobj'] = exp_statobj
            # X1['speed'] = log_speed
            # X1['rearing'] = rearings  # all_rearings
            # X1['groomig'] = groomings
            # X1 = sm.add_constant(X1)
            # model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            # result2 = model2.fit()
            # result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            # speed_coef=float(result_df2.iloc[4:5, 1:2].values)
            # groomings_coef = float(result_df2.iloc[6:7, 1:2].values)
            # rearings_coef = float(result_df2.iloc[5:6, 1:2].values)
            # exp_non_statobj_coef = float(result_df2.iloc[2:3, 1:2].values)
            # exp_statobj_coef = float(result_df2.iloc[3:4, 1:2].values)
            # constant_coef = float(result_df2.iloc[1:2, 1:2].values)


            behavior_walking = np.array(walking)
            behavior_grooming = np.array(groomings)
            behavior_rearings = np.array(rearings)
            behavior_exp_statobj = np.array(exp_statobj)
            behavior_exp_non_statobj = np.array(exp_non_statobj)

            list_behaviors=[behavior_walking,behavior_grooming,behavior_rearings,behavior_exp_statobj,behavior_exp_non_statobj]
            behavior_bk = (np.logical_or.reduce([behavior_walking,
                          behavior_grooming,
                          behavior_rearings,
                          behavior_exp_statobj,
                          behavior_exp_non_statobj]))

            ##  logical NOT on the OR result
            behavior_bk = np.logical_not(behavior_bk)
            list_behaviors.append(behavior_bk)

            name_behavior = "behavior_walking"
            behavior_sig = behavior_walking
            dfof1 = np.array(dfof)

            min_distance = int(0*sr_analysis)
            offset = int(5*sr_analysis)
            min_win_size = int(2*sr_analysis)
            # max_win_size= int(10*sr_analysis)
            min_dis_toShow = int(4*sr_analysis)

            dfof=dfof1
            # dfof = ((((dfof1
            #         - speed_coef * log_speed -groomings_coef * behavior_grooming - rearings_coef * behavior_rearings)
            #         - exp_statobj_coef * behavior_exp_statobj)
            #         - exp_non_statobj_coef * behavior_exp_non_statobj)
            #         - constant_coef)


            ##### predicting the dfof using speed
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
            total_win_persession.append(len(st_ed))

            ## find the max window size
            # max_window = 0
            # for t in range(0,len(st_ed)):
            #     win_d=st_ed[t][1] - st_ed[t][0] + 1
            #     if win_d > max_window:
            #         max_window=win_d
            # # print("max_window",max_window)
            max_window=5000

            ## functions

            def start_end_ind(st_ed, y, offset, dfof, behavior_sig, min_dis_toShow):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                flag = 0
                signal = []
                behavior_main =[]
                temp_speed =[]
                p_dfof =[]

                if  st_ed[y][1]-st_ed[y][0] > min_win_size:
                    if y < len(st_ed)-1:
                        if end_idx < len(dfof) and start_idx > 0 and (st_ed[y][0]-st_ed[y-1][1]) >= min_dis_toShow and (st_ed[y+1][0]-st_ed[y][1]) >= min_dis_toShow:
                            signal = zscore(dfof[start_idx:end_idx])
                            behavior_main = behavior_sig[start_idx:end_idx]
                            temp_speed = speed[start_idx:end_idx]
                            p_dfof = zscore(predicted_dfof[start_idx:end_idx])

                            flag = 1
                    else:
                        if end_idx < len(dfof) and (st_ed[y][0]-st_ed[y-1][1]) >= min_dis_toShow:
                            signal = zscore(dfof[start_idx: end_idx])
                            temp_speed = speed[start_idx: end_idx]
                            p_dfof = zscore(predicted_dfof[start_idx: end_idx])
                            behavior_main = behavior_sig[start_idx: end_idx]
                            flag = 1

                return signal, flag, behavior_main, temp_speed,p_dfof


            def extract_st_ends_behaviors(behavior_main,offset,len_signal):
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start == 1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start == 0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len_signal - offset: len_signal]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)
                return behavior_win_start,behavior_win_end

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
            tempflag=0
            for y in range(0,len(st_ed)):  #len(st_ed)

                signal, flag, behavior_main, temp_speed, p_dfof = start_end_ind(st_ed, y, offset, dfof, behavior_sig, min_dis_toShow)
                total_calculated_st += flag

                if flag == 1:
                    tempflag+=1
                    # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                    len_signal=len(signal)
                    streched_main_signal0, start_off0, end_off0, bewalking = start_end_ind_strechedmain(st_ed, offset, list_behaviors[0], len_signal)
                    streched_main_signal1, start_off1, end_off1, begrooming = start_end_ind_strechedmain(st_ed, offset, list_behaviors[1], len_signal)
                    streched_main_signal2, start_off2, end_off2, berearings = start_end_ind_strechedmain(st_ed, offset, list_behaviors[2], len_signal)
                    streched_main_signal3, start_off3, end_off3, beexp_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[3], len_signal)
                    streched_main_signal4, start_off4, end_off4, beexp_non_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[4], len_signal)
                    streched_main_bk, start_offbk, end_offbk, bebk = start_end_ind_strechedmain(st_ed, offset, list_behaviors[5], len_signal)

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

                    ### main signal
                    ### start and end of behavior for NaN detection
                    behavior_win_start, behavior_win_end=extract_st_ends_behaviors(behavior_main,offset,len_signal)

                    # split the siganl to start and end and main part
                    start_off = signal[:offset]
                    end_off = signal[len_signal - offset:len_signal]
                    main_part = signal[offset:len_signal - offset]

                    ## speed signal
                    speed_start_off = temp_speed[:offset]
                    speed_end_off = temp_speed[len_signal - offset:len_signal]
                    speed_main_part = temp_speed[offset:len_signal - offset]

                    # Create an array of main part signal values (time points)
                    original_x = np.arange(len(main_part))
                    # Define the new x values (time points) after stretching
                    new_x = np.linspace(0, len(main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(original_x, main_part, kind='linear')
                    # Interpolate the signal at new x values
                    new_signal = interp_func(new_x)

                    ## speed
                    # Create an array of main part signal values (time points)
                    speed_original_x = np.arange(len(speed_main_part))
                    # Define the new x values (time points) after stretching
                    speed_new_x = np.linspace(0, len(speed_main_part) - 1, max_window)
                    # Create a function for linear interpolation
                    interp_func = interp1d(speed_original_x, speed_main_part, kind='linear')
                    # Interpolate the signal at new x values
                    speed_new_signal = interp_func(speed_new_x)

                    ## save for avg
                    streched_win.append(np.nanmean(new_signal.reshape(-1, 20), axis=1))
                    start_offs.append(behavior_win_start * start_off)
                    end_offs.append(behavior_win_end * end_off)

                    ## save for avg
                    speed_streched_win.append(np.nanmean(speed_new_signal.reshape(-1, 20), axis=1))
                    speed_start_offs.append(behavior_win_start * speed_start_off)
                    speed_end_offs.append(behavior_win_end * speed_end_off)

                    ## predicted dfof
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

                    pdfof_streched_win.append(np.nanmean(pdfof_new_signal.reshape(-1, 20), axis=1))
                    pdfof_start_offs.append(behavior_win_start * pdfof_start_off)
                    pdfof_end_offs.append(behavior_win_end * pdfof_end_off)


            ## go to next expriment file
            total_calculated_st_persession.append(tempflag)
        i = i + 1

    print("total_bouts", total_win_be)
    print(total_win_persession)
    print("total_calculated_st",total_calculated_st)
    print(total_calculated_st_persession)

    ##main
    all_main=np.array(streched_win)
    all_sts=np.array(start_offs)
    all_eds=np.array(end_offs)

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


    plt.plot(final_sig_behavior0, color='#1f77b4', label='Locomotion')  # Blue
    plt.plot(final_sig_behavior1, color='#2ca02c', label='Grooming')  # Green
    plt.plot(final_sig_behavior2, color='#9467bd', label='Rearing')  # Purple
    plt.plot(final_sig_behavior3, color='#ff7f0e', label='Exp StatObj')  # Orange
    plt.plot(final_sig_behavior4, color='#8c564b', label='Exp NonStatObj')  # Brown
    plt.plot(final_sig_bk, color='#7f7f7f', label='Background Signal')  # Gray
    plt.legend(fontsize=10) #, title="Behaviors"
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
    plt.xlim(0,len(final_sig_behavior0_percent))
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(path + "behaviors_percentage" + name_behavior + ".svg", format="svg")

    plt.show()

    ###main dfof
    all_main_avg = np.nanmean(all_main, axis=0)
    std_main = np.nanstd(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.nanmean(all_sts, axis=0)
    std_st = np.nanstd(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.nanmean(all_eds, axis=0)
    std_ed = np.nanstd(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    avg_st= np.nanmean(all_sts_avg)
    avg_main= np.nanmean(all_main_avg)
    half_dfof=(avg_st+avg_main)/2
    index_half_dfof = np.where(merged >= half_dfof)[0][0]

    # Calculate the 90th percentile threshold
    # threshold = np.percentile(merged, 50)
    # index_half_dfof = np.where(merged >= threshold)[0]
    # print(index_half_dfof)


    ###### speed signal :
    all_main_avg_speed = np.nanmean(speed_all_main, axis=0)
    std_main_speed = np.nanstd(speed_all_main, axis=0)  # Standard deviation
    n = speed_all_main.shape[0]  # Number of observations
    all_main_sem_speed = std_main_speed / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg_speed = np.nanmean(speed_all_sts, axis=0)
    std_st_speed = np.nanstd(speed_all_sts, axis=0)  # Standard deviation
    n = speed_all_sts.shape[0]  # Number of observations
    all_st_sem_speed = std_st_speed / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg_speed = np.nanmean(speed_all_eds, axis=0)
    std_ed_speed = np.nanstd(speed_all_eds, axis=0)  # Standard deviation
    n = speed_all_eds.shape[0]  # Number of observations
    all_ed_sem_speed = std_ed_speed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged_speed = np.concatenate((all_sts_avg_speed, all_main_avg_speed, all_eds_avg_speed))
    merge_sem_speed = np.concatenate((all_st_sem_speed, all_main_sem_speed, all_ed_sem_speed))

    avg_st= np.nanmean(all_sts_avg_speed)
    avg_main= np.nanmean(all_main_avg_speed)
    half_speed=(avg_st+avg_main)/2
    index_half_speed = np.where(merged_speed >= half_speed)[0][0]

    ###### predicted dfof signal :
    all_main_avg_pdfof = np.nanmean(pdfof_all_main, axis=0)
    std_main_pdfof = np.nanstd(pdfof_all_main, axis=0)  # Standard deviation
    n = pdfof_all_main.shape[0]  # Number of observations
    all_main_sem_pdfof = std_main_pdfof / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg_pdfof = np.nanmean(pdfof_all_sts, axis=0)
    std_st_pdfof = np.nanstd(pdfof_all_sts, axis=0)  # Standard deviation
    n = pdfof_all_sts.shape[0]  # Number of observations
    all_st_sem_pdfof = std_st_pdfof / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg_pdfof = np.nanmean(pdfof_all_eds, axis=0)
    std_ed_pdfof = np.nanstd(pdfof_all_eds, axis=0)  # Standard deviation
    n = pdfof_all_eds.shape[0]  # Number of observations
    all_ed_sem_pdfof = std_ed_pdfof / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged_pdfof = np.concatenate((all_sts_avg_pdfof, all_main_avg_pdfof, all_eds_avg_pdfof))
    merge_sem_pdfof = np.concatenate((all_st_sem_pdfof, all_main_sem_pdfof, all_ed_sem_pdfof))

    ## 50 percent
    avg_st = np.nanmean(all_sts_avg_pdfof)
    avg_main = np.nanmean(all_main_avg_pdfof)
    half_pdfof = (avg_st + avg_main) / 2
    index_half_pdfof = np.where(merged_pdfof >= half_pdfof)[0][0]

    # Calculate the 90th percentile threshold
    # threshold = np.percentile(merged_pdfof, 50)
    # index_half_pdfof = np.where(merged_pdfof >= threshold)[0]
    # print(index_half_pdfof)


    ######### plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the first signal on the left y-axis
    ax1.set_ylim(0, 14)
    ax1.plot(merged_speed, label='Average speed', linewidth=3, color='#58585A')
    ax1.fill_between(np.arange(len(merged_speed)), merged_speed - merge_sem_speed, merged_speed + merge_sem_speed,
                     color='#D3D3D3', alpha=0.5)
    ax1.set_ylabel('avg(Speed)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Add vertical lines
    ax1.axvline(x=offset, color='gray', linestyle='--')
    ax1.axvline(x=len(merged) - offset, color='gray', linestyle='--')
    ax1.set_xlim([0, len(merged)])

    # Create a twin axis for the second signal on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.2, 1.2)
    ax2.plot(merged, label='Average dfof', linewidth=3, color='#38843F')
    ax2.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#D2E9CD', alpha=0.5)
    ax2.axvline(x=index_half_dfof,label=f"{(index_half_dfof/sr_analysis):.4f}", color='#38843F', linestyle='--')

    ## predicted
    ax2.plot(merged_pdfof, label='Average predicted dfof', linewidth=3, color='#F1605F')
    ax2.fill_between(np.arange(len(merged_pdfof)), merged_pdfof - merge_sem_pdfof, merged_pdfof + merge_sem_pdfof, color='#F9CDE1', alpha=0.5)
    ax2.axvline(x=index_half_pdfof, label=f"{(index_half_pdfof/sr_analysis):.4f}", color='red', linestyle='--')
    ax2.axvline(x=index_half_speed, label=f"{(index_half_speed / sr_analysis):.4f}", color='black', linestyle='--')

    # ax2.text(50, 0, f"{(index_half_pdfof - index_half_dfof) / sr_analysis:.4f}", fontsize=12, color='red')

    ax2.set_ylabel('avg(zscore(dfof))', color='#0F8140')
    ax2.tick_params(axis='y', labelcolor='#0F8140')
    fig.tight_layout()
    plt.legend()
    plt.savefig(path + "main_signals" + name_behavior + ".svg", format="svg")
    plt.show()

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal

    ### bar plots
    def average_n_points(data, n):
        end = len(data) - (len(data) % n)  # Adjust the length to be divisible by n
        reshaped_data = data[:end].reshape(-1, n)
        return reshaped_data.mean(axis=1)

    # Average every 10 points
    all_sts_avg_binned = average_n_points(interpolate_nans(all_sts_avg), 10)
    all_main_avg_binned = average_n_points(interpolate_nans(all_main_avg), 10)
    all_eds_avg_binned = average_n_points(interpolate_nans(all_eds_avg), 10)

    # Concatenate the averaged arrays
    merged_binned = np.concatenate((all_sts_avg_binned, all_main_avg_binned, all_eds_avg_binned))
    # Define the x-axis for the binned data
    x_sts = np.arange(len(all_sts_avg_binned))
    x_main = np.arange(len(all_sts_avg_binned), len(all_sts_avg_binned) + len(all_main_avg_binned))
    x_eds = np.arange(len(all_sts_avg_binned) + len(all_main_avg_binned), len(merged_binned))

    # Plot each section with bars
    bar_width = 0.8

    plt.figure(figsize=(10, 6))
    plt.bar(x_sts, all_sts_avg_binned, color='green', alpha=0.8, label='Section 1', width=bar_width)
    plt.bar(x_main, all_main_avg_binned, color='green', alpha=0.8, label='Section 2', width=bar_width)
    plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.8, label='Section 3', width=bar_width)

    # Add vertical dashed lines for discontinuity
    plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='blue', linestyle='--',
                linewidth=1.5)
    plt.ylim([-1.2, 1.2])
    # plt.xlim([0, len(merged)])
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Z score')
    plt.title('Bar Plot of Averaged Data (10 points per bar)')
    # Show the plot
    plt.savefig(path + "bar_plots" + name_behavior + ".svg", format="svg")
    plt.show()



if __name__ == '__main__':
    print("Cholinergic Activity")
    # time_analysis_washout()
    # plot_speed_dfof()
    # find_best_smoothing_corr()
    # novel_env()
    # behavioral_interactions()
    # GLM_func()
    # washout_win()
    # strech_time_Behaviors_only_4_with_speed_acrosswins()


