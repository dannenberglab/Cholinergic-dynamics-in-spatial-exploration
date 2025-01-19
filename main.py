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


def create_lineRegression(time_series):
    # Creating X and y for linear regression
    X = np.arange(len(time_series)).reshape(-1, 1)  # Reshape to column vector
    y = np.array(time_series)

    # Performing linear regression
    model = LinearRegression()
    results = model.fit(X, y)
    y_pred = model.predict(X)

    # Get the slope (coefficient) and intercept
    slope = results.coef_[0]
    intercept = results.intercept_

    # Perform a t-test for the slope
    n = len(time_series)
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Calculate the residuals
    residuals = y - y_pred
    residual_sum_of_squares = np.sum(residuals ** 2)

    # Standard error of the slope
    s_xx = np.sum((X.flatten() - X_mean) ** 2)
    standard_error = np.sqrt(residual_sum_of_squares / (n - 2)) / np.sqrt(s_xx)

    # t-statistic for the slope
    t_statistic = slope / standard_error

    # Degrees of freedom
    df = n - 2

    # p-value from the t-distribution
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

    # Plotting the time series with linear regression line
    plt.figure(figsize=(12, 5))
    plt.scatter(X, y, color='blue', label='Original data ')
    plt.plot(X, y_pred, color='red', label='Linear regression ')
    plt.legend()
    plt.title(f"Slope: {slope:.4f}, p-value: {p_value:.4f}")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim([-0.025, 0.045])
    plt.grid(True)

    return t_statistic,p_value,plt

def time_analysis():
    directory_path = 'E:/lab/Cholinergic Prj/Data/20/'
    sr_analysis = 20
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i=0
    print(folder_names)
    total_non_stat = []
    total_stat = []
    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data=filepath_exp+'/'+expriment_name+'_data_'+str(sr_analysis)+'fps.csv'
        labels_file=filepath_exp+'/'+expriment_name+'_labels.csv'
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
            analysis_win = 900*sr_analysis
            #720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 900*sr_analysis
        ##
############################## calulations
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        exp_non_statobj=data[3][onset:analysis_win]
        exp_statobj=data[4][onset:analysis_win]
        data2 = pd.read_csv(labels_file).values
        data2 = pd.DataFrame(data2)

        # non_stat_v=[]
        # stat_v=[]
        # diff=[]
        # filename = directory_path + "/times"+str(i)+".csv"
        # for w in range(0, 13, 1):
        #     non_stat_v.append(sum(exp_non_statobj[(w)*sr_analysis*60:(w+3)*sr_analysis*60])/sr_analysis)
        #     stat_v.append(sum(exp_statobj[(w)*sr_analysis*60:(w+3)*sr_analysis*60])/sr_analysis)
        #     diff.append(non_stat_v[w]-stat_v[w])
        # #save
        # data = {'Non_Stat': non_stat_v, 'Stat': stat_v, 'diff': diff}
        # df = pd.DataFrame(data)
        # df.to_csv(filename)

        total_non_stat.append(sum(exp_non_statobj) / sr_analysis)
        total_stat.append(sum(exp_statobj) / sr_analysis)

        i = i + 1
    data = {'Non_Stat': total_non_stat, 'Stat': total_stat}
    filename = directory_path + "/times_diff_15min_20.csv"
    df = pd.DataFrame(data)
    df.to_csv(filename)

def speed_correlation():
    path = 'C:/Users/ffarokhi/Desktop/updateresult/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    # directory_path = 'E:/lab/Cholinergic Prj/Data/30/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp = []
    dfof_all = []
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
            analysis_win = 900 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 900 * sr_analysis
        ##
        ############################## calulations
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        time = data[0][onset:analysis_win]
        dfof = data[1][onset:analysis_win]
        speed = data[2][onset:analysis_win]

        ## calculate log speed
        unique_speed = np.unique(speed)
        sorted_speed = np.sort(unique_speed)
        offset = sorted_speed[1]
        log_speed = np.log2(speed + offset)
        # Replace NaNs with the mean of the variable
        log_speed[np.isnan(log_speed)] = np.nanmean(log_speed)

        # Replace infs with a specific value, for example, replacing with the maximum value
        log_speed[np.isinf(log_speed)] = np.nanmax(log_speed)

        plt.scatter(log_speed, dfof, s=1)
        X = np.array(log_speed).reshape(-1, 1)
        y = np.array(dfof)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        plt.plot(X, y_pred, color='red')
        plt.show()

        plt.scatter(log_speed, stats.zscore(dfof), s=1)
        X = np.array(log_speed).reshape(-1, 1)
        y = np.array(stats.zscore(dfof))
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        plt.plot(X, y_pred, color='red')
        plt.show()

        data = pd.DataFrame()
        data['log_speed'] = np.array(log_speed)
        data['dfof'] = np.array(dfof)
        grouped_data = data.groupby('log_speed')['dfof'].mean()
        # Get the unique log_speed values
        unique_log_speed = grouped_data.index
        # Get th averaged dfof values
        averaged_dfof = grouped_data.values
        # Plot the averaged dfof against log_speed
        plt.scatter(unique_log_speed, averaged_dfof, s=1)
        plt.xlabel('log_speed')
        plt.ylabel('Averaged dfof')
        plt.title('Scatter plot of averaged dfof vs. log_speed')
        plt.show()

        i=i+12

def time_analysis():
    path = 'C:/Users/ffarokhi/Desktop/paper/2) time anlaysis/updated_restult_9-16-2024/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/only_patrick/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i=0
    print(folder_names)
    total_non_stat = []
    total_stat = []
    total_ratio=[]
    list_task=[]
    all_ratio = pd.DataFrame()
    while i < len(folder_names):
        filepath_exp = directory_path + str(folder_names[i])
        expriment_name = filepath_exp.split("/")[-1]
        task = expriment_name.split("_")[-1]
        file_data=filepath_exp+'/'+expriment_name+'_data.csv'
        # labels_file=filepath_exp+'/'+expriment_name+'_labels.csv'
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
            analysis_win = 60*15*sr_analysis
            #720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 60*15*sr_analysis
        ##
        list_task.append(task)
############################## calulations
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        # time=data[0][onset:analysis_win]
        # dfof=data[1][onset:analysis_win]
        # speed=data[2][onset:analysis_win]
        exp_non_statobj=data[3][onset:analysis_win]
        exp_statobj=data[4][onset:analysis_win]

        non_stat_v=[]
        stat_v=[]
        diff=[]
        ratio=[]
        for w in range(0, 13, 1):
            non_stat_v.append(sum(exp_non_statobj[(w)*sr_analysis*60:(w+3)*sr_analysis*60])/sr_analysis)
            stat_v.append(sum(exp_statobj[(w)*sr_analysis*60:(w+3)*sr_analysis*60])/sr_analysis)
            diff.append(non_stat_v[w]-stat_v[w])
            ratio.append((non_stat_v[w]-stat_v[w])/(non_stat_v[w]+stat_v[w]))
        #save
        # data = {'Non_Stat': non_stat_v, 'Stat': stat_v, 'diff': diff, 'ratio': ratio}
        # df = pd.DataFrame(data)
        # df.to_csv(filename)
        all_ratio[f'iter_{i}'] = pd.Series(ratio)
        # total_non_stat.append(sum(exp_non_statobj))
        # total_stat.append(sum(exp_statobj))
        # total_ratio.append((total_non_stat[i]-total_stat[i])/(total_non_stat[i]+total_stat[i]))
        i = i + 1
    # Save the DataFrame to a CSV file
    output_filename = path+"wash_time.csv"
    all_ratio.to_csv(output_filename, index=False)

    # # Save total_non_stat, total_stat, and total_ratio to a new CSV file
    # summary_data = pd.DataFrame({
    #     'Task': list_task,
    #     'Total_Non_Stat': total_non_stat,
    #     'Total_Stat': total_stat,
    #     'Total_Ratio': total_ratio
    # })
    #
    # summary_filename = path + "summary_time.csv"
    # summary_data.to_csv(summary_filename, index=False)

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

def plot_speed_dfof():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/scatter plots/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/' # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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

        # ## plot correlations
        # R, f1 = show_speed_dfof(time, dfof, speed, 1, sr_analysis, 1)
        # plt.savefig(path+"dfof_logspeed4_"+expriment_name+".svg",format="svg")
        # plt.show()
        # R_list.append(R)
        # df = pd.DataFrame({
        #     'Name_Exp': list_exp,
        #     'R': R_list
        # })
        # df.to_csv(path+'Rlog4.csv', index=False)

        # scatter plots
        # scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, 1)

        # avg scatter plot
        bin_size=0.2
        mean_dfof=avg_scatter_plot_dfof_speed(path, expriment_name, dfof, speed, 0.5, sr_analysis, 1, bin_size)
        #

        mean_dfof_exp.append(mean_dfof[0:30])
        if task == 'Learning' or task == 'learning':
            mean_dfof_exp_sample.append(mean_dfof[0:30])
        else:
            mean_dfof_exp_test.append(mean_dfof[0:30])
        print(len(mean_dfof))

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
    # print(df)
    # for j in np.arange(0,i,1):
    #
    #     plt.plot(x_values, mean_dfof_exp_sample[j])
    #     plt.plot(x_values, mean_dfof_exp_test[j])
    #     plt.show()
    # # avg Plot
    mean_dfof_across_experiments = np.nanmean(mean_dfof_exp, axis=0)  # Average across experiments
    sem_dfof_across_experiments = np.nanstd(mean_dfof_exp, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    x_values = np.arange(-0.5, 4.5, bin_size)
    # plt.errorbar(x_values, mean_dfof_across_experiments, yerr=sem_dfof_across_experiments,
    #              fmt='o-', capsize=3, label='Mean ± SEM')
    # plt.xlabel('Speed (cm/s)')
    # plt.ylabel('Mean dfof')
    # plt.title('Mean dfof vs Speed')
    # plt.legend()
    # plt.savefig(path + "mean_logspeed_binned" + ".svg", format="svg")
    # plt.show()
    #
    # mean_dfof_across_experiments_sample = np.nanmean(mean_dfof_exp_sample, axis=0)  # Average across experiments
    # sem_dfof_across_experiments_sample = np.nanstd(mean_dfof_exp_sample, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    # mean_dfof_across_experiments_test = np.nanmean(mean_dfof_exp_test, axis=0)  # Average across experiments
    # sem_dfof_across_experiments_test = np.nanstd(mean_dfof_exp_test, axis=0) / np.sqrt(len(mean_dfof_exp))  # SEM
    # plt.errorbar(x_values, mean_dfof_across_experiments_sample, yerr=sem_dfof_across_experiments_sample,
    #              fmt='o-', capsize=3, label='Sample sessions', color="#048786")
    # plt.errorbar(x_values, mean_dfof_across_experiments_test, yerr=sem_dfof_across_experiments_test,
    #              fmt='o-', capsize=3, label='Test sessions', color="#871719")
    # plt.xlabel('Speed (cm/s)')
    # plt.ylabel('Mean dfof')
    # plt.title('Mean dfof vs Speed')
    # plt.legend()
    # plt.savefig(path + "mean_logspeed_binned_seperate" + ".svg", format="svg")
    # plt.show()

    # # Sample group
    # mean_dfof_exp_sample=np.array(mean_dfof_exp_sample)
    # df_sample = pd.DataFrame({
    #     'dfof': mean_dfof_exp_sample.flatten(),  # Flatten the 2D array into 1D
    #     'speed': np.tile(x_values, mean_dfof_exp_sample.shape[0]),  # Repeat speed bins for each experiment
    #     'group': 0  # Label for sample group
    # })
    #
    # # Test group
    # mean_dfof_exp_test=np.array(mean_dfof_exp_test)
    # df_test = pd.DataFrame({
    #     'dfof': mean_dfof_exp_test.flatten(),  # Flatten the 2D array into 1D
    #     'speed': np.tile(x_values, mean_dfof_exp_test.shape[0]),  # Repeat speed bins for each experiment
    #     'group': 1  # Label for test group
    # })
    #
    # # Combine sample and test into one DataFrame
    # df = pd.concat([df_sample, df_test], ignore_index=True)
    # print(df)
    # # Fit an OLS model with interaction between speed and group
    # ols_model = smf.ols(
    #     formula="dfof ~ speed * group",  # Fixed effects: speed, group, and interaction
    #     data=df
    # ).fit()
    # # Print the summary
    # print(ols_model.summary())
    #
    # ## check the effect of test and sample sessions
    # # OLS
    # # Fit OLS model
    # model = smf.ols('dfof ~ speed + group', data=df).fit()
    # print(model.summary())
    #
    # # GLM
    # # Add a constant term for the intercept
    # X = sm.add_constant(df[['speed', 'group']])
    # y = df['dfof']
    # # Fit the GLM (Gaussian family with identity link function)
    # glm_model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    # # Print the summary
    # print(glm_model.summary())
    #
    # LMM
    model = smf.mixedlm(
        "dfof ~ speed * group",  # Fixed effects: speed, group, interaction
        data=df,
        groups=df["mouseid"],  # Random effects grouped by mouse id
        vc_formula={"experiment": "0 + C(experiment)" }
    )
    result = model.fit()
    # Print the summary
    print(result.summary())
    #
    # GLM
    # # Add a constant term for the intercept
    # df['speed_group_interaction'] = df['speed'] * pd.Categorical(df['group']).codes  # Multiply speed by encoded group
    # X = sm.add_constant(df[['speed', 'group', 'speed_group_interaction']])  # Include interaction
    # y = df['dfof']
    # # Fit the GLM (Gaussian family with identity link function)
    # glm_model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    # # Print the summary
    # print(glm_model.summary())


############ strech time :
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
def remove_small_win(st_ed, min_win_size):
    print("current st_ed", st_ed)
    new_st_ed=[]
    for i in range(0, len(st_ed)):
        print(st_ed[i], st_ed[i][1] - st_ed[i][0])
        if st_ed[i][1] - st_ed[i][0] > min_win_size:
            new_st_ed.append(st_ed[i])
    return new_st_ed

def remove_long_win(st_ed, min_win_size):
    new_st_ed=[]
    for i in range(0, len(st_ed)):
        print(st_ed[i], st_ed[i][1] - st_ed[i][0])
        if st_ed[i][1] - st_ed[i][0] < min_win_size:
            new_st_ed.append(st_ed[i])
    return new_st_ed



def strech_time():
    sr_analysis = 30
    # directory_path = 'E:/lab/Cholinergic Prj/Data/'+str(sr_analysis)+'/'
    # path = 'C:/Users/ffarokhi/Desktop/updateresult/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp=[]
    all_main=[]
    all_sts=[]
    all_eds=[]
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

            # data2 = pd.read_csv(labels_file).values
            # data2 = pd.DataFrame(data2)
            # all_rearings = (np.array(data2[3][onset:analysis_win]) | np.array(data2[4][onset:analysis_win]))

            ## calculate log speed
            smooth_speed = smooth_signal(speed, 1, sr_analysis)
            unique_speed = np.unique(smooth_speed)
            sorted_speed = np.sort(unique_speed)
            offset = sorted_speed[1]
            log_speed = np.log2(smooth_speed + offset)
            # Replace NaNs with the mean of the variable
            log_speed[np.isnan(log_speed)] = np.nanmean(log_speed)
            # Replace infs with a specific value, for example, replacing with the maximum value
            log_speed[np.isinf(log_speed)] = np.nanmax(log_speed)
            ########## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj
            X1['exp_statobj'] = exp_statobj
            X1['speed'] = log_speed
            X1['rearing'] = rearings  # all_rearings
            X1['groomig'] = groomings
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            # X1['speed_grooming_interaction'] = X1['speed'] * X1['groomig']
            # X1['speed_non_statobj_interaction'] = X1['speed'] * X1['exp_non_statobj']
            # X1['speed_statobj_interaction'] = X1['speed'] * X1['exp_statobj']
            # X1['nonobj_rearing'] = X1['rearing'] * X1['exp_non_statobj']
            # X1['obj_rearing'] = X1['rearing'] * X1['exp_statobj']
            X1 = sm.add_constant(X1)
            model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            speed_coef=float(result_df2.iloc[4:5, 1:2].values)
            ###########################

            name_behavior="walking"
            behavior_sig=np.array(walking)[0:1000] #[0:6000]
            speed=np.array(log_speed[0:1000]) #[0:6000]
            dfof1 = np.array(dfof[0:1000])#[8500:10000]

            plt.plot(behavior_sig[0:1000])  # [6700:7000])
            plt.plot(dfof1[0:1000])  # [6700:7000])
            plt.show()

            # smooth12=smooth_signal(dfof1,12,sr_analysis)
            # smooth_1_7 = smooth_signal(dfof1, 3.25, sr_analysis)
            # dfof = (dfof1) - smooth_1_7

            min_distance = int(0.1*sr_analysis)
            offset = int(3*sr_analysis)
            min_win_size = int(0.1*sr_analysis)
            # max_win_size= int(10*sr_analysis)
            dfof=dfof1 - speed_coef*speed
            # plt.axvline(x=8500, color='r', linestyle='--')
            # plt.axvline(x=10000, color='r', linestyle='--')
            # plt.legend()
            # plt.show()
            # plots


            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)
            total_num_behavior=len(starts)

            #### merge start and end points
            st_ed=merge_startANDend_points(starts, ends, min_distance)


            ### remove first and last bout if it is in the start or end offset of the session
            while 1:
                if st_ed[0][0]<offset:
                    print("yeeeeees")
                    st_ed=st_ed[1:]
                else: break
            if st_ed[len(st_ed)-1][1]>len(behavior_sig)-offset:
                st_ed = st_ed[:len(st_ed)-1]
            print("starts_ends after removing first or last one",st_ed)


            #remove winsdows with size < min_win_size
            st_ed=remove_small_win(st_ed, min_win_size)
            print("starts_ends after removing small windows", st_ed)

            # # remove winsdows with size > min_win_size
            # st_ed = remove_long_win(st_ed, max_win_size)
            # print("starts_ends after removing long windows", st_ed)
            plt.plot(behavior_sig)  # [6700:7000])
            plt.plot(dfof1)  # [6700:7000])
            for ppl in range(0,len(st_ed)):
                plt.axvline(x=(st_ed[ppl][0]), color='r', linestyle='--')
                plt.axvline(x=(st_ed[ppl][1]), color='green', linestyle='--')
            plt.show()

            ## find the max window size
            max_window = 0
            for t in range(0,len(st_ed)):
                win_d=st_ed[t][1] - st_ed[t][0] + 1
                if win_d > max_window:
                    max_window=win_d
            print("max_window",max_window)
            max_window=5000
            total_num_behavior_afterMerging=len(st_ed)
            streched_win = []
            start_offs = []
            end_offs = []
            dfof=zscore(dfof)
            for y in range(0,len(st_ed)):  #len(st_ed)
                signal=dfof[st_ed[y][0] - offset:st_ed[y][1] + offset]
                print(st_ed[y])
                # print(offset)
                behavior_main=behavior_sig[st_ed[y][0] - offset:st_ed[y][1] + offset]
                # plt.plot(signal)
                # plt.plot(behavior_main)
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(signal) - offset, color='r', linestyle='--')
                # plt.show()

                ### start and end of behavior for NaN detections
                win_start = (np.array(behavior_main[:offset]))
                win_start = [int(x) for x in win_start]
                win_start = np.array([x ^ 1 for x in win_start])
                behavior_win_start = np.where(win_start == 0, np.nan, win_start)
                ###
                win_end = (np.array(behavior_main[len(signal) - offset:len(signal)]))
                win_end = [int(x) for x in win_end]
                win_end = np.array([x ^ 1 for x in win_end])
                behavior_win_end = np.where(win_end == 0, np.nan, win_end)

                # split the siganl to start and end and main part
                start_off = signal[:offset]
                end_off = signal[len(signal) - offset:len(signal)]
                main_part = signal[offset:len(signal) - offset]
                # z zcore base on the start offset
                mean_strt=np.mean(start_off)
                std_strt=np.std(start_off)
                z_start_off=start_off #(start_off-mean_strt)/std_strt
                z_end_off=end_off #(end_off-mean_strt)/std_strt
                z_main=main_part #(main_part-mean_strt)/std_strt
                # z_start_off = (start_off-mean_strt)/std_strt
                # z_end_off =(end_off-mean_strt)/std_strt
                # z_main = (main_part-mean_strt)/std_strt
                # streching the main part
                # Create an array of main part signal values (time points)
                original_x = np.arange(len(z_main))
                # Define the new x values (time points) after stretching
                new_x = np.linspace(0, len(z_main) - 1, max_window)
                # Create a function for linear interpolation
                interp_func = interp1d(original_x, z_main, kind='linear')
                # Interpolate the signal at new x values
                new_signal = interp_func(new_x)
                ## save for avg
                streched_win.append(new_signal)
                start_offs.append(behavior_win_start * z_start_off)
                end_offs.append(behavior_win_end * z_end_off)

            average_signal = np.mean(streched_win, axis=0)
            average_starts = np.mean(start_offs, axis=0)
            average_ends = np.mean(end_offs, axis=0)

            if not np.any(np.isnan(average_signal)):
                all_main.append(np.mean(average_signal.reshape(-1, 20), axis=1))
                all_sts.append(average_starts)
                all_eds.append(average_ends)
                # merged = np.concatenate((average_starts, average_signal, average_ends))
                # # plt.text(max_window, 0, f'# = {total_num_behavior}')
                # # plt.text(max_window, -0.2, f'#Merging = {total_num_behavior_afterMerging}')
                # # plt.text(max_window, -0.4, f'dis(s) = {min_distance / sr_analysis}')
                # # plt.text(max_window, -0.6, f'ofst(s) = {offset / sr_analysis}')
                # plt.grid(True)
                # plt.plot(merged, label='Average Signal')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(average_signal) + offset, color='r', linestyle='--')
                # plt.show()

            ## go to next expriment file
        i = i + 20

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)

    # ##main
    all_main_avg = np.mean(all_main, axis=0)
    std_main = np.std(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.mean(all_sts, axis=0)
    std_st = np.std(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.mean(all_eds, axis=0)
    std_ed = np.std(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    # Smoothing function using a moving average
    def smooth_it(signal, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal


    # merged=np.array(smooth_signal(merged[0:150],0.5,sr_analysis))
    # merge_sem=np.array(smooth_signal(merge_sem[0:150],0.5,sr_analysis))
    plt.plot(merged, label='Average dfof all expriments for '+str(name_behavior))
    # Handle NaN values by interpolation
    # merged_cleaned = interpolate_nans(merged)
    # # Smoothed signal
    # smoothed_merged = np.concatenate((smooth_it(all_sts_avg, 30),smooth_it(all_main_avg, 30) , smooth_it(all_eds_avg, 30) ))
    # plt.plot(smoothed_merged)
    plt.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='lightgray')
    plt.axvline(x=offset, color='r', linestyle='--')
    plt.axvline(x=len(all_main_avg) + offset, color='r', linestyle='--')
    # plt.text(len(all_main_avg), -0.2, f'min_win_size = {min_win_size}')
    # plt.text(len(all_main_avg), -0.3, f'dis(s) = {min_distance / sr_analysis}')
    # plt.text(len(all_main_avg), -0.4, f'ofst(s) = {offset / sr_analysis}')
    # plt.ylim([-1 , 1])
    plt.xlim([0, len(merged)])
    plt.legend()
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/'
    # plt.savefig(path+str(name_behavior)+".svg",format="svg")
    plt.show()

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
    bar_width = 0.8  # You can adjust this as needed

    plt.figure(figsize=(10, 6))
    plt.bar(x_sts, all_sts_avg_binned, color='blue', alpha=0.6, label='Section 1', width=bar_width)
    plt.bar(x_main, all_main_avg_binned, color='red', alpha=0.6, label='Section 2', width=bar_width)
    plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.6, label='Section 3', width=bar_width)

    # Add vertical dashed lines for discontinuity
    plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='black', linestyle='--', linewidth=1.5)
    plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='black', linestyle='--',
                linewidth=1.5)
    # plt.ylim([-0.8, 0.8])
    # plt.xlim([0, len(merged)])
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Z score')
    plt.title('Bar Plot of Averaged Data (10 points per bar)')
    plt.legend()
    # plt.savefig(path + str(name_behavior) + "_bars.svg", format="svg")
    # Show the plot
    plt.show()


def GLM_func():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/GLM/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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
            # print(pseudo_r2_cs)
            # coeffs = result1.params
            # coeffs_dict = coeffs.to_dict()
            # coeffs_dict['exp'] = expriment_name  # Optional: Add run number for reference
            # coefficients_list.append(coeffs_dict)
            # print(coefficients_list)
            # filename = path + "/GLM_whole_coef.csv"
            # coefficients_df = pd.DataFrame(coefficients_list)
            # coefficients_df.to_csv(filename, index=False)

            # filename = path + "/GLM_whole_new.csv"
            # if not os.path.isfile(filename):
            #     # If the file does not exist, write the DataFrame with headers
            #     result_df.to_csv(filename, index=False, mode='w')
            # else:
            #     # If the file exists, append without headers
            #     result_df.to_csv(filename, index=False, mode='a', header=False)
            #
            # filename = path + "/GLM_whole_stats.csv"
            # if not os.path.isfile(filename):
            #     # If the file does not exist, write the DataFrame with headers
            #     model_stats.to_csv(filename, index=False, mode='w')
            # else:
            #     # If the file exists, append without headers
            #     model_stats.to_csv(filename, index=False, mode='a', header=False)
        i = i+1


def behavioral_interactions():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/interactions/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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
        if task_type == 0 : #or task_type == 0:
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

def washout_win():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/GLM_washout/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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
                X1['rearing'] = rearings[time_win:time_win + win_len]  # all_rearings
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
                    print("yes")
                else:
                    columns = ["const", "exp_non_statobj", "exp_statobj", "speed", "rearing", "groomig"]
                    data = [np.nan] * len(columns)
                    nan_df = pd.DataFrame([data], columns=columns)
                    nan_df.insert(0, 'task', None)
                    nan_df.at[1, 'task'] = str(task)
                    all_results = pd.concat([all_results, nan_df.iloc[1:, 0:]], ignore_index=True)
                    print("no")
                time_win = time_win + 1* 60 * sr_analysis
                w = w + 1

            # all_results=pd.concat([all_results, transposed_data.iloc[0:1,0:]], ignore_index=True)
            # all_results.to_csv(filename, index=False)

            nonstatObj_list=all_results.iloc[0:12,3:4].to_numpy().astype(float)
            statObj_list=all_results.iloc[0:12,4:5].to_numpy().astype(float)
            if task_type == 1:
                nonstat_stat_obj_list_test.append(nonstatObj_list-statObj_list)
            else:
                nonstat_stat_obj_list_sample.append(nonstatObj_list-statObj_list)
            num_exp+=1
            # group = 0 if task_type == 0 else 1
            # # Create a temporary DataFrame for the current experiment
            # temp_df = pd.DataFrame({
            #     'diff': (nonstatObj_list-statObj_list).squeeze(),
            #     'win': np.arange(1,13,1),
            #     'group': group,
            #     'experiment':i,
            #     'mouseid':mouseID
            # })
            # # Append the temporary DataFrame to the list
            # merged_data.append(temp_df)

        i = i + 1
    # df = pd.concat(merged_data, ignore_index=True)
    # df.to_csv(path + "nall_dataaa.csv", index=False)
    # print(df)

    nonstat_stat_obj_list_sample = np.array(nonstat_stat_obj_list_sample).squeeze()
    nonstat_stat_obj_list_test = np.array(nonstat_stat_obj_list_test).squeeze()
    # # # Convert to DataFrames
    # df_sample = pd.DataFrame(nonstat_stat_obj_list_sample)
    # df_test = pd.DataFrame(nonstat_stat_obj_list_test)
    # # # Save to CSV
    # # df_sample.to_csv(path+"nonstat_stat_obj_list_sample.csv", index=False)
    # # df_test.to_csv(path+"nonstat_stat_obj_list_test.csv", index=False)
    diff_test_sample= nonstat_stat_obj_list_test-nonstat_stat_obj_list_sample
    # diffrences=df_test-df_sample
    # avg_diffrences=np.nanmean(diffrences, axis=0)
    # sem_diffrences = np.nanstd(diffrences, axis=0) / np.sqrt(len(diffrences))
    # print(diffrences)

    # Print the resulting DataFrame

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

    x = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(regression_line_test, x).fit()
    print(model.summary())

    model = sm.OLS(regression_line_sample, x).fit()
    print(model.summary())

    # Perform linear regression
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
    # plt.grid(True)                                                                                       
    plt.savefig(path+"Diff_regressions.svg",format="svg")
    plt.show()
    #
    # # Scatter plot of the data
    # plt.scatter(filtered_df['win'], filtered_df['diff'], label='sample', color='#048786')
    # plt.plot(filtered_df['win'], model.predict(X), color='#048786', label='Regression Line for sample')
    # plt.scatter(filtered_df_test['win'], filtered_df_test['diff'], label='test', color='#871719')
    # plt.plot(filtered_df_test['win'], model_test.predict(X_test), color='#871719', label='Regression Line for test')
    # plt.xlabel('win')
    # plt.ylabel('diff')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()
    # #
    # diffrences["mouseid"]=[1,2,2,3,3,4,4,5,5,6]
    # diffrences["experiment"]=[1,2,3,4,5,6,7,8,9,10]
    # # Flatten the DataFrame
    # differences_long = pd.melt(diffrences,
    #                            id_vars=["mouseid", "experiment"],  # Keep these columns as identifiers
    #                            var_name="time_point",  # Name for the column representing time points
    #                            value_name="diff")  # Name for the column representing values
    # df_clean = differences_long.dropna()
    # # Prepare the data for OLS regression
    # df_clean["time_point"] = pd.to_numeric(df_clean["time_point"])
    # X_test = df_clean["time_point"]
    # X_test = sm.add_constant(X_test)  # Add a constant term for the intercept
    # y_test = df_clean["diff"]
    #
    # # Fit the OLS regression model
    # model_test = sm.OLS(y_test, X_test).fit()
    #
    # # Print the summary
    # print(model_test.summary())
    #
    # # Plot the scatter plot and regression line
    # plt.scatter(df_clean["time_point"], df_clean["diff"], label='Data Points', color='#871719')
    # plt.plot(df_clean["time_point"], model_test.predict(X_test), color='#048786', label='Regression Line')
    # plt.xlabel('Time Point (win)')
    # plt.ylabel('Difference (diff)')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()
    #
    #
    # avg_diffrences=np.nanmean(diffrences, axis=0)
    # print(avg_diffrences)
    # sem_diffrences = np.nanstd(diffrences, axis=0) / np.sqrt(len(diffrences))
    # # Perform linear regression
    # x = np.arange(1, 13)
    # x_reshaped = x.reshape(-1, 1)
    #
    # model_sample = LinearRegression()
    # model_sample.fit(x_reshaped, avg_diffrences)
    # regression_line_sample = model_sample.predict(x_reshaped)
    #
    # # Extract regression formulas
    # coef_sample, intercept_sample = model_sample.coef_[0], model_sample.intercept_
    #
    # formula_sample = f"y = {coef_sample:.5f}x + {intercept_sample:.5f}"
    #
    # # Plot the scatter plot and regression line
    # plt.figure(figsize=(15, 7))
    # plt.scatter(x, avg_diffrences, color='#048786', label='Mean Diff Sample (Signal)')
    # plt.plot(x, regression_line_sample, color='#048786', label=f'Regression Line ({formula_sample})')
    # plt.legend(fontsize=12)
    # plt.title('Scatter Plot with Regression Lines', fontsize=14)
    # plt.xlabel('X-axis', fontsize=12)
    # plt.ylabel('Y-axis', fontsize=12)
    # # plt.grid(True)
    # # plt.savefig(path+"AvgSEM_Diff_regressions.svg",format="svg")
    # plt.show()
    #
    # x = sm.add_constant(x)  # Adds a constant term to the predictor
    # model = sm.OLS(avg_diffrences, x).fit()
    # print(model.summary())
    #
    #
    # diffrences["mouseid"]=[1,2,2,3,3,4,4,5,5,6]
    # diffrences["experiment"]=[1,2,3,4,5,6,7,8,9,10]
    # # Flatten the DataFrame
    # differences_long = pd.melt(diffrences,
    #                            id_vars=["mouseid", "experiment"],  # Keep these columns as identifiers
    #                            var_name="time_point",  # Name for the column representing time points
    #                            value_name="diff")  # Name for the column representing values
    # df_clean = differences_long.dropna()
    # dff = pd.DataFrame()  # Instantiate a new DataFrame
    # dff["mouseid"] = df_clean["mouseid"].astype("category")
    # dff["experiment"] = df_clean["experiment"].astype("category")
    # dff["time_point"] = pd.to_numeric(df_clean["time_point"])
    # dff["diff"] = pd.to_numeric(df_clean["diff"])
    #
    # print(dff)
    # # Proceed with the linear mixed model
    # model = smf.mixedlm(
    #     "diff ~ time_point",  # Fixed effects: win, group, interaction
    #     data=dff,  # Use the cleaned DataFrame
    #     groups=dff["mouseid"],  # Random effects grouped by mouse id
    #     vc_formula={"experiment": "0 + C(experiment, Treatment(1))"}
    # )
    # result = model.fit()
    # print(result.summary())
    #
    # # Proceed with the linear mixed model
    # # Fit an OLS model including mouseid and experiment as categorical variables
    # ols_model = smf.ols("diff ~ time_point + C(mouseid, Treatment(reference=6)) ", data=dff)
    # ols_result = ols_model.fit()
    # print(ols_result.summary())



def corr_speed_dfof(dfof,smooth_time_window,input_speed,sr_analysis):
    ## find min as offset for log

    if smooth_time_window==0:
        smooth_speed = input_speed
        smoothed_dfof = dfof
        print("wrong smooth window size")
    else:
        smooth_speed = input_speed
        window_samples = int(smooth_time_window * sr_analysis)  # number of samples in the 1s window
        kernel = np.ones(window_samples) / window_samples
        smoothed_dfof = np.convolve(dfof, kernel, mode='same')

    ####### corr log
    # Compute Pearson's R
    Rlog, _ = pearsonr(smooth_speed, smoothed_dfof)

    return Rlog
def plot_avg_withSem(path,x,data_mean,data_sem):
    x_shape=np.log2(x)
    # Interpolation using Cubic Spline
    cs = CubicSpline(x_shape, data_mean)
    # Generate a finer set of x values for interpolation
    x_fine = np.linspace(x_shape.min(), x_shape.max(), 500)
    data_mean_fine = cs(x_fine)
    # Find the x value corresponding to the maximum interpolated data mean
    max_mean_index_fine = np.argmax(data_mean_fine)
    max_mean_value_fine = data_mean_fine[max_mean_index_fine]
    corresponding_x_value_fine = x_fine[max_mean_index_fine]
    print(pow(2, corresponding_x_value_fine))

    plt.figure(figsize=(6, 5))
    plt.plot(x_shape,data_mean)
    plt.fill_between(x_shape, data_mean - data_sem, data_mean + data_sem, color='lightgray')
    # plt.errorbar(x_shape, data_mean, yerr=data_sem, fmt='-o', color='blue',
    #               ecolor='#78BEE9', capsize=5, label="Learning")
    # plt.scatter(x_shape, data_mean)
    # plt.legend()
    # plt.axvline(x=corresponding_x_value_fine, color='r', linestyle='--')
    plt.xlabel("window size (s) for smoothing = pow(2,x)")
    plt.ylabel("R")
    # plt.savefig(path + "AvgSEM_max_nodots.svg", format="svg")
    plt.show()
def find_best_smoothing_corr():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/best_correlation/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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

        # x=[0.25,0.5,1,2,4,8,16,32,64,128,256]
        # plt.plot(x, rcorr)
        # plt.scatter(x, rcorr)
        # plt.xlabel("window size(s) for smoothing")
        # plt.ylabel("R")
        # plt.savefig(path+"R_30"+expriment_name+".svg",format="svg")
        # plt.xscale("log")
        # plt.savefig(path + "R_30_log" + expriment_name + ".svg", format="svg")
        # plt.show()

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

    plot_avg_withSem(path,x,mean_across_experiments,sem_across_experiments)





def extract_frame():
    directory_path = 'E:/lab/Cholinergic Prj/Data/30/Full_GCaMP7sChAT_611926_210604_Rec2_Light_15min_ObjectLocationMemory_Recall/'
    video_name='Full_GCaMP7sChAT_611926_210604_Rec2_Light_15min_ObjectLocationMemory_Recall.mp4'
    video_path=directory_path+video_name
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Frame number you want to extract
    frame_number = 1000  # Change this to the frame number you want

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as an image file
        cv2.imwrite(directory_path+'extracted_frame.jpg', frame)
        print("Frame saved as 'extracted_frame.jpg'")
    else:
        print("Error: Could not read the frame.")

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()




def washout_win_avg():
    sr_analysis = 30
    path = 'C:/Users/ffarokhi/Desktop/paper/6) GLM/updated_9-17-2024/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    diff_objs=[]
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
        analysis_win = analysis_win - onset
        data = pd.read_csv(file_data).values
        data = pd.DataFrame(data)
        time = np.array(data[0][onset:analysis_win])
        dfof = np.array(data[1][onset:analysis_win])
        speed = np.array(data[2][onset:analysis_win])
        exp_non_statobj = np.array(data[3][onset:analysis_win])
        exp_statobj = np.array(data[4][onset:analysis_win])
        walking = np.array(data[5][onset:analysis_win])
        rearings = np.array(data[6][onset:analysis_win])
        groomings = np.array(data[7][onset:analysis_win])

        # data2 = pd.read_csv(labels_file).values
        # data2 = pd.DataFrame(data2)
        # all_rearings = np.array(np.array(data2[3][onset:analysis_win]) | np.array(data2[4][onset:analysis_win]))

        smooth_speed = smooth_signal(speed, 1, sr_analysis)
        unique_speed = np.unique(smooth_speed)
        sorted_speed = np.sort(unique_speed)
        offset = sorted_speed[1]
        log_speed = np.log2(smooth_speed + offset)
        # Replace NaNs with the mean of the variable
        log_speed[np.isnan(log_speed)] = np.nanmean(log_speed)
        # Replace infs with a specific value, for example, replacing with the maximum value
        log_speed[np.isinf(log_speed)] = np.nanmax(log_speed)

        ## GLM
        w = 0
        time_win = 0
        win_len=3*60*sr_analysis
        all_results = pd.DataFrame()
        while time_win <= analysis_win-win_len:
            ## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj[time_win:time_win + win_len ]
            X1['exp_statobj'] = exp_statobj[time_win:time_win + win_len ]
            X1['speed'] = log_speed[time_win:time_win + win_len ]
            X1['rearing'] = rearings[time_win:time_win + win_len ]  # all_rearings
            X1['groomig'] = groomings[time_win:time_win + win_len ]
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            # X1['speed_grooming_interaction'] = X1['speed'] * X1['groomig']
            # X1['speed_non_statobj_interaction'] = X1['speed'] * X1['exp_non_statobj']
            # X1['speed_statobj_interaction'] = X1['speed'] * X1['exp_statobj']
            # X1['nonobj_rearing'] = X1['rearing'] * X1['exp_non_statobj']
            # X1['obj_rearing'] = X1['rearing'] * X1['exp_statobj']
            fof = dfof[time_win:time_win + win_len]
            X1 = sm.add_constant(X1)

            model2 = sm.GLM(fof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()

            # X1.to_csv(filepath_exp + '/X1_data.csv', index=False)
            # Interpret the model
            # print(result2.summary())
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            print(result_df2)
            first_two_columns = result_df2.iloc[:, :2]
            transposed_data = first_two_columns.T
            transposed_data.insert(0, 'task', None)
            transposed_data.at[1, 'task'] = str(task)
            filename = path + "/Avg_DiffObjs"+str(task)+"_GLM_smoothSpeed1S_all15min_washout3min" + ".csv"
            all_results = pd.concat([all_results, transposed_data.iloc[1:,0:]], ignore_index=True)
            time_win = time_win + 1* 60 * sr_analysis
            w = w + 1
        all_results=pd.concat([all_results, transposed_data.iloc[0:1,0:]], ignore_index=True)
        # print(all_results)
        nonstatObj_plot=all_results.iloc[0:12,3:4].to_numpy().astype(float)
        print(all_results.iloc[0:12,3:4].to_numpy().astype(float))
        statObj_plot=all_results.iloc[0:12,4:5].to_numpy().astype(float)
        diff_objs.append(nonstatObj_plot-statObj_plot)

        i = i + 1
    diff_objs_2d = diff_objs.reshape(diff_objs.shape[0], -1)  # Converts (20, 12, 1) to (20, 12)

    # Now create the DataFrame
    dataframe_diff = pd.DataFrame(diff_objs_2d)
    avg_diff_objs=np.mean(diff_objs,axis=0)
    # print(avg_diff_objs)
    plt.figure(figsize=(12, 5))
    plt.plot(avg_diff_objs)
    plt.ylim([-0.012, 0.035])
    plt.savefig(path + "Avg_DiffObjs"+str(task)+"_GLM_smoothSpeed1S_all15min_washout3min" + ".svg", format="svg")
    plt.show()
    reshaped_diff_objs = [x[0] for x in avg_diff_objs]
    df = pd.DataFrame(reshaped_diff_objs, columns=['Value'])
    df.to_csv(filename, index=False)


def plot_avg_withSem_GLM():
    path = 'C:/Users/ffarokhi/Desktop/paper/6) GLM/updated_9-17-2024/'
    df = pd.read_csv(path + 'diff_sample.csv')
    df2 = pd.read_csv(path + 'diff_test.csv')
    data_mean = df.mean(axis=0)
    data_sem = df.sem(axis=0)
    data_mean2 = df2.mean(axis=0)
    data_sem2 = df2.sem(axis=0)
    # Fill area between lines
    plt.figure(figsize=(12, 5))
    # plt.fill_between(np.arange(len(data_mean)), data_mean - data_sem, data_mean + data_sem, color='lightgray')
    plt.errorbar(np.arange(len(data_mean)), data_mean, yerr=data_sem, fmt='-o', color='blue',
                 ecolor='#78BEE9', capsize=5, label="Sample")
    # plt.fill_between(np.arange(len(data_mean2)), data_mean2 - data_sem2, data_mean2 + data_sem2, color='lightgray')
    plt.errorbar(np.arange(len(data_mean2)), data_mean2, yerr=data_sem2, fmt='-o', color='green',
                 ecolor='#86C547', capsize=5, label="test")
    plt.ylim([-0.025, 0.045])
    plt.grid()
    plt.legend()
    # plt.savefig(path+"AvgSEM_DiffObjs.svg",format="svg")
    plt.show()


def histo_distance_behaviors():
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/hist/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    st_ed=[]
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
            analysis_win = 900 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 900 * sr_analysis
        ##
        ############################## calulations
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

        ########################### streching time algorithm
        name_behavior = "exp_non_statobj"
        behavior_sig=np.array(exp_non_statobj)

        #### find start and endpoints
        count_st=0
        starts=[]
        ends=[]
        count_ed = 0
        for sweep in range(0,len(behavior_sig)):
            if np.array_equal([0, 1], np.array(behavior_sig[sweep: sweep + 2])):
                count_st=count_st+1
                starts.append(sweep)
            if np.array_equal([1, 0], np.array(behavior_sig[sweep: sweep + 2])):
                count_ed=count_ed+1
                ends.append(sweep)
        if ends[0]<starts[0]:
            ends=ends[1:]
        if len(starts)>len(ends):
            starts=starts[:len(starts)-1]
        total_num_behavior=len(starts)
        print(total_num_behavior)
        #### merge start and end points

        for se in range(0,len(starts)-1):
                st_ed.append(starts[se+1]-ends[se])
        # print(len(st_ed))
        i=i+1
    st_ed = [value / sr_analysis for value in st_ed if value <= 900 * sr_analysis]
    plt.hist(st_ed, bins=60, label=str(name_behavior))
    plt.xlabel("inter bout interval (s)")
    print(len(st_ed))
    plt.legend()
    # plt.savefig(path + str(name_behavior) + "_hist_distance.svg", format="svg")
    plt.show()

    # Define the bins for the histogram using log scale
    bins = np.logspace(np.log10(min(st_ed)), np.log10(max(st_ed)), 20)

    # Plot the histogram with the log-scaled bins
    plt.hist(st_ed, bins=bins, label=str(name_behavior))

    # Set the x-axis to log scale
    plt.xscale('log')
    plt.xlabel("inter bout interval (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(path + str(name_behavior) + "_hist_distance_log_less.svg", format="svg")
    # Display the plot
    plt.show()

def histo_winsize_behaviors():
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/hist/'
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    st_ed=[]
    win_sz = []
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
            analysis_win = 900 * sr_analysis
            # 720
        else:  ## Recall 3 min
            task_type = 1
            analysis_win = 900 * sr_analysis
        ##
        ############################## calulations
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

        ########################### streching time algorithm
        name_behavior = "rearings"
        behavior_sig=np.array(rearings)

        #### find start and endpoints
        count_st = 0
        starts = []
        ends = []
        count_ed = 0
        for sweep in range(0, len(behavior_sig)):
            if np.array_equal([0, 1], np.array(behavior_sig[sweep: sweep + 2])):
                count_st = count_st + 1
                starts.append(sweep)
            if np.array_equal([1, 0], np.array(behavior_sig[sweep: sweep + 2])):
                count_ed = count_ed + 1
                ends.append(sweep)
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(starts) - 1]
        print("starts", starts)
        print("ends", ends)

        for his in range(0, len(starts)):
            win_sz.append(ends[his] - starts[his])
        print(win_sz)
        i = i + 1
    win_sz = [value / sr_analysis for value in win_sz if value <= 50 * sr_analysis]
    plt.hist(win_sz, bins=60, label=str(name_behavior))
    plt.xlabel("win size (s)")
    print(len(win_sz))
    plt.legend()
    # plt.savefig(path + str(name_behavior) + "_hist_distance.svg", format="svg")
    plt.show()

    # Define the bins for the histogram using log scale
    bins = np.logspace(np.log10(min(win_sz)), np.log10(max(win_sz)), 60)

    # Plot the histogram with the log-scaled bins
    plt.hist(win_sz, bins=bins, label=str(name_behavior))

    # Set the x-axis to log scale
    plt.xscale('log')
    plt.xlabel("win size (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(path + str(name_behavior) + "_hist_winsize_log.svg", format="svg")
    # Display the plot
    plt.show()




########
def test_strech_time_newIdea():
    sr_analysis = 30
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp=[]
    all_main=[]
    all_sts=[]
    all_eds=[]
    total_calculated_end=0
    total_calculated_st=0
    total_num_beahviors=0

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

            ## calculate log speed
            smooth_speed = smooth_signal(speed, 1, sr_analysis)
            unique_speed = np.unique(smooth_speed)
            sorted_speed = np.sort(unique_speed)
            offset = sorted_speed[1]
            log_speed = np.log2(smooth_speed + offset)
            # Replace NaNs with the mean of the variable
            log_speed[np.isnan(log_speed)] = np.nanmean(log_speed)
            # Replace infs with a specific value, for example, replacing with the maximum value
            log_speed[np.isinf(log_speed)] = np.nanmax(log_speed)
            ########## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj
            X1['exp_statobj'] = exp_statobj
            X1['speed'] = log_speed
            X1['rearing'] = rearings  # all_rearings
            X1['groomig'] = groomings
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            X1 = sm.add_constant(X1)
            model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            speed_coef=float(result_df2.iloc[4:5, 1:2].values)
            ###########################

            name_behavior="walking"
            behavior_sig=np.array(walking)#[0:3000]
            speed=np.array(log_speed)#[0:3000]
            dfof1 = np.array(dfof) #[0:3000]

            min_distance = int(1*sr_analysis)
            min_dis_toShow= int(4*sr_analysis)
            offset = int(5*sr_analysis)
            min_win_size = int(1*sr_analysis)
            dfof=dfof1 - speed_coef*speed


            # plt.plot(behavior_sig)
            # plt.plot(dfof1)
            # plt.show()

            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)

            #### merge start and end points
            st_ed=merge_startANDend_points(starts, ends, min_distance)

            #remove winsdows with size < min_win_size
            st_ed=remove_small_win(st_ed, min_win_size)
            print("starts_ends after removing small windows", len(st_ed), st_ed)

            ## find the max window size
            max_window = 0
            for t in range(0,len(st_ed)):
                win_d=st_ed[t][1] - st_ed[t][0] + 1
                if win_d > max_window:
                    max_window=win_d

            max_window=5000
            streched_win = []
            start_offs = []
            end_offs = []
            dfof=zscore(dfof)



            # plt.plot(behavior_sig)  # [6700:7000])
            # plt.plot(dfof1)  # [6700:7000])
            # for ppl in range(0, len(st_ed)):
            #     plt.axvline(x=(st_ed[ppl][0]), color='r', linestyle='--')
            #     plt.axvline(x=(st_ed[ppl][1]), color='green', linestyle='--')
            # plt.title("after merging and removing small win size = "+ str(min_win_size/sr_analysis))
            # plt.show()



            def start_end_ind(st_ed, offset, dfof,y,min_dis_toShow):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                st=0
                end=0
                # Handle negative indices by inserting NaN
                if start_idx < 0 or (st_ed[y][0]-st_ed[y-1][1]) < min_dis_toShow:
                    stt = np.full(offset, np.nan)  # Fill NaN for negative range
                else:
                    stt = dfof[start_idx:st_ed[y][0]]
                    st += 1

                if y<len(st_ed)-1:
                    if end_idx > len(dfof) or (st_ed[y+1][0]-st_ed[y][1]) < min_dis_toShow:
                        ed = np.full(offset, np.nan)
                    else:
                        ed = dfof[st_ed[y][1]+1:end_idx]
                        end += 1
                else:
                    if end_idx > len(dfof):
                        ed = np.full(offset, np.nan)
                    else:
                        ed = dfof[st_ed[y][1] + 1:end_idx]
                        end += 1

                signal = np.concatenate([stt, dfof[st_ed[y][0]:st_ed[y][1] + 1],ed])

                return signal, stt, dfof[st_ed[y][0]:st_ed[y][1] + 1],ed, st, end


            for y in range(0,len(st_ed)):  #len(st_ed)

                signal, stt, mainnn,ed,st, end=start_end_ind(st_ed, offset, dfof,y,min_dis_toShow)
                total_calculated_end += st
                total_calculated_st += end
                behavior_main ,sttt, mainnnnn,edd,st, end=start_end_ind (st_ed, offset, behavior_sig,y,min_dis_toShow)
                # print(st_ed[y])
                # print("signal",signal, len(signal), "iiiiiiiii", i)
                # print("behavior_main",behavior_main, len(behavior_main))
                # plt.plot(signal)
                # plt.plot(behavior_main)
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(signal) - offset, color='r', linestyle='--')
                # plt.xlim([0, len(signal)])
                # plt.show()
                #
                ### start and end of behavior for NaN detections
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start ==1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start ==0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len(signal) - offset : len(signal)]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)

                # split the siganl to start and end and main part
                start_off = signal[:offset]
                end_off = signal[len(signal) - offset:len(signal)]
                main_part = signal[offset:len(signal) - offset]

                # Create an array of main part signal values (time points)
                original_x = np.arange(len(main_part))
                # Define the new x values (time points) after stretching
                new_x = np.linspace(0, len(main_part) - 1, max_window)
                # Create a function for linear interpolation
                interp_func = interp1d(original_x, main_part, kind='linear')
                # Interpolate the signal at new x values
                new_signal = interp_func(new_x)
                ## save for avg
                streched_win.append(new_signal)
                start_offs.append(behavior_win_start * start_off)
                end_offs.append(behavior_win_end * end_off)

            average_signal = np.nanmean(streched_win, axis=0)
            average_starts = np.nanmean(start_offs, axis=0)
            average_ends = np.nanmean(end_offs, axis=0)

            if not np.any(np.isnan(average_signal)):
                average_signal=np.mean(average_signal.reshape(-1, 20), axis=1)
                all_main.append(average_signal)
                all_sts.append(average_starts)
                all_eds.append(average_ends)
                # merged = np.concatenate((average_starts, average_signal, average_ends))
                # plt.text(max_window, 0, f'# = {total_num_behavior}')
                # plt.text(max_window, -0.2, f'#Merging = {total_num_behavior_afterMerging}')
                # plt.text(max_window, -0.4, f'dis(s) = {min_distance / sr_analysis}')
                # plt.text(max_window, -0.6, f'ofst(s) = {offset / sr_analysis}')
                # plt.grid(True)
                # plt.plot(merged, label='Average Signal')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(average_signal) + offset, color='r', linestyle='--')
                # plt.show()

            ## go to next expriment file
            total_num_beahviors += len(st_ed)
        i = i + 1

    print("total_num_beahviors", total_num_beahviors)  #
    print("calculated starts", total_calculated_st)
    print("calculated ends", total_calculated_end)

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)


    # ##main
    all_main_avg = np.mean(all_main, axis=0)
    std_main = np.std(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.mean(all_sts, axis=0)
    std_st = np.std(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.mean(all_eds, axis=0)
    std_ed = np.std(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    # Smoothing function using a moving average
    def smooth_it(signal, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal


    # merged=np.array(smooth_signal(merged[0:150],0.5,sr_analysis))
    # merge_sem=np.array(smooth_signal(merge_sem[0:150],0.5,sr_analysis))
    plt.plot(merged, label='Average dfof all expriments for '+str(name_behavior))
    # Handle NaN values by interpolation
    # merged_cleaned = interpolate_nans(merged)
    # # Smoothed signal
    # smoothed_merged = np.concatenate((smooth_it(all_sts_avg, 30),smooth_it(all_main_avg, 30) , smooth_it(all_eds_avg, 30) ))
    # plt.plot(smoothed_merged)
    plt.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='lightgray')
    plt.axvline(x=offset, color='r', linestyle='--')
    plt.axvline(x=len(all_main_avg) + offset, color='r', linestyle='--')
    # plt.text(len(all_main_avg), -0.2, f'min_win_size = {min_win_size}')
    # plt.text(len(all_main_avg), -0.3, f'dis(s) = {min_distance / sr_analysis}')
    # plt.text(len(all_main_avg), -0.4, f'ofst(s) = {offset / sr_analysis}')
    # plt.ylim([-1 , 1])
    plt.xlim([0, len(merged)])
    plt.legend()
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/'
    # plt.savefig(path+str(name_behavior)+".svg",format="svg")
    plt.show()

    # def average_n_points(data, n):
    #     end = len(data) - (len(data) % n)  # Adjust the length to be divisible by n
    #     reshaped_data = data[:end].reshape(-1, n)
    #     return reshaped_data.mean(axis=1)
    #
    # # Average every 10 points
    # all_sts_avg_binned = average_n_points(interpolate_nans(all_sts_avg), 10)
    # all_main_avg_binned = average_n_points(interpolate_nans(all_main_avg), 10)
    # all_eds_avg_binned = average_n_points(interpolate_nans(all_eds_avg), 10)
    #
    #
    # # Concatenate the averaged arrays
    # merged_binned = np.concatenate((all_sts_avg_binned, all_main_avg_binned, all_eds_avg_binned))
    # # Define the x-axis for the binned data
    # x_sts = np.arange(len(all_sts_avg_binned))
    # x_main = np.arange(len(all_sts_avg_binned), len(all_sts_avg_binned) + len(all_main_avg_binned))
    # x_eds = np.arange(len(all_sts_avg_binned) + len(all_main_avg_binned), len(merged_binned))
    #
    # # Plot each section with bars
    # bar_width = 0.8  # You can adjust this as needed
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(x_sts, all_sts_avg_binned, color='blue', alpha=0.6, label='Section 1', width=bar_width)
    # plt.bar(x_main, all_main_avg_binned, color='red', alpha=0.6, label='Section 2', width=bar_width)
    # plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.6, label='Section 3', width=bar_width)
    #
    # # Add vertical dashed lines for discontinuity
    # plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='black', linestyle='--', linewidth=1.5)
    # plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='black', linestyle='--',
    #             linewidth=1.5)
    # plt.ylim([-0.8, 0.8])
    # # plt.xlim([0, len(merged)])
    # # Add labels, title, and legend
    # plt.xlabel('Time (s)')
    # plt.ylabel('Z score')
    # plt.title('Bar Plot of Averaged Data (10 points per bar)')
    # plt.legend()
    # # plt.savefig(path + str(name_behavior) + "_bars.svg", format="svg")
    # # Show the plot
    # plt.show()



####### test streching
def test_strech_time():
    sr_analysis = 30
    # directory_path = 'E:/lab/Cholinergic Prj/Data/'+str(sr_analysis)+'/'
    # path = 'C:/Users/ffarokhi/Desktop/updateresult/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp=[]
    all_main=[]
    all_sts=[]
    all_eds=[]
    total_calculated_end=0
    total_calculated_st=0
    total_num_beahviors=0
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

            # data2 = pd.read_csv(labels_file).values
            # data2 = pd.DataFrame(data2)
            # all_rearings = (np.array(data2[3][onset:analysis_win]) | np.array(data2[4][onset:analysis_win]))

            ## calculate log speed
            smooth_speed = smooth_signal(speed, 1, sr_analysis)
            unique_speed = np.unique(smooth_speed)
            sorted_speed = np.sort(unique_speed)
            offset = sorted_speed[1]
            log_speed = np.log2(smooth_speed + offset)
            # Replace NaNs with the mean of the variable
            log_speed[np.isnan(log_speed)] = np.nanmean(log_speed)
            # Replace infs with a specific value, for example, replacing with the maximum value
            log_speed[np.isinf(log_speed)] = np.nanmax(log_speed)
            ########## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj
            X1['exp_statobj'] = exp_statobj
            X1['speed'] = log_speed
            X1['rearing'] = rearings  # all_rearings
            X1['groomig'] = groomings
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            # X1['speed_grooming_interaction'] = X1['speed'] * X1['groomig']
            # X1['speed_non_statobj_interaction'] = X1['speed'] * X1['exp_non_statobj']
            # X1['speed_statobj_interaction'] = X1['speed'] * X1['exp_statobj']
            # X1['nonobj_rearing'] = X1['rearing'] * X1['exp_non_statobj']
            # X1['obj_rearing'] = X1['rearing'] * X1['exp_statobj']
            X1 = sm.add_constant(X1)
            model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            speed_coef=float(result_df2.iloc[4:5, 1:2].values)
            ###########################

            name_behavior="groomings"
            behavior_sig=np.array(groomings)
            speed=np.array(log_speed)
            dfof1 = np.array(dfof)

            min_distance = int(1*sr_analysis)
            offset = int(15*sr_analysis)
            min_win_size = int(1*sr_analysis)
            # max_win_size= int(10*sr_analysis)
            dfof=dfof1-speed_coef*speed


            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)
            total_num_behavior=len(starts)

            #### merge start and end points
            st_ed=merge_startANDend_points(starts, ends, min_distance)


            #remove winsdows with size < min_win_size
            st_ed=remove_small_win(st_ed, min_win_size)
            print("starts_ends after removing small windows", len(st_ed), st_ed)

            # plt.plot(behavior_sig)
            # plt.plot(dfof1)
            # for ppl in range(0, len(st_ed)):
            #     plt.axvline(x=(st_ed[ppl][0]), color='r', linestyle='--')
            #     plt.axvline(x=(st_ed[ppl][1]), color='green', linestyle='--')
            # plt.title("after removing small win size = "+ str(min_win_size/sr_analysis))
            # plt.show()


            ## find the max window size
            max_window = 0
            for t in range(0,len(st_ed)):
                win_d=st_ed[t][1] - st_ed[t][0] + 1
                if win_d > max_window:
                    max_window=win_d
            # print("max_window",max_window)
            max_window=5000
            total_num_behavior_afterMerging=len(st_ed)
            streched_win = []
            start_offs = []
            end_offs = []
            dfof=zscore(dfof)

            def start_end_ind(st_ed, offset, dfof):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                st=0
                end=1
                # Handle negative indices by inserting NaN
                if start_idx < 0:
                    nan_fill = np.full(abs(start_idx), np.nan)  # Fill NaN for negative range
                    signal = np.concatenate([nan_fill, dfof[:end_idx]])
                else:
                    signal = dfof[start_idx:end_idx]
                    st = 1

                # If the end index exceeds the length of dfof, pad with NaN
                if end_idx > len(dfof):
                    nan_fill = np.full(end_idx - len(dfof), np.nan)
                    signal = np.concatenate([signal, nan_fill])
                    end = 0
                return signal, st, end


            for y in range(0,len(st_ed)):  #len(st_ed)

                signal, st, end =start_end_ind(st_ed, offset, dfof)
                behavior_main, st, end = start_end_ind(st_ed, offset, behavior_sig)
                total_calculated_st += st
                total_calculated_end += end
                # print(st_ed[y])
                # print("signal",signal, len(signal))
                # print("behavior_main",behavior_main, len(behavior_main))
                # plt.plot(signal)
                # plt.plot(behavior_main)
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(signal) - offset, color='r', linestyle='--')
                # plt.xlim([0, len(signal)])
                # plt.show()

                ### start and end of behavior for NaN detections
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start ==1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start ==0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len(signal) - offset : len(signal)]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)

                # split the siganl to start and end and main part
                start_off = signal[:offset]
                end_off = signal[len(signal) - offset:len(signal)]
                main_part = signal[offset:len(signal) - offset]

                # Create an array of main part signal values (time points)
                original_x = np.arange(len(main_part))
                # Define the new x values (time points) after stretching
                new_x = np.linspace(0, len(main_part) - 1, max_window)
                # Create a function for linear interpolation
                interp_func = interp1d(original_x, main_part, kind='linear')
                # Interpolate the signal at new x values
                new_signal = interp_func(new_x)
                ## save for avg
                streched_win.append(new_signal)
                start_offs.append(behavior_win_start * start_off)
                end_offs.append(behavior_win_end * end_off)
                # test_merged = np.concatenate((behavior_win_start * start_off, main_part, behavior_win_end * end_off))
                # plt.plot(test_merged)
                # plt.show()

            average_signal = np.nanmean(streched_win, axis=0)
            average_starts = np.nanmean(start_offs, axis=0)
            average_ends = np.nanmean(end_offs, axis=0)

            if not np.any(np.isnan(average_signal)):
                average_signal=np.mean(average_signal.reshape(-1, 20), axis=1)
                all_main.append(average_signal)
                all_sts.append(average_starts)
                all_eds.append(average_ends)
                # merged = np.concatenate((average_starts, average_signal, average_ends))
                # plt.text(max_window, 0, f'# = {total_num_behavior}')
                # plt.text(max_window, -0.2, f'#Merging = {total_num_behavior_afterMerging}')
                # plt.text(max_window, -0.4, f'dis(s) = {min_distance / sr_analysis}')
                # plt.text(max_window, -0.6, f'ofst(s) = {offset / sr_analysis}')
                # plt.grid(True)
                # plt.plot(merged, label='Average Signal')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(average_signal) + offset, color='r', linestyle='--')
                # plt.show()

            ## go to next expriment file
                total_num_beahviors += len(st_ed)
        i = i + 1

    print("total_num_beahviors", total_num_beahviors)  #
    print("calculated starts", total_calculated_st)
    print("calculated ends", total_calculated_end)

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)


    # ##main
    all_main_avg = np.mean(all_main, axis=0)
    std_main = np.std(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.mean(all_sts, axis=0)
    std_st = np.std(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.mean(all_eds, axis=0)
    std_ed = np.std(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    # Smoothing function using a moving average
    def smooth_it(signal, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal


    # merged=np.array(smooth_signal(merged[0:150],0.5,sr_analysis))
    # merge_sem=np.array(smooth_signal(merge_sem[0:150],0.5,sr_analysis))
    plt.plot(merged, label='Average dfof all expriments for '+str(name_behavior))
    # Handle NaN values by interpolation
    # merged_cleaned = interpolate_nans(merged)
    # # Smoothed signal
    # smoothed_merged = np.concatenate((smooth_it(all_sts_avg, 30),smooth_it(all_main_avg, 30) , smooth_it(all_eds_avg, 30) ))
    # plt.plot(smoothed_merged)
    plt.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='lightgray')
    plt.axvline(x=offset, color='r', linestyle='--')
    plt.axvline(x=len(all_main_avg) + offset, color='r', linestyle='--')
    # plt.text(len(all_main_avg), -0.2, f'min_win_size = {min_win_size}')
    # plt.text(len(all_main_avg), -0.3, f'dis(s) = {min_distance / sr_analysis}')
    # plt.text(len(all_main_avg), -0.4, f'ofst(s) = {offset / sr_analysis}')
    # plt.ylim([-1 , 1])
    plt.xlim([0, len(merged)])
    plt.legend()
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/'
    # plt.savefig(path+str(name_behavior)+".svg",format="svg")
    plt.show()

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
    bar_width = 0.8  # You can adjust this as needed

    plt.figure(figsize=(10, 6))
    plt.bar(x_sts, all_sts_avg_binned, color='blue', alpha=0.6, label='Section 1', width=bar_width)
    plt.bar(x_main, all_main_avg_binned, color='red', alpha=0.6, label='Section 2', width=bar_width)
    plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.6, label='Section 3', width=bar_width)

    # Add vertical dashed lines for discontinuity
    plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='black', linestyle='--', linewidth=1.5)
    plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='black', linestyle='--',
                linewidth=1.5)
    # plt.ylim([-0.8, 0.8])
    # plt.xlim([0, len(merged)])
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Z score')
    plt.title('Bar Plot of Averaged Data (10 points per bar)')
    plt.legend()
    # plt.savefig(path + str(name_behavior) + "_bars.svg", format="svg")
    # Show the plot
    plt.show()

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


def test_strech_time_otherBehaviors():
    sr_analysis = 30
    # directory_path = 'E:/lab/Cholinergic Prj/Data/'+str(sr_analysis)+'/'
    # path = 'C:/Users/ffarokhi/Desktop/updateresult/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/teokalmans/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp = []
    all_main = []
    all_sts = []
    all_eds = []

    speed_all_main = []
    speed_all_sts = []
    speed_all_eds = []

    total_calculated_end = 0
    total_calculated_st = 0
    total_num_beahviors = 0
    streched_win_behave = []
    start_offs_behave = []
    end_offs_behave = []

    all_main0 = []
    all_sts0 = []
    all_eds0 = []

    all_main1 = []
    all_sts1 = []
    all_eds1 = []

    all_main2 = []
    all_sts2 = []
    all_eds2 = []

    all_main3 = []
    all_sts3 = []
    all_eds3 = []

    all_main4 = []
    all_sts4 = []
    all_eds4 = []

    all_mainbk = []
    all_stsbk = []
    all_edsbk = []

    total_analyzed = 0

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

            # data2 = pd.read_csv(labels_file).values
            # data2 = pd.DataFrame(data2)
            # all_rearings = (np.array(data2[3][onset:analysis_win]) | np.array(data2[4][onset:analysis_win]))

            ## calculate log speed
            log_speed = np.log2(speed + 0.01)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ########## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj
            X1['exp_statobj'] = exp_statobj
            X1['speed'] = log_speed
            X1['rearing'] = rearings  # all_rearings
            X1['groomig'] = groomings
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            # X1['speed_grooming_interaction'] = X1['speed'] * X1['groomig']
            # X1['speed_non_statobj_interaction'] = X1['speed'] * X1['exp_non_statobj']
            # X1['speed_statobj_interaction'] = X1['speed'] * X1['exp_statobj']
            # X1['nonobj_rearing'] = X1['rearing'] * X1['exp_non_statobj']
            # X1['obj_rearing'] = X1['rearing'] * X1['exp_statobj']
            X1 = sm.add_constant(X1)
            model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            speed_coef = float(result_df2.iloc[4:5, 1:2].values)
            groomings_coef = float(result_df2.iloc[6:7, 1:2].values)
            rearings_coef = float(result_df2.iloc[5:6, 1:2].values)
            exp_non_statobj_coef = float(result_df2.iloc[2:3, 1:2].values)
            exp_statobj_coef = float(result_df2.iloc[3:4, 1:2].values)
            constant_coef = float(result_df2.iloc[1:2, 1:2].values)
            # print(result_df2)
            # print(speed_coef,groomings_coef,rearings_coef,exp_non_statobj_coef,exp_statobj_coef,constant_coef)
            ###########################

            behavior_walking = np.array(walking)
            behavior_grooming = np.array(groomings)
            behavior_rearings = np.array(rearings)
            behavior_exp_statobj = np.array(exp_statobj)
            behavior_exp_non_statobj = np.array(exp_non_statobj)

            list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,
                              behavior_exp_non_statobj]
            behavior_bk = (np.logical_or.reduce([behavior_walking,
                                                 behavior_grooming,
                                                 behavior_rearings,
                                                 behavior_exp_statobj,
                                                 behavior_exp_non_statobj]))

            # Perform logical NOT on the OR result
            behavior_bk = np.logical_not(behavior_bk)
            list_behaviors.append(behavior_bk)

            name_behavior = "behavior_walking"
            behavior_sig = behavior_walking
            # speed=np.array(log_speed)
            dfof1 = np.array(dfof)

            min_distance = int(0*sr_analysis)
            offset = int(5*sr_analysis)
            min_win_size = int(0*sr_analysis)
            # max_win_size= int(10*sr_analysis)

            # dfof = dfof1
            dfof=((((dfof1
                    - speed_coef*log_speed - groomings_coef*behavior_grooming - rearings_coef*behavior_rearings )
                     - exp_statobj_coef*behavior_exp_statobj)
                    - exp_non_statobj_coef*behavior_exp_non_statobj)
                    - constant_coef)
            ########################### streching time algorithm

            # window_samples = int(0.5 * sr_analysis)  # number of samples in the 0.5s window
            # kernel = np.ones(window_samples) / window_samples
            # speed = np.convolve(speed, kernel, mode='same')
            # dfof = np.convolve(dfof, kernel, mode='same')

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)
            total_num_behavior=len(starts)

            #### merge start and end points
            st_ed=merge_startANDend_points(starts, ends, min_distance)

            #remove winsdows with size < min_win_size
            st_ed = remove_small_win(st_ed, min_win_size)
            print("starts_ends after removing small windows", len(st_ed), st_ed)

            # Separate into two lists
            # sttttt = [item[0] for item in st_ed]
            # eddddd = [item[1] for item in st_ed]
            # st_ed = merge_startANDend_points(sttttt, eddddd, min_win_size)

            # plt.plot(behavior_sig)
            # plt.plot(dfof1)
            # for ppl in range(0, len(st_ed)):
            #     plt.axvline(x=(st_ed[ppl][0]), color='r', linestyle='--')
            #     plt.axvline(x=(st_ed[ppl][1]), color='green', linestyle='--')
            # plt.title("after removing small win size = "+ str(min_win_size/sr_analysis))
            # plt.show()


            ## find the max window size
            max_window = 0
            for t in range(0,len(st_ed)):
                win_d=st_ed[t][1] - st_ed[t][0] + 1
                if win_d > max_window:
                    max_window=win_d
            # print("max_window",max_window)
            max_window=5000
            total_num_behavior_afterMerging=len(st_ed)
            streched_win = []
            start_offs = []
            end_offs = []

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

            speed_streched_win=[]
            speed_start_offs=[]
            speed_end_offs=[]

            def start_end_ind(st_ed, offset, dfof):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                st=0
                end=1
                # Handle negative indices by inserting NaN
                if start_idx < 0:
                    nan_fill = np.full(abs(start_idx), np.nan)  # Fill NaN for negative range
                    signal = np.concatenate([nan_fill, zscore(dfof[:end_idx])])
                    temp_speed = np.concatenate([nan_fill, speed[:end_idx]])
                else:
                    signal = zscore(dfof[start_idx:end_idx])
                    temp_speed=speed[start_idx:end_idx]
                    st = 1

                # If the end index exceeds the length of dfof, pad with NaN
                if end_idx > len(dfof):
                    nan_fill = np.full(end_idx - len(dfof), np.nan)
                    signal = np.concatenate([signal, nan_fill])
                    temp_speed= np.concatenate([temp_speed, nan_fill])
                    end = 0
                return signal, st, end, temp_speed

            def extract_st_ends_behaviors(behavior_main,offset,len_signal):
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start == 1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start == 0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len_signal - offset: len_signal]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)
                return behavior_win_start,behavior_win_end

            def start_end_ind_strechedmain(st_ed, offset, sig,len_signal):
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


            for y in range(0,len(st_ed)):  #len(st_ed)

                behavior_main, st, end, temp_speed = start_end_ind(st_ed, offset, behavior_sig)
                signal, st, end, temp_speed =start_end_ind(st_ed, offset, dfof)
                total_calculated_st += st
                total_calculated_end += end

                # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                len_signal=len(signal)
                streched_main_signal0, start_off0, end_off0,bewalking = start_end_ind_strechedmain(st_ed, offset, list_behaviors[0], len_signal)
                streched_main_signal1, start_off1, end_off1,begrooming = start_end_ind_strechedmain(st_ed, offset, list_behaviors[1], len_signal)
                streched_main_signal2, start_off2, end_off2,berearings = start_end_ind_strechedmain(st_ed, offset, list_behaviors[2], len_signal)
                streched_main_signal3, start_off3, end_off3,beexp_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[3], len_signal)
                streched_main_signal4, start_off4, end_off4,beexp_non_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[4], len_signal)
                streched_main_bk, start_offbk, end_offbk,bebk = start_end_ind_strechedmain(st_ed, offset, list_behaviors[5], len_signal)


                streched_win_bk.append(streched_main_bk)
                start_bk.append(start_offbk)
                end_bk.append(end_offbk)

                streched_win_behave0.append(streched_main_signal0)
                start_offs_behave0.append(start_off0)
                end_offs_behave0.append(end_off0)

                streched_win_behave1.append(streched_main_signal1)
                start_offs_behave1.append(start_off1)
                end_offs_behave1.append(end_off1)

                streched_win_behave2.append(streched_main_signal2)
                start_offs_behave2.append(start_off2)
                end_offs_behave2.append(end_off2)

                streched_win_behave3.append(streched_main_signal3)
                start_offs_behave3.append(start_off3)
                end_offs_behave3.append(end_off3)

                streched_win_behave4.append(streched_main_signal4)
                start_offs_behave4.append(start_off4)
                end_offs_behave4.append(end_off4)

                # print(st_ed[y])
                # print("signal",signal, len(signal))
                # print("behavior_main",behavior_main, len(behavior_main))
                # plt.plot(signal)
                # plt.plot(bewalking,color='blue')
                # plt.plot(begrooming,color='green')
                # plt.plot(berearings,color='yellow')
                # plt.plot(beexp_statobj,color='orange')
                # plt.plot(beexp_non_statobj,color='gray')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len_signal - offset, color='r', linestyle='--')
                # plt.xlim([0, len_signal])
                # plt.show()

                # plt.plot(bebk, color='black')
                # plt.xlim([0, len_signal])
                # plt.show()
                ### start and end of behavior for NaN detection
                behavior_win_start,behavior_win_end=extract_st_ends_behaviors(behavior_main,offset,len_signal)



                ## speed signal
                speed_start_off = temp_speed[:offset]
                speed_end_off = temp_speed[len_signal - offset:len_signal]
                speed_main_part = temp_speed[offset:len_signal - offset]
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
                speed_streched_win.append(speed_new_signal)
                speed_start_offs.append(speed_start_off)
                speed_end_offs.append(speed_end_off)

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
                streched_win.append(new_signal)
                start_offs.append(behavior_win_start * start_off)
                end_offs.append(behavior_win_end * end_off)
                # test_merged = np.concatenate((behavior_win_start * start_off, main_part, behavior_win_end * end_off))
                # plt.plot(test_merged)
                # plt.show()




            average_signal = np.nanmean(streched_win, axis=0)
            average_starts = np.nanmean(start_offs, axis=0)
            average_ends = np.nanmean(end_offs, axis=0)

            speed_average_signal = np.nanmean(speed_streched_win, axis=0)
            speed_average_starts = np.nanmean(speed_start_offs, axis=0)
            speed_average_ends = np.nanmean(speed_end_offs, axis=0)

            streched_win_behave0_sum = np.nansum(streched_win_behave0, axis=0)
            start_offs_behave0_sum = np.nansum(start_offs_behave0, axis=0)
            end_offs_behave0_sum = np.nansum(end_offs_behave0, axis=0)

            streched_win_behave1_sum = np.nansum(streched_win_behave1, axis=0)
            start_offs_behave1_sum = np.nansum(start_offs_behave1, axis=0)
            end_offs_behave1_sum = np.nansum(end_offs_behave1, axis=0)

            streched_win_behave2_sum = np.nansum(streched_win_behave2, axis=0)
            start_offs_behave2_sum = np.nansum(start_offs_behave2, axis=0)
            end_offs_behave2_sum = np.nansum(end_offs_behave2, axis=0)

            streched_win_behave3_sum = np.nansum(streched_win_behave3, axis=0)
            start_offs_behave3_sum = np.nansum(start_offs_behave3, axis=0)
            end_offs_behave3_sum = np.nansum(end_offs_behave3, axis=0)

            streched_win_behave4_sum = np.nansum(streched_win_behave4, axis=0)
            start_offs_behave4_sum = np.nansum(start_offs_behave4, axis=0)
            end_offs_behave4_sum = np.nansum(end_offs_behave4, axis=0)

            streched_win_bk_sum = np.nansum(streched_win_bk, axis=0)
            start_offs_bk_sum = np.nansum(start_bk, axis=0)
            end_offs_bk_sum = np.nansum(end_bk, axis=0)


            if not np.any(np.isnan(average_signal)):
                average_signal=np.mean(average_signal.reshape(-1, 20), axis=1)
                all_main.append(average_signal)
                all_sts.append(average_starts)
                all_eds.append(average_ends)

                speed_average_signal = np.mean(speed_average_signal.reshape(-1, 20), axis=1)
                speed_all_main.append(speed_average_signal)
                speed_all_sts.append(speed_average_starts)
                speed_all_eds.append(speed_average_ends)

                # merged = np.concatenate((average_starts, average_signal, average_ends))
                # plt.text(max_window, 0, f'# = {total_num_behavior}')
                # plt.text(max_window, -0.2, f'#Merging = {total_num_behavior_afterMerging}')
                # plt.text(max_window, -0.4, f'dis(s) = {min_distance / sr_analysis}')
                # plt.text(max_window, -0.6, f'ofst(s) = {offset / sr_analysis}')
                # plt.grid(True)
                # plt.plot(merged, label='Average Signal')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(average_signal) + offset, color='r', linestyle='--')
                # plt.show()

                streched_win_behave0_sum = np.mean(streched_win_behave0_sum.reshape(-1, 20), axis=1)
                all_main0.append(streched_win_behave0_sum)
                all_sts0.append(start_offs_behave0_sum)
                all_eds0.append(end_offs_behave0_sum)

                streched_win_behave1_sum = np.mean(streched_win_behave1_sum.reshape(-1, 20), axis=1)
                all_main1.append(streched_win_behave1_sum)
                all_sts1.append(start_offs_behave1_sum)
                all_eds1.append(end_offs_behave1_sum)

                streched_win_behave2_sum = np.mean(streched_win_behave2_sum.reshape(-1, 20), axis=1)
                all_main2.append(streched_win_behave2_sum)
                all_sts2.append(start_offs_behave2_sum)
                all_eds2.append(end_offs_behave2_sum)

                streched_win_behave3_sum = np.mean(streched_win_behave3_sum.reshape(-1, 20), axis=1)
                all_main3.append(streched_win_behave3_sum)
                all_sts3.append(start_offs_behave3_sum)
                all_eds3.append(end_offs_behave3_sum)

                streched_win_behave4_sum = np.mean(streched_win_behave4_sum.reshape(-1, 20), axis=1)
                all_main4.append(streched_win_behave4_sum)
                all_sts4.append(start_offs_behave4_sum)
                all_eds4.append(end_offs_behave4_sum)

                streched_win_bk_sum = np.mean(streched_win_bk_sum.reshape(-1, 20), axis=1)
                all_mainbk.append(streched_win_bk_sum)
                all_stsbk.append(start_offs_bk_sum)
                all_edsbk.append(end_offs_bk_sum)


            ## go to next expriment file
                total_num_beahviors += len(st_ed)
        i = i + 1

    print("total_num_beahviors", total_num_beahviors)  #
    print("calculated starts", total_calculated_st)
    print("calculated ends", total_calculated_end)

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)

    speed_all_main = np.array(speed_all_main)
    speed_all_sts = np.array(speed_all_sts)
    speed_all_eds = np.array(speed_all_eds)


    final_sig_behavior0 = look_final_behavior(all_main0, all_sts0, all_eds0)
    final_sig_behavior1 = look_final_behavior(all_main1, all_sts1, all_eds1)
    final_sig_behavior2 = look_final_behavior(all_main2, all_sts2, all_eds2)
    final_sig_behavior3 = look_final_behavior(all_main3, all_sts3, all_eds3)
    final_sig_behavior4 = look_final_behavior(all_main4, all_sts4, all_eds4)
    final_sig_bk = look_final_behavior(all_mainbk, all_stsbk, all_edsbk)


    plt.plot(final_sig_behavior0, color='blue')
    plt.plot(final_sig_behavior1, color='green')
    plt.plot(final_sig_behavior2, color='yellow')
    plt.plot(final_sig_behavior3, color='orange')
    plt.plot(final_sig_behavior4, color='gray')
    plt.plot(final_sig_bk, color='black')
    #
    plt.show()


    # ##main
    all_main_avg = np.mean(all_main, axis=0)
    std_main = np.std(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.mean(all_sts, axis=0)
    std_st = np.std(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.mean(all_eds, axis=0)
    std_ed = np.std(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    ## speed signal :
    all_main_avg_speed = np.mean(speed_all_main, axis=0)
    std_main_speed = np.std(speed_all_main, axis=0)  # Standard deviation
    n = speed_all_main.shape[0]  # Number of observations
    all_main_sem_speed = std_main_speed / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg_speed = np.mean(speed_all_sts, axis=0)
    std_st_speed = np.std(speed_all_sts, axis=0)  # Standard deviation
    n = speed_all_sts.shape[0]  # Number of observations
    all_st_sem_speed = std_st_speed / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg_speed = np.mean(speed_all_eds, axis=0)
    std_ed_speed = np.std(speed_all_eds, axis=0)  # Standard deviation
    n = speed_all_eds.shape[0]  # Number of observations
    all_ed_sem_speed = std_ed_speed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged_speed = np.concatenate((all_sts_avg_speed, all_main_avg_speed, all_eds_avg_speed))
    merge_sem_speed = np.concatenate((all_st_sem_speed, all_main_sem_speed, all_ed_sem_speed))


    # Smoothing function using a moving average
    def smooth_it(signal, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal


    # # merged=np.array(smooth_signal(merged[0:150],0.5,sr_analysis))
    # # merge_sem=np.array(smooth_signal(merge_sem[0:150],0.5,sr_analysis))
    # plt.plot(merged, label='Average dfof all expriments for '+str(name_behavior))
    # # Handle NaN values by interpolation
    # # merged_cleaned = interpolate_nans(merged)
    # # # Smoothed signal
    # # smoothed_merged = np.concatenate((smooth_it(all_sts_avg, 30),smooth_it(all_main_avg, 30) , smooth_it(all_eds_avg, 30) ))
    # # plt.plot(smoothed_merged)
    # plt.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='lightgray')
    # plt.axvline(x=offset, color='r', linestyle='--')
    # plt.axvline(x=len(all_main_avg) + offset, color='r', linestyle='--')
    # # plt.text(len(all_main_avg), -0.2, f'min_win_size = {min_win_size}')
    # # plt.text(len(all_main_avg), -0.3, f'dis(s) = {min_distance / sr_analysis}')
    # # plt.text(len(all_main_avg), -0.4, f'ofst(s) = {offset / sr_analysis}')
    # # plt.ylim([-1 , 1])
    # plt.xlim([0, len(merged)])
    # plt.legend()
    # path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/'
    # # plt.savefig(path+str(name_behavior)+".svg",format="svg")
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the first signal on the left y-axis
    ax1.set_ylim(4, 16)
    ax1.plot(merged_speed, label='Average speed', linewidth=3, color='#87898A')
    ax1.fill_between(np.arange(len(merged_speed)), merged_speed - merge_sem_speed, merged_speed + merge_sem_speed,
                     color='#D3D3D3', alpha=0.5)
    ax1.set_ylabel('avg(Speed)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Add vertical lines
    ax1.axvline(x=offset, color='b', linestyle='--')
    ax1.axvline(x=len(merged) - offset, color='b', linestyle='--')
    ax1.set_xlim([0, len(merged)])

    # Create a twin axis for the second signal on the right y-axis
    ax2 = ax1.twinx()
    # Plot the second signal on the right y-axis

    ax2.set_ylim(-0.25, 0.25)
    ax2.plot(merged, label='Average dfof', linewidth=3, color='#38843F')
    ax2.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#90CB82', alpha=0.5)
    ax2.set_ylabel('avg(zscore(dfof))', color='#0F8140')
    ax2.tick_params(axis='y', labelcolor='#0F8140')

    fig.tight_layout()  # Adjust layout so labels fit nicely
    plt.show()


    # Stack the signals into a 2D array (each row is a signal)
    all_signals = np.vstack(
        [final_sig_behavior0, final_sig_behavior1, final_sig_behavior2, final_sig_behavior3, final_sig_behavior4,final_sig_bk])

    # Calculate the sum of all signals at each time point
    sum_signals = np.nansum(all_signals, axis=0)
    # background_signal=np.full(len(sum_signals), total_num_beahviors)-sum_signals
    # print("total_num_beahviors", max(final_sig_behavior0))  #
    print("max sum", max(sum_signals))

    # Calculate the percentage for each signal at each time point
    percentage_behavior0 = (final_sig_behavior0 / sum_signals) * 100
    percentage_behavior1 = (final_sig_behavior1 / sum_signals) * 100
    percentage_behavior2 = (final_sig_behavior2 / sum_signals) * 100
    percentage_behavior3 = (final_sig_behavior3 / sum_signals) * 100
    percentage_behavior4 = (final_sig_behavior4 / sum_signals) * 100
    percentage_BK = (final_sig_bk / sum_signals) * 100

    # Create a list of x-values (time points)
    time_points = np.arange(len(final_sig_behavior0))
    bar_width = 0.8  # You can adjust this as needed
    # Plot the stacked bar chart
    plt.bar(time_points, percentage_behavior0, color='blue',alpha=0.6, label='walking',width=bar_width)
    plt.bar(time_points, percentage_behavior1, bottom=percentage_behavior0, color='green',alpha=0.6, label='grooming',width=bar_width)
    plt.bar(time_points, percentage_behavior2, bottom=percentage_behavior0 + percentage_behavior1, color='yellow',alpha=0.6,
            label='rearing',width=bar_width)
    plt.bar(time_points, percentage_behavior3,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2, color='orange',alpha=0.6,
            label='exp_stat_obj',width=bar_width)
    plt.bar(time_points, percentage_behavior4,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2 + percentage_behavior3,
            color='gray',alpha=0.6, label='exp_nonstat_obj',width=bar_width)

    plt.bar(time_points, percentage_BK,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2 + percentage_behavior3 + percentage_behavior4,
            color='black', alpha=0.6, label='background', width=bar_width)


    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Percentage')
    plt.title('Percentage Contribution of Each Behavior at Each Time Point')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


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
    bar_width = 0.8  # You can adjust this as needed

    plt.figure(figsize=(10, 6))
    plt.bar(x_sts, all_sts_avg_binned, color='blue', alpha=0.6, label='Section 1', width=bar_width)
    plt.bar(x_main, all_main_avg_binned, color='red', alpha=0.6, label='Section 2', width=bar_width)
    plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.6, label='Section 3', width=bar_width)

    # Add vertical dashed lines for discontinuity
    plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='black', linestyle='--', linewidth=1.5)
    plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='black', linestyle='--',
                linewidth=1.5)
    plt.ylim([-0.8 , 0.8])
    # plt.xlim([0, len(merged)])
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Z score')
    plt.title('Bar Plot of Averaged Data (10 points per bar)')
    plt.legend()
    # plt.savefig(path + str(name_behavior) + "_bars.svg", format="svg")
    # Show the plot
    plt.show()

def test_strech_time_otherBehaviors_only_4():
    sr_analysis = 30
    # directory_path = 'E:/lab/Cholinergic Prj/Data/'+str(sr_analysis)+'/'
    # path = 'C:/Users/ffarokhi/Desktop/updateresult/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/teokalmans/'
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path, folder))]
    i = 0
    all_exp = []
    all_main = []
    all_sts = []
    all_eds = []

    speed_all_main = []
    speed_all_sts = []
    speed_all_eds = []

    total_calculated_end = 0
    total_calculated_st = 0
    total_num_beahviors = 0
    streched_win_behave = []
    start_offs_behave = []
    end_offs_behave = []

    all_main0 = []
    all_sts0 = []
    all_eds0 = []

    all_main1 = []
    all_sts1 = []
    all_eds1 = []

    all_main2 = []
    all_sts2 = []
    all_eds2 = []

    all_main3 = []
    all_sts3 = []
    all_eds3 = []

    all_main4 = []
    all_sts4 = []
    all_eds4 = []

    all_mainbk = []
    all_stsbk = []
    all_edsbk = []

    total_analyzed = 0

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

            # data2 = pd.read_csv(labels_file).values
            # data2 = pd.DataFrame(data2)
            # all_rearings = (np.array(data2[3][onset:analysis_win]) | np.array(data2[4][onset:analysis_win]))

            ## calculate log speed
            log_speed = np.log2(speed + 0.01)
            if np.isnan(log_speed).any() or np.isinf(log_speed).any():
                df_log = pd.DataFrame({'log_speed': log_speed})
                log_speed = df_log.interpolate(method='linear')['log_speed'].values

            ########## GLM
            X1 = pd.DataFrame()
            X1['exp_non_statobj'] = exp_non_statobj
            X1['exp_statobj'] = exp_statobj
            X1['speed'] = log_speed
            X1['rearing'] = rearings  # all_rearings
            X1['groomig'] = groomings
            # interactions
            # X1['speed_rearing_interaction'] = X1['speed'] * X1['rearing']
            # X1['speed_grooming_interaction'] = X1['speed'] * X1['groomig']
            # X1['speed_non_statobj_interaction'] = X1['speed'] * X1['exp_non_statobj']
            # X1['speed_statobj_interaction'] = X1['speed'] * X1['exp_statobj']
            # X1['nonobj_rearing'] = X1['rearing'] * X1['exp_non_statobj']
            # X1['obj_rearing'] = X1['rearing'] * X1['exp_statobj']
            X1 = sm.add_constant(X1)
            model2 = sm.GLM(dfof, X1, family=sm.families.Gaussian())
            result2 = model2.fit()
            result_df2 = pd.DataFrame(result2.summary().tables[1].data)
            speed_coef = float(result_df2.iloc[4:5, 1:2].values)
            groomings_coef = float(result_df2.iloc[6:7, 1:2].values)
            rearings_coef = float(result_df2.iloc[5:6, 1:2].values)
            exp_non_statobj_coef = float(result_df2.iloc[2:3, 1:2].values)
            exp_statobj_coef = float(result_df2.iloc[3:4, 1:2].values)
            constant_coef = float(result_df2.iloc[1:2, 1:2].values)
            # print(result_df2)
            # print(speed_coef,groomings_coef,rearings_coef,exp_non_statobj_coef,exp_statobj_coef,constant_coef)
            ###########################

            behavior_walking = np.array(walking)
            behavior_grooming = np.array(groomings)
            behavior_rearings = np.array(rearings)
            behavior_exp_statobj = np.array(exp_statobj)
            behavior_exp_non_statobj = np.array(exp_non_statobj)

            list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,
                              behavior_exp_non_statobj]
            behavior_bk = (np.logical_or.reduce([behavior_walking,
                                                 behavior_grooming,
                                                 behavior_rearings,
                                                 behavior_exp_statobj,
                                                 behavior_exp_non_statobj]))

            # Perform logical NOT on the OR result
            behavior_bk = np.logical_not(behavior_bk)
            list_behaviors.append(behavior_bk)

            name_behavior = "behavior_walking"
            behavior_sig = behavior_walking
            # speed=np.array(log_speed)
            dfof1 = np.array(dfof)

            min_distance = int(0 * sr_analysis)
            offset = int(5 * sr_analysis)
            min_win_size = int(2 * sr_analysis)
            # max_win_size= int(10*sr_analysis)
            min_dis_toShow = int(4 * sr_analysis)

            # dfof=dfof1
            dfof = ((((dfof1
                       - speed_coef * log_speed - groomings_coef * behavior_grooming - rearings_coef * behavior_rearings)
                      - exp_statobj_coef * behavior_exp_statobj)
                     - exp_non_statobj_coef * behavior_exp_non_statobj)
                    - constant_coef)

            ########################### streching time algorithm

            #### find start and endpoints
            starts, ends = find_startANDendpoints(behavior_sig)
            total_num_behavior=len(starts)

            #### merge start and end points
            st_ed=merge_startANDend_points(starts, ends, min_distance)


            # # remove winsdows with size < min_win_size
            # st_ed=remove_small_win(st_ed, min_win_size)
            # print("starts_ends after removing small windows", len(st_ed), st_ed)

            # plt.plot(behavior_sig)
            # plt.plot(dfof1)
            # for ppl in range(0, len(st_ed)):
            #     plt.axvline(x=(st_ed[ppl][0]), color='r', linestyle='--')
            #     plt.axvline(x=(st_ed[ppl][1]), color='green', linestyle='--')
            # plt.title("after removing small win size = "+ str(min_win_size/sr_analysis))
            # plt.show()


            ## find the max window size
            max_window = 0
            for t in range(0,len(st_ed)):
                win_d=st_ed[t][1] - st_ed[t][0] + 1
                if win_d > max_window:
                    max_window=win_d
            # print("max_window",max_window)
            max_window=5000
            total_num_behavior_afterMerging=len(st_ed)
            streched_win = []
            start_offs = []
            end_offs = []

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

            def start_end_ind(st_ed, y, offset, dfof,behavior_sig,min_dis_toShow):
                start_idx = st_ed[y][0] - offset
                end_idx = st_ed[y][1] + offset + 1
                flag=0
                signal = []
                behavior_main=[]

                if  st_ed[y][1]- st_ed[y][0]> min_win_size:
                    if y<len(st_ed)-1:
                        if end_idx < len(dfof) and start_idx > 0 and (st_ed[y][0]-st_ed[y-1][1]) >= min_dis_toShow and (st_ed[y+1][0]-st_ed[y][1]) >= min_dis_toShow:
                            signal = zscore(dfof[start_idx : end_idx])
                            behavior_main = behavior_sig[start_idx : end_idx]
                            flag=1
                    else:
                        if end_idx < len(dfof) and (st_ed[y][0]-st_ed[y-1][1]) >= min_dis_toShow:
                            signal = zscore(dfof[start_idx: end_idx])
                            behavior_main = behavior_sig[start_idx: end_idx]
                            flag = 1


                return signal, flag , behavior_main


            def extract_st_ends_behaviors(behavior_main,offset,len_signal):
                win_start = (np.array(behavior_main[:offset]))
                behavior_win_start = np.where(win_start == 1, np.nan, win_start)
                behavior_win_start = np.where(behavior_win_start == 0, 1, behavior_win_start)

                win_end = (np.array(behavior_main[len_signal - offset: len_signal]))
                behavior_win_end = np.where(win_end == 1, np.nan, win_end)
                behavior_win_end = np.where(behavior_win_end == 0, 1, behavior_win_end)
                return behavior_win_start,behavior_win_end

            def start_end_ind_strechedmain(st_ed, offset, sig,len_signal):
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

            for y in range(0,len(st_ed)):  #len(st_ed)

                signal, flag , behavior_main =start_end_ind(st_ed,y, offset, dfof, behavior_sig,min_dis_toShow)
                total_calculated_st += flag
                if flag==1:

                    # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                    len_signal=len(signal)
                    streched_main_signal0, start_off0, end_off0,bewalking = start_end_ind_strechedmain(st_ed, offset, list_behaviors[0], len_signal)
                    streched_main_signal1, start_off1, end_off1,begrooming = start_end_ind_strechedmain(st_ed, offset, list_behaviors[1], len_signal)
                    streched_main_signal2, start_off2, end_off2,berearings = start_end_ind_strechedmain(st_ed, offset, list_behaviors[2], len_signal)
                    streched_main_signal3, start_off3, end_off3,beexp_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[3], len_signal)
                    streched_main_signal4, start_off4, end_off4,beexp_non_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[4], len_signal)
                    streched_main_bk, start_offbk, end_offbk,bebk = start_end_ind_strechedmain(st_ed, offset, list_behaviors[5], len_signal)


                    streched_win_bk.append(streched_main_bk)
                    start_bk.append(start_offbk)
                    end_bk.append(end_offbk)

                    streched_win_behave0.append(streched_main_signal0)
                    start_offs_behave0.append(start_off0)
                    end_offs_behave0.append(end_off0)

                    streched_win_behave1.append(streched_main_signal1)
                    start_offs_behave1.append(start_off1)
                    end_offs_behave1.append(end_off1)

                    streched_win_behave2.append(streched_main_signal2)
                    start_offs_behave2.append(start_off2)
                    end_offs_behave2.append(end_off2)

                    streched_win_behave3.append(streched_main_signal3)
                    start_offs_behave3.append(start_off3)
                    end_offs_behave3.append(end_off3)

                    streched_win_behave4.append(streched_main_signal4)
                    start_offs_behave4.append(start_off4)
                    end_offs_behave4.append(end_off4)

                    # print(st_ed[y])
                    # print("signal",signal, len(signal))
                    # print("behavior_main",behavior_main, len(behavior_main))
                    # plt.plot(signal)
                    # plt.plot(bewalking,color='blue')
                    # # plt.plot(begrooming,color='green')
                    # # plt.plot(berearings,color='yellow')
                    # # plt.plot(beexp_statobj,color='orange')
                    # # plt.plot(beexp_non_statobj,color='gray')
                    # plt.axvline(x=offset, color='r', linestyle='--')
                    # plt.axvline(x=len_signal - offset, color='r', linestyle='--')
                    # plt.xlim([0, len_signal])
                    # plt.show()

                    # plt.plot(bebk, color='black')
                    # plt.xlim([0, len_signal])
                    # plt.show()
                    ### start and end of behavior for NaN detection
                    behavior_win_start,behavior_win_end=extract_st_ends_behaviors(behavior_main,offset,len_signal)

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
                    streched_win.append(new_signal)
                    start_offs.append(behavior_win_start * start_off)
                    end_offs.append(behavior_win_end * end_off)
                    # test_merged = np.concatenate((behavior_win_start * start_off, main_part, behavior_win_end * end_off))
                    # plt.plot(test_merged)
                    # plt.show()


            average_signal = np.nanmean(streched_win, axis=0)
            average_starts = np.nanmean(start_offs, axis=0)
            average_ends = np.nanmean(end_offs, axis=0)


            streched_win_behave0_sum = np.nansum(streched_win_behave0, axis=0)
            start_offs_behave0_sum = np.nansum(start_offs_behave0, axis=0)
            end_offs_behave0_sum = np.nansum(end_offs_behave0, axis=0)

            streched_win_behave1_sum = np.nansum(streched_win_behave1, axis=0)
            start_offs_behave1_sum = np.nansum(start_offs_behave1, axis=0)
            end_offs_behave1_sum = np.nansum(end_offs_behave1, axis=0)

            streched_win_behave2_sum = np.nansum(streched_win_behave2, axis=0)
            start_offs_behave2_sum = np.nansum(start_offs_behave2, axis=0)
            end_offs_behave2_sum = np.nansum(end_offs_behave2, axis=0)

            streched_win_behave3_sum = np.nansum(streched_win_behave3, axis=0)
            start_offs_behave3_sum = np.nansum(start_offs_behave3, axis=0)
            end_offs_behave3_sum = np.nansum(end_offs_behave3, axis=0)

            streched_win_behave4_sum = np.nansum(streched_win_behave4, axis=0)
            start_offs_behave4_sum = np.nansum(start_offs_behave4, axis=0)
            end_offs_behave4_sum = np.nansum(end_offs_behave4, axis=0)

            streched_win_bk_sum = np.nansum(streched_win_bk, axis=0)
            start_offs_bk_sum = np.nansum(start_bk, axis=0)
            end_offs_bk_sum = np.nansum(end_bk, axis=0)


            if not np.any(np.isnan(average_signal)):
                average_signal=np.mean(average_signal.reshape(-1, 20), axis=1)
                all_main.append(average_signal)
                all_sts.append(average_starts)
                all_eds.append(average_ends)
                # merged = np.concatenate((average_starts, average_signal, average_ends))
                # plt.text(max_window, 0, f'# = {total_num_behavior}')
                # plt.text(max_window, -0.2, f'#Merging = {total_num_behavior_afterMerging}')
                # plt.text(max_window, -0.4, f'dis(s) = {min_distance / sr_analysis}')
                # plt.text(max_window, -0.6, f'ofst(s) = {offset / sr_analysis}')
                # plt.grid(True)
                # plt.plot(merged, label='Average Signal')
                # plt.axvline(x=offset, color='r', linestyle='--')
                # plt.axvline(x=len(average_signal) + offset, color='r', linestyle='--')
                # plt.show()

                streched_win_behave0_sum = np.mean(streched_win_behave0_sum.reshape(-1, 20), axis=1)
                all_main0.append(streched_win_behave0_sum)
                all_sts0.append(start_offs_behave0_sum)
                all_eds0.append(end_offs_behave0_sum)

                streched_win_behave1_sum = np.mean(streched_win_behave1_sum.reshape(-1, 20), axis=1)
                all_main1.append(streched_win_behave1_sum)
                all_sts1.append(start_offs_behave1_sum)
                all_eds1.append(end_offs_behave1_sum)

                streched_win_behave2_sum = np.mean(streched_win_behave2_sum.reshape(-1, 20), axis=1)
                all_main2.append(streched_win_behave2_sum)
                all_sts2.append(start_offs_behave2_sum)
                all_eds2.append(end_offs_behave2_sum)

                streched_win_behave3_sum = np.mean(streched_win_behave3_sum.reshape(-1, 20), axis=1)
                all_main3.append(streched_win_behave3_sum)
                all_sts3.append(start_offs_behave3_sum)
                all_eds3.append(end_offs_behave3_sum)

                streched_win_behave4_sum = np.mean(streched_win_behave4_sum.reshape(-1, 20), axis=1)
                all_main4.append(streched_win_behave4_sum)
                all_sts4.append(start_offs_behave4_sum)
                all_eds4.append(end_offs_behave4_sum)

                streched_win_bk_sum = np.mean(streched_win_bk_sum.reshape(-1, 20), axis=1)
                all_mainbk.append(streched_win_bk_sum)
                all_stsbk.append(start_offs_bk_sum)
                all_edsbk.append(end_offs_bk_sum)


            ## go to next expriment file
                total_num_beahviors += len(st_ed)
        i = i + 1

    print("total_num_beahviors", total_num_beahviors)  #
    print("calculated starts", total_calculated_st)
    print("calculated ends", total_calculated_end)

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)


    final_sig_behavior0 = look_final_behavior(all_main0, all_sts0, all_eds0)
    final_sig_behavior1 = look_final_behavior(all_main1, all_sts1, all_eds1)
    final_sig_behavior2 = look_final_behavior(all_main2, all_sts2, all_eds2)
    final_sig_behavior3 = look_final_behavior(all_main3, all_sts3, all_eds3)
    final_sig_behavior4 = look_final_behavior(all_main4, all_sts4, all_eds4)
    final_sig_bk = look_final_behavior(all_mainbk, all_stsbk, all_edsbk)


    plt.plot(final_sig_behavior0, color='blue')
    plt.plot(final_sig_behavior1, color='green')
    plt.plot(final_sig_behavior2, color='yellow')
    plt.plot(final_sig_behavior3, color='orange')
    plt.plot(final_sig_behavior4, color='gray')
    plt.plot(final_sig_bk, color='black')
    #
    plt.show()


    # ##main
    all_main_avg = np.mean(all_main, axis=0)
    std_main = np.std(all_main, axis=0)  # Standard deviation
    n = all_main.shape[0]  # Number of observations
    all_main_sem = std_main / np.sqrt(n)  # Standard error of the mean

    # ## start
    all_sts_avg = np.mean(all_sts, axis=0)
    std_st = np.std(all_sts, axis=0)  # Standard deviation
    n = all_sts.shape[0]  # Number of observations
    all_st_sem = std_st / np.sqrt(n)  # Standard error of the mean

    # ## end
    all_eds_avg = np.mean(all_eds, axis=0)
    std_ed = np.std(all_eds, axis=0)  # Standard deviation
    n = all_eds.shape[0]  # Number of observations
    all_ed_sem = std_ed / np.sqrt(n)  # Standard error of the mean

    ### merge
    merged = np.concatenate((all_sts_avg, all_main_avg, all_eds_avg))
    merge_sem = np.concatenate((all_st_sem, all_main_sem, all_ed_sem))

    # Smoothing function using a moving average
    def smooth_it(signal, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(signal, window, mode='same')

    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal


    # merged=np.array(smooth_signal(merged[0:150],0.5,sr_analysis))
    # merge_sem=np.array(smooth_signal(merge_sem[0:150],0.5,sr_analysis))
    plt.plot(merged, label='Average dfof ', linewidth=3, color='#38843F') #all expriments for '+str(name_behavior)
    # Handle NaN values by interpolation
    # merged_cleaned = interpolate_nans(merged)
    # # Smoothed signal
    # smoothed_merged = np.concatenate((smooth_it(all_sts_avg, 30),smooth_it(all_main_avg, 30) , smooth_it(all_eds_avg, 30) ))
    # plt.plot(smoothed_merged)
    plt.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#D8ECDA')
    plt.axvline(x=offset, color='b', linestyle='--')
    plt.axvline(x=len(all_main_avg) + offset, color='b', linestyle='--')
    # plt.text(len(all_main_avg), -0.2, f'min_win_size = {min_win_size}')
    # plt.text(len(all_main_avg), -0.3, f'dis(s) = {min_distance / sr_analysis}')
    # plt.text(len(all_main_avg), -0.4, f'ofst(s) = {offset / sr_analysis}')
    plt.ylim([-1 , 1])
    plt.xlim([0, len(merged)])
    # plt.legend()
    path = 'C:/Users/ffarokhi/Desktop/paper/4)onset_offset/updated_9-24-2024/'
    plt.show()

    # Stack the signals into a 2D array (each row is a signal)
    all_signals = np.vstack(
        [final_sig_behavior0, final_sig_behavior1, final_sig_behavior2, final_sig_behavior3, final_sig_behavior4,final_sig_bk])

    # Calculate the sum of all signals at each time point
    sum_signals = np.nansum(all_signals, axis=0)
    # background_signal=np.full(len(sum_signals), total_num_beahviors)-sum_signals
    # print("total_num_beahviors", max(final_sig_behavior0))  #
    print("max sum", max(sum_signals))

    # Calculate the percentage for each signal at each time point
    percentage_behavior0 = (final_sig_behavior0 / sum_signals) * 100
    percentage_behavior1 = (final_sig_behavior1 / sum_signals) * 100
    percentage_behavior2 = (final_sig_behavior2 / sum_signals) * 100
    percentage_behavior3 = (final_sig_behavior3 / sum_signals) * 100
    percentage_behavior4 = (final_sig_behavior4 / sum_signals) * 100
    percentage_BK = (final_sig_bk / sum_signals) * 100

    # Create a list of x-values (time points)
    time_points = np.arange(len(final_sig_behavior0))
    bar_width = 0.8  # You can adjust this as needed
    # Plot the stacked bar chart
    plt.bar(time_points, percentage_behavior0, color='blue',alpha=0.6, label='walking',width=bar_width)
    plt.bar(time_points, percentage_behavior1, bottom=percentage_behavior0, color='green',alpha=0.6, label='grooming',width=bar_width)
    plt.bar(time_points, percentage_behavior2, bottom=percentage_behavior0 + percentage_behavior1, color='yellow',alpha=0.6,
            label='rearing',width=bar_width)
    plt.bar(time_points, percentage_behavior3,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2, color='orange',alpha=0.6,
            label='exp_stat_obj',width=bar_width)
    plt.bar(time_points, percentage_behavior4,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2 + percentage_behavior3,
            color='gray',alpha=0.6, label='exp_nonstat_obj',width=bar_width)
    plt.bar(time_points, percentage_BK,
            bottom=percentage_behavior0 + percentage_behavior1 + percentage_behavior2 + percentage_behavior3 + percentage_behavior4,
            color='black', alpha=0.6, label='background', width=bar_width)


    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Percentage')
    plt.title('Percentage Contribution of Each Behavior at Each Time Point')

    # Add a legend
    # plt.legend(loc='upper right')

    # Show the plot
    plt.show()


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
    bar_width = 0.8  # You can adjust this as needed

    plt.figure(figsize=(10, 6))
    plt.bar(x_sts, all_sts_avg_binned, color='green', alpha=0.8, label='Section 1', width=bar_width)
    plt.bar(x_main, all_main_avg_binned, color='green', alpha=0.8, label='Section 2', width=bar_width)
    plt.bar(x_eds, all_eds_avg_binned, color='green', alpha=0.8, label='Section 3', width=bar_width)

    # Add vertical dashed lines for discontinuity
    plt.axvline(x=len(all_sts_avg_binned) - 0.5, color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(x=len(all_sts_avg_binned) + len(all_main_avg_binned) - 0.5, color='blue', linestyle='--',
                linewidth=1.5)
    plt.ylim([-0.8 , 0.8])
    # plt.xlim([0, len(merged)])
    # Add labels, title, and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Z score')
    plt.title('Bar Plot of Averaged Data (10 points per bar)')
    # plt.legend()
    # plt.savefig(path + str(name_behavior) + "_bars.svg", format="svg")
    # Show the plot
    plt.savefig("E:/lab/published/BMES2024/" + "nonstat" + expriment_name + ".svg", format="svg")
    plt.show()

def double_exponential_func(t, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * t) + a2 * np.exp(b2 * t) + c

def fit_exp_double(sig, fps_video):
    # Fit the exponential model to the data
    time_values = np.linspace(0, len(sig) - 1, len(sig))
    popt, pcov = curve_fit(double_exponential_func, time_values, sig, p0=[1, -0.5, 0.5, -0.3, 0.1],
                           bounds=([0, -np.inf, 0, -np.inf, -np.inf], [np.inf, 0, np.inf, 0, np.inf]),
                           maxfev=10000)
    a1, b1, a2, b2, c = popt
    fitted_signal = double_exponential_func(time_values, a1, b1, a2, b2, c)
    # Print the optimized parameters
    print(f"Fitted parameters: a1 = {a1}, b1 = {b1}, a2 = {a2}, b2= {b2}, c= {c}")
    tau1 = (-1 / b1)/ fps_video
    tau2 = (-1 / b2) / fps_video
    print(f"Time constant (tau): {tau1}, {tau2}")
    return fitted_signal,tau1, tau2
def show_speed_dfofdoubletau_decays(time_vec,dfof,smooth_time_window,input_speed,sr_analysis):

    smooth_speed = smooth_signal(input_speed, smooth_time_window, sr_analysis)
    fitted_speed, tau_speed = fit_exp(input_speed, sr_analysis)
    smoothed_dfof = smooth_signal(dfof, smooth_time_window, sr_analysis)
    fitted_dfof, tau1_dfof,tau2_dfof = fit_exp_double(dfof, sr_analysis)
    z_speed = stats.zscore(input_speed)
    z_dfof = stats.zscore(dfof)
    zfitted_speed, ztau_speed = fit_exp(z_speed, sr_analysis)
    zfitted_dfof, ztau1_dfof,ztau2_dfof = fit_exp_double(z_dfof, sr_analysis)


    f1, (s1, s2, s3) = plt.subplots(3, 1, figsize=(8, 10))

    # s1.plot(time_vec, input_speed, linewidth=1, color='k')
    s1.plot(time_vec, smooth_speed, linewidth=1, color='#666869') #, color=[0, 0.4470, 0.7410]
    s1.plot(time_vec, fitted_speed, color='gray', linewidth=5)
    s1.set_title(f'{smooth_time_window}-sec smoothed running speed')
    s1.set_ylabel('Running speed (cm/s)')
    s1.set_xticklabels([])
    s1.text(100, 4, f'τ = {tau_speed:.2f}' + " (s)", fontsize=16, color='black')
    s1.tick_params(direction='out')

    # s2.plot(time_vec, dfof, linewidth=1, color='k')
    s2.plot(time_vec, smoothed_dfof, linewidth=1 ,color='#38843F') # ,color=[0.8500, 0.3250, 0.0980]
    s2.plot(time_vec,fitted_dfof, color='Blue', linewidth=5)
    s2.set_title(f'{smooth_time_window}-sec smoothed cholinergic activity')
    s2.set_ylabel(r'$\Delta$'+'F/F')
    s2.set_xticklabels([])
    s2.text(100, 0.06, f'τ = {tau1_dfof:.2f}, {tau2_dfof:.2f}' + " (s)", fontsize=16, color='black')
    s2.tick_params(direction='out')

    s3.plot(time_vec, smooth_signal(z_speed, smooth_time_window, sr_analysis) , linewidth=1, color='#666869')
    s3.plot(time_vec, zfitted_speed, linewidth=4, color='black')
    s3.plot(time_vec, smooth_signal(z_dfof, smooth_time_window, sr_analysis)  , linewidth=1.1,color='#38843F')
    s3.plot(time_vec, zfitted_dfof, linewidth=4, color='orange')
    s3.text(100, -2, f'τ = {ztau1_dfof:.2f},{ztau2_dfof:.2f}' + " (s)", fontsize=16, color='orange')
    s3.text(600, -2, f'τ = {ztau_speed:.2f}' + " (s)", fontsize=16, color='black')
    s3.set_title('Superimposition')
    s3.set_xlabel('Time (s)')
    s3.set_ylabel('Z-score')
    s3.tick_params(direction='out')

    ####### corr log
    # Compute Pearson's R
    Rlog, _ = pearsonr(input_speed, dfof)
    dim = [.87, .1, .1, .1]
    str = f'R = {Rlog:.2f}'
    s3.text(dim[0], dim[1], str, transform=s3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    # Link x-axes of the subplots
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    return Rlog


def novel_env():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/novelty_decay/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/' # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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
            # Reshape the signals to fit the model

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


            # plt each single session
            # signals = [dfof[:win_plt], corrected_dfof[:win_plt], pdfof[:win_plt], speed[:win_plt]]
            # tau_speed,tau_dfof,tau_corrected_dfof,tau_predicted_dfof = show_speed_dfof_decays_signle(time[:win_plt], signals, smooth_time_window, sr_analysis)
            # # plt.savefig(path + "decays" + str(expriment_name) + ".svg", format="svg")
            # plt.show()
            # tau_speed_list.append(tau_speed)
            # tau_dfof_list.append(tau_dfof)
            # tau_corrected_dfof_list.append(tau_corrected_dfof)
            # tau_predicted_dfof_list.append(tau_predicted_dfof)

        i+=1

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
    # plt.savefig("E:/lab/published/BMES2024/" + "decay" + expriment_name + ".svg", format="svg")
    # Rlog = show_speed_dfof_decays(time[:win_plt], signals, sems, 1, sr_analysis)
    Rlog = show_speed_dfof_decays_seperate(time[:win_plt], signals, sems, 1, sr_analysis, path)
    # plt.savefig(path + "decays_avg.svg", format="svg")
    # plt.show()



def test_strech_time_otherBehaviors_only_4_with_speed():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/onset_offset/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
    sr_analysis = 30
    folder_names = [folder for folder in os.listdir(directory_path) if
                    os.path.isdir(os.path.join(directory_path,
                                               folder))]  # Folder containing different sessions of the experiment.
    i = 0
    all_exp=[]
    all_main=[]
    all_sts=[]
    all_eds=[]

    speed_all_main=[]
    speed_all_sts=[]
    speed_all_eds=[]

    pdfof_all_main=[]
    pdfof_all_sts=[]
    pdfof_all_eds=[]

    total_calculated_end=0
    total_calculated_st=0
    total_num_beahviors=0
    streched_win_behave=[]
    start_offs_behave=[]
    end_offs_behave=[]

    all_main0 = []
    all_sts0 = []
    all_eds0 = []

    all_main1 = []
    all_sts1 = []
    all_eds1 = []

    all_main2 = []
    all_sts2 = []
    all_eds2 = []

    all_main3 = []
    all_sts3 = []
    all_eds3 = []

    all_main4 = []
    all_sts4 = []
    all_eds4 = []

    all_mainbk = []
    all_stsbk = []
    all_edsbk = []

    total_analyzed=0

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

            ## find the max window size
            # max_window = 0
            # for t in range(0,len(st_ed)):
            #     win_d=st_ed[t][1] - st_ed[t][0] + 1
            #     if win_d > max_window:
            #         max_window=win_d
            # # print("max_window",max_window)
            max_window=5000


            streched_win = []
            start_offs = []
            end_offs = []

            speed_streched_win=[]
            speed_start_offs=[]
            speed_end_offs=[]

            pdfof_streched_win=[]
            pdfof_start_offs=[]
            pdfof_end_offs=[]

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
            for y in range(0,len(st_ed)):  #len(st_ed)

                signal, flag, behavior_main, temp_speed, p_dfof = start_end_ind(st_ed, y, offset, dfof, behavior_sig, min_dis_toShow)
                total_calculated_st += flag

                if flag == 1:
                    # list_behaviors = [behavior_walking, behavior_grooming, behavior_rearings, behavior_exp_statobj,behavior_exp_non_statobj]
                    len_signal=len(signal)
                    streched_main_signal0, start_off0, end_off0, bewalking = start_end_ind_strechedmain(st_ed, offset, list_behaviors[0], len_signal)
                    streched_main_signal1, start_off1, end_off1, begrooming = start_end_ind_strechedmain(st_ed, offset, list_behaviors[1], len_signal)
                    streched_main_signal2, start_off2, end_off2, berearings = start_end_ind_strechedmain(st_ed, offset, list_behaviors[2], len_signal)
                    streched_main_signal3, start_off3, end_off3, beexp_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[3], len_signal)
                    streched_main_signal4, start_off4, end_off4, beexp_non_statobj = start_end_ind_strechedmain(st_ed, offset, list_behaviors[4], len_signal)
                    streched_main_bk, start_offbk, end_offbk, bebk = start_end_ind_strechedmain(st_ed, offset, list_behaviors[5], len_signal)

                    streched_win_bk.append(streched_main_bk)
                    start_bk.append(start_offbk)
                    end_bk.append(end_offbk)

                    streched_win_behave0.append(streched_main_signal0)
                    start_offs_behave0.append(start_off0)
                    end_offs_behave0.append(end_off0)

                    streched_win_behave1.append(streched_main_signal1)
                    start_offs_behave1.append(start_off1)
                    end_offs_behave1.append(end_off1)

                    streched_win_behave2.append(streched_main_signal2)
                    start_offs_behave2.append(start_off2)
                    end_offs_behave2.append(end_off2)

                    streched_win_behave3.append(streched_main_signal3)
                    start_offs_behave3.append(start_off3)
                    end_offs_behave3.append(end_off3)

                    streched_win_behave4.append(streched_main_signal4)
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
                    streched_win.append(new_signal)
                    start_offs.append(behavior_win_start * start_off)
                    end_offs.append(behavior_win_end * end_off)

                    ## save for avg
                    speed_streched_win.append(speed_new_signal)
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

                    pdfof_streched_win.append(pdfof_new_signal)
                    pdfof_start_offs.append(behavior_win_start * pdfof_start_off)
                    pdfof_end_offs.append(behavior_win_end * pdfof_end_off)


            average_signal = np.nanmean(streched_win, axis=0)
            average_starts = np.nanmean(start_offs, axis=0)
            average_ends = np.nanmean(end_offs, axis=0)

            speed_average_signal = np.nanmean(speed_streched_win, axis=0)
            speed_average_starts = np.nanmean(speed_start_offs, axis=0)
            speed_average_ends = np.nanmean(speed_end_offs, axis=0)

            pdfof_average_signal = np.nanmean(pdfof_streched_win, axis=0)
            pdfof_average_starts = np.nanmean(pdfof_start_offs, axis=0)
            pdfof_average_ends = np.nanmean(pdfof_end_offs, axis=0)


            streched_win_behave0_sum = np.nansum(streched_win_behave0, axis=0)
            start_offs_behave0_sum = np.nansum(start_offs_behave0, axis=0)
            end_offs_behave0_sum = np.nansum(end_offs_behave0, axis=0)

            streched_win_behave1_sum = np.nansum(streched_win_behave1, axis=0)
            start_offs_behave1_sum = np.nansum(start_offs_behave1, axis=0)
            end_offs_behave1_sum = np.nansum(end_offs_behave1, axis=0)

            streched_win_behave2_sum = np.nansum(streched_win_behave2, axis=0)
            start_offs_behave2_sum = np.nansum(start_offs_behave2, axis=0)
            end_offs_behave2_sum = np.nansum(end_offs_behave2, axis=0)

            streched_win_behave3_sum = np.nansum(streched_win_behave3, axis=0)
            start_offs_behave3_sum = np.nansum(start_offs_behave3, axis=0)
            end_offs_behave3_sum = np.nansum(end_offs_behave3, axis=0)

            streched_win_behave4_sum = np.nansum(streched_win_behave4, axis=0)
            start_offs_behave4_sum = np.nansum(start_offs_behave4, axis=0)
            end_offs_behave4_sum = np.nansum(end_offs_behave4, axis=0)

            streched_win_bk_sum = np.nansum(streched_win_bk, axis=0)
            start_offs_bk_sum = np.nansum(start_bk, axis=0)
            end_offs_bk_sum = np.nansum(end_bk, axis=0)

            print("yes")
            if not np.any(np.isnan(average_signal)):
                print("yes")
                average_signal=np.nanmean(average_signal.reshape(-1, 20), axis=1)
                all_main.append(average_signal)
                all_sts.append(average_starts)
                all_eds.append(average_ends)

                speed_average_signal = np.nanmean(speed_average_signal.reshape(-1, 20), axis=1)
                speed_all_main.append(speed_average_signal)
                speed_all_sts.append(speed_average_starts)
                speed_all_eds.append(speed_average_ends)

                pdfof_average_signal = np.nanmean(pdfof_average_signal.reshape(-1, 20), axis=1)
                pdfof_all_main.append(pdfof_average_signal)
                pdfof_all_sts.append(pdfof_average_starts)
                pdfof_all_eds.append(pdfof_average_ends)

                streched_win_behave0_sum = np.nanmean(streched_win_behave0_sum.reshape(-1, 20), axis=1)
                all_main0.append(streched_win_behave0_sum)
                all_sts0.append(start_offs_behave0_sum)
                all_eds0.append(end_offs_behave0_sum)

                streched_win_behave1_sum = np.nanmean(streched_win_behave1_sum.reshape(-1, 20), axis=1)
                all_main1.append(streched_win_behave1_sum)
                all_sts1.append(start_offs_behave1_sum)
                all_eds1.append(end_offs_behave1_sum)

                streched_win_behave2_sum = np.nanmean(streched_win_behave2_sum.reshape(-1, 20), axis=1)
                all_main2.append(streched_win_behave2_sum)
                all_sts2.append(start_offs_behave2_sum)
                all_eds2.append(end_offs_behave2_sum)

                streched_win_behave3_sum = np.nanmean(streched_win_behave3_sum.reshape(-1, 20), axis=1)
                all_main3.append(streched_win_behave3_sum)
                all_sts3.append(start_offs_behave3_sum)
                all_eds3.append(end_offs_behave3_sum)

                streched_win_behave4_sum = np.nanmean(streched_win_behave4_sum.reshape(-1, 20), axis=1)
                all_main4.append(streched_win_behave4_sum)
                all_sts4.append(start_offs_behave4_sum)
                all_eds4.append(end_offs_behave4_sum)

                streched_win_bk_sum = np.nanmean(streched_win_bk_sum.reshape(-1, 20), axis=1)
                all_mainbk.append(streched_win_bk_sum)
                all_stsbk.append(start_offs_bk_sum)
                all_edsbk.append(end_offs_bk_sum)

            ## go to next expriment file
        i = i + 1

    print("total_num_beahviors", total_num_beahviors)  #
    print("calculated starts", total_calculated_st)
    print("calculated ends", total_calculated_end)

    ##main
    all_main=np.array(all_main)
    all_sts=np.array(all_sts)
    all_eds=np.array(all_eds)

    speed_all_main = np.array(speed_all_main)
    speed_all_sts = np.array(speed_all_sts)
    speed_all_eds = np.array(speed_all_eds)

    pdfof_all_main = np.array(pdfof_all_main)
    pdfof_all_sts = np.array(pdfof_all_sts)
    pdfof_all_eds = np.array(pdfof_all_eds)


    final_sig_behavior0 = look_final_behavior(all_main0, all_sts0, all_eds0)
    final_sig_behavior1 = look_final_behavior(all_main1, all_sts1, all_eds1)
    final_sig_behavior2 = look_final_behavior(all_main2, all_sts2, all_eds2)
    final_sig_behavior3 = look_final_behavior(all_main3, all_sts3, all_eds3)
    final_sig_behavior4 = look_final_behavior(all_main4, all_sts4, all_eds4)
    final_sig_bk = look_final_behavior(all_mainbk, all_stsbk, all_edsbk)


    plt.plot(final_sig_behavior0, color='#1f77b4', label='Walking')  # Blue
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
    plt.title("Behaviors")
    plt.xlabel("Time")
    plt.ylabel("Percentage of Each Behavior")
    plt.ylim(0, 110)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig(path + "behaviors_percentage" + name_behavior + ".svg", format="svg")

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


    # Function to interpolate NaN values in the signal
    def interpolate_nans(signal):
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        return signal

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the first signal on the left y-axis
    ax1.set_ylim(0, 14)
    ax1.plot(merged_speed, label='Average speed', linewidth=3, color='#87898A')
    ax1.fill_between(np.arange(len(merged_speed)), merged_speed - merge_sem_speed, merged_speed + merge_sem_speed,
                     color='#D3D3D3', alpha=0.5)
    ax1.set_ylabel('avg(Speed)', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # Add vertical lines
    ax1.axvline(x=offset, color='b', linestyle='--')
    ax1.axvline(x=len(merged) - offset, color='b', linestyle='--')
    ax1.set_xlim([0, len(merged)])

    # Create a twin axis for the second signal on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(-1.2, 1.2)
    ax2.plot(merged, label='Average dfof', linewidth=3, color='#38843F')
    ax2.fill_between(np.arange(len(merged)), merged - merge_sem, merged + merge_sem, color='#90CB82', alpha=0.5)

    ## predicted
    ax2.plot(merged_pdfof, label='Average predicted dfof', linewidth=3, color='red')
    ax2.fill_between(np.arange(len(merged_pdfof)), merged_pdfof - merge_sem_pdfof, merged_pdfof + merge_sem_pdfof, color='#F8C1D9', alpha=0.5)

    ax2.set_ylabel('avg(zscore(dfof))', color='#0F8140')
    ax2.tick_params(axis='y', labelcolor='#0F8140')
    fig.tight_layout()
    # plt.savefig(path + "main_signals" + name_behavior + ".svg", format="svg")
    plt.show()

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
    # plt.savefig(path + "bar_plots" + name_behavior + ".svg", format="svg")
    plt.show()



def test_strech_time_otherBehaviors_only_4_with_speed_acrosswins():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/onset_offset/'  # for saving the results
    directory_path = 'C:/Users/ffarokhi/Desktop/BlancaData/all_30/'  # Path to directories of the .csv files including cholinergic activity, speed, and behavioral signals
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

def box_plotting():
    path = 'C:/Users/ffarokhi/Desktop/final draft results/GLM/'  # for saving the results
    name_file="test_box_adj"
    file_path = path+ name_file + ".csv"  # Replace with the path to your CSV file
    data = pd.read_csv(file_path)

    # Step 2: Specify custom color codes (HEX format)
    # color_codes = ['#2278B5', '#2278B5', '#2278B5', '#2278B5', '#2278B5'] # blue sample
    # color_codes = ['#FBAB18', '#FBAB18', '#FBAB18', '#FBAB18', '#FBAB18'] # orange test
    color_codes = ['#8D574C', '#F57F20', '#2278B5', '#9268AD', '#2FA148']

    # Step 3: Adjust x-axis positions to make boxes closer
    x_positions = range(1, len(data.columns) + 1)  # Generate x-axis positions for boxes

    # Step 4: Create a figure and box plot without outliers
    plt.figure(figsize=(7, 6))  # Set a smaller figure size for closer boxes
    box = plt.boxplot(
        data.values,  # Use the values from the CSV file
        positions=x_positions,  # Set the x positions of the boxes
        patch_artist=True,  # Enable color filling for boxes
        labels=data.columns,  # Set the column names as x-axis labels
        showfliers=False,  # Hide the outliers
        widths = 0.5,
        showmeans=True,      # Show the mean
        meanline=True,       # Use a line to represent the mean
        meanprops={'color': 'black', 'linewidth': 1},
        medianprops={'color': 'none'}  # Hide the median line

    )

    # Step 5: Apply colors to each box
    for patch, color in zip(box['boxes'], color_codes):
        patch.set_facecolor(color)  # Set the face color of each box


    # Step 6: Add titles, labels, and grid
    plt.title('Box Plot with Tight Spacing', fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines for clarity

    # Step 5: Overlay individual points with random jitter
    for i, col in enumerate(data.columns):
        jittered_x = np.random.normal(x_positions[i], 0.02, len(data[col]))  # Add random jitter around the x position
        plt.scatter(
            jittered_x-0.35, data[col], alpha=0.8, s=50, color=color_codes[i], edgecolor='k', zorder=3, label=f'{col} points'
        )  # zorder ensures points are drawn in front

    # Step 7: Display the plot
    plt.tight_layout()  # Adjust layout for better appearance
    plt.xlim(0.25, 5.5)
    plt.ylim(-16, 9)
    plt.savefig(path + name_file + ".svg", format="svg")
    plt.show()

def box_plotting_paired():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # File paths and configuration
    path = 'C:/Users/ffarokhi/Desktop/final draft results/GLM/'  # for saving the results
    name_file = "diff_adj"
    file_path = path + name_file + ".csv"  # Replace with the path to your CSV file
    data = pd.read_csv(file_path)

    # Step 2: Specify custom color codes (HEX format)
    color_codes = ['#048786', '#871719']  # Only two colors for the two columns

    # Step 3: Adjust x-axis positions for the box plot
    x_positions = range(1, len(data.columns) + 1)  # Generate x-axis positions for boxes

    # Step 4: Create a figure and box plot without outliers
    plt.figure(figsize=(7, 6))  # Set a smaller figure size for closer boxes
    box = plt.boxplot(
        data.values,  # Use the values from the CSV file
        positions=x_positions,  # Set the x positions of the boxes
        patch_artist=True,  # Enable color filling for boxes
        labels=data.columns,  # Set the column names as x-axis labels
        showfliers=False,  # Hide the outliers
        widths=0.5,
        showmeans=True,      # Show the mean
        meanline=True,       # Use a line to represent the mean
        meanprops={'color': 'black', 'linewidth': 1},
        medianprops={'color': 'none'}  # Hide the median line
    )

    # Step 5: Apply colors to each box
    for patch, color in zip(box['boxes'], color_codes):
        patch.set_facecolor(color)  # Set the face color of each box

    # Step 6: Overlay individual points and connect paired points with lines
    for i, (x1, x2) in enumerate(zip(data.iloc[:, 0], data.iloc[:, 1])):  # Iterate through paired points
        plt.plot(
            [1, 2], [x1, x2], color='gray', alpha=0.6, zorder=2  # Connect the paired points with a line
        )
        # Add scatter points for the two columns
        plt.scatter(
            1 , x1, alpha=0.8, s=50, color=color_codes[0], edgecolor='k', zorder=3
        )
        plt.scatter(
            2 , x2, alpha=0.8, s=50, color=color_codes[1], edgecolor='k', zorder=3
        )

    # Step 7: Add titles, labels, and grid
    plt.title('Box Plot with Paired Points', fontsize=16)
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines for clarity

    # Step 8: Save and display the plot
    plt.tight_layout()  # Adjust layout for better appearance
    # plt.xlim(0.5, 2.5)  # Set x-axis limits for two columns
    plt.savefig(path + name_file + ".svg", format="svg")
    plt.show()

if __name__ == '__main__':
    print("Cholinergic Activity")
    # box_plotting_paired()
    # plot_speed_dfof()
    # find_best_smoothing_corr()
    # novel_env()
    # GLM_func()
    # behavioral_interactions()
    # washout_win()
    # plot_avg_withSem_GLM()
    # extract_frame()
    # washout_win()
    # washout_win_avg()
    # plot_avg_withSem()
    # behavioral_interactions()
    # wilcoxin_test()
    # strech_time()
    # histo_distance_behaviors()
    # histo_winsize_behaviors()
    # plot_error()
    # plotting()
    # plot_poly()
    # plt_regression_lines()
    # time_analysis()
    # washout_win()
    # speed_correlation()
    # plot_washed_times_errorbars()
    # anova_washed_times()
    # test_strech_time()
    # test_strech_time_newIdea()
    # test_strech_time_otherBehaviors()
    # test_strech_time_otherBehaviors_only_4()
    # test_strech_time_otherBehaviors_only_4_with_speed_acrosswins()
    # test_strech_time_otherBehaviors_only_4_with_speed()