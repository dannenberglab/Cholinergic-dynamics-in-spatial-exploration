df_long = readtable('E:/lab/Cholinergic Prj/final files/results/all20sessions_allbehaviors.csv');
df_long.task = categorical(df_long.task);
df_long.mouse_id = categorical(df_long.mouse_id);
df_long.session_id = categorical(df_long.session_id);
df_long.trial_id = categorical(df_long.trial);
df_long.exploration_diff = df_long.exp_non_statobj - df_long.exp_statobj;

formula = ['dfof ~ task * (speed + exp_non_statobj + exp_statobj + rearing + grooming) ' ...
'+ (1 + speed + exp_non_statobj + exp_statobj + rearing + grooming | mouse_id)'...
'+ (1 + speed + exp_non_statobj + exp_statobj + rearing + grooming | mouse_id:trial_id)'...
'+ (1 + speed + exp_non_statobj + exp_statobj + rearing + grooming | session_id)'];

% Sampling rate and window settings
fs = 30; % Hz
window_minutes = 3;
overlap_minutes = 2;
window_samples = window_minutes * 60 * fs
step_samples = (window_minutes - overlap_minutes) * 60 * fs

% time range
max_time = max(df_long.time)
total_samples = ceil(max_time * fs)
max_window_start = floor((max_time * fs - window_samples) / step_samples)

% initialize results table
results_all = table();
contrast_results_sample = table();
contrast_results_test = table();
contrast_results_betweentwophase = table();

% Loop through sliding windows
for i = 0:12 %13  time windows
    start_time = i * step_samples / fs
    end_time = start_time + window_minutes * 60
    % Extract data for this window from all sessions
    win_data = df_long(df_long.time >= start_time & df_long.time < end_time, :);

    % Identify sessions with sufficient exploration
    session_list = unique(win_data.session_id);
    valid_rows = false(height(win_data), 1);
    
    for s = 1:length(session_list)
        session_mask = win_data.session_id == session_list(s);
    
        % Check if both behaviors occur >= 15 (1s in total) times in this session
        non_stat_count = sum(win_data.exp_non_statobj(session_mask) > 0);
        stat_count = sum(win_data.exp_statobj(session_mask) > 0);
    
        if non_stat_count >= 15 && stat_count >= 15
            valid_rows(session_mask) = true;
        end
    end
    
    % Filter valid session data
    valid_win_data = win_data(valid_rows, :);

    % Skip if nothing left
    if height(valid_win_data) < 1000
        fprintf("Skipping window %.1f–%.1f min due to insufficient valid session data.\n", start_time/60, end_time/60);
        continue;
    end
    try

        % LLM
        lme = fitlme(valid_win_data, formula);
        % fixed effect details
        fixed = dataset2table(lme.Coefficients);  
        % Add window start to the table
        fixed.WindowStart = repmat(start_time, height(fixed), 1);
        if i == 0
            fixed_all = fixed;
        else
            fixed_all = [fixed_all; fixed];
        end
        disp(lme)
    catch ME
        warning("Skipping window %.1f–%.1f min due to error:\n%s", start_time/60, end_time/60, ME.message);
    end

end

% Rename for clarity
fixed_all.Properties.VariableNames{'Estimate'} = 'Estimate';
fixed_all.Properties.VariableNames{'SE'} = 'SE';
fixed_all.Properties.VariableNames{'tStat'} = 'tStat';
fixed_all.Properties.VariableNames{'pValue'} = 'pValue';

% wide format
est_wide  = unstack(fixed_all(:, {'WindowStart', 'Name', 'Estimate'}), 'Estimate', 'Name');
se_wide   = unstack(fixed_all(:, {'WindowStart', 'Name', 'SE'}), 'SE', 'Name');
tstat_wide = unstack(fixed_all(:, {'WindowStart', 'Name', 'tStat'}), 'tStat', 'Name');
pval_wide = unstack(fixed_all(:, {'WindowStart', 'Name', 'pValue'}), 'pValue', 'Name');

% merging all into one table
results_wide_all = join(est_wide, se_wide, 'Keys', 'WindowStart');
results_wide_all = join(results_wide_all, tstat_wide, 'Keys', 'WindowStart');
results_wide_all = join(results_wide_all, pval_wide, 'Keys', 'WindowStart');

% Save results
writetable(results_wide_all, 'E:/lab/Cholinergic Prj/final files/results/LMM_windowed.csv');

