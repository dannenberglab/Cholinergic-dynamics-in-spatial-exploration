
df_long = readtable('E:/lab/Cholinergic Prj/final files/results/all20sessions_allbehaviors.csv.csv');
df_long.task = categorical(df_long.task);
df_long.mouse_id = categorical(df_long.mouse_id);
df_long.session_id = categorical(df_long.session_id);
df_long.trial_id = categorical(df_long.trial);
df_long.exploration_diff = df_long.exp_non_statobj - df_long.exp_statobj;

% formula ='effect ~ -1 + task*behavior + (1 + behavior | session_id) + (1 + behavior | mouse_id)';
formula = ['dfof ~ task * (speed + exp_non_statobj + exp_statobj + rearing + grooming) ' ...
'+ (1 + task * (speed + exp_non_statobj + exp_statobj + rearing + grooming) | mouse_id)'...
'+ (1 + task * (speed + exp_non_statobj + exp_statobj + rearing + grooming) | mouse_id:trial_id)'...
'+ (1 + task * (speed + exp_non_statobj + exp_statobj + rearing + grooming) | session_id)'];

% run LLM
lme = fitlme(df_long, formula);
disp(lme)

% save
save_path = 'E:/lab/Cholinergic Prj/final files/results/llmzscoreallsession.mat'; 
save(save_path, 'lme');


%% a linear contrast analysis within the fitted LMM
idx_1 = find(strcmp(lme.CoefficientNames, 'task_1:exp_non_statobj'));
idx_2 = find(strcmp(lme.CoefficientNames, 'task_1:exp_statobj'));
contrast = zeros(1, length(lme.CoefficientNames));
contrast(idx_1) = 1;
contrast(idx_2) = -1;
% run FTest
[pval, Fstat, df1, df2] = coefTest(lme, contrast)
