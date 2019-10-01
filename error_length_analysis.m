raw_table = readtable('~/ctm2-stable-baselines/rl-baselines-zoo/test2.csv');

% ranges = 0:0.01:0.05;
ranges = 0:0.02:0.1;
total_steps = [];
g_steps = [];
total_rewards = [];
g_rewards = [];
total_success = [];
g_success = [];
total_errors = [];
g_errors = [];

for i=1:length(ranges) - 1
    % bin = raw_table(raw_table.qGoal_2 > ranges(i) & raw_table.qGoal_2 < ranges(i+1), :); % one tube bins
    bin = raw_table((raw_table.qGoal_2 + raw_table.qGoal_4) > ranges(i) & (raw_table.qGoal_2 + raw_table.qGoal_4) < ranges(i+1), :); % two tube total extension boxplot
    % bin = raw_table((raw_table.qGoal_2) > ranges(i) & (raw_table.qGoal_2) < ranges(i+1), :); % two tube, first tube extension boxplot
    % bin = raw_table((raw_table.qGoal_4) > ranges(i) & (raw_table.qGoal_4) < ranges(i+1), :); % two tube, second tube extension boxplot
    total_steps = [total_steps; bin.totalSteps];
    g_steps = [g_steps; i * ones(size(bin.totalSteps))];
    total_rewards = [total_rewards; bin.rewards];
    g_rewards = [g_rewards; i * ones(size(bin.rewards))];
    total_success = [total_success; bin.success];
    g_success = [g_success; i * ones(size(bin.success))];
    total_errors = [total_errors; bin.errors];
    g_errors = [g_errors; i * ones(size(bin.errors))];
end
% 2-tube boxplots
boxplot(total_errors * 10^3, g_errors, 'Labels',{'0 to 20','20 to 40', '40 to 60','60 to 80','80 to 100'})
% boxplot(total_errors * 10^3, g_errors, 'Labels',{'0 to 10','10 to 20', '20 to 30','30 to 40','40 to 50'}) % first or second tube labelling
title("Error as a function of Length, 2-tube  total extension")
xlabel("Extension groups (mm)")
ylabel("Error(mm)")

% 1-tube boxplot
% boxplot(total_errors * 10^3, g_errors, 'Labels',{'0 to 10','10 to 20', '20 to 30','30 to 40','40 to 50'})
% title("Error as a function of Length, 1-tube")
% xlabel("Extension groups (mm)")
% ylabel("Error(mm)")