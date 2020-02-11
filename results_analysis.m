% Training analysis for IPCAI 2020 Paper
% Keshav Iyengar

% Load in learning data
% 2-tube
exp_1 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_1/progress.csv');
exp_2 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_2/progress.csv');
exp_4 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_4/progress.csv');
exp_5 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_5/progress.csv');
% 3-tube
exp_6 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_6/progress.csv');
exp_7 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_7/progress.csv');
exp_9 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_9/progress.csv');
exp_10 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_10/progress.csv');
% 4-tube
exp_11 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_11/progress.csv');
exp_12 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_12/progress.csv');
exp_14 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_14/progress.csv');
exp_15 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_15/progress.csv');
%% 2 tube plotting
% Success rate and error in single plot.
fig = figure;
set(fig,'defaultAxesColorOrder',[[0, 0, 0]; [0, 0, 0]]);
plot(exp_1.total_steps, exp_1.successRate * 100,'r', 'DisplayName','Type 1 success rate')
hold on
plot(exp_2.total_steps, exp_2.successRate * 100, 'b', 'DisplayName','Type 2 success rate')
hold on
plot(exp_4.total_steps, exp_4.successRate * 100, 'g','DisplayName','Type 3 success rate')
hold on
plot(exp_5.total_steps, exp_5.successRate * 100, 'k','DisplayName','Type 4 success rate')
hold on
xlabel('Training steps')
ylabel('Success rate (%)')
yyaxis right
plot(exp_1.total_steps, exp_1.eval_errors * 1000, 'r--', 'DisplayName','Type 1 evaluation error')
hold on
plot(exp_2.total_steps, exp_2.eval_errors * 1000, 'b--', 'DisplayName','Type 2 evaluation error')
hold on
plot(exp_4.total_steps, exp_4.eval_errors * 1000, 'g--', 'DisplayName','Type 3 evaluation error')
hold on
plot(exp_5.total_steps, exp_5.eval_errors * 1000, 'k--', 'DisplayName','Type 4 evaluation error')
ylabel('Error (mm)')
legend
%% 3 tube plotting
% Success rate and error in single plot.
fig = figure;
set(fig,'defaultAxesColorOrder',[[0, 0, 0]; [0, 0, 0]]);
plot(exp_6.total_steps, exp_6.successRate * 100,'r', 'DisplayName','Type 1 success rate')
hold on
plot(exp_7.total_steps, exp_7.successRate * 100, 'b', 'DisplayName','Type 2 success rate')
hold on
plot(exp_9.total_steps, exp_9.successRate * 100, 'g','DisplayName','Type 3 success rate')
hold on
plot(exp_10.total_steps, exp_10.successRate * 100, 'k','DisplayName','Type 4 success rate')
hold on
xlabel('Training steps')
ylabel('Success rate')
yyaxis right
plot(exp_6.total_steps, exp_6.eval_errors * 1000, 'r--', 'DisplayName','Type 1 evaluation error')
hold on
plot(exp_7.total_steps, exp_7.eval_errors * 1000, 'b--', 'DisplayName','Type 2 evaluation error')
hold on
plot(exp_9.total_steps, exp_9.eval_errors * 1000, 'g--', 'DisplayName','Type 3 evaluation error')
hold on
plot(exp_10.total_steps, exp_10.eval_errors * 1000, 'k--', 'DisplayName','Type 4 evaluation error')
ylabel('Error (mm)')

%% 4 tube plotting
% Success rate and error in single plot.
fig = figure;
set(fig,'defaultAxesColorOrder',[[0, 0, 0]; [0, 0, 0]]);
plot(exp_11.total_steps, exp_11.successRate * 100,'r', 'DisplayName','Type 1 success rate')
hold on
plot(exp_12.total_steps, exp_12.successRate * 100, 'b', 'DisplayName','Type 2 success rate')
hold on
plot(exp_14.total_steps, exp_14.successRate * 100, 'g','DisplayName','Type 3 success rate')
hold on
plot(exp_15.total_steps, exp_15.successRate * 100, 'k','DisplayName','Type 4 success rate')
hold on
xlabel('Training steps')
ylabel('Success rate')
yyaxis right
plot(exp_11.total_steps, exp_11.eval_errors * 1000, 'r--', 'DisplayName','Type 1 evaluation error')
hold on
plot(exp_12.total_steps, exp_12.eval_errors * 1000, 'b--', 'DisplayName','Type 2 evaluation error')
hold on
plot(exp_14.total_steps, exp_14.eval_errors * 1000, 'g--', 'DisplayName','Type 3 evaluation error')
hold on
plot(exp_15.total_steps, exp_15.eval_errors * 1000, 'k--', 'DisplayName','Type 4 evaluation error')
ylabel('Error (mm)')
legend
%% Load in evaluation data
% Load in eval data
% 2-tube
exp_1 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_1/eval.csv');
exp_2 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_2/eval.csv');
exp_4 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_4/eval.csv');
exp_5 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_5/eval.csv');
% 3-tube
exp_6 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_6/eval.csv');
exp_7 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_7/eval.csv');
exp_9 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_9/eval.csv');
exp_10 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_10/eval.csv');
% 4-tube
exp_11 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_11/eval.csv');
exp_12 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_12/eval.csv');
exp_14 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_14/eval.csv');
exp_15 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_15/eval.csv');

%% 2-tube plotting
errorbar(exp_1.goal_tolerance * 1000, exp_1.mean_error * 1000, exp_1.std_error / 2 * 1000, exp_1.std_error / 2 * 1000, 'r', 'DisplayName','Single Gaussian mean evaluation error')
hold on
errorbar(exp_2.goal_tolerance * 1000, exp_2.mean_error * 1000, exp_2.std_error / 2 * 1000, exp_2.std_error / 2 * 1000, 'b', 'DisplayName','Multivariate Gaussian mean evaluation error')
hold on
errorbar(exp_4.goal_tolerance * 1000, exp_4.mean_error * 1000, exp_4.std_error / 2 * 1000, exp_4.std_error / 2 * 1000, 'g', 'DisplayName','Parameter Noise mean evaluation error')
hold on
errorbar(exp_5.goal_tolerance * 1000, exp_5.mean_error * 1000, exp_5.std_error / 2 * 1000, exp_5.std_error / 2 * 1000, 'k', 'DisplayName','OU mean evaluation error')
xlabel('Goal tolerance (mm)')
ylabel('Error (mm)')
xlim([0 1.1 ])
%% 3-tube plotting
errorbar(exp_6.goal_tolerance * 1000, exp_6.mean_error * 1000, exp_6.std_error / 2 * 1000, exp_6.std_error / 2 * 1000, 'r', 'DisplayName','Single Gaussian mean evaluation error')
hold on
errorbar(exp_7.goal_tolerance * 1000, exp_7.mean_error * 1000, exp_7.std_error / 2 * 1000, exp_7.std_error / 2 * 1000, 'b', 'DisplayName','Multivariate Gaussian mean evaluation error')
hold on
errorbar(exp_9.goal_tolerance * 1000, exp_9.mean_error * 1000, exp_9.std_error / 2 * 1000, exp_9.std_error / 2 * 1000, 'g', 'DisplayName','Parameter Noise mean evaluation error')
hold on
errorbar(exp_10.goal_tolerance * 1000, exp_10.mean_error * 1000, exp_10.std_error / 2 * 1000, exp_10.std_error / 2 * 1000, 'k', 'DisplayName','OU mean evaluation error')
xlabel('Goal tolerance (mm)')
ylabel('Error (mm)')
xlim([0 1.1 ])
%% 4-tube plotting
errorbar(exp_11.goal_tolerance * 1000, exp_11.mean_error * 1000, exp_11.std_error / 2 * 1000, exp_11.std_error / 2 * 1000, 'r', 'DisplayName','Single Gaussian mean evaluation error')
hold on
errorbar(exp_12.goal_tolerance * 1000, exp_12.mean_error * 1000, exp_12.std_error / 2 * 1000, exp_12.std_error / 2 * 1000, 'b', 'DisplayName','Multivariate Gaussian mean evaluation error')
hold on
errorbar(exp_14.goal_tolerance * 1000, exp_14.mean_error * 1000, exp_14.std_error / 2 * 1000, exp_14.std_error / 2 * 1000, 'g', 'DisplayName','Parameter Noise mean evaluation error')
hold on
errorbar(exp_15.goal_tolerance * 1000, exp_15.mean_error * 1000, exp_15.std_error / 2 * 1000, exp_15.std_error / 2 * 1000, 'k', 'DisplayName','OU mean evaluation error')
xlabel('Goal tolerance (mm)')
ylabel('Error (mm)')
xlim([0 1.1 ])
%% Load in trajectory data
exp_1 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_1/square_traj.csv');
exp_1(1,:) = [];
exp_2 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_2/square_traj.csv');
exp_2(1:128,:) = [];
exp_4 = readtable('~/ctm2-stable-baselines/saved_results/traj_exp/exp_4_triangle_traj.csv');
exp_4(1,:) = [];
exp_5 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_5/square_traj.csv');
exp_5(1:81,:) = [];

exp_6 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_6/square_traj.csv');
exp_7 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_7/square_traj.csv'); % Redo test
exp_7(1:962,:) = [];
exp_9 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_9/square_traj.csv');
exp_9(1:538,:) = [];
exp_10 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_10/square_traj.csv');
exp_10(1,:) = [];

exp_11 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_11/square_traj.csv');
exp_11(1:499,:) = [];
exp_12 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_12/square_traj.csv');
exp_12(1:500,:) = [];
exp_14 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_14/square_traj.csv');
exp_14(1:496,:) = [];
exp_15 = readtable('~/ctm2-stable-baselines/saved-runs/results/exp_15/square_traj.csv');
exp_15(1:516,:) = [];
%% 2-tube plotting (z = 100)
plot(exp_1.achieved_goal_x * 1000, exp_1.achieved_goal_y * 1000, 'r')
hold on
plot(exp_1.desired_goal_x * 1000, exp_1.desired_goal_y * 1000, 'g')

plot(exp_2.achieved_goal_x * 1000, exp_2.achieved_goal_y * 1000, 'r')
hold on
plot(exp_2.desired_goal_x * 1000, exp_2.desired_goal_y * 1000, 'g')

plot(exp_4.achieved_goal_x(31:end) * 1000, exp_4.achieved_goal_y(31:end) * 1000, 'r')
hold on
plot(exp_4.desired_goal_x(31:end) * 1000, exp_4.desired_goal_y(31:end) * 1000, 'g')

plot(exp_5.achieved_goal_x * 1000, exp_5.achieved_goal_y * 1000, 'r')
hold on
plot(exp_5.desired_goal_x * 1000, exp_5.desired_goal_y * 1000, 'g')

%% 3-tube plotting
plot(exp_6.achieved_goal_x * 1000, exp_6.achieved_goal_y * 1000, 'r')
hold on
plot(exp_6.desired_goal_x * 1000, exp_6.desired_goal_y * 1000, 'g')

plot(exp_7.achieved_goal_x * 1000, exp_7.achieved_goal_y * 1000, 'r')
hold on
plot(exp_7.desired_goal_x * 1000, exp_7.desired_goal_y * 1000, 'g')

plot(exp_9.achieved_goal_x * 1000, exp_9.achieved_goal_y * 1000, 'r')
hold on
plot(exp_9.desired_goal_x * 1000, exp_9.desired_goal_y * 1000, 'g')

plot(exp_10.achieved_goal_x * 1000, exp_10.achieved_goal_y * 1000, 'r')
hold on
plot(exp_10.desired_goal_x * 1000, exp_10.desired_goal_y * 1000, 'g')

%% 4-tube plotting
plot(exp_11.desired_goal_x * 1000, exp_11.desired_goal_y * 1000, 'o')
hold on
text(exp_11.desired_goal_x(1) * 1000, exp_11.desired_goal_y(1) * 1000, 'start')
text(exp_11.desired_goal_x(end) * 1000, exp_11.desired_goal_y(end) * 1000, 'end')
plot(exp_11.achieved_goal_x * 1000, exp_11.achieved_goal_y * 1000, 'r')
hold on
plot(exp_11.desired_goal_x * 1000, exp_11.desired_goal_y * 1000, 'g')

plot(exp_12.achieved_goal_x * 1000, exp_12.achieved_goal_y * 1000, 'r')
hold on
plot(exp_12.desired_goal_x * 1000, exp_12.desired_goal_y * 1000, 'g')

plot(exp_14.achieved_goal_x * 1000, exp_14.achieved_goal_y * 1000, 'r')
hold on
plot(exp_14.desired_goal_x * 1000, exp_14.desired_goal_y * 1000, 'g')

plot(exp_15.achieved_goal_x * 1000, exp_15.achieved_goal_y * 1000, 'r')
hold on
plot(exp_15.desired_goal_x * 1000, exp_15.desired_goal_y * 1000, 'g')

%% Load best tolerance and strategy for each tube and plot
two_tube = readtable('~/ctm2-stable-baselines/saved_results/exp-4-0-6-square.csv');
two_tube(1:9,:) = [];
three_tube = readtable('~/ctm2-stable-baselines/saved_results/exp-10-0-4-square_test.csv');
three_tube(1:10,:) = [];
four_tube = readtable('~/ctm2-stable-baselines/saved_results/exp-14-0-3-square.csv');
four_tube(1:12,:) = [];
%% 2-tube plot
plot(two_tube.achieved_goal_x * 1000, two_tube.achieved_goal_y * 1000, 'r', 'DisplayName','achieved')
hold on
plot(two_tube.desired_goal_x * 1000, two_tube.desired_goal_y * 1000, 'g', 'DisplayName','desired')
hold on
plot(two_tube.desired_goal_x(1) * 1000, two_tube.desired_goal_y(1) * 1000, 'p', 'DisplayName','start')
hold on
plot(two_tube.desired_goal_x(end) * 1000, two_tube.desired_goal_y(end) * 1000, 'p', 'DisplayName','end')
xlabel('x (mm)')
ylabel('y (mm)')
ylim([26 44])
axis equal
legend

% plot errors
plot(two_tube.errors * 1000, 'r')
xlabel('steps')
ylabel('errors (mm)')
%% 3-tube plot
plot(three_tube.achieved_goal_x * 1000, three_tube.achieved_goal_y * 1000, 'r', 'DisplayName','achieved')
hold on
plot(three_tube.desired_goal_x * 1000, three_tube.desired_goal_y * 1000, 'g', 'DisplayName','desired')
hold on
plot(three_tube.desired_goal_x(1) * 1000, three_tube.desired_goal_y(1) * 1000, 'p', 'DisplayName','start')
hold on
plot(three_tube.desired_goal_x(end) * 1000, three_tube.desired_goal_y(end) * 1000, 'p', 'DisplayName','end')
xlabel('x (mm)')
ylabel('y (mm)')
axis equal
ylim([-4 14])
legend

% plot errors
plot(three_tube.errors * 1000, 'r')
xlabel('steps')
ylabel('errors (mm)')
%% 4-tube plot
plot(four_tube.achieved_goal_x * 1000, four_tube.achieved_goal_y * 1000, 'r', 'DisplayName','achieved')
hold on
plot(four_tube.desired_goal_x * 1000, four_tube.desired_goal_y * 1000, 'g', 'DisplayName','desired')
hold on
plot(four_tube.desired_goal_x(1) * 1000, four_tube.desired_goal_y(1) * 1000, 'p', 'DisplayName','start')
hold on
plot(four_tube.desired_goal_x(end) * 1000, four_tube.desired_goal_y(end) * 1000, 'p', 'DisplayName','end')
xlabel('x (mm)')
ylabel('y (mm)')
ylim([6 24])
axis equal
legend

% plot errors
plot(four_tube.errors * 1000, 'r')
xlabel('steps')
ylabel('errors (mm)')

%% Point cloud data
% Some plotting to compare
[achieved_goals_1, errors_1, q_values_1] = process_hd5("1");
[achieved_goals_5, errors_5, q_values_5] = process_hd5("5");
norm_q = normalize([q_values_1;q_values_5], 'range');   
norm_q_1 = norm_q(1:size(q_values_1));
norm_q_5 = norm_q(size(q_values_1)+1:end);
figure;
pcshow(transpose(achieved_goals_1), norm_q_1)
figure;
pcshow(transpose(achieved_goals_5), norm_q_5)
%%  exp_1
[achieved_goals, errors, q_values] = process_hd5("1");
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_2
[achieved_goals, errors, q_values] = process_hd5("2"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_4
[achieved_goals, errors, q_values] = process_hd5("4"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_5
[achieved_goals, errors, q_values] = process_hd5("5"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_6
[achieved_goals, errors, q_values] = process_hd5("6"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_7
[achieved_goals, errors, q_values] = process_hd5("7"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_8
[achieved_goals, errors, q_values] = process_hd5("9"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_10
[achieved_goals, errors, q_values] = process_hd5("10"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_11
[achieved_goals, errors, q_values] = process_hd5("11"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_12
[achieved_goals, errors, q_values] = process_hd5("12"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_14
[achieved_goals, errors, q_values] = process_hd5("14"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% exp_15
[achieved_goals, errors, q_values] = process_hd5("15"); 
figure;
pcshow(transpose(achieved_goals), normalize(errors, 'range'))
figure;
pcshow(transpose(achieved_goals), normalize(q_values, 'range'))
%% Functions
function [achieved_goals, errors, q_values] = process_hd5(experiment)
achieved_goals = [h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_500000.h5", "/achieved_goals"), ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1000000.h5", "/achieved_goals"), ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1500000.h5", "/achieved_goals"), ... 
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1999995.h5", "/achieved_goals")];

errors = [h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_500000.h5", "/errors"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1000000.h5", "/errors"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1500000.h5", "/errors"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1999995.h5", "/errors")];

q_values = [h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_500000.h5", "/q_values"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1000000.h5", "/q_values"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1500000.h5", "/q_values"); ...
    h5read("~/ctm2-stable-baselines/saved-runs/results/exp_" + experiment + "/data_1999995.h5", "/q_values")];
end