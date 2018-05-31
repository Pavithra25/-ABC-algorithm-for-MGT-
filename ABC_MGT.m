# -ABC-algorithm-for-MGT-
Machey glass time series prediction using ABC
clear all
close all
clc
% Control Parameters of ABC algorithm
population=20; %The number of colony size (employed bees+onlooker bees)
FoodNumber=population/2; %The number of food sources equals the half of the colony size
limit=100; % A food source which could not be improved through "limit" trials is abandoned by its employed bee
max_iterations=1000; %The number of cycles for foraging
load mgdata.dat
x=mgdata(:,2);
input_x = x(1:600);
%Problem specific variables
d=15; %The number of parameters of the problem to be optimized
ub=ones(1,d)*5.12; %lower bounds of the parameters.
lb=ones(1,d)*(-5.12);%upper bound of the parameters.
runtime=10;%No of runs in order to see its robustness
Global_gbest=zeros(1,runtime);
for r=1:runtime
% All food sources are initialized
% Variables are initialized in the range [lb,ub]
Range = repmat((ub-lb),[FoodNumber 1]);
Lower = repmat(lb, [FoodNumber 1]);
%Foods = rand(FoodNumber,d) .* Range + Lower;
for j= 1 : FoodNumber
w_sigma_1 = 8*randn; %weight 1
w_sigma_2 = 8*randn; %weight 2
w_sigma_3 = 8*randn; %weight 3
w_sigma_4 = 8*randn; %weight 4
w_sigma_5 = 8*randn; %weight_5 Feedback input to Sigma
w_pi_1 = 8*randn; %weight 6
w_pi_2 = 8*randn; %weight 7
w_pi_3 = 8*randn; %weight 8
w_pi_4 = 8*randn; %weight 9
w_pi_5 = 8*randn; %weight_10 Feedback input to Pi
w = 8*randn; %weight 11
x_os = 1; %weight 12
Neural Network and Deep learning Algorithm Page 24
x_opi = 1; %weight 13
lambda_s = -1; %weight 14
lambda_p = 12.5; %weight 15
Foods(j,:)=[w_sigma_1,w_sigma_2,w_sigma_3,w_sigma_4,w_sigma_5, w_pi_1,w_pi_2,w_pi_3,w_pi_4,w_pi_5,w,x_os,x_opi,lambda_s,lambda_p];
end
end
%%%%fbest=10000* ones(no_of_particles,1)
% Foods is the population of food sources.
% Each row of Foods matrix is a vector holding d parameters to be optimized.
% The number of rows of Foods matrix equals to the FoodNumber
costfood = 1000*ones(FoodNumber,1)
Fun_Cost = costfood; % Result from the function
Fitness = calculateFitness(Fun_Cost); % Fitness of cost
% reset trial counters
% trial vector holds trial numbers through which solutions can not be improved
trial=zeros(1,FoodNumber);
%The best food source is memorized
BestInd=find(Fun_Cost==min(Fun_Cost));
BestInd=BestInd(end);
gbest=Fun_Cost(BestInd); % Optimal solution
gbest_Params=Foods(BestInd,:); % Parameters of Optimal Solution
iter=1;
while ((iter <= max_iterations))
tic;
%%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
for i=1:FoodNumber
%The parameter to be changed is determined randomly
k=fix(rand*d)+1;
% A randomly chosen solution is used in producing a mutant solution of the solution i
j=fix(rand*(FoodNumber))+1;
%Randomly selected solution must be different from the solution i
while(j==i)
j=fix(rand*(FoodNumber))+1;
end;
for j = 1 : FoodNumber
weights = Foods(j,:);
output(18) = randn;
total_error_square = 0;
for kk = 19:(numel(input_x)-6)
s_net_1 = input_x(kk-18) * weights(1);
s_net_2 = input_x(kk-12) * weights(2);
s_net_3 = input_x(kk-6) * weights(3);
s_net_4 = input_x(kk) * weights(4);
s_net_5 = output(kk-1) * weights(5);
s_net = s_net_1 + s_net_2 + s_net_3 + s_net_4 + s_net_5 + weights(12);
Neural Network and Deep learning Algorithm Page 25
p_net_1 = input_x(kk-18) * weights(6);
p_net_2 = input_x(kk-12) * weights(7);
p_net_3 = input_x(kk-6) * weights(8);
p_net_4 = input_x(kk) * weights(9);
p_net_5 = output(kk-1) * weights(10);
p_net = p_net_1 * p_net_2 * p_net_3 * p_net_4 * p_net_5 * weights(13);
o_sigma = 1/(1+ exp(-1*weights(14)*s_net));
o_pi = exp(-1*weights(15)*p_net*p_net) ;
output_pi = (1-weights(11))*o_pi;
output_s = weights(11) * o_sigma;
output(kk) = output_pi + output_s;
desired_output(kk) = input_x(kk+6);
error_square(kk) = (output(kk) - desired_output(kk))^2;
total_error_square = (total_error_square + error_square(kk));
end % END of k = 19:(numel(input_x)-6) loop
mean_square_error = total_error_square/numel(input_x);
costfood(i,:) = mean_square_error
% Generate a new solution
new_sol=Foods(i,:);
% v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
new_sol(k)=Foods(i,k)+(Foods(i,k)-Foods(j,k))*(rand-0.5)*2;
% if generated parameter value is out of boundaries, it is shifted onto the boundaries
ind=find(new_sol<lb);
new_sol(ind)=lb(ind);
ind=find(new_sol>ub);
new_sol(ind)=ub(ind);
%evaluate new solution
Sol_cost = rastrigin(new_sol);
% Fitness value of new solution
FitnessSol=calculateFitness(Sol_cost);
% Apply greedy selection between the current solution i and its mutant
% If the mutant solution is better than the current solution i,
% replace the solution with the mutant and reset the trial counter of solution
if (FitnessSol>Fitness(i))
Foods(i,:)=new_sol;
Fitness(i)=FitnessSol;
Fun_Cost(i)=Sol_cost;
trial(i)=0;
else
trial(i)=trial(i)+1; %if the solution i can not be improved, increase its trial counter
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Neural Network and Deep learning Algorithm Page 26
% A food source is chosen with the probability proportioal to its quality
prob=(0.9.*Fitness./max(Fitness))+0.1;
%%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
t=0;
while(t<FoodNumber)
if(rand<prob(i))
t=t+1;
%The parameter to be changed is determined randomly
k=fix(rand*d)+1;
%A randomly chosen solution is used in producing a mutant solution of the solution i
j=fix(rand*(FoodNumber))+1;
%Randomly selected solution must be different from the solution i
while(j==i)
j=fix(rand*(FoodNumber))+1;
end;
Foods(i,:)= mean_square_error;
new_sol=Foods(i,:);
% v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
new_sol(k)=Foods(i,k)+(Foods(i,k)-Foods(j,k))*(rand-0.5)*2;
% if generated parameter value is out of boundaries, it is shifted onto the boundaries
ind=find(new_sol<lb);
new_sol(ind)=lb(ind);
ind=find(new_sol>ub);
new_sol(ind)=ub(ind);
% evaluate new solution
Sol_cost=rastrigin(new_sol);
FitnessSol=calculateFitness(Sol_cost);
% greedy selection is applied between the current solution i and its mutant
% If the mutant solution is better than the current solution i,
% replace the solution with the mutant and reset the trial counter of solution i
if (FitnessSol>Fitness(i))
Foods(i,:)=new_sol;
Fitness(i)=FitnessSol;
Fun_Cost(i)=Sol_cost;
trial(i)=0;
else
trial(i)=trial(i)+1; %if the solution i can not be improved, increase its trial counter
end;
end;
i=i+1;
if (i==(FoodNumber)+1)
i=1;
Neural Network and Deep learning Algorithm Page 27
end;
end;
% best food source is memorized
ind=find(Fun_Cost==min(Fun_Cost));
ind=ind(end);
if (Fun_Cost(ind)<gbest)
gbest=Fun_Cost(ind);
gbest_Params=Foods(ind,:);
end;
%%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%determine the food sources whose trial counter exceeds the "limit" value.
ind=find(trial==max(trial));
ind=ind(end);
if (trial(ind)>limit)
trial(ind)=0;
new_sol=(ub-lb).*rand(1,d)+lb;
Sol_cost=rastrigin(new_sol);
FitnessSol=calculateFitness(Sol_cost);
Foods(ind,:)=new_sol;
Fitness(ind)=FitnessSol;
Fun_Cost(ind)=Sol_cost;
end;
BestCost(iter)=gbest;
time_taken = toc;
fprintf('Iter=%d ObjVal=%g Time Taken =%d\n',iter,gbest,time_taken);
iter=iter+1;
end % End of ABC
Global_gbest(r)=gbest; % gbest of each run
end; %end of runs
hold on;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
title('Rastrigin function = 10d + sigma(x^2-10cos(2*pi*x))')
grid on
m_gbest = mean(Global_gbest);
std_dev = std(Global_gbest);
fprintf(1,'Statistics for 10 Runs with 100 iterations in each run\n');
fprintf(1,'Mean Global Best : %f and Std Devioation : %f',m_gbest,std_dev);
%********************************************* Testing Phase ****************************************
input_x = x(576:1201);
%weights = global_best_position;
output(18) = randn;
total_error_square = 0;
for k = 19:(numel(input_x)-6)
s_net_1 = input_x(k-18) * weights(1);
Neural Network and Deep learning Algorithm Page 28
s_net_2 = input_x(k-12) * weights(2);
s_net_3 = input_x(k-6) * weights(3);
s_net_4 = input_x(k) * weights(4);
s_net_5 = output(k-1) * weights(5);
s_net = s_net_1 + s_net_2 + s_net_3 + s_net_4 + s_net_5 + weights(12);
p_net_1 = input_x(k-18) * weights(6);
p_net_2 = input_x(k-12) * weights(7);
p_net_3 = input_x(k-6) * weights(8);
p_net_4 = input_x(k) * weights(9);
p_net_5 = output(k-1) * weights(10);
p_net = p_net_1 * p_net_2 * p_net_3 * p_net_4 * p_net_5 * weights(13);
o_sigma = 1/(1+ exp(-1*weights(14)*s_net));
o_pi = exp(-1*weights(15)*p_net*p_net) ;
output_pi = (1-weights(11))*o_pi;
output_s = weights(11) * o_sigma;
output(k) = output_pi + output_s;
desired_output(k) = input_x(k+6);
error_square(k) = (output(k) - desired_output(k))^2;
total_error_square = (total_error_square + error_square(k));
end % END of k = 19:(numel(input_x)-6) loop
op_mean_square_error = total_error_square/numel(input_x);
sprintf('Testing MSE = %d', op_mean_square_error)
% Determine MSE over training set
%
plot(1:600,output(19:618),'r',1:600,desired_output(19:618),'b')
xlabel('t')
ylabel('x(t)')
grid on
legend('Output', 'Desired Output','Location', 'Best' )
title('MG Time Series')
figure
plot(1:iter,Global_gbest,'r')
xlabel('Iteration')
ylabel('Mean Square ')
title('Variation of Mean Square Error')
grid on
