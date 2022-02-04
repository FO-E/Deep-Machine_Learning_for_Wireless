%%% iNITIALIZATION
clear variables; close all; clc;

%% GENERATION OF TRAINING DATA

dBToLinear = @(x)( 10.^(x./10) );
LinearTodB = @(x)( 10*log10(x) );
SNR_EVE_dB = 10;
dim = 10;
SNR_dB = linspace(0,25,dim);
SNR_linear = dBToLinear( SNR_dB );


N_itr = 1000;
N_itr_new = 10;
Ns = 4; % Source Antennas
Ne = 1; % Eavesdropper Antennas
Nr = 1; % Legit Antennas
Rs = 2;
N_bob = Ns*Nr;
N_eve = Ns*Ne;


secrecycapacity_new = zeros(N_itr_new,dim); %%% New data capacities at every snr
ant_idx_new = zeros(N_itr_new,dim);
featurevector = zeros(1,Ns*Nr + Ns*Ne);
train_data = zeros(N_itr, Ns*Nr + Ns*Ne + 1,dim);


for i =1:1:N_itr
    
     h = sqrt(1/2)*(randn(Nr,Ns) + 1j*randn(Nr,Ns));
     h_abs = abs(h);
     h2_abs = h_abs.^2;
     
     g = sqrt(1/2)*(randn(Ne,Ns) + 1j*randn(Ne,Ns));
     g_abs = abs(g);
     g2_abs = g_abs.^2;
     
     d = [reshape(h_abs,1,[]),reshape(g_abs,1,[])];
     feature_vector = (d - mean(d))/(max(d)-min(d));
       
    %%% Secrecy capacity at each SNR Value
    
     for k = 1:1:dim
        
         capacity_bob = log2(1+ h2_abs*SNR_linear(k));
         capacity_eve = log2(1+ g2_abs*dBToLinear(SNR_EVE_dB));

         % Selecting antenna that achieves max secrecy (Max KPI) for every
         % SNR value of BOB
         
         [secrecy_capacity, idx_ant] = max( capacity_bob - capacity_eve, [], 2);
         
%          ant_idx(:,k) = idx_ant;
%          secrecycapacity(:,k) = secrecy_capacity;
         
         train_data(i,:,k)= [feature_vector, idx_ant];

     end 
     

end 

%% FITTING TRAINING DATA TO SVM MODEL

% Generate Antenna Index for every dimension of data
index = zeros(N_itr,dim);
for w = 1:1:dim
    
    anttind = train_data(:,end,w);
    index(:,w) = anttind;
    
end


SVMModels = cell(Ns,1);
classes = unique(index);

for j = 1:numel(classes)
   
    for i = 1:1:dim
        
        indx = [index(:,i) == classes(j)];
        SVMModels{j} = fitcsvm(train_data(:,1:end-1,i), indx, 'ClassNames',[false true],'Standardize', true,...
        'KernelFunction','rbf','BoxConstraint',1);
        
    end 

end


%% GENERATION OF NEW DATA

new_data = zeros(N_itr_new, Ns*Nr + Ns*Ne,dim);
generated_g_h = zeros(N_itr_new, Ns*Nr + Ns*Ne,dim);
outage = zeros(1,dim);

for z = 1:1:N_itr_new
    
     h = sqrt(1/2)*(randn(Nr,Ns) + 1j*randn(Nr,Ns));
     h_abs = abs(h);
     h2_abs = h_abs.^2;
     h2_abs_n = h2_abs';
     
     
     g = sqrt(1/2)*(randn(Ne,Ns) + 1j*randn(Ne,Ns));
     g_abs = abs(g);
     g2_abs = g_abs.^2;
     g2_abs_n = g2_abs';
     
     d_new = [reshape(h_abs,1,[]),reshape(g_abs,1,[])];
     feature_vector_new = (d_new - mean(d_new))/(max(d_new)-min(d_new));
     
     for w = 1:1:dim
         
         new_data(z,:,w) = [feature_vector_new];
         generated_g_h(z,:,w) = [d_new]; 
         
     end
     

         capacity_bob_new = log2(1+ h2_abs_n*SNR_linear);
         capacity_eve_new = log2(1+ g2_abs_n*dBToLinear(SNR_EVE_dB));

         % Selecting antenna that achieves max secrecy (Max KPI) for every
         % SNR value of BOB
         
         [secrecy_capacity, idx_ant] = max( capacity_bob_new - capacity_eve_new, [], 1); 
         
         secrecy_capacity_ = max(secrecy_capacity,0);
         
         secrecycapacity_new(z,:) = secrecy_capacity_;
        
         ant_idx_new (z,:) = idx_ant;
         
         outage = outage + (secrecy_capacity_ <= Rs);
       
end 

%% PREDICTION OF NEW DATA

N = size(new_data,1);
Scores = zeros(N,numel(classes),dim);

for j = 1:numel(classes)
    
    for i = 1:1:dim
        
        [~, score] = predict(SVMModels{j},new_data(:,:,i));
        Scores(:,j,i) = score(:,2);
        
    end
    
end

%% ASSOCIATION EACH NEW OBSERVATION WITH THE CLASSIFIER THAT GIVES MAXIMUM SCORE

[~,maxScore] = max(Scores,[],2);


predicted_index = squeeze(maxScore);



%% FINDING SECRECY CAPACITY WITH PREDICTED INDEX FOR ANALYSIS

analysis_data = zeros(N_itr_new,Ns*Nr + Ns*Ne + 1,dim);

for i = 1:1:dim
    
    analysis_data = [generated_g_h(:,:,i), predicted_index(:,i)];
            
end


secrecy_capacity_predicted = zeros([],[]);
outage_predict_ = zeros(1,dim);

for i = 1:1:N_itr_new
    
    h_feature = analysis_data(i, analysis_data(i,end));
    g_feature = analysis_data(i, analysis_data(i,end)+Ns); 
    
    h_abs = abs(h_feature).^2;
    g_abs = abs(g_feature).^2;
    
    capacity_bob = log2(1+h_abs*SNR_linear);
    capacity_eve = log2(1+g_abs*dBToLinear(SNR_EVE_dB));
    
    [secrecy_capacity] = (capacity_bob - capacity_eve); 
         
    secrecy_capacity_ =  max(secrecy_capacity,0);
    secrecy_capacity_predicted(i,:) = secrecy_capacity_; 
    
    outage_predict_ = outage_predict_ + (secrecy_capacity_ <= Rs);
        
end

%% PLOTTING SECRECY CAPACITY FOR NEW DATA

mean_secrecycapacity = zeros([],[]);
mean_secrecycapacity_predicted = zeros([],[]);
for i = 1:1:dim
    
    mean_secrecy_capacity = mean(secrecycapacity_new(:,i));
    mean_secrecycapacity(i) = mean_secrecy_capacity; 
    
    mean_secrecy_capacity_predicted = mean(secrecy_capacity_predicted(:,i));
    mean_secrecycapacity_predicted(i) = mean_secrecy_capacity_predicted;
     
end

outage_new = outage/N_itr_new;
outage_predict = outage_predict_/N_itr_new;

figure(1);
hold on; grid on;
plot(SNR_dB,mean_secrecycapacity,'k^-');
plot(SNR_dB,mean_secrecycapacity_predicted, 'ro-');
legend('conventional', 'SVM')
title('Secrecy Capacity')
xlabel('SNR [dB]')
ylabel('Secrecy Capacity [bps/Hz]')

