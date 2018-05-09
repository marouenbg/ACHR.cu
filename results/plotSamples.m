M = csvread('File0');
%%
figure;
for i=1:81
   subplot(9,9,i)
   hist(M(i,:))
end