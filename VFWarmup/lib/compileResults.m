tableVF=readtable('VFWarmup.csv');
Res=zeros(5,6);
sumVF=0;
for j=1:7
    k=1;
   for i=1:15
       sumVF=sumVF+tableVF{i,j};
       if mod(i,3)==0
           Res(k,j)=sumVF/3;
           sumVF=0;
           k=k+1;
       end
   end
end