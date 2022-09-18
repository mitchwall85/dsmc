clear
close all
% generate points in a cylindrical face randomly

tic
r_rand = [];
t_rand = [];
for i = 1:10000
    r_rand = [r_rand, 2*sqrt(rand(1,1))];
    t_rand = [t_rand, rand(1)*2*pi];
   
end
toc



polarscatter(t_rand,r_rand,'.')
rlim([0 2])


% saveas('random_distribution.png')