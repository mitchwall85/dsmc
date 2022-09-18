
n = 10000;
blk = [0, 300, 1000]
dir = ['X', 'Y', 'Z']

ci = zeros(3,n);
for b = 1:length(blk)
    for i = 1:n

        r1 = rand();
        r2 = rand();
        k = 1.380649e-23;
        Ttr = 300;
        m = 28.0134/1000/6.02e23;

        ci(b,i) = 0 + sqrt(2*k*Ttr/m)*sin(2*pi*r1)*sqrt(-log(r2));
    end
    figure()
    histogram(abs(ci(b,:)),200)
    ylabel(strcat(dir(b), 'Velocity'))
    saveas(gcf, strcat('hist_',dir(b),'.png'))
end

