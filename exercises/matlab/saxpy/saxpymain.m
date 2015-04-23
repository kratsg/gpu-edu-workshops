%% initial matrix
g = gpuDevice()
%maxNumCompThreads( 'automatic' );
size = 1024 * 1024 * 128;
mbytes = size*8 / 1000000.0;
fprintf('Vector size is %d elements or %g MB\n',size,mbytes);
X = rand( size, 1 );
Y = rand( size, 1 );
alpha = 2.0;

%% CPU saxpy
t = tic;
C = ( alpha .* X) + Y ;
cpuT = toc(t);
fprintf('Elapsed time on CPU is %g s\n', cpuT);
gflops = 2.0 * size * 1.e-9;
cpuPerf = gflops / cpuT;
fprintf('CPU performance is %g GFlop/s\n\n', cpuPerf );


%% using gpuArray + MATLAB builtins
wait(g);
t1 = tic;
d_X = gpuArray(X);
d_Y = gpuArray(Y);
t2 = toc(t1);
%% insert the proper call to arrayfun
d_C = FIXME;
wait(g);
t3 = toc(t1);
h_C = gather( d_C );
gpuT = toc(t1);

gpuKernelPerf = gflops / (t3 - t2);
gpuTotalPerf = gflops / gpuT;

fprintf('Host to device copy time is %g s\n', t2 );
fprintf('H2D bandwidth is            %g GB/s\n', 2.0 * mbytes / t2 * .001 );
fprintf('Kernel time is              %g s\n', t3 - t2 );
fprintf('Device to host copy time is %g s\n', gpuT - t3 );
fprintf('D2H bandwidth is            %g GB/s\n', mbytes / ( gpuT - t3 ) * .001 );
fprintf('Total time is               %g s\n\n', gpuT );
fprintf('Kernel performance is       %g GFlop/s\n', gpuKernelPerf );
fprintf('Total GPU performance is    %g GFlop/s\n\n', gpuTotalPerf );

fprintf('Kernel to CPU speedup is    %g \n', gpuKernelPerf / cpuPerf );
fprintf('Total GPU to CPU speedup is %g \n', gpuTotalPerf / cpuPerf );
%% check results

tf = abs( ( h_C -  C ) ./ C ) < 0.001;

if all( tf(:) )
  fprintf('Vectors are equivalent!\n\n');
else
  fprintf('Error in computation!!!\n\n');
end

%reset( g );
clearvars;
