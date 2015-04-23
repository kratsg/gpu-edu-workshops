%% initial matrix
g = gpuDevice()
%maxNumCompThreads( 'automatic' );
size = 4096;
mbytes = size*size*8 / 1000000.0;
fprintf('Matrix size is %d by %d or %g MB\n',size,size,mbytes);
A = rand( size, size );
B = rand( size, size );

%% CPU matmult
t = tic;
C = A * B;
cpuT = toc(t);
fprintf('Elapsed time on CPU is %g s\n', cpuT);
gflops = 2.0 * size * size * size * 1.e-9;
cpuPerf = gflops / cpuT;
fprintf('CPU performance is %g GFlop/s\n\n', cpuPerf );


%% using gpuArray + MATLAB builtins
wait(g);
t1 = tic;
%% copy A and B from host to device, assiging to d_A and d_B
d_A = FIXME;
d_B = FIXME;
t2 = toc(t1);
d_C = d_A * d_B;
wait(g);
t3 = toc(t1);
%% Copy d_C back to host to array h_C
h_C = FIXME;
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
  fprintf('Matrices are equivalent!\n\n');
else
  fprintf('Error in computation!!!\n\n');
end

%reset( g );
clearvars;
