%% cleanup
close all;
clear; 
clc;

%% flags to avoid some computations 
PROC_ARTIFIMG = 0 ; % gradient descent loop on artifical image
PROC_CORTEXIM = 1 ; % gradient descent loop on cortex image
PROC_CHANVESE = 1 ; % Chan-Vese segmentation

%% changing path
% cd /media/Data/Users/murd/Documents/gitLinBoxes/gpeyre-numtours/matlab
%% adding toolbox to path
getd = @(p)path(p,path); % scilab users must *not* execute this
getd('toolbox_signal/');
getd('toolbox_general/');
getd('toolbox_graph/');

%% img param
n = 200;
[Y,X] = meshgrid(1:n,1:n);
r = n/3;
c = [r r] +10;

%% first field - circle
phi1 = sqrt( (X-c(1)).^2 + (Y-c(2)).^2 ) - r;
%% second field
c2 = 2*[r r] - [10 5];
phi2 = max( abs(X-c2(1)) , abs(Y-c2(2)) ) - r;

%% display the two fields separately
figure('Name', 'gpeyre segmentation 3 snakes levelset') ;
subplot(1,2,1);
 plot_levelset(phi1);
subplot(1,2,2);
 plot_levelset(phi2);

%% union of the two fields
phi0 = min( phi1 , phi2 ) ;
subplot(1,2,1);
 plot_levelset( phi0 );
 title('Original') ;

%% snake parameters
Tmax = 200 ;
tau = 0.5 ;
niter = round( Tmax / tau ) ;
options.order = 2 ;  %% gpeyre's comment : We use centered differences for the discretization of the gradient. WTF
%% initial shape 
phi = phi0 ;
%% plot initial shape
subplot( 1 , 2 , 2 );
plot_levelset( phi ) ;
title('Original') ;

%% gradient descent loop
if( ( exist( 'PROC_ARTIFIMG', 'var' ) == 1 ) && ( PROC_ARTIFIMG > 0 ) )
    for ii = 1:niter
       %% compute gradient
        g0 = grad( phi , options ) ;
        d = max( eps, sqrt( sum( g0 .^2 , 3 ) ) ); %% gradient norm
        g = g0 ./ repmat( d , [ 1 1 2 ] ); %% normalized gradient
        K = -d .* div( g , options ) ; % curvature term
        phi = phi - tau * K ;
        subplot( 1 , 2 , 2 ) ;
        plot_levelset( phi ) ;
        title( sprintf( 'Iteration %03i' , ii ) );
        drawnow
        if( mod( ii , 10 ) == 0 )
            %         keyboard ;
        end
    end
end

%%%
%% re-distancing
%%%
if( ( exist( 'PROC_REDIST', 'var' ) == 1 ) && ( PROC_REDIST > 0 ) )
    phi = phi0 .^ 3 ;
    phi1 = perform_redistancing( phi0 ) ;

    %% display redistancing
    subplot( 1 , 2 , 1 ) ;
     plot_levelset( phi ) ;
     title( 'Before redistancing' ) ;
    subplot( 1 , 2 , 2 ) ;
     plot_levelset( phi1 ) ;
     title( 'After redistancing' );
end

%%% 
%% edge-based segmentation with geodesic active contour
%%%
%% input image
n = 200 ;
name = 'cortex' ;
f0 = rescale( sum( load_image( name , n ) , 3 ) ) ;
 
%% compute gradient
g = grad( f0 , options ) ;
d0 = sqrt( sum( g .^2 , 3 ) ) ;
%% blurring
a = 5 ; % blur size
d = perform_blurring( d0 , a ) ;

%% weights : invert of blurred gradient
epsilon = 1e-1 ; % safety value for zero gradient-regions 
W = 1 ./ ( epsilon + d ) ;
W = rescale( -d , .1 ,1) ;

%% display weights
figure('Name', 'gpeyre cortex weights') ;
imageplot( f0 , 'Image to segment', 1 , 2 , 1 ) ;
imageplot(  W , 'Weights'         , 1 , 2 , 2 ) ;

%% initial contour, centered square
[ Y , X ] = meshgrid( 1:n ,1:n ) ;
r = n / 3 ;
c = [ n n ] / 2 ;
phi0 = max( abs( X - c( 1 ) ) , abs( Y - c( 2 ) ) ) - r ;
%% display initial contour
% clf ;
% plot_levelset(phi0 , 0 , f0 ) ;

%% active contour parameters
tau = .4;
Tmax = 2000;
niter = round(Tmax/tau);
%% init
phi = phi0 ;
%% precompute weight gradients
gW = grad( W , options ) ;


%% gradient descent loop
if( ( exist( 'PROC_CORTEXIM', 'var' ) == 1 ) && ( PROC_CORTEXIM > 0 ) )
    figure('Name', 'gpeyre cortex segmentation') ;
    for ii = 1:niter
        gD = grad( phi , options ) ;
        d = max( eps , sqrt( sum( gD .^2 , 3 ) ) ); % normalized gradient
        g = gD ./ repmat( d , [ 1 1 2 ] ); 
        G = - W .* d .* div( g , options ) - sum( gW .* gD , 3 ) ; % gradient
        phi = phi - tau * G ; % gradient descent step
        if( mod( ii , 5 ) == 0 )
            clf;
            plot_levelset( phi , 0 , f0 ) ;
            title( sprintf( 'Iteration %03i' , ii ) );
            drawnow
        end
        if( mod( ii , 30 ) == 0 )
            phi = perform_redistancing( phi ) ;
        end
    end
    sprintf('Cortex segmentation finished after %04i iterations', ii )  ;
end



%%%
%% edge-based segmentation with Chan-Vese
%%%

%% using small circles as init
[ Y , X ] = meshgrid( 1:n , 1:n ) ;
k = 4 ; %number of circles
r = .3 * n / k ;
phi0 = zeros( n , n ) + Inf ;
for i=1:k
    for j=1:k
        c = ( [ i j ] - 1 ) * ( n / k ) + ( n / k ) * .5 ;
        phi0 = min( phi0 , sqrt( ( X - c( 1 ) ).^2 + ( Y - c( 2 ) ).^2 ) - r ) ;
    end
end
figure('Name', 'gpeyre chan-vese cortex segmentation') ;
subplot( 1 , 2 , 1 ) ;
 plot_levelset( phi0 ) ;
subplot( 1 , 2 , 2) ;
 plot_levelset( phi0, 0 , f0) ;

%% chan-vese parameters
lambda = 0.8 ;
c1 = 0.7 ;
c2 = 0.0 ;
tau = 0.4 ;
Tmax = 3000;
niter = round(Tmax/tau);

%% chan-vese loop
if( ( exist( 'PROC_CHANVESE', 'var' ) == 1 ) && ( PROC_CHANVESE > 0 ) )
    phi = phi0;
    clf ;
    for ii = 1:niter
        gD = grad( phi , options ) ;
        d = max( eps , sqrt( sum( gD .^2 , 3 ) ) ); % find normalizatino factor
        g = gD ./ repmat( d , [ 1 1 2 ] ); % normalized gradient
        G = - d .* div( g , options ) ...
                - ( lambda * ( f0 - c1 ).^2 ) ...
                + ( lambda * ( f0 - c2 ).^2 ) ; % gradient
        phi = phi - tau * G ; % gradient descent step
        if( mod( ii , 5 ) == 0 )
            clf;
            plot_levelset( phi , 0 , f0 ) ;
            title( sprintf( 'Iteration %03i' , ii ) );
            drawnow
        end
        if( mod( ii , 30 ) == 0 )
            phi = perform_redistancing( phi ) ;
        end
    end
    fprintf('Chan-vese finished after %04i iterations', ii )  ;
end










