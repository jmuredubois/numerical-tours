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

%% display
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
 
%% gradient descent step
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