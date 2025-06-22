% Load point cloud from PLY file
ptCloud = pcread("saved_pointclouds\epoch50_sample19_target.ply"); % input point cloud (.ply)

% Extract coordinates
points = double(ptCloud.Location);

% Create an alpha shape
alpha = 0.1;  % adjust based on your point cloud density
shp = alphaShape(points, alpha);

% Plot the reconstructed surface
figure;
plot(shp);
title('Reconstructed Surface using alphaShape');
axis equal;

% Extract triangulated surface from alpha shape
[faces, vertices] = boundaryFacets(shp);
tr = triangulation(faces, vertices);  % Create triangulation object
% 
% Save as STL file
stlwrite(tr, "reconstructed\test_recon\femur_011_target.stl"); % file path for reconstructed mesh (.stl)

disp('STL file saved as reconstructed_surface.stl');
