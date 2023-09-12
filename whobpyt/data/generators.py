

import torch


def gen_cube(device = torch.device("cpu")):
    """
    # Generates a synthetic dataset based on the structure of a cube
    # Not biologically realistic, but useful for demos and testing
    """
    
    # %%
    # Generating a physically possible (in 3D Space) Structural Connectivity Matrix
    # ---------------------------------------------------
    #
    # First, get corner points on a cube and project onto a sphere
    square_points = torch.tensor([[1.,1.,1.],
                                  [-1.,1.,1.],
                                  [1.,-1.,1.],
                                  [-1.,-1.,1.],
                                  [1.,1.,-1.],
                                  [-1.,1.,-1.],
                                  [1.,-1.,-1.],
                                  [-1.,-1.,-1.]]).to(device)
    sphere_points = square_points / torch.sqrt(torch.sum(torch.square(square_points), axis = 1)).repeat(3, 1).t()

    # Second, find the distance between all pairs of points
    num_regions = 8
    dist_mtx = torch.zeros(num_regions, num_regions).to(device)
    for x in range(num_regions):
        for y in range(num_regions):
            dist_mtx[x,y]= torch.linalg.norm(sphere_points[x,:] - sphere_points[y,:])

    # Third, Structural Connectivity defined to be 1/dist and remove self-connection values
    SC_mtx = 1/dist_mtx
    for z in range(num_regions):
        SC_mtx[z,z] = 0.0

    # Fourth, Normalize the matrix
    SC_mtx_norm = (1/torch.linalg.matrix_norm(SC_mtx, ord = 2)) * SC_mtx
    Con_Mtx = SC_mtx_norm
    
    
    # %%
    # Generating a Lead Field Matrix
    # ---------------------------------------------------
    #
    # Placing an EEG Electrode in the middle of each cube face. 
    # Then electrode is equally distance from four courner on cube face squre.
    # Assume no signal from further four points. 

    Lead_Field = torch.tensor([[1,1,0,0,1,1,0,0],
                               [1,1,1,1,0,0,0,0],
                               [0,1,0,1,0,1,0,1],
                               [0,0,0,0,1,1,1,1],
                               [1,0,1,0,1,0,1,0],
                               [0,0,1,1,0,0,1,1]], dtype = torch.float).to(device)
    LF_Norm = (1/torch.linalg.matrix_norm(Lead_Field, ord = 2)) * Lead_Field
    
    # %%
    # Generating a "Connectivity Matrix" for Channel Space
    # ---------------------------------------------------
    #
    # Generating a physically possible (in 3D Space) "Channel" Connectivity Matrix
    # That is a theoretical matrix for the EEG SC to be fit to

    # First, get face points on a cube and project onto a sphere
    LF_square_points = torch.tensor([[0.,1.,0.],
                                     [0.,0.,1.],
                                     [-1.,0.,0.],
                                     [0.,0.,-1.],
                                     [1.,0.,0.],
                                     [0.,-1.,0.]])
    # Note: this does nothing as the points are already on the r=1 sphere
    LF_sphere_points = LF_square_points / torch.sqrt(torch.sum(torch.square(LF_square_points), axis = 1)).repeat(3, 1).t()


    # Second, find the distance between all pairs of channel points
    num_channels = 6
    LF_dist_mtx = torch.zeros(num_channels, num_channels).to(device)
    for x in range(num_channels):
        for y in range(num_channels):
            LF_dist_mtx[x,y]= torch.linalg.norm(LF_sphere_points[x,:] - LF_sphere_points[y,:])

    # Third, Structural Connectivity defined to be 1/dist and remove self-connection values
    LF_SC_mtx = 1/LF_dist_mtx
    for z in range(num_channels):
        LF_SC_mtx[z,z] = 0.0

    # Fourth, Normalize the matrix
    LF_SC_mtx_norm = (1/torch.linalg.matrix_norm(LF_SC_mtx, ord = 2)) * LF_SC_mtx
    LF_Con_Mtx = LF_SC_mtx_norm
    
    return {"SC" : SC_mtx_norm, "dist" : dist_mtx, "LF" : LF_Norm, "Source FC" : SC_mtx_norm, "Channel FC" : LF_Con_Mtx}
    
    