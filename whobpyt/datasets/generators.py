import torch
import numpy as np


def gen_cube(device = torch.device("cpu")):
    """"Generates a synthetic dataset based on the structure of a cube. Not biologically realistic, but useful for demos and testing.

    Args:
        device (torch.device, optional): Specify the PyTorch device to use, e.g., CPU, GPU, etc. Defaults to torch.device("cpu").

    Returns:
        dict: A deictionary containing the following keys:
            "SC" (torch.tensor): A synthetic structural connectivity matrix
            "dist" (torch.tensor): A matrix of distances between regions
            "LF" (torch.tensor): A normalized lead field matrix
            "Source FC" (torch.tensor): A synthetic Functional Connectivity Matrix
            "Channel FC" (torch.tensor): A synthetic Functional Connectivity Matrix for Channel Space
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
    

def syntheticSC(numRegions, seed = None, maxConDist = 50):
    """Returns a synetheic Structural Connectivity Matrix with associated region locations in 3D Space

    Args:
        numRegions (int): The number of regions in the connectome (must be an even number).
        seed (int, optional): value to use as np.random.seed() for reproducability.. Defaults to None.
        maxConDist (int, optional): The max distance between regions such that less than this distance there can still be a connection strength. May wish to scale this based on `numRegions`. Defaults to 50.

    Raises:
        ValueError: If `numRegions` is not an even number.

    Returns:
        con (np.array): Structural Connectivity Matrix
        loc (list): List of region locations in 3D space
    """    
    # Recommendation: try different seeds and pick good SC matrices based on visual inspection - using nilearn.plotting.view_connectome(con, loc).
    
    if seed != None:
        np.random.seed(seed)
    
    regionsPerHemi = numRegions//2
    if (regionsPerHemi*2 != numRegions):
        raise ValueError("numRegions should be an even number.")
    
    loc = [] # List of region locations in 3D space
    for x in range(regionsPerHemi):
        # Divide hemisphere of brain into quadrants and randomly select region points in each quadrant
        if x <= regionsPerHemi//4:
            loc.append([5+np.random.rand()*40, np.random.rand()*65-15, np.random.rand()*40+20]) #side, forward, height
        elif x <= 2*(regionsPerHemi//4):
            loc.append([5+np.random.rand()*40, np.random.rand()*65-15, np.random.rand()*40-20]) #side, forward, height
        elif x <= 3*(regionsPerHemi//4):
            loc.append([5+np.random.rand()*40, np.random.rand()*65-80, np.random.rand()*40+20]) #side, forward, height
        else:
            loc.append([5+np.random.rand()*40, np.random.rand()*65-80, np.random.rand()*40-20]) #side, forward, height
    for x in range(regionsPerHemi):
        # Make the brain symmetric
        loc.append([-loc[x][0], loc[x][1], loc[x][2]])

    con = np.zeros((regionsPerHemi*2,regionsPerHemi*2)) # Structural Connectivity Matrix
    for x in range(len(loc)):
        for y in range(len(loc)):
            if x == y:
                continue
            dist = np.linalg.norm(np.array(loc[x]) - np.array(loc[y])) #Distance between two regions
            if dist < maxConDist:
                # If the distance between two regions is less than maxConDist, then connection strenth is calculated as follows
                con[x,y] = (maxConDist-dist)/maxConDist
    
    return con, loc
