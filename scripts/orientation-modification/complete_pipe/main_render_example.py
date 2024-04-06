import os
import sys
import torch
import os
import torch
import matplotlib.pyplot as plt
import cv2

# Util function for loading meshes
from plyfile import PlyData, PlyElement
from pytorch3d.renderer.mesh.textures import Textures
import numpy as np 

#Load function render 
from render_torch import Image_render

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


output_dir = ""
name = "Image"

# Scaling factor mesh 
scaling = 10 # define to scale mesh to the camera view 
img_size = 512 # image desired size

# load mesh in ply 
plydata = PlyData.read('mesh.ply')

# texture should be normalized in the rage [0 255]
texture_rgb =[]
texture_rgb.append(plydata['vertex']['red']/255)
texture_rgb.append(plydata['vertex']['green']/255)
texture_rgb.append(plydata['vertex']['blue']/255)

texture_rgb = torch.tensor(np.transpose(texture_rgb)).to(torch.float32)
texture_rgb = texture_rgb[None,:,:]

faces = []
for i in range(len(plydata['face'].data['vertex_indices'][:])):
  faces.append(np.transpose(plydata['face'].data['vertex_indices'][i]))

faces = torch.tensor(faces)
faces = faces[:,:]
print(np.shape(faces))

verts = []
verts.append(plydata['vertex']['x'])
verts.append(plydata['vertex']['y'])
verts.append(plydata['vertex']['z'])
verts = torch.tensor(np.transpose(verts))
verts = verts[:,:]

# Frontal, left and right render

image_f, image_render = Image_render(verts, faces, texture_rgb, "frontal", device, size = img_size, scaling=scaling )
image_l, image_render = Image_render(verts, faces, texture_rgb, "left", device, size = img_size ,scaling = scaling)
image_r, image_render = Image_render(verts, faces, texture_rgb, "right", device, size = img_size ,scaling = scaling)


img_w = cv2.cvtColor(image_f[0,:,:,:].cpu().numpy()*255, cv2.COLOR_BGR2RGB) 
cv2.imwrite(os.path.join(output_dir, 'Image_render_frontal_'+  name + '.jpg'), img_w)

img_w = cv2.cvtColor(image_l[0,:,:,:].cpu().numpy()*255, cv2.COLOR_BGR2RGB) 
cv2.imwrite(os.path.join(output_dir, 'Image_render_left_'+  name + '.jpg'), img_w)

img_w = cv2.cvtColor(image_r[0,:,:,:].cpu().numpy()*255, cv2.COLOR_BGR2RGB) 
cv2.imwrite(os.path.join(output_dir, 'Image_render_right_'+  name + '.jpg'), img_w)



'''If same mshe wants to be rotated bases on a vectors of rotations modify the function to worch with batches using the following : # Set batch size - this is the number of different viewpoints from which we want to render the mesh.
batch_size =  #number batches 

# Create a batch of meshes by repeating the cow mesh and associated textures. 
# Meshes has a useful `extend` method which allows us do this very easily. 
# This also extends the textures. 
meshes = mesh.extend(batch_size)

# Get a batch of viewing angles. 
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)

# All the cameras helper methods support mixed type inputs and broadcasting. So we can 
# view the camera from the same distance and specify dist=2.7 as a float,
# and then specify elevation and azimuth angles for each viewpoint as tensors. 
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Move the light back in front of the cow which is facing the -z direction.
lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

# We can pass arbitrary keyword arguments to the rasterizer/shader via the renderer
# so the renderer does not need to be reinitialized if any of the settings change.
images = renderer(meshes, cameras=cameras, lights=lights)

image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)

'''

