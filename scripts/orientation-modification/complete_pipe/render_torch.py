import torch
torch.cuda.empty_cache()

import sys, getopt, os

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.mesh.textures import Textures

import torchvision.transforms.functional as F
from torchvision import transforms

import torchvision

def mse_torch(imageA, imageB):
  	# the 'Mean Squared Error' between the two images is the
  	# sum of the squared difference between the two images;
  	# NOTE: the two images must have the same dimension
  	err = torch.sum(torch.pow(imageA.type(torch.FloatTensor) - imageB.type(torch.FloatTensor), 2))

  	err /= float(imageA.shape[1] * imageA.shape[2])
  	
  	# return the MSE, the lower the error, the more "similar"
  	# the two images are
  	return err

def Renderer_3DFace(device, size = 2048, dist =30, elev=0, azim = 0, face_pixel= 8, blur_radius= 0.0 ):
 
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z
    # Control distance or posititon of the camera, azim move camera pose as we change the yaw, elve move pitch

    R, T = look_at_view_transform(dist, elev, azim) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # sizexsize. 
    #As we are rendering images for having accurate reconstruction we will set faces_per_pixel=8 ( Number of faces to save per pixel, returning the nearest faces_per_pixel points along the z-axis.)
    # and blur_radius=1 .0, where used to expand the face bounding boxes for rasterization. Setting 
    # blur radius results in blurred edges around the shape instead of a hard boundary. Set to 0 for no blur. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 

    raster_settings = RasterizationSettings(
        image_size=size, 
        blur_radius= blur_radius, 
        faces_per_pixel=face_pixel, 
    )
    
    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    # 
    lights = PointLights(device=device,ambient_color=torch.tensor([[1.0, 1.0, 1.0]], device= device), diffuse_color = torch.tensor([[0.8, 0.8, 0.8]], device= device), specular_color=torch.tensor([[0.0, 0.0, 0.0]],device= device), location=torch.tensor([[2.0, 2.0, -2.0]],device= device))
    
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    return renderer
    
def Image_render(verts, faces, texture, view, device, size = 2048, scaling = 1 ):

    #Render 3D mesh
    tex = Textures(verts_rgb= texture/255) 
    faces = torch.tensor(np.int16(faces))
    verts= verts.to(device)
    verts = verts*scaling # scale mesh to camera view
    
    pi = np.pi# 3.1415927410125732 
    rad_x = 11*pi/180;
    Rx = torch.tensor([[1,0,0] ,[0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
    Rx= Rx.float()
    
    print(Rx.dtype)
    print(verts.dtype)
    verts = torch.matmul(verts , Rx.to(device) )
    
    # Mesh object
    mesh = Meshes(verts=[verts], faces=[faces.to(device)], textures= tex.to(device))
    
    # Render view 
    if view == "frontal":
        render_f = Renderer_3DFace(device, size = 2048, dist =30, elev=0, azim = 0, face_pixel= 8, blur_radius= 0.0 )
        image_render = render_f(mesh)
           
    if view == "left":
        render_l = Renderer_3DFace(device, size = 2048, dist =20, elev=0, azim = -85, face_pixel= 8, blur_radius= 0.0 )
        image_render = render_l(mesh)
        
    if view == "right":
        render_r = Renderer_3DFace(device, size = 2048, dist =20, elev=0, azim = 85, face_pixel= 8, blur_radius= 0.0 )
        image_render = render_r(mesh)
        
    image_render_permute = torch.permute(image_render[:,:,:,:3], (0, 3, 1,2))
    tensor_s = image_render_permute.size()
    desired_size=224
    
    image_render_batch = torch.ones((tensor_s[0], tensor_s[1], desired_size, desired_size))
     
    for i in range(0,image_render.shape[0]):
     
        pos = torch.nonzero(torch.sum(image_render_permute[i,:,:,:],dim=0) != 3)
              
        x_min =min(pos[:,0])
        x_max = max(pos[:,0])
        y_min = min(pos[:,1])
        y_max = max(pos[:,1])
        
        image_crop = image_render_permute[i,:3, x_min : x_max, y_min : y_max]
        shape_t = image_crop.size()
        old_size = (shape_t[1], shape_t[2])  # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        img = torchvision.transforms.functional.resize(image_crop, (new_size[0], new_size[1]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
      
        image_render_batch[i,:,top:desired_size-bottom,left:desired_size-right] = img #torchvision.transforms.functional.pad(img,(left, top, right, bottom))
        
    resized_images  =  torch.permute(image_render_batch, (0, 2, 3,1))
           
    
    return resized_images,  image_render_batch
    
        
    