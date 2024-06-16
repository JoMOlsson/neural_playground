import os
import bpy
import sys

basedir = os.path.dirname(bpy.data.filepath)
if basedir not in sys.path:
    print("Appending sys path")
    module_path = r"C:\Users\Joakim\AppData\Local\Programs\Python\Python37\Lib\site-packages"
    sys.path.append(basedir)
    sys.path.append(module_path)

import matplotlib as mpl
import numpy as np
import math


def color_fader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1 - mix) * c1 + mix * c2)


def draw_picture(picture, w=1, d=0.5):
    print("displaying picture ... ")

    c1 = '#000000'  # black
    c2 = '#61A4C9'  # blue

    n = picture.shape[0]
    m = picture.shape[1]
    maximum_y = pixel_w * m + pixel_d * (m - 1)
    maximum_z = pixel_w * n + pixel_d * (n - 1)
    x = 0
    input_positions = []
    for i in range(0, m):
        z = maximum_z - (w / 2 + (w + d) * i)
        # z = (w / 2 + (w + d) * i)
        pos = []
        for j in range(0, m):
            y = maximum_y - (w / 2 + (w + d) * j)
            # y = (w / 2 + (w + d) * j)

            v = picture[i, j]
            c_rgb = color_fader(c1, c2, v)   # Get rgb color

            # Add pixel
            bpy.ops.mesh.primitive_cube_add(size=w, location=(x, y, z))
            activeObject = bpy.context.active_object  # Set active object to variable
            mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
            activeObject.data.materials.append(mat)  # add the material to the object
            bpy.context.object.active_material.diffuse_color = (c_rgb[0], c_rgb[1], c_rgb[2], 1)  # change color

            pos.append((x, y, z))
            # Draw neuron connectors
            # neuron_connector(0, y, z, 10, y, z, 0.1)
        input_positions.append(pos)
    return input_positions


def neuron_connector(x1, y1, z1, x2, y2, z2, r):

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=dist,
        location=(dx/2 + x1, dy/2 + y1, dz/2 + z1)
    )

    alpha = math.atan2(dy, dx)
    beta = math.acos(dz/dist)

    bpy.context.object.rotation_euler[1] = beta
    bpy.context.object.rotation_euler[2] = alpha

    # Set material
    activeObject = bpy.context.active_object  # Set active object to variable
    mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
    activeObject.data.materials.append(mat)  # add the material to the object
    bpy.context.object.active_material.diffuse_color = (97/255, 164/255, 201/255, 0.3)  # change color


def draw_neuron_layers(network_size, max_y, max_z, activations, layer_distance=20, neuron_width=1):
    print("Drawing neuron layers ... ")

    c1 = '#000000'  # black
    c2 = '#61A4C9'  # blue

    x = layer_distance
    neuron_positions = []
    for i_layer, num_of_neurons in enumerate(network_size):
        if i_layer and i_layer < len(network_size)-1:
            n_neurons_y = round(math.sqrt(num_of_neurons))
            n_neurons_z = round(math.sqrt(num_of_neurons))
            dy = max_y / (n_neurons_y - 1)
            dz = max_z / (n_neurons_z - 1)
            y = 0
            a_layer = activations[i_layer]

            positions = []
            i_neuron = 0
            for i in range(0, n_neurons_y):
                z = 0
                p = []
                for j in range(0, n_neurons_z):
                    a = a_layer[i_neuron]

                    c_rgb = color_fader(c1, c2, a)  # Get rgb color

                    bpy.ops.mesh.primitive_uv_sphere_add(radius=neuron_width, location=(x, y, z))
                    activeObject = bpy.context.active_object  # Set active object to variable
                    mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
                    activeObject.data.materials.append(mat)  # add the material to the object
                    bpy.context.object.active_material.diffuse_color = (c_rgb[0], c_rgb[1], c_rgb[2], 1)
                    z += dz
                    i_neuron += 1
                    p.append((x, y, z))
                y += dy
                positions.append(p)
            x += layer_distance
            neuron_positions.append(positions)

    return neuron_positions


def draw_output_layer(prediction, layer_position, layer_height, y_max):
    c1 = '#000000'  # black
    c2 = '#61A4C9'  # blue

    x = layer_position
    z = layer_height
    y = y_max

    output_positions = []
    for p_val in prediction:
        c_rgb = color_fader(c1, c2, p_val)  # Get rgb color

        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x, y, z))
        activeObject = bpy.context.active_object  # Set active object to variable
        mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        activeObject.data.materials.append(mat)  # add the material to the object
        # bpy.context.object.active_material.diffuse_color = (97 / 255, 164 / 255, 201 / 255, p_val)  # change color
        bpy.context.object.active_material.diffuse_color = (c_rgb[0], c_rgb[1], c_rgb[2], 1)   # change color
        output_positions.append((x, y, z))
        y -= y_max / (len(prediction) - 1)
    return output_positions


def clean_blender():
    print('cleaning blender environment ... ')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def normalize(picture):
    d_max = np.max(picture)
    d_min = np.min(picture)

    picture = (2*(picture - d_min)) / (d_max - d_min)
    return picture


if __name__ == '__main__':
    # Load data
    cur_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(cur_path, 'data_dump')
    data = np.load(os.path.join(data_path, 'data.npy'), allow_pickle=True)

    # Extract data
    pic = data.item().get('picture')
    theta = data.item().get('theta')
    net_size = data.item().get('network_size')
    activation_values = data.item().get('a')
    neuron_values = data.item().get('z')
    pred = data.item().get('prediction')

    # Resize theta
    for iLayer, t in enumerate(theta):
        t = t.reshape(net_size[iLayer] + 1, net_size[iLayer + 1])
        theta[iLayer] = t[1:, :]  # Skip bias unit

    # Normalize pixels (0-1)
    pic = normalize(pic)

    # Calculate max y and z value
    pixel_w = 1
    pixel_d = 0.5
    maxy = pixel_w * pic.shape[1] + pixel_d * (pic.shape[1] - 1)
    maxz = pixel_w * pic.shape[0] + pixel_d * (pic.shape[0] - 1)
    l_dist = 20

    # ---- Draw ----
    clean_blender()
    input_positions = draw_picture(pic, pixel_w, pixel_d)
    neuron_pos = draw_neuron_layers(net_size, maxy, maxz, activation_values, layer_distance=l_dist)
    P = neuron_pos[-1][0][0]
    output_positions = draw_output_layer(pred, P[0]+l_dist, maxz/2, maxy)

    # Collect positions of all neurons
    positions = []
    positions.append(input_positions)
    positions.extend(neuron_pos)
    positions.append(output_positions)
