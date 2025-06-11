import os.path
import json
import torch

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow
from utils.scene_validation import is_visible

from CNN.GeneratorModule import GeneratorModule

class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model = GeneratorModule.load_from_checkpoint("../../CNN_model/model-epoch=314-val_loss=0.00.ckpt")
        self.model.eval().cuda()  # or .to("cuda")

    def min_max_normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def normalize_scene_params(self, scene_params):
        frame = scene_params[0]  # Keep as is

        # Extract components
        model_x, model_y, model_z = scene_params[1:4]
        material_diffuse = scene_params[4:7]  # Leave unchanged
        shininess = scene_params[7]
        light_x, light_y, light_z = scene_params[8:11]

        # Normalize
        model_x = self.min_max_normalize(model_x, -12., 8.)
        model_y = self.min_max_normalize(model_y, -12., 8.)
        model_z = self.min_max_normalize(model_z, -22., -2.)
        shininess = self.min_max_normalize(shininess, 3., 20.)
        light_x = self.min_max_normalize(light_x, -25., 15.)
        light_y = self.min_max_normalize(light_y, -25., 15.)
        light_z = self.min_max_normalize(light_z, -22., 0.)

        # Reconstruct normalized scene_params
        normalized_scene_params = [frame] + [model_x, model_y, model_z] + material_diffuse + [shininess] + [light_x,
                                                                                                            light_y,
                                                                                                            light_z]
        return normalized_scene_params

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        

        # todo: Randomize
        # model_translation = [0.0, 5.0, 0.0]
        # material_diffuse = [1.0, 0.0, 0.0]
        # material_shininess = float(np.random.randint(3, 20))

        lookat_point = (0.0, 0.0, 0.0)
        world_up_vec = (0.0, 1.0, 0.0)
        camera_position = [5.0, 5.0, 15.0]
        fov = 60.0

        model_translation = np.round(np.random.uniform(-7., 13., size=3), decimals=1)
        material_diffuse = np.round(np.random.uniform(0., 1., size=3), decimals=1)
        material_shininess = np.random.randint(3, 20)
        light_position = np.array([
            np.round(np.random.uniform(-20., 20.), decimals=1),
            np.round(np.random.uniform(-20., 20.), decimals=1),
            np.round(np.random.uniform(model_translation[2], camera_position[2]), decimals=1)])

        while(not is_visible(
                model_translation,
                camera_position,
                lookat_point,
                world_up_vec,
                fov,
                self.aspect_ratio,
        )):
            model_translation = np.round(np.random.uniform(-7., 13., size=3), decimals=1)


        relative_model_translation = model_translation - camera_position
        relative_light_position = light_position - camera_position
        scene_params = [self.frame] + relative_model_translation.tolist() + material_diffuse.tolist() + [material_shininess] + relative_light_position.tolist()
        
        #scene_params = [self.frame] + model_translation.tolist() + material_diffuse.tolist() + [material_shininess] + light_position.tolist()

        ########################################################
        #CNN part
        normalized_scene_params = self.normalize_scene_params(scene_params)
        device = next(self.model.parameters()).device
        normalized_scene_params_tensor = torch.tensor(normalized_scene_params[1:11], dtype=torch.float32, device=device)
        normalized_scene_params_tensor = normalized_scene_params_tensor.unsqueeze(0)
        texture = self.model(normalized_scene_params_tensor).cpu().squeeze(0)
        texture = texture.permute(1, 2, 0).detach().numpy()
        texture = np.flipud((texture + 1) / 2)
        texture = texture.astype('f4')
        texture = np.ascontiguousarray(texture)

        # Create a texture from the neural network output (128x128)
        self.texture = self.ctx.texture((128, 128), 3, texture, dtype='f4')
        self.texture.use(location=0)  # Bind texture to texture unit 0


        lookat = Matrix44.look_at(
                    camera_position,
                    lookat_point,
                    world_up_vec,
                )
        proj = Matrix44.perspective_projection(fov, self.aspect_ratio, 0.1, 1000.0)

        model_matrix = Matrix44.from_translation(model_translation)
        

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(os.path.join(self.output_path, "img"), f'image_{self.frame:04}.png'))
            with open(os.path.join(self.output_path, f'scene_params.csv'), 'a+') as f:
                f.write(','.join(map(str, scene_params)) + '\n')
            self.frame += 1
            if self.frame % 100 == 0:
                print(f'Progress: {int(self.frame / self.max_frames * 100.)}%', flush=True)

        if self.frame >= self.max_frames:
            self.wnd.close()
            return
