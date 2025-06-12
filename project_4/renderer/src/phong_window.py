import os.path
import json

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow
from utils.scene_validation import is_visible

class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        

        self.scene_params = np.loadtxt(self.argv.scene_data_path, delimiter=',') if self.argv.scene_data_path is not None else None

        lookat_point = (0.0, 0.0, 0.0)
        world_up_vec = (0.0, 1.0, 0.0)
        camera_position = [5.0, 5.0, 15.0]
        fov = 60.0

        if self.scene_params is not None:
            data = self.scene_params[self.frame]
            model_translation = data[1:4]  # kolumny 1, 2, 3
            material_diffuse = data[4:7]  # kolumny 4, 5, 6
            material_shininess = data[7]  # kolumna 7
            light_position = data[8:11]
        else:
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

        if not self.dataset:
            img_path = os.path.join(self.output_path, "renderer")
            scene_params = [self.frame] + model_translation.tolist() + material_diffuse.tolist() + [material_shininess] + light_position.tolist()
        else:
            img_path = os.path.join(self.output_path, "img")
            relative_model_translation = model_translation - camera_position
            relative_light_position = light_position - camera_position
            scene_params = [self.frame] + relative_model_translation.tolist() + material_diffuse.tolist() + [material_shininess] + relative_light_position.tolist()
        
        #scene_params = [self.frame] + model_translation.tolist() + material_diffuse.tolist() + [material_shininess] + light_position.tolist()


        lookat = Matrix44.look_at(
                    camera_position,
                    lookat_point,
                    world_up_vec,
                )
        proj = Matrix44.perspective_projection(fov, self.aspect_ratio, 0.1, 1000.0)

        model_matrix = Matrix44.from_translation(model_translation)
        

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(img_path, f'image_{self.frame:04}.png'))
            if self.dataset:
                with open(os.path.join(self.output_path, f'scene_params.csv'), 'a+') as f:
                    f.write(','.join(map(str, scene_params)) + '\n')
            self.frame += 1
            if self.frame % 100 == 0:
                print(f'Progress: {int(self.frame / self.max_frames * 100.)}%', flush=True)

        if self.frame >= self.max_frames:
            self.wnd.close()
            return
