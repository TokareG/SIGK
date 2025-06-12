from collections import namedtuple
from enum import Enum

import moderngl_window

from phong_window import PhongWindow

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG_DATASET = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../../res/",
            "--max_frames=10",
            "--dataset",
        ],
        PhongWindow
    )

    PHONG_REF = Task(
        [
            "--shaders_dir_path=../resources/shaders/phong",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            "--output_path=../../results/",
            "--scene_data_path=../resources/scene_params.csv",
            "--max_frames=10",
        ],
        PhongWindow
    )



if __name__ == '__main__':
    task = TaskType.PHONG_REF
    moderngl_window.run_window_config(task.window_cls, args=task.window_args)
