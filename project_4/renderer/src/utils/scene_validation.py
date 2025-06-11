import numpy as np

def normalize(v):
    """
    Normalize a vector.
    
    Args:
        v (np.ndarray): The vector to normalize.
        
    Returns:
        np.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def is_visible(model_translation,
               camera_position,
               lookat_point,
               world_up_vec,
               fov,
               aspect_ratio,
               near_plane=0.1,
               far_plane=1000.0):
    """
    Check if the model is within the camera's view frustum.
    
    Args:
        model_translation (np.ndarray): The translation of the model in 3D space.
        camera_position (np.ndarray): The position of the camera in 3D space.
        fov (float): The field of view of the camera in degrees.
        aspect_ratio (float): The aspect ratio of the camera.
        
    Returns:
        bool: True if the model is visible, False otherwise.
    """
    if not isinstance(model_translation, np.ndarray):
        model_translation = np.array(model_translation, dtype=np.float32)
    if not isinstance(camera_position, np.ndarray):
        camera_position = np.array(camera_position, dtype=np.float32)
    if not isinstance(lookat_point, np.ndarray):
        lookat_point = np.array(lookat_point, dtype=np.float32)
    if not isinstance(world_up_vec, np.ndarray):
        world_up_vec = np.array(world_up_vec, dtype=np.float32)
    if not isinstance(fov, (int, float)):
        raise ValueError("Field of view (fov) must be a number.")
    if not isinstance(aspect_ratio, (int, float)):
        raise ValueError("Aspect ratio must be a number.")
    if not isinstance(near_plane, (int, float)):
        raise ValueError("Near plane must be a number.")
    if not isinstance(far_plane, (int, float)):
        raise ValueError("Far plane must be a number.")
    if model_translation.shape != (3,):
        raise ValueError("Model translation must be a 3D vector.")
    if camera_position.shape != (3,):
        raise ValueError("Camera position must be a 3D vector.")
    if lookat_point.shape != (3,):
        raise ValueError("Lookat point must be a 3D vector.")
    if world_up_vec.shape != (3,):
        raise ValueError("World up vector must be a 3D vector.")
    if not (0 < fov < 180):
        raise ValueError("Field of view (fov) must be between 0 and 180 degrees.")
    if aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be a positive number.")
    if near_plane <= 0:
        raise ValueError("Near plane must be a positive number.")
    if far_plane <= near_plane:
        raise ValueError("Far plane must be greater than near plane.")
    

    forward_vec = normalize(lookat_point - camera_position)
    right_vec = normalize(np.cross(forward_vec, world_up_vec))
    up_vec = normalize(np.cross(right_vec, forward_vec))

    relative_position = model_translation - camera_position
    z = np.dot(relative_position, forward_vec)
    x = np.dot(relative_position, right_vec)
    y = np.dot(relative_position, up_vec)

    if z < near_plane or z > far_plane:
        return False
    
    fov_rad = np.radians(fov)
    half_height = np.tan(fov_rad / 2.0) * z
    half_width = half_height * aspect_ratio

    return (-half_width <= x <= half_width and
            -half_height <= y <= half_height)