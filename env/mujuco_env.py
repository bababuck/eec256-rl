import mujoco
from rope_xml import rope_xml

model = mujoco.MjModel.from_xml_string(rope_xml)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
media.show_image(renderer.render())