def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]

def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    import torch
    # obtain image from observation dictionary returned by ManiSkill environment
    if camera_name is None:
        if "google_robot" in env.unwrapped.robot_uids.uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uids.uid:
            camera_name = "3rd_view_camera"
        elif "panda" in env.unwrapped.robot_uids.uid:
            # img = env.render().detach() # [1, H, W ,C]
            img = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8) # [1, H, W, C]
            # import PIL.Image as Image
            # Image.fromarray(img.squeeze().cpu().numpy()).save("output.jpg", "JPEG")
            return img.to(torch.uint8)
        else:
            raise NotImplementedError()
    img = obs["sensor_data"][camera_name]["rgb"] # [1, H, W, C]
    return img.to(torch.uint8)