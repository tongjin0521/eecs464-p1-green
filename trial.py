import numpy as np


def nearest_turn(curr_ang,target_ang):
    front_near_angle = target_ang - curr_ang
    print(front_near_angle / np.pi * 180)
    if abs(front_near_angle) > np.pi:
        front_near_angle =  (2 * np.pi - abs(front_near_angle) ) * -1 * np.sign(front_near_angle)
    back_angle = 0
    if curr_ang > 0:
        back_angle = curr_ang - np.pi
    else:
        back_angle = curr_ang + np.pi
    back_near_angle = target_ang - back_angle
    if abs(back_near_angle) > np.pi:
        back_near_angle =  (2 * np.pi - abs(back_near_angle) ) * -1 * np.sign(back_near_angle)
    if abs(front_near_angle) > abs(back_near_angle):
        return back_near_angle / np.pi * 180,False
    else:
        return front_near_angle / np.pi * 180,True


print(nearest_turn(np.pi, -23 / 180 * np.pi))