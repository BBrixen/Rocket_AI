import math
import numpy as np

# -------------------------------physics constants-------------------------------
acceleration_from_thrust = 20  # how much the acceleration increases from thrust
acceleration_from_gravity = -9.8
thrust_per_unit_fuel = 1  # how much thrust from 1 unit of fuel
tilt_ratio = 0.1
bounds = {'max_acceleration': 100,  # this is also used for preprocessing
          'max_velocity': 1000,
          'max_y': 10000,
          'min_y': 0,
          'max_fuel': 10,
          'max_time': 100,
          'max_x': 10000,
          'min_x': 0}

all_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x', 'x_tilt']
max_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x']
# the names of all data fields that have a max
min_value_names = ['y', 'x']  # the names of all data fields that have a min


def apply_physics(state, actions):
    # recalculating fuel consumption when low on fuel
    thrust = actions['thrust']
    max_thrust = state['fuel'] * thrust_per_unit_fuel
    if thrust >= max_thrust:
        thrust = max_thrust
        state['fuel'] = 0
    else:
        state['fuel'] -= thrust / thrust_per_unit_fuel

    state['x_tilt'] += tilt_ratio*(actions['x_tilt_gain'] - 0.5)  # x_tilt_gain is between 0 and 1,
    # so we change it to be between -0.5 and +0.5 and add it to the current x_tilt
    state['x_tilt'] %= 1  # applies unit circle

    # TODO: i need to store the actual angle of the rocket's velocity and acceleration separately from x-tilt,
    # because the rocket can have a different angle of velocity than the x-tilt, so this is just wrong.
    # rocket falling down at angle of 250 degrees could have an x-tilt of 0.25, in which case
    # this code just assumes that the rocket is flying up at an angle of 0.5pi radians
    angle = state['x_tilt'] * 2 * math.pi  # this angle is stored in radians, ranging from 0-2pi
    x_component = math.cos(angle)
    y_component = math.sin(angle)  # multiply by these for correct components

    cur_acceleration = thrust * acceleration_from_thrust
    acceleration_y = (cur_acceleration * y_component) + acceleration_from_gravity
    acceleration_x = cur_acceleration * x_component

    cur_acceleration = math.sqrt((acceleration_x**2 + acceleration_y**2))
    cur_acceleration *= np.sign(acceleration_y)
    state['acceleration'] = cur_acceleration

    # ----- y -----
    # updating velocity, and y:
    velocity_y = state['velocity'] * y_component
    if state['has_left_ground'] or acceleration_y > 0:
        # this keeps the rocket from gaining negative velocity and y while on the ground
        velocity_y += acceleration_y
        state['y'] += velocity_y

    # ----- x -----
    # updating velocity, and x:
    velocity_x = state['velocity'] * x_component
    velocity_x += acceleration_x
    state['x'] += velocity_x

    cur_velocity = math.sqrt((velocity_x**2 + velocity_y**2))
    cur_velocity *= np.sign(velocity_y)
    state['velocity'] = cur_velocity

    state['time'] += 1

    return apply_bounds(state)  # this can be called earlier if data needs to be changed after applying bounds


def apply_bounds(state):
    # keeping values at or under max
    for name in max_value_names:
        max_str = 'max_' + name
        if state[name] > bounds[max_str]:
            state[name] = bounds[max_str]

    # keeping values at or above min
    # currently all min values are 0, but if this changes i might have to rework the squish state function
    for name in min_value_names:
        min_str = 'min_' + name
        if state[name] < bounds[min_str]:
            state[name] = bounds[min_str]
    return state
