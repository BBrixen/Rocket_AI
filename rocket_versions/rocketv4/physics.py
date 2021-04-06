import math
import numpy as np
from perlin_noise import PerlinNoiseFactory

# -------------------------------physics constants-------------------------------
ACCELERATION_FROM_THRUST = 20  # how much the acceleration increases from thrust
ACCELERATION_FROM_GRAVITY = -9.8
THRUST_PER_UNIT_FUEL = 1  # how much thrust from 1 unit of fuel
TILT_RATIO = 0.2
ACCELERATION_FROM_WIND = 1

# -------------------------------classification-------------------------------
bounds = {'max_acceleration': 100,  # this is also used for preprocessing
          'max_velocity': 1000,
          'max_y': 10000,
          'min_y': 0,
          'max_fuel': 10,
          'max_time': 100,
          'max_x': 10000,
          'min_x': 0,
          'max_z': 10000,
          'min_z': 0,
          'max_wind': 1,
          'min_wind': 0}

all_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x', 'x_tilt', 'x_direction', 'z', 'z_tilt',
                   'z_direction', 'wind', 'wind_direction']
# x_tilt and z_tilt is the direction the rocket is tilted, which is the direction the acceleration is applied
# x_direction and z_direction is the direction the rocket is actually moving, so this determines the velocity components
# before adding acceleration
max_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x', 'z', 'wind']
# the names of all data fields that have a max
min_value_names = ['y', 'x', 'z']  # the names of all data fields that have a min


# -------------------------------environment constants-------------------------------
wind_mags = []
wind_angles = []
rt2 = math.sqrt(2)


def init_env():
    global wind_mags, wind_angles
    x_noise_generator = PerlinNoiseFactory(1, octaves=2, unbias=False)
    z_noise_generator = PerlinNoiseFactory(1, octaves=3, unbias=False)
    div = 10
    x_noise = [(x_noise_generator(i/div)+1)/2 for i in range(bounds['max_time'])]
    z_noise = [(z_noise_generator(i/div)+1)/2 for i in range(bounds['max_time'])]
    wind_mags = []
    wind_angles = []
    for i in range(bounds['max_time']):
        x_point = x_noise_generator(i/div) * rt2
        z_point = x_noise_generator(i/div) * rt2
        wind_mags.append(math.sqrt(x_point ** 2 + z_point ** 2))
        wind_angles.append(math.atan2(z_noise[i], x_noise[i]))


def apply_physics(state, actions):
    # this is called after the ai has made its decision
    # recalculating fuel consumption when low on fuel
    thrust = actions['thrust']
    max_thrust = state['fuel'] * THRUST_PER_UNIT_FUEL
    if thrust >= max_thrust:
        thrust = max_thrust
        state['fuel'] = 0
    else:
        state['fuel'] -= thrust / THRUST_PER_UNIT_FUEL

    # changing angles from tilt gain
    # converting tilt_gain from 0-1 into -0.5 to 0.5 and adding it to current tilt in a given dimension
    state['x_tilt'] += TILT_RATIO * (actions['x_tilt_gain'] - 0.5)
    state['z_tilt'] += TILT_RATIO * (actions['z_tilt_gain'] - 0.5)
    state = apply_unit_circle(state)

    # converting from 0-1 to 0-2pi
    vel_angle_x_projection = state['x_direction'] * 2 * math.pi
    vel_angle_z_projection = state['z_direction'] * 2 * math.pi
    acc_angle_x_projection = state['x_tilt'] * 2 * math.pi
    acc_angle_z_projection = state['z_tilt'] * 2 * math.pi

    # calculating angles for velocity using direction
    vel_z_component = math.cos(vel_angle_z_projection)
    vel_xy_projection = math.sin(vel_angle_z_projection)
    vel_y_component = vel_xy_projection * math.sin(vel_angle_x_projection)
    vel_x_component = vel_xy_projection * math.cos(vel_angle_x_projection)

    # calculating angles for acceleration using tilt
    acc_z_component = math.cos(acc_angle_z_projection)
    acc_xy_projection = math.sin(vel_angle_z_projection)
    acc_y_component = acc_xy_projection * math.sin(acc_angle_x_projection)
    acc_x_component = acc_xy_projection * math.cos(acc_angle_x_projection)

    # calculating x, y, z acceleration from components
    cur_acceleration = thrust * ACCELERATION_FROM_THRUST
    acceleration_y = (cur_acceleration * acc_y_component) + ACCELERATION_FROM_GRAVITY
    acceleration_x = cur_acceleration * acc_x_component
    acceleration_z = cur_acceleration * acc_z_component

    # calculating and storing new magnitude
    cur_acceleration = math.sqrt((acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2))
    cur_acceleration *= np.sign(acceleration_y)
    cur_acceleration = round(cur_acceleration, 10)
    state['acceleration'] = cur_acceleration

    # ----- y -----
    # updating velocity_y, and y:
    velocity_y = state['velocity'] * vel_y_component
    if state['has_left_ground'] or acceleration_y > 0:
        # this keeps the rocket from gaining negative velocity and y while on the ground
        velocity_y += acceleration_y
        velocity_y = round(velocity_y, 10)
        state['y'] += velocity_y

    # ----- x -----
    # updating velocity_x, and x:
    velocity_x = state['velocity'] * vel_x_component
    velocity_x += acceleration_x
    velocity_x = round(velocity_x, 10)
    state['x'] += velocity_x

    # ----- z -----
    # updating velocity_z, and z:
    velocity_z = state['velocity'] * vel_z_component
    velocity_z += acceleration_z
    velocity_z = round(velocity_z, 10)
    state['z'] += velocity_z

    # calculating and storing new velocity magnitude
    cur_velocity = math.sqrt((velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2))
    cur_velocity *= np.sign(velocity_y)
    cur_velocity = round(cur_velocity, 10)
    state['velocity'] = cur_velocity

    # finding x and z angles
    vel_angle_x = math.atan2(velocity_y, velocity_x) / (2 * math.pi)
    xy_projection_magnitude = math.sqrt(velocity_y ** 2 + velocity_x ** 2)
    xy_projection_magnitude *= np.sign(velocity_y)
    vel_angle_z = math.atan2(xy_projection_magnitude, velocity_z) / (2 * math.pi)
    state['x_direction'] = vel_angle_x
    state['z_direction'] = vel_angle_z

    state['time'] += 1

    return apply_bounds(state)  # this can be called earlier if data needs to be changed after applying bounds


def apply_environment(state):
    pass


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
    return apply_unit_circle(state)


def apply_unit_circle(state):
    state['x_tilt'] %= 1  # applies unit circle
    state['z_tilt'] %= 1
    return state
