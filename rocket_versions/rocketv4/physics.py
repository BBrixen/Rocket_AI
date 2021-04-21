import math
import numpy as np
from perlin_noise import PerlinNoiseFactory

# -------------------------------physics constants-------------------------------
# from now on this is measured in meters per frame
# if we want meters per second we must define that 1 frame is 1 second
FORCE_FROM_THRUST = 800  # how much the acceleration increases from thrust, measured in N
THRUST_PER_UNIT_FUEL = 1  # how much thrust from 1 unit of fuel
TILT_RATIO = 0.5  # how much the rocket tilts in frame
FORCE_FROM_WIND = 100  # multiplied by wind (value from 0-1), measured in N
MASS = 20  # kg
ACCELERATION_FROM_GRAVITY = -9.8
FORCE_FROM_GRAVITY = MASS * ACCELERATION_FROM_GRAVITY  # currently -196N
# drag
DRAG_COEFFICIENT = 0.55
RADIUS = 0.1  # measured in meters. radius is 10cm
CROSS_SECTIONAL_AREA = math.pi * RADIUS**2  # this is measured in meters
AIR_DENSITY = 1.225  # kg/m^3

# -------------------------------classification-------------------------------
# terminal velocity moving straight down is:
# math.sqrt((-2*FORCE_FROM_GRAVITY) / (AIR_DENSITY*CROSS_SECTIONAL_AREA*DRAG_COEFFICIENT))
# which is a positive number. if the rocket is moving downwards, its max velocity_y cannot exceed this
bounds = {'max_acceleration': 100,  # this is also used for preprocessing
          'max_velocity': 300,
          'max_y': 10000,
          'min_y': 0,
          'max_fuel': 10,
          'max_time': 100,
          'max_x': 100000,
          'min_x': 0,
          'max_z': 100000,
          'min_z': 0,
          'max_wind': FORCE_FROM_WIND,
          'min_wind': 0}

all_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x', 'x_tilt', 'x_direction', 'z', 'z_tilt',
                   'z_direction', 'wind', 'wind_direction']
# x_tilt and z_tilt is the direction the rocket is tilted, which is the direction the acceleration is applied
# x_direction and z_direction is the direction the rocket is actually moving, so this determines the velocity components
# before adding acceleration
max_value_names = ['acceleration', 'velocity', 'y', 'fuel', 'time', 'x', 'z', 'wind']
# the names of all data fields that have a max
min_value_names = ['y', 'x', 'z']  # the names of all data fields that have a min
angles = ['x_tilt', 'x_direction', 'z_tilt', 'z_direction', 'wind_direction']


# -------------------------------environment constants-------------------------------
wind_mags = []
wind_angles = []
rt2 = math.sqrt(2)


def init_env():
    # i might want to transform the curves to better fit a realistic environment
    global wind_mags, wind_angles
    x_noise_generator = PerlinNoiseFactory(1, octaves=2, unbias=True)
    z_noise_generator = PerlinNoiseFactory(1, octaves=3, unbias=True)
    div = 100
    # 50* instead of 100/2*
    wind_x = [(x_noise_generator(i/div))/rt2 for i in range(bounds['max_time'])]
    wind_z = [(z_noise_generator(i/div))/rt2 for i in range(bounds['max_time'])]
    wind_mags = []
    wind_angles = []
    for i in range(len(wind_x)):
        magnitude = math.sqrt(wind_x[i]**2 + wind_z[i]**2)
        angle = math.atan2(wind_z[i], wind_x[i]) / (2*math.pi)
        angle %= 1
        wind_mags.append(magnitude)
        wind_angles.append(angle)


def apply_physics(state, actions):
    # this is called after the ai has made its decision
    # recalculating fuel consumption when low on fuel

    # actions['thrust'] = 1  # sets the rocket to turbo mode
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
    force_angle_x_projection = state['x_tilt'] * 2 * math.pi
    force_angle_z_projection = state['z_tilt'] * 2 * math.pi
    wind_angle = state['wind_direction'] * 2 * math.pi

    # calculating angles for velocity using direction
    vel_z_component = math.cos(vel_angle_z_projection)
    vel_xy_projection = math.sin(vel_angle_z_projection)
    vel_y_component = vel_xy_projection * math.sin(vel_angle_x_projection)
    vel_x_component = vel_xy_projection * math.cos(vel_angle_x_projection)

    # calculating angles for acceleration using tilt
    force_z_component = math.cos(force_angle_z_projection)
    force_xy_projection = math.sin(force_angle_z_projection)
    force_y_component = force_xy_projection * math.sin(force_angle_x_projection)
    force_x_component = force_xy_projection * math.cos(force_angle_x_projection)

    # calculating wind components
    wind_z_component = math.sin(wind_angle)
    wind_x_component = math.cos(wind_angle)
    wind_z = FORCE_FROM_WIND * state['wind'] * wind_z_component
    wind_x = FORCE_FROM_WIND * state['wind'] * wind_x_component

    # calculating drag force:
    drag = 0.5 * DRAG_COEFFICIENT * CROSS_SECTIONAL_AREA * state['velocity']**2 * AIR_DENSITY
    drag *= -1 * np.sign(state['velocity'])  # this way drag is constantly working against the direction of the rocket

    # calculating x, y, z force from components
    cur_force = thrust * FORCE_FROM_THRUST
    force_y = cur_force * force_y_component + FORCE_FROM_GRAVITY + drag
    force_x = cur_force * force_x_component + wind_x
    force_z = cur_force * force_z_component + wind_z

    # translating force into acceleration
    acceleration_y = force_y / MASS
    acceleration_x = force_x / MASS
    acceleration_z = force_z / MASS

    # calculating and storing new magnitude
    cur_acceleration = math.sqrt((acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2))
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
    cur_velocity = round(cur_velocity, 10)
    state['velocity'] = cur_velocity

    # finding x and z angles
    print('x:', velocity_x)
    print('y:', velocity_y)
    print('z:', velocity_z)
    vel_angle_x = math.atan2(velocity_y, velocity_x) / (2 * math.pi)
    xy_projection_magnitude = math.sqrt(velocity_y ** 2 + velocity_x ** 2)
    vel_angle_z = math.atan2(xy_projection_magnitude, velocity_z) / (2 * math.pi)
    state['x_direction'] = vel_angle_x
    state['z_direction'] = vel_angle_z
    print('r:', xy_projection_magnitude)
    print('angle x:', str(vel_angle_x * math.pi * 2))
    print('angle z:', str(vel_angle_z * math.pi * 2))

    state['time'] += 1

    return apply_bounds(state)  # this can be called earlier if data needs to be changed after applying bounds


def apply_environment(state):
    cur_time = state['time']
    state['wind'] = wind_mags[cur_time]
    state['wind_direction'] = wind_angles[cur_time]
    return state


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
    for angle in angles:
        state[angle] %= 1
    return state
