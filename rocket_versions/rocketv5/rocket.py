from simulation import launch
from physics import *

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt
import time

# this fixed memory growth problem that terminates the program before training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# -------------------------------training-------------------------------
MAX_GAMES = 10
num_of_games = 0
TRAINING_FREQUENCY = 5
RANDOMIZE_CONSTANT = MAX_GAMES  # stops being random after [randomize_constant] many games
previous_x = 0
previous_z = 0
all_max_ys = []  # so i can see the progression if there is any
all_max_xs = []
all_max_zs = []
overall_training_data = set()

# timing
decision_times = []
frame_times = []
prev_time = None

MULTIPLE_AI = True  # if false, there is only 1 ai
LOADING_MODEL = True
SAVING_MODEL = True
TRAINING_METHOD = 2
# method 1: set training_data to false if the rocket is moving away from the vertical origin
# method 2: set training_data to true if the rocket it moving towards the vertical origin

# -------------------------------model architecture-------------------------------
INPUT_SIZE = len(all_value_names) + 1  # +1 to account for has_left_ground
if MULTIPLE_AI:
    engine_model = models.Sequential([
        layers.Flatten(input_shape=(INPUT_SIZE,)),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        # softmax is for classification of many different things
        # (like classifying between 10 different animals)
        # sigmoid compresses the value between 0 and 1 but not for classification
    ])
    engine_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         # categorical_crossentropy for 1 AI, sparse_categorical_crossentropy for multiple AIs
                         metrics=['accuracy'])

    x_tilt_model = models.Sequential([
        layers.Flatten(input_shape=(INPUT_SIZE,)),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        # softmax is for classification of many different things
        # (like classifying between 10 different animals)
        # sigmoid compresses the value between 0 and 1 but not for classification
    ])
    x_tilt_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         # categorical_crossentropy for 1 AI, sparse_categorical_crossentropy for multiple AIs
                         metrics=['accuracy'])

    z_tilt_model = models.Sequential([
        layers.Flatten(input_shape=(INPUT_SIZE,)),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        # softmax is for classification of many different things
        # (like classifying between 10 different animals)
        # sigmoid compresses the value between 0 and 1 but not for classification
    ])
    z_tilt_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         # categorical_crossentropy for 1 AI, sparse_categorical_crossentropy for multiple AIs
                         metrics=['accuracy'])

    if LOADING_MODEL:
        engine_model = models.load_model('rocket_engine_v5.h5')
        x_tilt_model = models.load_model('rocket_x_tilt_v5.h5')
        z_tilt_model = models.load_model('rocket_z_tilt_v5.h5')
else:
    model = models.Sequential([
        layers.Flatten(input_shape=(INPUT_SIZE,)),
        layers.Dense(30, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(3, activation='sigmoid')
        # softmax is for classification of many different things
        # (like classifying between 10 different animals)
        # sigmoid compresses the value between 0 and 1 but not for classification
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  # categorical_crossentropy for 1 AI, sparse_categorical_crossentropy for multiple AIs
                  metrics=['accuracy'])
    if LOADING_MODEL:
        model = models.load_model('rocket_v5.h5')


class Rocket:
    def __init__(self):
        # current state data
        self.state = {}
        self.reset_state()
        # training data
        self.training_data = set()

        launch(self, 0, MULTIPLE_AI)

    def reset_state(self):
        # resets all values for first or next run
        self.state['y'] = 0
        self.state['velocity'] = 0
        self.state['acceleration'] = 0
        self.state['has_left_ground'] = False
        self.state['fuel'] = bounds['max_fuel']

        # x axis
        self.state['x'] = int(bounds['max_x'] / 2)
        self.state['x_tilt'] = 0.25  # this is a value from 0 to 1,
        self.state['x_direction'] = 0.25
        # unit circle from 0-1, where 0 and 1 are 2pi (directly right), 0.5 is directly left, and 0.25 is straight up

        # z axis
        self.state['z'] = int(bounds['max_z'] / 2)
        self.state['z_tilt'] = 0.25
        self.state['z_direction'] = 0.25

        # overall data
        self.state['time'] = 0  # measured in frames, not seconds
        self.state['max_y'] = 0
        self.state['max_x'] = 0
        self.state['max_z'] = 0
        self.state['max_velocity'] = 0
        self.state['max_acceleration'] = 0

        # environmental data
        init_env()
        self.state['wind'] = 0
        self.state['wind_direction'] = 0

    def update(self, actions, predictions, squished_state):
        # this function updates the current state of the rocket depending on the ai output
        global previous_x, previous_z, prev_time

        cur_time = time.perf_counter()
        if prev_time is not None:
            diff = cur_time-prev_time
            frame_times.append(diff)
        prev_time = cur_time

        self.state = apply_physics(self.state, actions)  # big boi math

        use_data = False
        if self.state['acceleration'] > 0:
            use_data = True
        elif self.state['acceleration'] == 0 and self.state['velocity'] > 0:
            use_data = True
        # 1: if the x-distance from the middle value is increasing, set use_data to False
        # 2: if the x-distance from the middle value is decreasing, set use_data to True
        cur_x = abs(self.state['x'] - (0.5 * bounds['max_x']))
        cur_z = abs(self.state['z'] - (0.5 * bounds['max_z']))
        if TRAINING_METHOD == 1:
            if cur_x > previous_x or cur_z > previous_z:
                use_data = False
        else:
            # method 2
            if cur_x <= previous_x and cur_z <= previous_z:
                use_data = True
        previous_x = cur_x
        previous_z = cur_z

        if use_data is True:
            # adding state and output to current rocket training data
            data_point = (tuple(squished_state), tuple(predictions))
            self.training_data.add(data_point)

        # updating overall data
        if self.state['acceleration'] > self.state['max_acceleration']:
            self.state['max_acceleration'] = self.state['acceleration']
        if self.state['velocity'] > self.state['max_velocity']:
            self.state['max_velocity'] = self.state['velocity']
        if self.state['y'] > self.state['max_y']:
            self.state['max_y'] = self.state['y']
        if cur_x > self.state['max_x']:
            self.state['max_x'] = cur_x
        if cur_z > self.state['max_z']:
            self.state['max_z'] = cur_z

        # updating crashes
        if not self.state['has_left_ground'] and self.state['y'] > 0:
            self.state['has_left_ground'] = True
        elif self.state['has_left_ground'] and self.state['y'] <= 0:
            return self.state, True
        elif self.state['time'] >= bounds['max_time']:
            return self.state, True

        return self.state, False

    def fly(self):

        self.state = apply_environment(self.state)

        # this is where the ai will give an output
        squished_state = self.squish_state()  # accessing current state of rocket for inputs
        predictions = []

        time_before_predictions = time.perf_counter()

        if MULTIPLE_AI:
            engine_prediction = engine_model.predict(np.array([squished_state]))  # generating command based on state
            engine_prediction = engine_prediction[0][0]  # accessing array of values from 2d array given
            predictions.append(engine_prediction)

            x_tilt_prediction = x_tilt_model.predict(np.array([squished_state]))
            x_tilt_prediction = x_tilt_prediction[0][0]
            predictions.append(x_tilt_prediction)

            z_tilt_prediction = z_tilt_model.predict(np.array([squished_state]))
            z_tilt_prediction = z_tilt_prediction[0][0]
            predictions.append(z_tilt_prediction)
        else:
            predictions = model.predict(np.array([squished_state]))  # generating command based on state
            predictions = predictions[0]  # accessing array of values from 2d array given

        time_after_predictions = time.perf_counter()
        diff = time_after_predictions - time_before_predictions
        decision_times.append(diff)

        predictions = randomize_output(predictions)  # adding a level of randomness for ai training

        actions = {'thrust': predictions[0],
                   'x_tilt_gain': predictions[1],
                   'z_tilt_gain': predictions[2]}

        # calling the update function to change the rockets current state
        # inside update, we will use the environmental values from before and applies the physics to the rocket
        return self.update(actions, predictions, squished_state)

    def has_crashed(self):
        # the rocket has crashed and we are now moving onto another iteration
        # this is where we will fit the model to the training data

        global num_of_games
        global overall_training_data
        global all_max_ys, all_max_xs, all_max_zs

        # printing overall data for me
        print('rocket #', str(num_of_games + 1), 'crashed')
        print('max y:', self.state['max_y'])
        print('max_x:', self.state['max_x'])
        print('max_z:', self.state['max_z'])
        print('max velocity:', self.state['max_velocity'])
        print('max acceleration:', self.state['max_acceleration'])
        print('time:', self.state['time'])

        all_max_ys.append(self.state['max_y'])
        all_max_xs.append(self.state['max_x'])
        all_max_zs.append(self.state['max_z'])

        # adding all values from current training data into overall training data
        overall_training_data = overall_training_data.union(self.training_data)

        if num_of_games is not 0 and num_of_games % TRAINING_FREQUENCY is 0 and len(overall_training_data) is not 0:
            # we only train every few games, and not every rocket is added to training data,
            train(overall_training_data)
            # resetting the training arrays to not over train
            overall_training_data = set()

        # preparing for next launch
        self.reset_state()
        self.training_data = set()

        num_of_games += 1
        if num_of_games >= MAX_GAMES:
            return
        print('finished game #', num_of_games)

        # continuing to play
        launch(self, num_of_games, MULTIPLE_AI)

    def squish_state(self):
        # squishing values between 0 and 1
        left_ground_bool = 0
        if self.state['has_left_ground']:
            left_ground_bool = 1

        squished_array = np.array([left_ground_bool])

        for name in all_value_names:
            max_str = 'max_' + name
            cur_val = self.state[name]
            # the only values without a max are tilt and directions, which are already between 0 and 1
            if max_str in bounds:
                cur_val /= bounds[max_str]
            squished_array = np.append(squished_array, cur_val)

        return squished_array


def train(training_set):
    # trains the model to the current training data and saves it
    global engine_model, x_tilt_model, z_tilt_model, model

    # converting set of tuples of lists into 2 2d lists
    inputs = [list(input_data) for input_data, output_data in training_set]
    all_outputs = [list(output_data) for input_data, output_data in training_set]

    if MULTIPLE_AI:
        engine_outputs = [i[0] for i in all_outputs]
        x_tilt_outputs = [i[1] for i in all_outputs]
        z_tilt_outputs = [i[2] for i in all_outputs]

        engine_model.fit(inputs, engine_outputs, epochs=10, verbose=1)
        x_tilt_model.fit(inputs, x_tilt_outputs, epochs=10, verbose=1)
        z_tilt_model.fit(inputs, z_tilt_outputs, epochs=10, verbose=1)

        if SAVING_MODEL:
            engine_model.save('rocket_engine_v5.h5')
            x_tilt_model.save('rocket_x_tilt_v5.h5')
            z_tilt_model.save('rocket_z_tilt_v5.h5')
    else:
        model.fit(inputs, all_outputs, epochs=10, verbose=1)
        if SAVING_MODEL:
            model.save('rocket_v5.h5')


def randomize_output(predicted_outputs):
    # this adds some randomization to the beginning of training
    # because without it the ai will keep the same output constantly and wont learn
    new_outputs = np.array([])
    for predicted_output in predicted_outputs:
        rand = randint(0, 100)
        random_rate = RANDOMIZE_CONSTANT * (1 - num_of_games / RANDOMIZE_CONSTANT)
        if rand < random_rate:
            new_outputs = np.append(new_outputs, abs(1 - predicted_output))
        else:
            new_outputs = np.append(new_outputs, predicted_output)

    return new_outputs


al = Rocket()

total = 0
counter = 0
for t in decision_times:
    total += t
    counter += 1
if counter == 0:
    print('no decision times to calculate')
    counter = 1
avg = total / counter
print('V5 | average time for 1 decision:', avg)
total = 0
counter = 0
for t in frame_times:
    total += t
    counter += 1
if counter == 0:
    print('no frame times to calculate')
    counter = 1
avg = total / counter
print('V5 | average time for 1 loop: ', avg)

# graphing
y = np.linspace(1, len(all_max_ys), len(all_max_ys))
plt.plot(y, all_max_ys, 'o-', color='g')

x = np.linspace(1, len(all_max_xs), len(all_max_xs))
plt.plot(x, all_max_xs, 'o-', color='r')

z = np.linspace(1, len(all_max_zs), len(all_max_zs))
plt.plot(z, all_max_zs, 'o-', color='b')
plt.xlabel("Rocket Iteration")
plt.ylabel("Value r, g, b = x, y, z\n(m)")
if MULTIPLE_AI:
    plt.title('Multiple AIs')
    plt.savefig('multiple_ais_v5.png')
else:
    plt.title('Singular AI')
    plt.savefig('singular_ai_v5.png')
