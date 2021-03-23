from simulation import launch
from physics import apply_physics, bounds, max_value_names

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt

# this fixed memory growth problem that terminates the program before training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# -------------------------------training-------------------------------
max_games = 20
num_of_games = 0
training_frequency = 5
randomize_constant = 20  # stops being random after [randomize_constant] many games
previous_x = 0
all_max_ys = []  # so i can see the progression if there is any
all_max_xs = []
overall_training_data = set()

loading_model = False
saving_model = True

# -------------------------------model architecture-------------------------------
model = models.Sequential([
    layers.Flatten(input_shape=(8,)),
    layers.Dense(30, activation='relu'),
    layers.Dense(2, activation='sigmoid')
    # softmax is for classification of many different things
    # (like classifying between 10 different animals)
    # sigmoid compresses the value between 0 and 1 but not for classification
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------checkpoints-------------------------------
checkpoint_dir = '../../rocket_v5_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

if loading_model:
    model = models.load_model('rocket_v5.h5')


class Rocket:
    # TODO: add new components for 3d, everywhere there is x variables, there also need to be z
    def __init__(self):
        # current state data
        self.state = {}
        self.reset_state()

        # training data
        self.training_data = set()

        launch(self, 0)

    def reset_state(self):
        # resets all values for first or next run
        # current state
        self.state['y'] = 0
        self.state['velocity'] = 0
        self.state['acceleration'] = 0
        self.state['has_left_ground'] = False
        self.state['fuel'] = bounds['max_fuel']
        
        # new components for 2d:
        self.state['x'] = int(bounds['max_x'] / 2)
        self.state['x_tilt'] = 0.25  # this is a value from 0 to 1,
        # unit circle from 0-1, where 0 and 1 are 2pi (directly right), 0.5 is directly left, and 0.25 is straight up

        # overall data
        self.state['time'] = 0  # measured in frames, not seconds
        self.state['max_y'] = 0
        self.state['max_x'] = 0
        self.state['max_velocity'] = 0
        self.state['max_acceleration'] = 0

    def update(self, actions, predictions, squished_state):
        # this function updates the current state of the rocket depending on the ai output
        global previous_x

        self.state = apply_physics(self.state, actions)

        use_data = False
        if self.state['acceleration'] > 0:
            use_data = True
        elif self.state['acceleration'] == 0 and self.state['velocity'] > 0:
            use_data = True
        # 1: if the x-distance from the middle value is increasing, set use_data to False
        # 2: if the x-distance from the middle value is decreasing, set use_data to True
        cur_x = abs(self.state['x'] - (0.5 * bounds['max_x']))
        if cur_x <= previous_x:
            # method 2
            use_data = True
        previous_x = cur_x

        if use_data is True:
            # adding state and output to current rocket training data
            data_point = (tuple(squished_state), tuple(predictions))
            self.training_data.add(data_point)

        # updating overall data
        if self.state['acceleration'] > self.state['max_acceleration']:
            self.state['max_acceleration'] = self.state['acceleration']
        if self.state['y'] > self.state['max_y']:
            self.state['max_y'] = self.state['y']
        if self.state['velocity'] > self.state['max_velocity']:
            self.state['max_velocity'] = self.state['velocity']
        if cur_x > self.state['max_x']:
            self.state['max_x'] = cur_x

        # updating crashes
        if not self.state['has_left_ground'] and self.state['y'] > 0:
            print('the rocket has left the ground')
            self.state['has_left_ground'] = True
        elif self.state['has_left_ground'] and self.state['y'] <= 0:
            # the rocket has crashed or took too long
            return self.state, True
        elif self.state['time'] >= bounds['max_time']:
            return self.state, True

        return self.state, False

    def fly(self):
        # this is where the ai will give an output

        actions = {}
        squished_state = self.squish_state()  # accessing current state of rocket for inputs
        predictions = model.predict(np.array([squished_state]))  # generating command based on state
        predictions = predictions[0]  # accessing array of values from 2d array given
        predictions = randomize_output(predictions)  # adding a level of randomness for ai training
        # print('predictions after randomization:', predictions)
        actions['thrust'] = predictions[0]  # accessing the variable needed from the array(s) given
        actions['x_tilt_gain'] = predictions[1]

        return self.update(actions, predictions, squished_state)
        # calling the update function to change the rockets current state

    def has_crashed(self):
        # the rocket has crashed and we are now moving onto another iteration
        # this is where we will fit the model to the training data

        global num_of_games
        global overall_training_data
        global model
        global all_max_ys
        global all_max_xs

        # printing overall data for me
        print('rocket #', str(num_of_games+1), 'crashed')
        print('max y:', self.state['max_y'])
        print('max_x:', self.state['max_x'])
        print('max velocity:', self.state['max_velocity'])
        print('max acceleration:', self.state['max_acceleration'])
        print('time:', self.state['time'])

        all_max_ys.append(self.state['max_y'])
        all_max_xs.append(self.state['max_x'])
        for data in self.training_data:
            overall_training_data.add(data)

        if num_of_games is not 0 and num_of_games % training_frequency is 0 and len(overall_training_data) is not 0:
            # we only train every few games, and not every rocket is added to training data,
            # so we first check if we should train after the current game, and we check if there is data to train with
            train(overall_training_data)
            # resetting the training arrays to not over train
            overall_training_data = set()

        # preparing for next launch
        self.reset_state()

        # resetting rocket training data
        self.training_data = set()

        num_of_games += 1
        if num_of_games >= max_games:
            return
        print('finished game #', num_of_games)

        # continuing to play
        launch(self, num_of_games)

    def squish_state(self):
        # squishing values between 0 and 1
        # all training and prediction values must be between 0 and 1
        left_ground_bool = 0
        if self.state['has_left_ground']:
            left_ground_bool = 1

        squished_array = np.array([left_ground_bool])

        for name in max_value_names:
            max_str = 'max_' + name
            cur_val = self.state[name] / bounds[max_str]
            squished_array = np.append(squished_array, cur_val)

        squished_array = np.append(squished_array, self.state['x_tilt'])

        return squished_array


def train(training_set):
    # trains the model to the current training data and saves it
    global model

    # converting set of tuples of lists into 2 2d lists
    inputs = [list(input_data) for input_data, output_data in training_set]
    outputs = [list(output_data) for input_data, output_data in training_set]

    print(inputs)
    print(outputs)

    model.fit(inputs, outputs, epochs=10, verbose=1, callbacks=[checkpoint_callback])
    if saving_model:
        model.save('rocket_v5.h5')


def randomize_output(predicted_outputs):
    # this adds some randomization to the beginning of training
    # because without it the ai will keep the same output constantly and wont learn
    new_outputs = np.array([])
    for predicted_output in predicted_outputs:
        rand = randint(0, 100)
        random_rate = randomize_constant*(1-num_of_games/randomize_constant)
        if rand < random_rate:
            new_outputs = np.append(new_outputs, abs(1-predicted_output))
        else:
            new_outputs = np.append(new_outputs, predicted_output)

    return new_outputs


myself = Rocket()
# plt.subplot(4, 5, 1)
y = np.linspace(1, len(all_max_ys), len(all_max_ys))
plt.plot(y, all_max_ys, 'o-', color='b')

x = np.linspace(1, len(all_max_xs), len(all_max_xs))
plt.plot(x, all_max_xs, 'o-', color='r')
plt.xlabel("Rocket")
plt.ylabel("Value (blue = y, red = x)")
plt.title('Rocket data')
plt.savefig('rocket_v5.png')
