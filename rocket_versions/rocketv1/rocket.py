from simulation import launch
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

# -------------------------------physics constants-------------------------------
acceleration_from_thrust = 20  # how much the acceleration increases from thrust
acceleration_from_gravity = -9.8
thrust_per_unit_fuel = 1  # how much thrust from 1 unit of fuel

# -------------------------------training-------------------------------
max_games = 50
num_of_games = 0
training_frequency = 5
randomize_constant = 50  # stops being random after [randomize_constant] many games
previous_height = 0
all_max_heights = []  # so i can see the progression if there is any
training_states = np.array([])
training_outputs = np.array([])

# -------------------------------preprocessing-------------------------------
max_acceleration = 100
max_velocity = 1000
max_height = 100000
max_fuel = 10
max_time = 1000

# -------------------------------model architecture-------------------------------
# try making the shape (None, 6) so I can put the entire rocket
# state/output history as 1 data point instead of putting many
model = models.Sequential([
    layers.Flatten(input_shape=(6,)),
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    # softmax is for classification of many different things
    # (like classifying between 10 different animals)
    # sigmoid compresses the value between 0 and 1 but not for classification
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------checkpoints-------------------------------
checkpoint_dir = '../../rocket_v1_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)


class Rocket:

    def __init__(self):
        # current state data
        self.velocity = 0
        self.height = 0
        self.has_left_ground = False
        self.fuel = max_fuel  # setting this to a lower value for my testing cases
        self.acceleration = 0

        # overall data
        self.time = 0  # not measured in seconds, measured in frames
        self.max_height = 0
        self.max_velocity = 0

        # training data
        self.states = np.array([])
        self.outputs = np.array([])

        launch(self, 0)

    def update(self, actions, squished_state):
        # this function updates the current state of the rocket depending on the ai output

        # recalculating fuel consumption when low on fuel
        thrust = actions['thrust']
        max_thrust = self.fuel * thrust_per_unit_fuel
        if thrust >= max_thrust:
            thrust = max_thrust
            self.fuel = 0
        else:
            self.fuel -= thrust / thrust_per_unit_fuel

        # updating velocity and height:
        self.acceleration = (thrust * acceleration_from_thrust) + acceleration_from_gravity
        if self.has_left_ground or self.acceleration > 0:
            # this keeps the rocket from gaining negative velocity and height while on the ground
            self.velocity += self.acceleration
            self.height += self.velocity

        # keeping values at or under max:
        self.time += 1
        if self.height >= max_height:
            print('rocket reached max height')
            self.height = max_height
        if self.velocity >= max_velocity:
            print('rocket reached max velocity')
            self.velocity = max_velocity
        if self.acceleration >= max_acceleration:
            print('rocket reached max velocity')
            self.acceleration = max_acceleration

        use_data = False
        if self.acceleration > 0:
            use_data = True
        elif self.acceleration == 0 and self.velocity > 0:
            use_data = True

        if use_data is True:
            # adding state and output to current rocket training data
            # self.states is a 2D array of squished states that can be added to the training data
            if len(self.states) == 0:
                self.states = np.array([squished_state])
                # if the array is empty we cannot v-stack
            else:
                self.states = np.vstack((self.states, squished_state))
                # v-stack keeps the dimensions of the 2d array, which we need for proper training data
            self.outputs = np.append(self.outputs, thrust)

        # updating overall data
        if self.height > self.max_height:
            self.max_height = self.height
        if self.velocity > self.max_velocity:
            self.max_velocity = self.velocity

        # changing state to be passed to display
        state = {'height': self.height,
                 'velocity': self.velocity,
                 'acceleration': self.acceleration,
                 'fuel': self.fuel}

        # updating crashes
        if not self.has_left_ground and self.height > 0:
            self.has_left_ground = True
        elif self.has_left_ground and self.height <= 0:
            # the rocket has crashed or took too long
            return state, True
        elif self.time > max_time:
            return state, True

        return state, False

    def fly(self):
        # this is where the ai will give an output

        actions = {}
        squished_state = self.squish_state()  # accessing current state of rocket for inputs
        thrust = model.predict(np.array([squished_state]))  # generating command based on state
        thrust = thrust[0][0]  # accessing the variable needed from the array(s) given
        thrust = randomize_output(thrust)
        actions['thrust'] = thrust

        return self.update(actions, squished_state)  # calling the update function to change the rockets current state

    def has_crashed(self):
        # the rocket has crashed and we are now moving onto another iteration
        # this is where we will fit the model to the training data

        global num_of_games
        global previous_height
        global training_states
        global training_outputs
        global model
        global all_max_heights

        # printing overall data for me
        print('rocket #', str(num_of_games+1), 'crashed')
        print('max height:', self.max_height)
        print('max velocity:', self.max_velocity)
        print('time:', self.time)

        all_max_heights.append(self.max_height)
        for state in self.states:
            if len(training_states) == 0:
                training_states = state
                # if the array is empty we cannot v-stack, so we add the fist state
            else:
                training_states = np.vstack((training_states, state))
                # we have to use v-stack like before to keep the 2d array of states
        for output in self.outputs:
            training_outputs = np.append(training_outputs, output)

        if num_of_games is not 0 and num_of_games % training_frequency is 0 and len(training_states) is not 0:
            # we only train every few games, and not every rocket is added to training data,
            # so we first check if we should train after the current game,
            # and we check if there is data to train with
            train(training_states, training_outputs)
            # resetting the training arrays to not over train
            training_states = np.array([])
            training_outputs = np.array([])

        # changing the average_height for determining if future rockets are viable for training
        # previous_height = self.max_height

        # preparing for next launch
        # resetting all values
        self.velocity = 0
        self.height = 0
        self.has_left_ground = False
        self.fuel = max_fuel
        self.acceleration = 0

        # resetting rocket overall data
        self.time = 0
        self.max_height = 0
        self.max_velocity = 0

        # resetting rocket training data
        self.states = np.array([])
        self.outputs = np.array([])

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
        if self.has_left_ground:
            left_ground_bool = 1

        cur_height = self.height / max_height
        cur_velocity = self.velocity / max_velocity
        cur_acceleration = self.acceleration / max_acceleration
        cur_time = self.time / max_time
        cur_fuel = self.fuel / max_fuel

        return np.array([left_ground_bool, cur_height, cur_velocity, cur_acceleration, cur_fuel, cur_time])


def train(inputs, outputs):
    # trains the model to the current training data and saves it
    global model
    model.fit(inputs, outputs, epochs=10, verbose=1, callbacks=[checkpoint_callback])
    model.save('rocket_v1.h5')


def load():
    # loads a model, wow
    global model
    model = tf.keras.models.load_model('rocket_v1.h5')


def randomize_output(predicted_output):
    # this adds some randomization to the beginning of training
    # because without it the ai will keep the same output constantly and wont learn
    rand = randint(0, 100)
    random_rate = randomize_constant*(1-num_of_games/randomize_constant)
    if rand < random_rate:
        return abs(1-predicted_output)

    return predicted_output


myself = Rocket()
plt.subplot(3, 1, 1)
x = np.linspace(1, len(all_max_heights), len(all_max_heights))
plt.plot(x, all_max_heights, 'o-', color='r')
plt.xlabel("Rocket")
plt.ylabel("Max Height")
plt.title('All Max Heights')
plt.savefig('rocket_v1.png')
