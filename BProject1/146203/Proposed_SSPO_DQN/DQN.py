import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from Proposed_SSPO_DQN import SSPO

# Constants
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '2x256'
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

# Suppress TensorFlow logging
import logging
tf.get_logger().setLevel(logging.ERROR)

class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing for better view

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[2]
        env[self.enemy.x][self.enemy.y] = self.d[3]
        env[self.player.x][self.player.y] = self.d[1]
        img = Image.fromarray(env, 'RGB')
        return img

 # Own Tensorboard class
    class ModifiedTensorBoard(TensorBoard):

        # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
        def __init__(self, **kwargs):
            super( ).__init__(**kwargs)
            self.step = 1
            self.writer = tf.summary.FileWriter(self.log_dir)

        # Overriding this method to stop creating default log writer
        def set_model(self, model):
            pass

        # Overrided, saves logs with our step number
        # (otherwise every .fit() will start writing from 0th step)
        def on_epoch_end(self, epoch, logs=None):
            self.update_stats(**logs)

        # Overrided
        # We train for one batch only, no need to save anything at epoch end
        def on_batch_end(self, batch, logs=None):
            pass

        # Overrided, so won't close writer
        def on_train_end(self, _):
            pass

        # Custom method for saving own metrics
        # Creates writer, writes custom metrics and closes writer
        def update_stats(self, **stats):
            self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self, train_data, train_label, test_data, test_label, pred):
        self.model = self.create_model(train_data, train_label, test_data, test_label, pred)

    def create_model(self, train_data, train_label, test_data, test_label, pred):
      # Normalize the input data
      train_data = np.array(train_data).astype(np.float32)
      test_data = np.array(test_data).astype(np.float32)
      train_data /= np.max(train_data) if np.max(train_data) != 0 else 1
      test_data /= np.max(test_data) if np.max(test_data) != 0 else 1

      # Expand dimensions for Conv1D
      train_x = train_data[:, :, np.newaxis]
      test_x = test_data[:, :, np.newaxis]

      train_y = to_categorical(train_label)
      test_y = to_categorical(test_label)

      # Build the CNN model
      model = Sequential()
      model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
      model.add(Conv1D(64, kernel_size=3, activation='relu'))
      model.add(Dropout(0.5))
      model.add(MaxPooling1D(pool_size=1))
      model.add(Flatten())
      model.add(Dense(100, activation='relu'))
      model.add(Dense(50, activation='relu'))
      model.add(Dense(train_y.shape[1], activation='softmax'))
      model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

      # Get initial weights and total weight count
      init_weights = model.get_weights()
      total_weights = sum(np.prod(w.shape) for w in init_weights)

      # Call SSPO optimizer with expected weight count
      flat_opt_weights = SSPO.algm(total_weights=total_weights)
      flat_opt_weights = np.array(flat_opt_weights).flatten()

      if flat_opt_weights.ndim != 1:
          raise ValueError("SSPO.algm() must return a 1D array of weights.")
      if flat_opt_weights.size != total_weights:
          raise ValueError(f"Mismatch in number of weights. Expected: {total_weights}, Got: {flat_opt_weights.size}")

      # Reshape and set weights
      reshaped_weights = []
      pointer = 0
      for w in init_weights:
          size = np.prod(w.shape)
          reshaped_weights.append(flat_opt_weights[pointer:pointer+size].reshape(w.shape))
          pointer += size

      model.set_weights(reshaped_weights)

      # Train the model
      model.fit(train_x, train_y, epochs=5, batch_size=1000, verbose=0)

      # Predict and return class indices
      predictions = np.argmax(model.predict(test_x), axis=1)
      pred.extend(predictions.tolist())

      return predictions



# Calculate performance metrics
def cal_metrics(xx, yy, tpr, A, Tpr, Tnr):
    tr = tpr / 100
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr, stratify=yy)
    pred = []
    DQNAgent(x_train, y_train, x_test, y_test, pred)

    target = y_test
    predict = np.array(pred)

    tp, tn, fn, fp = 0, 0, 0, 0
    unique_classes = np.unique(yy)

    for c in unique_classes:
        tp += np.sum((target == c) & (predict == c))
        tn += np.sum((target != c) & (predict != c))
        fn += np.sum((target == c) & (predict != c))
        fp += np.sum((target != c) & (predict == c))

    tn /= len(unique_classes)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)  # avoid division by zero
    A.append(acc)
    Tpr.append(tp / (tp + fn + 1e-10))
    Tnr.append(tn / (tn + fp + 1e-10))