import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import exarl as erl

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPG(erl.ExaAgent):
    is_learner: bool

    def __init__(self, env):

        # Distributed variables
        self.is_learner = False
        self.epsilon = 0 # Not used by agent but required by the learner class
        # Environment space and action parameters
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]

        self.gamma = 0.99
        self.tau = 0.005
        std_dev = 0.02
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        buffer_capacity = 100000
        batch_size = 64
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.memory = self.state_buffer # BAD

        # Required by the learner
        if self.is_learner:
            self.actor_model = self.get_actor()
            self.critic_model = self.get_critic()

        # Every agent needs this
        self.target_critic = self.get_critic()
        self.target_actor = self.get_actor()

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done

        self.buffer_counter += 1

    @tf.function
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        outputs = outputs #* self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.num_states)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.num_actions)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def generate_data(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        yield state_batch, action_batch, reward_batch, next_state_batch

    def train(self, batch):
        # Get sampling range
        # record_range = min(self.buffer_counter, self.buffer_capacity)
        # # Randomly sample indices
        # batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        # state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        # action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        # reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        # reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        # next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        #if self.buffer_counter >= self.batch_size:
        self.update_grad(batch[0], batch[1], batch[2], batch[3])

    def target_train(self):
        # Update the target model
        #if self.buffer_counter >= self.batch_size:
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def action(self, state):#, noise_object):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.target_actor(tf_state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions_wn = sampled_actions.numpy()*(1 + noise)
        print('sampled_actions/noise/sampled_actions_wn: {}/{}/{}'.format(sampled_actions,noise,sampled_actions_wn))
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions_wn, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)], 1

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def set_learner(self):
        self.is_learner = True

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def load(self):
        print("Implement load method in ddpg.py")

    def save(self):
        print("Implement load method in ddpg.py")

    def monitor(self):
        print("Implement monitor method in ddpg.py")

    def set_agent(self):
        print("Implement set_agent method in ddpg.py")

    def print_timers(self):
        print("Implement print_timers method in ddpg.py")
