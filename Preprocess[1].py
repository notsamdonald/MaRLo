import gym
import cv2
import time, datetime
import numpy as np
from gym import ObservationWrapper
from gym.wrappers import FrameStack

class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Each frame is converted to PyTorch tensors
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        #>>> import gym
        #>>> env = gym.make('PongNoFrameskip-v0')
        #>>> env = FrameStack(env, 4)
        #>>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()


class BufferWrapper(gym.ObservationWrapper):
    """
    Only every k-th frame is collected by the buffer
    """

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class FrameSkip(gym.Wrapper):
    def __init__(self, env=None, frames=1, limit=False, render_game=False):
        """
        Wrapper for open AI environment to skip frames, by applying the same acton within all

        Args:
            env (TODO - complete description):
            frames (int): number of frames to be skipped
        """
        super(FrameSkip, self).__init__(env)
        self.frames = frames
        self.tic = 0
        self.limit = limit
        self.render_game = render_game

    def step(self, action):
        """
        Modifying the default step action

        Args:
            action (str): Action string (eg 'right')

        Returns:
            (tuple): modified tuple containing env.step(action) response:

                obs (TODO - complete description)
                total_reward (int): cumulative reward over skipped frames
                done (bool): Flag for if the environment is complete
                info (TODO - complete description)

        """

        net_reward = 0
        for _ in range(self.frames):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            net_reward += reward
            self.toc = time.perf_counter()

            if self.render_game:
                self.render()

            if self.tic!=0 and self.limit != False:
                time.sleep(max(self.limit-(self.toc-self.tic),0))
            self.tic = time.perf_counter()

            if done:
                break

        return obs, net_reward, done, info


class Rescale(gym.ObservationWrapper):
    """
    Downsamples/Rescales each frame to size 84x84 with greyscale
    """

    # TODO - tidy up (have re-written most by not the first section/names)

    def __init__(self, env=None, shape=84):
        super(Rescale, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return Rescale.process(obs)

    @staticmethod
    def process(frame):


        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        # https://e2eml.school/convert_rgb_to_grayscale.html  (greyscale approximation)

        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        img = 0.299 * r + 0.587 * g + 0.114 * b


        resized_screen = cv2.resize(img[40:222, :], (84, 84), interpolation=cv2.INTER_AREA)

        resized_screen *= 1.0 / resized_screen.max()  # Marginally faster than divide (https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range)


        resized_screen = np.reshape(resized_screen, [84, 84, 1])

        return resized_screen

