from absl import app
from absl import flags
from prescience.labelling import get_property
import gym

flags.DEFINE_string('env', 'FreewayNoFrameskip-v4', 'Env string')
flags.DEFINE_string('prop', 'hit', 'Name of property to check')
flags.DEFINE_bool('human', False, 'whether to test using human play')
flags.DEFINE_float('pause', 1, 'time to pause on violated property')
flags.DEFINE_integer('grace', 30, 'pauses will not be triggered until this many frames have passed since last pause')
flags.DEFINE_float('speed', 0, 'time waiting per frame')
flags.DEFINE_integer('max_frames', 10000, 'max number of frames before ending')
flags.DEFINE_integer('fps', 60, 'Max frame per second for human play')
FLAGS = flags.FLAGS


def main(_):
    env = gym.make(FLAGS.env+'NoFrameskip-v4')
    labeller = get_property(env, FLAGS.prop)
    if FLAGS.human:
        labeller.test_human(fps=FLAGS.fps)
    else:
        labeller.test_random(pause=FLAGS.pause, grace=FLAGS.grace, max_steps=FLAGS.max_frames, speed=FLAGS.speed)


if __name__ == "__main__":
    app.run(main)
