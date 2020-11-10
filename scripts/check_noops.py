from absl import flags
from absl import app
import os

from prescience.model_check_noop import check_noops
from prescience import get_wrapped
flags.DEFINE_boolean('verbose',True,'Whether to print progress')
flags.DEFINE_integer('lookahead',0,'Number of frames to lookahead for shielding')
flags.DEFINE_string('env', 'FreewayNoFrameskip-v4', 'Which Atari env to test (note that the checker is designed around NoFramekip-v4 variants).')
flags.DEFINE_string('method', 'DQN', 'The method to use')
flags.DEFINE_integer('max_noops',10,'Number of traces to check')
flags.DEFINE_boolean('render', False, 'Render traces')
flags.DEFINE_string('prop', 'hit', 'name of property to check')
flags.DEFINE_integer('max_frames', 100000, 'maximum number of frames per episode')
flags.DEFINE_string('output', 'violations.csv', 'name of output file')
flags.DEFINE_string('model_path', '../models', 'path to model dir')
flags.DEFINE_integer('run_id', 1, 'which run to use (for uber model zoo)')
flags.DEFINE_integer('min_noops', 0, 'Start at this number of noops')
flags.DEFINE_boolean('stop_at_violation',False,'Stop checking further traces if a violation is found')
flags.DEFINE_boolean('record',False,'Record video')
FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.method in ['DQN-D','Rainbow-D','APEX','IMPALA-U','A2C']:
        import lucid
        from lucid.modelzoo.vision_base import Model
        from lucid.misc.io import show
        import lucid.optvis.objectives as objectives
        import lucid.optvis.param as param
        import lucid.optvis.transform as transform
        import lucid.optvis.render as render
        import tensorflow as tf

        from atari_zoo import MakeAtariModel
        from lucid.optvis.render import import_model
        model_name_map = {
                'A2C': 'a2c',
                'DQN-D': 'dqn',
                'Rainbow-D': 'rainbow',
                'IMPALA-U':'impala',
                'APEX': 'apex'
                }
        m = MakeAtariModel(model_name_map[FLAGS.method],FLAGS.env+'NoFrameskip-v4',1,tag='final')()
        if FLAGS.verbose:
            print('Loading Model-Zoo graph for %s on %s.' % (FLAGS.method,FLAGS.env))
        m.load_graphdef()
        print('Finished loading graph')
        config = tf.ConfigProto(
                device_count = {'GPU': 1}
            )
        config.gpu_options.allow_growth=True
        with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
            env = get_wrapped(FLAGS.env+'NoFrameskip-v4',FLAGS.method,FLAGS.prop)

            nA = env.action_space.n
            X_t = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))
            T = import_model(m,X_t,X_t)
            policy = T(m.layers[-1]['name'])
            order = tf.argsort(-policy,axis=-1)
            action_function=lambda x: sess.run([order],feed_dict={X_t:x[None]})[0][0]
            print(action_function(env.reset()))
            violation_list, noop_violation_list, reward_list = check_noops(FLAGS.env+'NoFrameskip-v4',FLAGS.method,FLAGS.prop,action_function,max_frames = FLAGS.max_frames, max_noops = FLAGS.max_noops,render = FLAGS.render,min_noops = FLAGS.min_noops,shield=FLAGS.lookahead,verbose=FLAGS.verbose,demand_full_safety=FLAGS.stop_at_violation,record=FLAGS.record)
    else:
        from prescience.agents import AtariAgent
        agent = AtariAgent(FLAGS.method,FLAGS.env+'NoFrameskip-v4',FLAGS.model_path)
        violation_list, noop_violation_list, reward_list = check_noops(FLAGS.env+'NoFrameskip-v4',FLAGS.method,FLAGS.prop,agent.action_order,max_frames = FLAGS.max_frames, max_noops = FLAGS.max_noops,render = FLAGS.render,min_noops = FLAGS.min_noops,shield=FLAGS.lookahead,verbose=FLAGS.verbose,demand_full_safety=FLAGS.stop_at_violation,record=FLAGS.record)

    info = [FLAGS.env,FLAGS.method,FLAGS.prop,FLAGS.max_frames,FLAGS.lookahead]
    info_noop = [FLAGS.env,'no-ops',FLAGS.prop,FLAGS.max_frames,FLAGS.lookahead]
    info_reward = [FLAGS.env, FLAGS.method, 'Reward', FLAGS.max_frames,FLAGS.lookahead]
    with open(FLAGS.output,"a") as f:
        if f.tell()==0:
            labels = ['Environment','Agent','Property','Max Frames','Lookahead']
            noops = [(str(x) + " noops") for x in range(FLAGS.max_noops)]
            f.write(format_list(labels + noops))
        f.write(format_list(info + violation_list))
        f.write(format_list(info_noop + noop_violation_list))
        f.write(format_list(info_reward + reward_list))
def format_list(ls):
    string = str(ls[0])
    for item in ls[1:]:
        string += ',' + str(item)
    return string + '\n'
if __name__=="__main__":
    app.run(main)

