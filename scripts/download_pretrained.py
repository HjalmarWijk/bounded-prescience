import chainer
import chainerrl
from chainerrl import misc
from urllib.error import HTTPError
from absl import app
from absl import flags

flags.DEFINE_string('alg', 'DQN', 'Algorithm')
flags.DEFINE_string('env', 'Freeway', 'Environment')
FLAGS = flags.FLAGS

def main(argv): 
    env = FLAGS.env + "NoFrameskip-v4"
    try:
        misc.download_model(FLAGS.alg,env,model_type="final")[0]
    except HTTPError:
        print("ERROR: Could not download %s for %s" % (alg,env))
if __name__ == "__main__":
    app.run(main)

