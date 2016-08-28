import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams
import plotting

from theano.compile.nanguardmode import NanGuardMode

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
parser.add_argument('--name', type=str, default='./mnist_alt/my_alternative')
args = parser.parse_args()
print(args)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# specify generative model
noise = theano_rng.uniform(size=(args.batch_size, 100))
gen_layers = [ll.InputLayer(shape=(args.batch_size, 100), input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = ll.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
layers = [ll.InputLayer(shape=(None, 28**2))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=1000, W=Normal(0.05))))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=500, W=Normal(0.05))))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=250, W=Normal(0.05))))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=250, W=Normal(0.05))))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=250, W=Normal(0.05))))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.weight_norm(ll.DenseLayer(layers[-1], num_units=1, W=Normal(0.05), nonlinearity=None), init_stdv=0.1, train_g=True))

# costs
labels = T.ivector()
x_unl = T.matrix()

temp = ll.get_output(gen_layers[-1], init=True)
temp = ll.get_output(layers[-1], x_unl, deterministic=False, init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_before_softmax_unl = ll.get_output(layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(layers[-1], gen_dat, deterministic=False)

loss_unl = T.mean(T.nnet.softplus(-output_before_softmax_unl)) + T.mean(T.nnet.softplus(-output_before_softmax_gen) + output_before_softmax_gen)

loss_gen = T.mean(-output_before_softmax_gen)

# Theano functions for training and testing
lr = T.scalar()
disc_params = ll.get_all_params(layers, trainable=True)
#disc_param_updates = nn.adam_updates(disc_params, loss_unl, lr=lr, mom1=0.5)
disc_param_updates = lasagne.updates.adam(loss_unl, disc_params, learning_rate=lr, beta1=0.5)

aa = []
for key in disc_param_updates:
    aa.append((key, disc_param_updates[key]))

disc_param_updates = aa;

disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = ll.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_unl], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_unl,lr], outputs=loss_unl, updates=disc_param_updates+disc_avg_updates)
#train_batch_disc = th.function(inputs=[x_unl,lr], outputs=loss_unl, updates=disc_param_updates+disc_avg_updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
train_batch_gen = th.function(inputs=[lr], outputs=loss_gen, updates=gen_param_updates)
samplefun = th.function(inputs=[],outputs=gen_dat)

# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
nr_batches_test = int(testx.shape[0]/args.batch_size)

init_param(trainx[:500]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.003
f = open("./" + args.name + ".log", "w")
for epoch in range(300):
    begin = time.time()

    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]

    # train
    loss_unl = 0.
    loss_gen = 0.
    for t in range(nr_batches_train):
        ran_from = t * args.batch_size
        ran_to = (t+1) * args.batch_size
        lu = train_batch_disc(trainx_unl[ran_from:ran_to],lr)

        loss_unl += lu
        e = train_batch_gen(lr)
        loss_gen += e
        #e = train_batch_gen(trainx_unl[t*args.batch_size:(t+1)*args.batch_size], lr)
    loss_unl /= nr_batches_train
    loss_gen /= nr_batches_train

    
    # report
    print("Iteration %d, time = %ds, loss_unl = %.4f, loss_gen = %.4f" %(epoch, time.time()-begin, loss_unl, loss_gen))
    f.write("Iteration %d, time = %ds, loss_unl = %.4f, loss_gen = %.4f\n" %(epoch, time.time()-begin, loss_unl, loss_gen))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.reshape(sample_x, (args.batch_size, 28, 28))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='MNIST samples')

    if epoch < 10:
        c = '00' + str(epoch)
    elif epoch < 100:
        c = '0' + str(epoch)
    else:
        c = str(epoch)

    plotting.plt.savefig(args.name + '_' + c + '.png')
    plotting.plt.close('all')
    
    # save params
    #np.savez('disc_params.npz',*[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz',*[p.get_value() for p in gen_params])

f.close()
