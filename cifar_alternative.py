import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cifar10_data

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1)
parser.add_argument('--seed_data', default=1)
parser.add_argument('--count', default=400)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--data_dir', type=str, default='/home/mlg/ihcho/data/cifar-10-python')
parser.add_argument('--name', type=str, default='./cifar_alt/cifar_alternative')
args = parser.parse_args()
print(args)

# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# load CIFAR-10
trainx, trainy = cifar10_data.load(args.data_dir, subset='train')
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
testx, testy = cifar10_data.load(args.data_dir, subset='test')
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# specify generative model
noise_dim = (args.batch_size, 100)
noise = theano_rng.uniform(size=noise_dim)
gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4)))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
gen_dat = ll.get_output(gen_layers[-1])

# specify discriminative model
disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
disc_params = ll.get_all_params(disc_layers, trainable=True)

# costs
labels = T.ivector()
x_unl = T.tensor4()
temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_unl, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]

output_before_softmax_unl = ll.get_output(disc_layers[-1], x_unl, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)

loss_unl = T.mean(T.nnet.softplus(-output_before_softmax_unl)) + T.mean(T.nnet.softplus(-output_before_softmax_gen) + output_before_softmax_gen)

loss_gen = T.mean(-output_before_softmax_gen)

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
#disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_updates = lasagne.updates.adam(loss_unl, disc_params, learning_rate=lr, beta1=0.5)

aa = []
for key in disc_param_updates:
    aa.append((key, disc_param_updates[key]))

disc_param_updates = aa

disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
init_param = th.function(inputs=[x_unl], outputs=None, updates=init_updates) # data based initialization
#train_batch_disc = th.function(inputs=[x_lab,labels,x_unl,lr], outputs=[loss_lab, loss_unl, train_err], updates=disc_param_updates+disc_avg_updates)
train_batch_disc = th.function(inputs=[x_unl,lr], outputs=loss_unl, updates=disc_param_updates+disc_avg_updates)
samplefun = th.function(inputs=[],outputs=gen_dat)

# Theano functions for training the gen net
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
#train_batch_gen = th.function(inputs=[x_unl,lr], outputs=None, updates=gen_param_updates)
train_batch_gen = th.function(inputs=[lr], outputs=loss_gen, updates=gen_param_updates)

# //////////// perform training //////////////
f = open('./' + args.name + '.log', 'w')
for epoch in range(1200):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))

    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    
    if epoch==0:
        print(trainx.shape)
        init_param(trainx[:500]) # data based initialization

    # train
    loss_unl = 0.
    loss_gen = 0.
    for t in range(nr_batches_train):
        ran_from = t*args.batch_size
        ran_to = (t+1)*args.batch_size
        lu = train_batch_disc(trainx_unl[ran_from:ran_to],lr)
        loss_unl += lu
        
        e = train_batch_gen(lr)
        loss_gen += e

    loss_unl /= nr_batches_train
    loss_gen /= nr_batches_train
    
    # report
    print("Iteration %d, time = %ds, loss_unl = %.4f, loss_gen = %.4f" % (epoch, time.time()-begin, loss_unl, loss_gen))
    f.write("Iteration %d, time = %ds, loss_unl = %.4f, loss_gen = %.4f\n" % (epoch, time.time()-begin, loss_unl, loss_gen))
    sys.stdout.flush()

    # generate samples from the model
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')

    if epoch < 10:
        c = '000' + str(epoch)
    elif epoch < 100:
        c = '00' + str(epoch)
    elif epoch < 1000:
        c = '0' + str(epoch)
    else:
        c = str(epoch)

    plotting.plt.savefig(args.name + '_' + c + '.png')
    plotting.plt.close('all')

    # save params
    #np.savez('disc_params.npz', *[p.get_value() for p in disc_params])
    #np.savez('gen_params.npz', *[p.get_value() for p in gen_params])
imgst = plotting.img_stretch(img_bhwc)
imgst = (imgst * 255).astype(int)
np.savez('./cifar_alt/cifar_alternative_img.npz', imgst)
