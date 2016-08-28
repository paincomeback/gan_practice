# Generative Adversarial Network practice
This is a practice of GAN

**Theano and lasagne is needed**

* origin.py is original GAN
* alternative.py is alternative update rule GAN by [this blog post](http://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/)

**It turns out that alternative update rule is just to remove softplus function at the end of the generator**

After the iteration, each learning process save images and you can calculate inception score using `sc.py`
