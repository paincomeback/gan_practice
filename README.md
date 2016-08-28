# gan_pracrice
This is a practice of gan

** Theano and lasagne is needed **

* origin.py is original GAN
* alternative.py is alternative update rule GAN by [this blog](http://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/)

** It turns out that alternative update rule is just adding softplus function at the end of the discriminator **

After the iteration, they save images and you can calculate inception score using `sc.py`
