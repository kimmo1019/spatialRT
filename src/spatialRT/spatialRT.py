import tensorflow as tf
from .model import Generator, Encoder, Discriminator
import numpy as np
from .util import Mixture_sampler, Spatial_DLPFC_sampler
import dateutil.tz
import datetime
import sys
import copy
import os
import json
tf.keras.utils.set_random_seed(123)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


class spatialRT(object):
    """ spatialRT model.
    """
    def __init__(self, params):
        super(spatialRT, self).__init__()
        self.params = params
        self.g_net = Generator(input_dim=params['z_dim']+params['nb_classes']+2,z_dim = params['z_dim'], output_dim = params['x_dim'],nb_layers=10, nb_units=512, concat_every_fcl=False)
        self.h_net = Encoder(input_dim=params['x_dim']+2, output_dim = params['z_dim']+params['nb_classes'],feat_dim=params['z_dim'],nb_layers=10,nb_units=256)
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',nb_layers=2,nb_units=256)
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',nb_layers=2,nb_units=256)
        self.g_h_optimizer = tf.keras.optimizers.Adam(params['lr']/2, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(params['lr']/2, beta_1=0.5, beta_2=0.9)
        self.pre_d_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.pre_g_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Mixture_sampler(nb_classes=params['nb_classes'],N=10000,dim=params['z_dim'],sd=1)
        self.x_sampler = Spatial_DLPFC_sampler(slice_id=params['id'])
        self.initilize_nets()
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "checkpoints/%s" % self.timestamp
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "results/%s" % self.timestamp
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   
        ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   h_net = self.h_net,
                                   dz_net = self.dz_net,
                                   dx_net = self.dx_net,
                                   g_h_optimizer = self.g_h_optimizer,
                                   d_optimizer = self.d_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        
    def get_config(self):
        return {
                "params": self.params,
        }
    
    def initilize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim']+self.params['nb_classes']+2)))
        self.h_net(np.zeros((1, self.params['x_dim']+2)))
        self.dz_net(np.zeros((1, self.params['z_dim'])))
        self.dx_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())
            print(self.dz_net.summary())
            print(self.dx_net.summary())

    @tf.function
    def train_gen_step(self, data_z, data_z_onehot, data_coor, data_x, adj_hexigon):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
                Fourth item: 0: update generators, 1: update discriminators
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            data_x = tf.cast(data_x, tf.float32)
            data_z_combine = tf.concat([data_z, data_z_onehot, data_coor], axis=-1)
            data_x_ = self.g_net(data_z_combine)

            data_z_latent_, data_z_onehot_ = self.h_net(tf.concat([data_x, data_coor], axis=-1))
            data_z_ = data_z_latent_[:,:self.params['z_dim']]

            data_z_latent__, data_z_onehot__ = self.h_net(tf.concat([data_x_, data_coor], axis=-1))
            data_z__ = data_z_latent__[:,:self.params['z_dim']]
            data_z_logits__ = data_z_latent__[:,self.params['z_dim']:]
            
            data_z_combine_ = tf.concat([data_z_, data_z_onehot_,data_coor], axis=-1)
            data_x__ = self.g_net(data_z_combine_)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)

            cce = tf.keras.losses.CategoricalCrossentropy()
            entropy_loss = cce(data_z_onehot_, data_z_onehot_)

            data_z_onehot1_ = tf.reshape(data_z_onehot_, (1,self.params['bs'],self.params['nb_classes']))
            data_z_onehot2_ = tf.reshape(data_z_onehot_, (self.params['bs'],1,self.params['nb_classes']))
            distance_y = tf.norm(data_z_onehot1_- data_z_onehot2_, ord=1, axis=2) #(bs, bs)
            #smoothness loss
            tv_loss = tf.reduce_sum(distance_y * adj_hexigon)/(self.params['bs']*5)

            
            CE_loss_z = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_z_onehot, logits=data_z_logits__))
            
            g_loss_adv = -tf.reduce_mean(data_dx_)
            h_loss_adv = -tf.reduce_mean(data_dz_)
            g_h_loss = g_loss_adv+h_loss_adv+self.params['alpha']*(l2_loss_x + l2_loss_z)+ \
                        self.params['beta']*CE_loss_z + tv_loss + entropy_loss
            
        # Calculate the gradients for generators and discriminators
        g_h_gradients = gen_tape.gradient(g_h_loss, self.g_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_h_optimizer.apply_gradients(zip(g_h_gradients, self.g_net.trainable_variables+self.h_net.trainable_variables))
        return g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss, tv_loss, entropy_loss

    @tf.function
    def train_disc_step(self, data_z, data_z_onehot, data_coor, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: latent onehot tensor with shape [batch_size, nb_classes].
                Third item: obervation data with shape [batch_size, x_dim].
                Fourth item: 0: update generators, 1: update discriminators
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            data_x = tf.cast(data_x, tf.float32)
            data_z_combine = tf.concat([data_z, data_z_onehot,data_coor], axis=-1)
            #print('1',data_z, data_z_onehot, data_x, data_z_combine)
            data_x_ = self.g_net(data_z_combine)
            data_z_latent_, data_z_onehot_ = self.h_net(tf.concat([data_x, data_coor], axis=-1))
            data_z_ = data_z_latent_[:,:self.params['z_dim']]
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            dx_loss = -tf.reduce_mean(data_dx) + tf.reduce_mean(data_dx_)
            
            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
            data_dx_hat = self.dx_net(data_x_hat)
            grad_x = tf.gradients(data_dx_hat, data_x_hat)[0] #(bs,x_dim)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            
            d_loss = dx_loss + dz_loss + self.params['gamma']*(gpz_loss+gpx_loss)
        
        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        return dx_loss, dz_loss, d_loss

    @tf.function
    def pretrain_gen_step(self, data_x, data_coor, data_one_hot):
        with tf.GradientTape(persistent=True) as pre_g_tape:
            data_z_latent_, data_z_onehot_ = self.h_net(tf.concat([data_x, data_coor], axis=-1))
            data_z_ = data_z_latent_[:,:self.params['z_dim']]
            data_z_logits_ = data_z_latent_[:,self.params['z_dim']:]
            CE_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=data_one_hot, logits=data_z_logits_))
            data_dz_ = self.dz_net(data_z_)
            pre_g_loss_adv = -tf.reduce_mean(data_dz_)
            pre_g_loss = pre_g_loss_adv +  CE_loss
        pre_g_gradients = pre_g_tape.gradient(pre_g_loss, self.h_net.trainable_variables)
        # Apply the gradients to the optimizer
        self.pre_g_optimizer.apply_gradients(zip(pre_g_gradients, self.h_net.trainable_variables))
        return pre_g_loss_adv, CE_loss

    @tf.function
    def pretrain_disc_step(self, data_x, data_coor, data_z):
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as pre_d_tape:
            data_z_latent_, data_z_onehot_ = self.h_net(tf.concat([data_x, data_coor], axis=-1))
            data_z_ = data_z_latent_[:,:self.params['z_dim']]

            data_dz_ = self.dz_net(data_z_)
            data_dz = self.dz_net(data_z)
            pre_d_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)

            #gradient penalty for z
            data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
            data_dz_hat = self.dz_net(data_z_hat)
            grad_z = tf.gradients(data_dz_hat, data_z_hat)[0] #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            pre_d_loss += gpz_loss*self.params['gamma']
        # Calculate the gradients for generators and discriminators
        pre_d_gradients = pre_d_tape.gradient(pre_d_loss, self.dz_net.trainable_variables)
        # Apply the gradients to the optimizer
        self.pre_d_optimizer.apply_gradients(zip(pre_d_gradients, self.dz_net.trainable_variables))
        return pre_d_loss

    def pretrain(self):
        from sklearn.metrics import accuracy_score
        batches_per_eval = 100
        batch_size = self.params['bs']
        for batch_idx in range(8000):
            for _ in range(5):
                batch_z, _ = self.z_sampler.train(batch_size)
                batch_x, batch_adj, batch_adj_hexigon, batch_x_neighbors, batch_adj_neighbors, batch_coor = self.x_sampler.get_batch(batch_size) 
                pre_d_loss = self.pretrain_disc_step(batch_x, batch_coor, batch_z)

            batch_z, batch_z_onehot = self.z_sampler.train(batch_size)
            batch_x, batch_adj, batch_adj_hexigon, batch_x_neighbors, batch_adj_neighbors, batch_coor, batch_one_hot = self.x_sampler.get_batch(batch_size, use_label=True)            
            pre_g_loss_adv, CE_loss = self.pretrain_gen_step(batch_x, batch_coor, batch_one_hot)
            if batch_idx % batches_per_eval == 0:
                print("Batch_idx [%d] pre_d_loss [%.4f] pre_g_loss_adv [%.4f] CE_loss [%.4f]" %(batch_idx, pre_d_loss, pre_g_loss_adv, CE_loss))
                data_x, _, data_psedo = self.x_sampler.load_all(use_label=True)
                data_coor = self.x_sampler.coor
                #_, data_z_onehot_ = self.h_net(tf.concat([data_x, data_coor], axis=-1))
                _, data_z_onehot_ = self.h_net(np.concatenate([data_x, data_coor], axis=-1))
                label_pre = np.argmax(data_z_onehot_, axis=1)
                acc = accuracy_score(data_psedo, label_pre)
                print("Batch_idx [%d] Accuracy [%.4f]"%(batch_idx, acc))

    def train(self): 
        print('use spatialRT model')
        print('pretrain')
        self.pretrain()
        batches_per_eval = 100
        ratio = 0.2
        batch_size = self.params['bs']
        weights = np.ones(self.params['nb_classes'], dtype=np.float64) / float(self.params['nb_classes'])
        last_weights = np.ones(self.params['nb_classes'], dtype=np.float64) / float(self.params['nb_classes'])
        diff_history=[]
        for batch_idx in range(self.params['nb_batches']):
            for _ in range(6):
                batch_z, batch_z_onehot = self.z_sampler.train(batch_size, weights)
                #batch_z, batch_z_onehot = self.z_sampler.train(batch_size)
                batch_x, batch_adj, batch_adj_hexigon, batch_x_neighbors, batch_adj_neighbors, batch_coor = self.x_sampler.get_batch(batch_size) 
                dx_loss, dz_loss, d_loss = self.train_disc_step(batch_z, batch_z_onehot, batch_coor, batch_x)
            batch_z, batch_z_onehot = self.z_sampler.train(batch_size, weights)
            #batch_z, batch_z_onehot = self.z_sampler.train(batch_size)
            
            batch_x, batch_adj, batch_adj_hexigon, batch_x_neighbors, batch_adj_neighbors, batch_coor = self.x_sampler.get_batch(batch_size)            
            g_loss_adv, h_loss_adv, l2_loss_x, l2_loss_z, CE_loss_z, g_h_loss, tv_loss, entropy_loss = self.train_gen_step(batch_z, batch_z_onehot,batch_coor,batch_x, batch_adj_hexigon)
            if batch_idx % batches_per_eval == 0:
                #print(batch_idx, g_loss_adv, h_loss_adv, CE_loss_z, l2_loss_z, l2_loss_x, g_h_loss, dz_loss, dx_loss, d_loss)
                print("Batch_idx [%d] g_loss_adv [%.4f] h_loss_adv [%.4f] CE_loss_z [%.4f] \
                    l2_loss_z [%.4f] l2_loss_x [%.4f] g_h_loss [%.4f] dz_loss [%.4f] \
                        dx_loss [%.4f] d_loss [%.4f] tv_loss [%.4f], entropy_loss [%.4f]"  % (batch_idx, 
                        g_loss_adv, h_loss_adv, CE_loss_z, l2_loss_z, l2_loss_x, 
                        g_h_loss, dz_loss, dx_loss, d_loss, tv_loss, entropy_loss))
                self.evaluate(batch_idx)
                #ckpt_save_path = self.ckpt_manager.save()
                #print ('Saving checkpoint for epoch {} at {}'.format(batch_idx,ckpt_save_path))
                tol = 0.02
                estimated_weights = self.estimate_weights()
                weights = ratio*weights + (1-ratio)*estimated_weights
                weights = weights/np.sum(weights)
                diff_weights = np.mean(np.abs(last_weights-weights))
                diff_history.append(diff_weights)
                if np.min(weights)<tol:
                    weights = self.adjust_tiny_weights(weights,tol)
                last_weights = copy.copy(weights)
            if len(diff_history)>100 and np.mean(diff_history[-10:]) < 5e-3 and batch_idx>30000:
                print('Reach a stable cluster')
                self.evaluate(batch_idx)
                sys.exit()

    def adjust_tiny_weights(self,weights,tol):
        idx_less = np.where(weights<tol)[0]
        idx_greater = np.where(weights>=tol)[0]
        weights[idx_less] = np.array([np.random.uniform(2*tol,1./self.params['nb_classes']) for item in idx_less])
        weights[idx_greater] = weights[idx_greater]*(1-np.sum(weights[idx_less]))/np.sum(weights[idx_greater])
        return weights   

    def estimate_weights(self):
        data_x, data_adj = self.x_sampler.load_all()
        data_coor = self.x_sampler.coor
        data_z_, data_z_onehot_ = self.h_net.predict(tf.concat([data_x, data_coor], axis=-1))
        label_infer = np.argmax(data_z_onehot_, axis=1)
        weights = np.empty(self.params['nb_classes'], dtype=np.float32)
        for i in range(self.params['nb_classes']):
            weights[i] = list(label_infer).count(i)  
        return weights/float(np.sum(weights)) 

    def evaluate(self,batch_idx):
        data_x, data_adj = self.x_sampler.load_all()
        data_coor = self.x_sampler.coor
        label_true = self.x_sampler.label_annot
        data_z_, data_z_onehot_ = self.h_net.predict(tf.concat([data_x, data_coor], axis=-1))
        label_infer = np.argmax(data_z_onehot_, axis=1)
        from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score,homogeneity_score
        nmi = normalized_mutual_info_score(label_true, label_infer)
        ari = adjusted_rand_score(label_true, label_infer)
        homo = homogeneity_score(label_true,label_infer)
        print('spatialRT: NMI = {}, ARI = {}, Homogeneity = {}'.format(nmi,ari,homo))
        f = open('%s/log.txt'%self.save_dir,'a+')
        f.write('NMI = {}\tARI = {}\tHomogeneity = {}\t batch_idx = {}\n'.format(nmi,ari,homo,batch_idx))
        f.close()
        np.savez('{}/data_at_{}.npz'.format(self.save_dir, batch_idx+1),data_z_,data_z_onehot_,label_true)
