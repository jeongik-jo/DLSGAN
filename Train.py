import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


@tf.function
def _gan_train_step(dis: kr.Model, gen: kr.Model, real_imgs: tf.Tensor):
    ltn_scl_vecs = hp.get_ltn_scl_vecs()
    batch_size = real_imgs.shape[0]

    ltn_vecs = hp.ltn_dist_func(batch_size)
    fake_imgs = gen(ltn_vecs * ltn_scl_vecs)

    with tf.GradientTape() as dis_tape:
        with tf.GradientTape() as reg_tape:
            reg_tape.watch(real_imgs)
            real_adv_vals, real_ltn_vecs, _ = dis(real_imgs)
        reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reg_tape.gradient(real_adv_vals, real_imgs)), axis=[1, 2, 3]))
        fake_adv_vals, rec_ltn_vecs, rec_ltn_logvars = dis(fake_imgs)

        ltn_diff = tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs)
        rec_ltn_traces = rec_ltn_vecs

        if hp.use_logvar:
            enc_loss = tf.reduce_mean(rec_ltn_logvars + ltn_diff / (tf.exp(rec_ltn_logvars) + 1e-7))
        else:
            enc_loss = tf.reduce_mean(ltn_diff)

        dis_adv_loss = tf.reduce_mean(tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals))
        dis_loss = dis_adv_loss + hp.d_enc_w * enc_loss + hp.reg_w * reg_loss

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_tape)

    with tf.GradientTape() as gen_tape:
        ltn_vecs = hp.ltn_dist_func(batch_size)
        fake_imgs = gen(ltn_vecs * ltn_scl_vecs)
        fake_adv_vals, rec_ltn_vecs, rec_ltn_logvars = dis(fake_imgs)

        ltn_diff = tf.square((ltn_vecs - rec_ltn_vecs) * ltn_scl_vecs)
        rec_ltn_traces = tf.concat([rec_ltn_traces, rec_ltn_vecs], axis=0)

        if hp.use_logvar:
            enc_loss = tf.reduce_mean(rec_ltn_logvars + ltn_diff / (tf.exp(rec_ltn_logvars) + 1e-7))
        else:
            enc_loss = tf.reduce_mean(ltn_diff)

        gen_adv_loss = tf.reduce_mean(tf.nn.softplus(-fake_adv_vals))
        gen_loss = gen_adv_loss + hp.g_enc_w * enc_loss

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, tape=gen_tape)

    hp.ltn_var_trace.assign(hp.ltn_var_trace * hp.ltn_var_decay_rate +
                            tf.reduce_mean(tf.square(rec_ltn_traces), axis=0) * (1.0 - hp.ltn_var_decay_rate))

    results = {
        'real_adv_val': tf.reduce_mean(real_adv_vals), 'fake_adv_val': tf.reduce_mean(fake_adv_vals),
        'reg_loss': reg_loss, 'enc_loss': enc_loss
    }
    return results


@tf.function
def _enc_train_step(dis: kr.Model, gen: kr.Model):
    batch_size = hp.batch_size
    ltn_vecs = hp.ltn_dist_func(batch_size)
    fake_imgs = gen(ltn_vecs)

    with tf.GradientTape() as dis_tape:
        fake_adv_vals, rec_ltn_vecs, rec_ltn_logvars = dis(fake_imgs)
        ltn_diff = tf.square(ltn_vecs - rec_ltn_vecs)

        if hp.use_logvar:
            enc_loss = tf.reduce_mean(rec_ltn_logvars + ltn_diff / (tf.exp(rec_ltn_logvars) + 1e-7))
        else:
            enc_loss = tf.reduce_mean(ltn_diff)

        dis_loss = hp.d_enc_w * enc_loss

    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, tape=dis_tape)

    results = {
        'enc_loss': enc_loss
    }
    return results


def train(dis: kr.Model, gen: kr.Model, dataset):
    results = {}
    for data in dataset:
        if hp.train_only_enc:
            batch_results = _enc_train_step(dis, gen)
        else:
            batch_results = _gan_train_step(dis, gen, data)
        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.convert_to_tensor(results[key]), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    temp_results['ltn_ent'] = hp.get_ltn_ent()
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results

